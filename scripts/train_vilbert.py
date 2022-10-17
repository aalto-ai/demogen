import argparse
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.utils.data import DataLoader, Subset
from positional_encodings.torch_encodings import PositionalEncoding1D
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger

from gscan_metaseq2seq.models.embedding import BOWEmbedding
from gscan_metaseq2seq.util.dataset import PaddingDataset, ReshuffleOnIndexZeroDataset
from gscan_metaseq2seq.util.load_data import load_data
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        hidden_size,
        output_size,
        nlayers,
        nhead,
        norm_first,
        pad_action_idx,
        dropout_p=0.1,
    ):
        #
        # Input
        #  hidden_size : number of hidden units in Transformer, and embedding size for output symbols
        #  output_size : number of output symbols
        #  nlayers : number of hidden layers
        #  dropout_p : dropout applied to symbol embeddings and Transformers
        #
        super().__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.pos_encoding = PositionalEncoding1D(hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.pad_action_idx = pad_action_idx
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                dim_feedforward=hidden_size * 4,
                dropout=dropout_p,
                nhead=nhead,
                norm_first=norm_first,
                activation='gelu'
            ),
            num_layers=nlayers,
        )
        self.norm_first = norm_first
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, encoder_outputs, encoder_padding):
        # Run batch decoder forward for a single time step.
        #
        # Input
        #  input: LongTensor of length batch_size x seq_len (left-shifted targets)
        #  memory: encoder state
        #
        # Output
        #   output : unnormalized output probabilities, batch_size x output_size
        #
        # Embed each input symbol
        # state, state_padding_bits = extract_padding(state)
        input_padding_bits = inputs == self.pad_action_idx

        embedding = self.embedding(inputs)  # batch_size x hidden_size
        embedding = self.embedding_projection(
            torch.cat([embedding, self.pos_encoding(embedding)], dim=-1)
        )
        embedding = self.dropout(embedding)

        decoded = self.decoder(
            tgt=embedding.transpose(0, 1),
            memory=encoder_outputs,
            memory_key_padding_mask=encoder_padding,
            tgt_key_padding_mask=input_padding_bits,
            tgt_mask=torch.triu(
                torch.full((inputs.shape[-1], inputs.shape[-1]), float("-inf")),
                diagonal=1,
            ).to(inputs.device),
        ).transpose(0, 1)

        return self.out(decoded)


class StateCNN(nn.Module):
    def __init__(
        self,
        n_input_channels,
        n_output_channels,
        emb_channels,
        conv_kernel_sizes,
        dropout_p,
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=n_input_channels,
                    out_channels=n_output_channels,
                    kernel_size=size,
                    padding="same",
                )
                for size in conv_kernel_sizes
            ]
        )
        self.mlp = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(len(conv_kernel_sizes) * n_output_channels, emb_channels),
        )
        self.residual = nn.Linear(n_input_channels, emb_channels)

    def forward(self, x):
        orig = x
        # NHWC => NCWH => NCHW
        x = x.transpose(-1, -3).transpose(-1, -2)
        x_multiscale = [layer(x) for layer in self.conv_layers]
        x_multiscale = torch.cat(x_multiscale, dim=-3)
        # NCHW => NWHC => NHWC
        x_multiscale = x_multiscale.transpose(-1, -3).transpose(-2, -3)

        # Original implementation does not use a residual connection, but it
        # probably should
        return self.mlp(x_multiscale) + self.residual(orig)


class TransformerMLP(nn.Module):
    def __init__(self, emb_dim, ff_dim, norm_first, dropout_p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_dim, ff_dim), nn.Dropout(dropout_p))
        self.norm = nn.LayerNorm(emb_dim if norm_first else ff_dim)
        self.norm_first = norm_first

    def forward(self, hidden, residual):
        if self.norm_first:
            return residual + self.net(self.norm(hidden))

        return self.norm(residual + self.net(hidden))


class TransformerCrossAttentionLayer(nn.Module):
    def __init__(self, emb_dim, ff_dim, nhead=4, norm_first=False, dropout_p=0.0):
        super().__init__()
        self.norm_in_x = nn.LayerNorm(emb_dim)
        self.norm_in_y = nn.LayerNorm(emb_dim)
        self.mha_x_to_y = nn.MultiheadAttention(emb_dim, nhead, dropout=dropout_p)
        self.mha_y_to_x = nn.MultiheadAttention(emb_dim, nhead, dropout=dropout_p)
        self.dense_x_to_y = TransformerMLP(emb_dim, ff_dim, norm_first, dropout_p)
        self.dense_y_to_x = TransformerMLP(emb_dim, ff_dim, norm_first, dropout_p)

    def forward(self, x, y, x_key_padding_mask=None, y_key_padding_mask=None):
        x = self.norm_in_x(x)
        y = self.norm_in_y(y)
        mha_x, _ = self.mha_x_to_y(x, y, y, key_padding_mask=y_key_padding_mask)
        mha_y, _ = self.mha_y_to_x(y, x, x, key_padding_mask=x_key_padding_mask)

        return self.dense_x_to_y(mha_x, x), self.dense_y_to_x(mha_y, y)


class TransformerIntermediateLayer(nn.Module):
    def __init__(self, emb_dim, ff_dim):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.linear = nn.Linear(emb_dim, ff_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.linear(self.norm(x)))


class TransformerCrossEncoderLayer(nn.Module):
    def __init__(self, emb_dim, ff_dim, nhead=4, norm_first=False, dropout_p=0.0):
        super().__init__()
        self.cross_attn = TransformerCrossAttentionLayer(
            emb_dim, emb_dim, nhead=nhead, norm_first=norm_first, dropout_p=dropout_p
        )
        self.intermediate1 = TransformerIntermediateLayer(emb_dim, ff_dim)
        self.intermediate2 = TransformerIntermediateLayer(emb_dim, ff_dim)
        self.output1 = TransformerMLP(ff_dim, emb_dim, norm_first, dropout_p)
        self.output2 = TransformerMLP(ff_dim, emb_dim, norm_first, dropout_p)

    def forward(self, x, y, x_key_padding_mask=None, y_key_padding_mask=None):
        attn_x, attn_y = self.cross_attn(x, y, x_key_padding_mask, y_key_padding_mask)
        intermediate_x = self.intermediate1(attn_x)
        intermediate_y = self.intermediate2(attn_y)
        output_x = self.output1(intermediate_x, attn_x)
        output_y = self.output2(intermediate_y, attn_y)

        return output_x, output_y


class TransformerCrossEncoder(nn.Module):
    def __init__(
        self, nlayers, emb_dim, ff_dim, nhead=4, norm_first=False, dropout_p=0.0
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerCrossEncoderLayer(
                    emb_dim,
                    ff_dim,
                    nhead=nhead,
                    norm_first=norm_first,
                    dropout_p=dropout_p,
                )
                for _ in range(nlayers)
            ]
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.norm_first = norm_first

    def forward(self, x, y, x_key_padding_mask=None, y_key_padding_mask=None):
        x_key_padding_mask = (
            torch.zeros_like(x[..., 0]).bool()
            if x_key_padding_mask is None
            else x_key_padding_mask
        )
        y_key_padding_mask = (
            torch.zeros_like(y[..., 0]).bool()
            if y_key_padding_mask is None
            else y_key_padding_mask
        )

        # Seq-first
        x = x.transpose(1, 0)
        y = y.transpose(1, 0)
        for layer in self.layers:
            x, y = layer(
                x,
                y,
                x_key_padding_mask=x_key_padding_mask,
                y_key_padding_mask=y_key_padding_mask,
            )

        encoded = torch.cat([x, y], dim=0)

        # Important that we normalize the encoding here, otherwise
        # the scales from norm-first in the decoder will be different
        encoded = self.norm(encoded)

        return encoded, torch.cat([x_key_padding_mask, y_key_padding_mask], dim=-1)


class TransformerEmbeddings(nn.Module):
    def __init__(self, n_inp, embed_dim, dropout_p=0.0):
        super().__init__()
        self.embedding = nn.Embedding(n_inp, embed_dim)
        self.pos_embedding = nn.Embedding(n_inp, embed_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, instruction):
        projected_instruction = self.embedding(instruction)
        projected_instruction = (
            self.pos_embedding(torch.ones_like(instruction).cumsum(dim=-1) - 1)
            + projected_instruction
        )

        return self.dropout(self.norm(projected_instruction))


def nullable_one_hot(vec, cats):
    # Allows for zeros in a field where the category is zero
    return F.one_hot(vec, cats + 1)[..., 1:].float()


class OneHotEmbedding(nn.Module):
    def __init__(self, component_sizes):
        super().__init__()
        self.component_sizes = component_sizes
        self.output_size = np.sum(component_sizes)

    def forward(self, x):
        return torch.cat(
            [
                nullable_one_hot(x[..., i], s)
                for i, s in enumerate(self.component_sizes)
            ],
            dim=-1,
        )


class ViLBERTStateEncoderTransformer(nn.Module):
    def __init__(
        self,
        state_component_sizes,
        embed_dim=128,
        nlayers=6,
        nhead=8,
        norm_first=False,
        dropout_p=0.1,
    ):
        super().__init__()
        self.state_embedding = OneHotEmbedding(state_component_sizes)
        self.embedding = TransformerEmbeddings(64, embed_dim, dropout_p=dropout_p)
        self.state_encoder = StateCNN(
            self.state_embedding.output_size,
            50,
            embed_dim,
            [1, 5, 7],
            dropout_p=dropout_p,
        )
        self.cross_encoder = TransformerCrossEncoder(
            nlayers,
            embed_dim,
            embed_dim * 2,
            nhead=nhead,
            norm_first=norm_first,
            dropout_p=dropout_p,
        )

    def forward(self, state, instruction, instruction_key_padding_mask=None):
        projected_state = self.state_embedding(state)
        state_seq_dim = projected_state.shape[-2]
        state_w_dim = int(math.sqrt(state_seq_dim))
        state_h_dim = state_w_dim

        projected_state = projected_state.view(
            -1, state_w_dim, state_h_dim, projected_state.shape[-1]
        )

        projected_state = self.state_encoder(projected_state)
        projected_state = projected_state.flatten(-3, -2)

        projected_instruction = self.embedding(instruction)

        encoding, encoding_mask = self.cross_encoder(
            projected_state,
            projected_instruction,
            x_key_padding_mask=None,
            y_key_padding_mask=instruction_key_padding_mask,
        )

        return encoding, encoding_mask


def init_parameters(module, scale=1):
    if type(module) in [nn.LayerNorm]:
        return

    if type(module) in [nn.MultiheadAttention]:
        torch.nn.init.xavier_normal_(module.in_proj_weight, scale)
        return

    if type(module) in [nn.Conv2d]:
        torch.nn.init.normal_(module.weight, 0, 1e-2)
        torch.nn.init.zeros_(module.bias)
        return

    if getattr(module, "weight", None) is not None:
        torch.nn.init.xavier_normal_(module.weight, scale)

    if getattr(module, "bias", None) is not None:
        torch.nn.init.zeros_(module.bias)


class ViLBERTLeaner(pl.LightningModule):
    def __init__(
        self,
        state_component_sizes,
        x_categories,
        y_categories,
        embed_dim,
        dropout_p,
        nlayers,
        nhead,
        norm_first,
        pad_word_idx,
        pad_action_idx,
        sos_action_idx,
        eos_action_idx,
        lr=16e-3,
        wd=1e-2,
        warmup_proportion=0.1,
        decay_power=-1,
        predict_steps=64,
        no_lr_decay=False,
    ):
        super().__init__()
        self.encoder = ViLBERTStateEncoderTransformer(
            state_component_sizes, embed_dim, nlayers, nhead, norm_first, dropout_p
        )
        self.decoder = DecoderTransformer(
            embed_dim,
            y_categories,
            nlayers,
            nhead,
            norm_first,
            pad_action_idx,
            dropout_p,
        )
        self.y_categories = y_categories
        self.pad_word_idx = pad_word_idx
        self.pad_action_idx = pad_action_idx
        self.sos_action_idx = sos_action_idx
        self.eos_action_idx = eos_action_idx

        self.apply(init_parameters)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return transformer_optimizer_config(
            self,
            self.hparams.lr,
            warmup_proportion=self.hparams.warmup_proportion,
            weight_decay=self.hparams.wd,
            decay_power=self.hparams.decay_power,
            no_lr_decay=self.hparams.no_lr_decay,
        )

    def encode(self, states, queries):
        return self.encoder(states, queries)

    def decode_autoregressive(self, decoder_in, encoder_outputs, encoder_padding):
        return self.decoder(decoder_in, encoder_outputs, encoder_padding)

    def forward(self, states, queries, decoder_in):
        instruction_mask = queries == self.pad_word_idx
        encoded, encoding_mask = self.encoder(
            states, queries, instruction_key_padding_mask=instruction_mask
        )
        padding = torch.cat(
            [torch.zeros_like(states[..., 0]).bool(), instruction_mask],
            dim=-1,
        )
        return self.decode_autoregressive(decoder_in, encoded, padding)

    def training_step(self, x, idx):
        query, targets, state = x
        actions_mask = targets == self.pad_action_idx

        decoder_in = torch.cat(
            [torch.ones_like(targets)[:, :1] * self.sos_action_idx, targets], dim=-1
        )

        # Now do the training
        preds = self.forward(state, query, decoder_in)[:, :-1]

        # Ultimately we care about the cross entropy loss
        loss = F.cross_entropy(
            preds.flatten(0, -2),
            targets.flatten().long(),
            ignore_index=self.pad_action_idx,
        )

        argmax_preds = preds.argmax(dim=-1)
        argmax_preds[actions_mask] = self.pad_action_idx
        exacts = (argmax_preds == targets).all(dim=-1).to(torch.float).mean()

        self.log("tloss", loss, prog_bar=True)
        self.log("texact", exacts, prog_bar=True)
        self.log(
            "tacc",
            (preds.argmax(dim=-1)[~actions_mask] == targets[~actions_mask])
            .float()
            .mean(),
            prog_bar=True,
        )

        return loss

    def validation_step(self, x, idx, dl_idx):
        query, targets, state = x
        actions_mask = targets == self.pad_action_idx

        decoder_in = torch.cat(
            [torch.ones_like(targets)[:, :1] * self.sos_action_idx, targets], dim=-1
        )

        # Now do the training
        preds = self.forward(state, query, decoder_in)[:, :-1]

        # Ultimately we care about the cross entropy loss
        loss = F.cross_entropy(
            preds.flatten(0, -2),
            targets.flatten().long(),
            ignore_index=self.pad_action_idx,
        )

        argmax_preds = preds.argmax(dim=-1)
        argmax_preds[actions_mask] = self.pad_action_idx
        exacts = (argmax_preds == targets).all(dim=-1).to(torch.float).mean()

        self.log("vloss", loss, prog_bar=True)
        self.log("vexact", exacts, prog_bar=True)
        self.log(
            "vacc",
            (preds.argmax(dim=-1)[~actions_mask] == targets[~actions_mask])
            .float()
            .mean(),
            prog_bar=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-demonstrations", type=str, required=True)
    parser.add_argument("--valid-demonstrations-directory", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--valid-batch-size", type=int, default=128)
    parser.add_argument("--batch-size-mult", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--nlayers", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--norm-first", action="store_true")
    parser.add_argument("--precision", type=int, choices=(16, 32), default=16)
    parser.add_argument("--norm-first", action="store_true")
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-lr-decay", action="store_true")
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1)
    parser.add_argument("--decay-power", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=2500000)
    parser.add_argument("--disable-shuffle", action="store_true")
    parser.add_argument("--check-val-every", type=int, default=1000)
    parser.add_argument("--enable-progress", action="store_true")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--tag", type=str, default="none")
    parser.add_argument("--swa", action="store_true")
    args = parser.parse_args()

    exp_name = "gscan"
    model_name = f"vilbert_cross_encoder_decode_actions_l_{args.nlayers}_h_{args.nhead}_d_{args.hidden_size}"
    dataset_name = "gscan"
    effective_batch_size = args.train_batch_size * args.batch_size_mult
    exp_name = f"{exp_name}_s_{args.seed}_m_{model_name}_it_{args.iterations}_b_{effective_batch_size}_d_gscan_t_{args.tag}_drop_{args.dropout_p}"
    model_dir = f"models/{exp_name}/{model_name}"
    model_path = f"{model_dir}/{exp_name}.pt"
    print(model_path)
    print(
        f"Batch size {args.train_batch_size}, mult {args.batch_size_mult}, total {args.train_batch_size * args.batch_size_mult}"
    )

    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(f"{model_path}"):
        print(f"Skipping {model_path} as it already exists")
        return

    seed = args.seed
    iterations = args.iterations

    pl.seed_everything(seed)

    (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (train_demonstrations, valid_demonstrations_dict),
    ) = load_data(
        args.train_demonstrations, args.valid_demonstrations_directory, args.dictionary
    )

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}
    IDX2ACTION = {i: w for w, i in ACTION2IDX.items()}

    pad_word = WORD2IDX["[pad]"]
    pad_action = ACTION2IDX["[pad]"]
    sos_action = ACTION2IDX["[sos]"]
    eos_action = ACTION2IDX["[eos]"]

    train_dataset = ReshuffleOnIndexZeroDataset(
        PaddingDataset(
            train_demonstrations,
            (8, 72, None),
            (pad_word, pad_action, None),
        )
    )

    pl.seed_everything(seed)
    meta_module = ViLBERTLeaner(
        [4, len(color_dictionary), len(noun_dictionary), 1, 4],
        len(IDX2WORD),
        len(IDX2ACTION),
        args.hidden_size,
        args.dropout_p,
        args.nlayers,
        args.nhead,
        args.norm_first,
        pad_word,
        pad_action,
        sos_action,
        eos_action,
        lr=args.lr,
        decay_power=args.decay_power,
        warmup_proportion=args.warmup_proportion,
        no_lr_decay=args.no_lr_decay,
    )
    print(meta_module)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        pin_memory=True,
    )

    check_val_opts = {}
    interval = args.check_val_every / len(train_dataloader)

    # Every check_val_interval steps, regardless of how large the training dataloader is
    if interval > 1.0:
        check_val_opts["check_val_every_n_epoch"] = math.floor(interval)
    else:
        check_val_opts["val_check_interval"] = interval

    checkpoint_cb = ModelCheckpoint(
        monitor="vexact/dataloader_idx_0",
        auto_insert_metric_name=False,
        save_top_k=5,
        mode="max",
    )

    logs_root_dir = f"logs/{exp_name}/{model_name}/{dataset_name}/{seed}"
    most_recent_version = args.version

    trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(logs_root_dir, version=most_recent_version),
            LoadableCSVLogger(
                logs_root_dir, version=most_recent_version, flush_logs_every_n_steps=10
            ),
        ],
        callbacks=[pl.callbacks.LearningRateMonitor(), checkpoint_cb] + ([
            StochasticWeightAveraging(
                swa_lrs=1e-2,
                annealing_epochs=int((iterations * args.batch_size_mult) // len(train_dataloader) * 0.2)
            )
        ] if args.swa else []),
        max_steps=iterations,
        num_sanity_val_steps=10,
        gpus=1 if torch.cuda.is_available() else 0,
        precision=args.precision if torch.cuda.is_available() else 32,
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=args.batch_size_mult,
        enable_progress_bar=sys.stdout.isatty() or args.enable_progress,
        gradient_clip_val=0.2,
        **check_val_opts,
    )

    trainer.fit(
        meta_module,
        train_dataloader,
        [
            DataLoader(
                PaddingDataset(
                    Subset(demonstrations, torch.randperm(len(demonstrations))[:1024]),
                    (8, 72, None),
                    (pad_word, pad_action, None),
                ),
                batch_size=max([args.train_batch_size, args.valid_batch_size]),
                pin_memory=True,
            )
            for demonstrations in valid_demonstrations_dict.values()
        ],
    )
    print(f"Done, saving {model_path}")
    trainer.save_checkpoint(f"{model_path}")


if __name__ == "__main__":
    main()
