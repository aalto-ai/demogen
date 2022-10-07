import argparse
import os
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.utils.data import DataLoader, Subset
from positional_encodings.torch_encodings import PositionalEncoding1D
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from gscan_metaseq2seq.models.embedding import BOWEmbedding
from gscan_metaseq2seq.util.dataset import PaddingDataset, ReshuffleOnIndexZeroDataset
from gscan_metaseq2seq.util.load_data import load_data
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        n_state_components,
        hidden_size,
        output_size,
        nlayers,
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
        self.n_state_components = n_state_components
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
                nhead=4,
            ),
            num_layers=nlayers,
        )
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
    def __init__(self, n_input_channels, conv_kernel_sizes, dropout_p):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=n_input_channels,
                    out_channels=n_input_channels,
                    kernel_size=size,
                    padding="same",
                )
                for size in conv_kernel_sizes
            ]
        )
        self.mlp = nn.Sequential(
            nn.Linear(
                len(conv_kernel_sizes) * n_input_channels,
                len(conv_kernel_sizes) * n_input_channels,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(len(conv_kernel_sizes) * n_input_channels, n_input_channels),
        )

    def forward(self, x):
        x = x.transpose(-1, -3).transpose(-1, -2)
        x_multiscale = [layer(x) for layer in self.conv_layers]
        x_multiscale = torch.cat(x_multiscale, dim=-3)
        x_multiscale = x_multiscale.transpose(-1, -3).transpose(-2, -3)

        return self.mlp(x_multiscale)


class TransformerMLP(nn.Module):
    def __init__(self, emb_dim, ff_dim, dropout_p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, emb_dim),
            nn.Dropout(dropout_p),
        )
        self.norm = nn.LayerNorm(emb_dim, eps=1e-5)

    def forward(self, x, attn_output):
        return self.norm(x + self.net(attn_output))


class TransformerCrossEncoderLayer(nn.Module):
    def __init__(self, emb_dim, ff_dim, nhead=4, dropout_p=0.0):
        super().__init__()
        self.mha_x_to_y = nn.MultiheadAttention(emb_dim, nhead, dropout=dropout_p)
        self.mha_y_to_x = nn.MultiheadAttention(emb_dim, nhead, dropout=dropout_p)
        self.dense_x_to_y = TransformerMLP(emb_dim, ff_dim, dropout_p)
        self.dense_y_to_x = TransformerMLP(emb_dim, ff_dim, dropout_p)

    def forward(self, x, y, x_key_padding_mask=None, y_key_padding_mask=None):
        mha_x, _ = self.mha_x_to_y(x, y, y, key_padding_mask=y_key_padding_mask)
        mha_y, _ = self.mha_y_to_x(y, x, x, key_padding_mask=x_key_padding_mask)

        return self.dense_x_to_y(mha_x, x), self.dense_y_to_x(mha_y, y)


class TransformerCrossEncoder(nn.Module):
    def __init__(self, nlayers, emb_dim, ff_dim, nhead=4, dropout_p=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerCrossEncoderLayer(
                    emb_dim, ff_dim, nhead=4, dropout_p=dropout_p
                )
                for _ in range(nlayers)
            ]
        )

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

        return encoded, torch.cat([x_key_padding_mask, y_key_padding_mask], dim=-1)


class ViLBERTStateEncoderTransformer(nn.Module):
    def __init__(
        self,
        n_state_components,
        text_dictionary_size,
        embed_dim=128,
        nlayers=6,
        dropout_p=0.1,
    ):
        super().__init__()
        self.state_embedding = BOWEmbedding(64, n_state_components, embed_dim)
        self.state_projection = nn.Linear(n_state_components * embed_dim, embed_dim)
        self.embedding = nn.Embedding(text_dictionary_size, embed_dim)
        self.pos_encoding = PositionalEncoding1D(embed_dim)
        self.state_encoder = StateCNN(embed_dim, [1, 5, 7], dropout_p=dropout_p)
        self.cross_encoder = TransformerCrossEncoder(
            nlayers, embed_dim, embed_dim * 2, nhead=4, dropout_p=dropout_p
        )

    def forward(self, state, instruction, instruction_key_padding_mask=None):
        projected_state = self.state_projection(self.state_embedding(state))
        state_seq_dim = projected_state.shape[-2]
        state_w_dim = int(math.sqrt(state_seq_dim))
        state_h_dim = state_w_dim

        projected_state = projected_state.view(
            -1, state_w_dim, state_h_dim, projected_state.shape[-1]
        )

        projected_state = self.state_encoder(projected_state)
        projected_state = projected_state.flatten(-3, -2)

        projected_instruction = self.embedding(instruction)
        projected_instruction = (
            self.pos_encoding(projected_instruction) + projected_instruction
        )

        encoding, encoding_mask = self.cross_encoder(
            projected_state,
            projected_instruction,
            x_key_padding_mask=None,
            y_key_padding_mask=instruction_key_padding_mask,
        )

        return encoding, encoding_mask


class ViLBERTLeaner(pl.LightningModule):
    def __init__(
        self,
        n_state_components,
        x_categories,
        y_categories,
        embed_dim,
        dropout_p,
        nlayers,
        pad_word_idx,
        pad_action_idx,
        sos_action_idx,
        eos_action_idx,
        lr=1e-4,
        wd=1e-2,
        warmup_proportion=0.001,
        decay_power=-1,
        predict_steps=64,
    ):
        super().__init__()
        self.encoder = ViLBERTStateEncoderTransformer(
            n_state_components, x_categories, embed_dim, nlayers, dropout_p
        )
        self.decoder = DecoderTransformer(
            n_state_components,
            embed_dim,
            y_categories,
            nlayers,
            pad_action_idx,
            dropout_p,
        )
        self.y_categories = y_categories
        self.pad_word_idx = pad_word_idx
        self.pad_action_idx = pad_action_idx
        self.sos_action_idx = sos_action_idx
        self.eos_action_idx = eos_action_idx
        self.save_hyperparameters()

    def configure_optimizers(self):
        return transformer_optimizer_config(
            self,
            self.hparams.lr,
            warmup_proportion=self.hparams.warmup_proportion,
            weight_decay=self.hparams.wd,
            decay_power=self.hparams.decay_power,
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
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1)
    parser.add_argument("--decay-power", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=2500000)
    parser.add_argument("--disable-shuffle", action="store_true")
    parser.add_argument("--check-val-every", type=int, default=1000)
    parser.add_argument("--enable-progress", action="store_true")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--tag", type=str, default="none")
    args = parser.parse_args()

    exp_name = "gscan"
    model_name = "vilbert_cross_encoder_decode_actions"
    dataset_name = "gscan"
    effective_batch_size = args.train_batch_size * args.batch_size_mult
    exp_name = f"{exp_name}_s_{args.seed}_m_{model_name}_it_{args.iterations}_b_{effective_batch_size}_d_gscan_t_{args.tag}"
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
        7,
        len(IDX2WORD),
        len(IDX2ACTION),
        args.hidden_size,
        args.dropout_p,
        args.nlayers,
        pad_word,
        pad_action,
        sos_action,
        eos_action,
        lr=args.lr,
        decay_power=args.decay_power,
        warmup_proportion=args.warmup_proportion,
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
        callbacks=[pl.callbacks.LearningRateMonitor(), checkpoint_cb],
        max_steps=iterations,
        num_sanity_val_steps=10,
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16 if torch.cuda.is_available() else None,
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
