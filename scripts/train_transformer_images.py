import argparse
import os
import math

# Import pillow and matplotlib first before torch pulls in a different libc
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from positional_encodings.torch_encodings import (
    PositionalEncoding1D,
    PositionalEncoding2D,
)


from gscan_metaseq2seq.util.dataset import PaddingDataset, ReshuffleOnIndexZeroDataset
from gscan_metaseq2seq.util.load_data import load_data_directories
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.models.enc_dec_transformer.enc_dec_transformer_model import (
    DecoderTransformer,
    autoregressive_model_unroll_predictions,
    compute_encoder_decoder_model_loss_and_stats,
    init_parameters,
)
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config

from gscan_metaseq2seq.util.solver import (
    create_vocabulary,
    create_world,
    state_to_situation,
    reinitialize_world,
)


class ImagePatchEncoding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        patches = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return patches.flatten(1, 2)


class TransformerEmbeddings(nn.Module):
    def __init__(self, n_inp, embed_dim, dropout_p=0.0):
        super().__init__()
        self.embedding = nn.Embedding(n_inp, embed_dim)
        self.pos_embedding = PositionalEncoding1D(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, instruction):
        projected_instruction = self.embedding(instruction)
        projected_instruction = (
            self.pos_embedding(projected_instruction) + projected_instruction
        )

        return self.dropout(self.norm(projected_instruction))


class ImageEncoderTransformer(nn.Module):
    def __init__(
        self,
        n_input_channels,
        patch_size,
        vocab_size,
        embed_dim=128,
        nlayers=6,
        nhead=8,
        norm_first=False,
        dropout_p=0.1,
    ):
        super().__init__()
        self.state_embedding = ImagePatchEncoding(
            n_input_channels, embed_dim, patch_size
        )
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_encoding = PositionalEncoding1D(embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 4,
                norm_first=norm_first,
                dropout=dropout_p,
            ),
            num_layers=nlayers,
        )

    def forward(
        self,
        state,
        instruction,
        instruction_key_padding_mask=None,
    ):
        projected_state = self.state_embedding(state)
        projected_instruction = self.embedding(instruction)
        inputs = torch.cat([projected_state, projected_instruction], dim=-2)
        inputs = self.pos_encoding(inputs) + inputs
        inputs = self.dropout(self.norm(inputs))

        key_padding_mask = torch.cat(
            [
                torch.zeros_like(projected_state[..., 0].bool()),
                instruction_key_padding_mask,
            ],
            dim=-1,
        )

        encoding = self.encoder(
            inputs.transpose(0, 1), key_padding_mask=key_padding_mask
        )

        return encoding, key_padding_mask


class TransformerImgLearner(pl.LightningModule):
    def __init__(
        self,
        n_input_channels,
        patch_size,
        x_categories,
        y_categories,
        embed_dim,
        dropout_p,
        nlayers,
        nhead,
        pad_word_idx,
        pad_action_idx,
        sos_action_idx,
        eos_action_idx,
        norm_first=False,
        lr=1e-4,
        wd=1e-2,
        warmup_proportion=0.001,
        decay_power=-1,
        predict_steps=64,
    ):
        super().__init__()
        self.encoder = ImageEncoderTransformer(
            n_input_channels,
            patch_size,
            x_categories,
            embed_dim,
            nlayers,
            nhead,
            dropout_p,
            norm_first,
            pad_word_idx,
        )
        self.decoder = DecoderTransformer(
            embed_dim,
            y_categories,
            nlayers,
            nhead,
            pad_action_idx,
            dropout_p=dropout_p,
            norm_first=norm_first,
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
        )

    def encode(self, states, queries):
        return self.encoder(states, queries)

    def decode_autoregressive(self, decoder_in, encoder_outputs, encoder_padding):
        return self.decoder(decoder_in, encoder_outputs, encoder_padding)

    def forward(self, states, queries, decoder_in):
        encoded, encoder_padding = self.encoder(states, queries)
        return self.decode_autoregressive(decoder_in, encoded, encoder_padding)

    def training_step(self, x, idx):
        query, targets, state = x
        stats = compute_encoder_decoder_model_loss_and_stats(
            self, (state, query), targets, self.sos_action_idx, self.pad_action_idx
        )
        self.log("tloss", stats["loss"], prog_bar=True)
        self.log("texact", stats["exacts"], prog_bar=True)
        self.log("tacc", stats["acc"], prog_bar=True)

        return stats["loss"]

    def validation_step(self, x, idx, dataloader_idx=0):
        query, targets, state = x
        stats = compute_encoder_decoder_model_loss_and_stats(
            self, (state, query), targets, self.sos_action_idx, self.pad_action_idx
        )
        self.log("vloss", stats["loss"], prog_bar=True)
        self.log("vexact", stats["exacts"], prog_bar=True)
        self.log("vacc", stats["acc"], prog_bar=True)

    def predict_step(self, x, idx, dataloader_idx=0):
        instruction, target, state = x[:3]

        decoded, logits, exacts, _ = autoregressive_model_unroll_predictions(
            self,
            (state, instruction),
            target,
            self.sos_action_idx,
            self.eos_action_idx,
            self.pad_action_idx,
        )

        return tuple([instruction, state, decoded, logits, exacts, target] + x[3:])


class ImageRenderingDemonstrationsDataset(Dataset):
    def __init__(
        self, train_demonstrations, word2idx, colors, nouns, image_downsample=5
    ):
        super().__init__()
        self.train_demonstrations = train_demonstrations
        self.word2idx = word2idx
        self.colors = sorted(colors)
        self.nouns = sorted(nouns)

        vocabulary = create_vocabulary()
        world = create_world(vocabulary)

        self.world = world
        self.vocabulary = vocabulary
        self.image_downsample = image_downsample

    def __getitem__(self, i):
        instruction, actions, state = self.train_demonstrations[i]
        words, situation = state_to_situation(
            instruction,
            state,
            self.word2idx,
            self.colors,
            self.nouns,
            need_target=False,
        )

        self.world = reinitialize_world(self.world, situation, self.vocabulary)

        img = (
            self.world.render(mode="rgb_array")[
                :: self.image_downsample, :: self.image_downsample
            ]
            / 255.0
        )

        return instruction, actions, img.astype(np.float32)

    def __len__(self):
        return len(self.train_demonstrations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-demonstrations", type=str, required=True)
    parser.add_argument("--valid-demonstrations-directory", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--valid-batch-size", type=int, default=128)
    parser.add_argument("--batch-size-mult", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--nlayers", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--norm-first", action="store_true")
    parser.add_argument("--precision", type=int, choices=(16, 32), default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1)
    parser.add_argument("--decay-power", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=300000)
    parser.add_argument("--check-val-every", type=int, default=1000)
    parser.add_argument("--limit-val-size", type=int, default=None)
    parser.add_argument("--enable-progress", action="store_true")
    parser.add_argument("--restore-from-checkpoint", action="store_true")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--tag", type=str, default="none")
    parser.add_argument("--dataset-name", type=str, default="gscan")
    parser.add_argument("--pad-instructions-to", type=int, default=32)
    parser.add_argument("--pad-actions-to", type=int, default=128)
    parser.add_argument("--pad-state-to", type=int, default=36)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--dataloader-ncpus", type=int, default=1)
    parser.add_argument(
        "--state-profile", choices=("gscan", "reascan"), default="gscan"
    )
    parser.add_argument("--image-downsample", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=12)
    args = parser.parse_args()

    exp_name = "gscan"
    model_name = f"transformer_img_encoder_only_decode_actions_l_{args.nlayers}_h_{args.nhead}_d_{args.hidden_size}"
    dataset_name = args.dataset_name
    effective_batch_size = args.train_batch_size * args.batch_size_mult
    exp_name = f"{exp_name}_s_{args.seed}_m_{model_name}_it_{args.iterations}_b_{effective_batch_size}_d_{dataset_name}_t_{args.tag}_drop_{args.dropout_p}"
    model_dir = f"models/{exp_name}/{model_name}"
    model_path = f"{model_dir}/{exp_name}.pt"
    print(model_path)
    print(
        f"Batch size {args.train_batch_size}, mult {args.batch_size_mult}, total {args.train_batch_size * args.batch_size_mult}"
    )

    torch.set_float32_matmul_precision("medium")
    print("Flash attention:", torch.backends.cuda.flash_sdp_enabled())

    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(f"{model_path}"):
        print(f"Skipping {model_path} as it already exists")
        return

    seed = args.seed
    iterations = args.iterations

    (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (train_demonstrations, valid_demonstrations_dict),
    ) = load_data_directories(args.train_demonstrations, args.dictionary)

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}
    IDX2ACTION = {i: w for w, i in ACTION2IDX.items()}

    pad_word = WORD2IDX["[pad]"]
    pad_action = ACTION2IDX["[pad]"]
    sos_action = ACTION2IDX["[sos]"]
    eos_action = ACTION2IDX["[eos]"]

    pl.seed_everything(0)
    train_dataset = ReshuffleOnIndexZeroDataset(
        PaddingDataset(
            ImageRenderingDemonstrationsDataset(
                train_demonstrations, WORD2IDX, color_dictionary, noun_dictionary
            ),
            (
                args.pad_instructions_to,
                args.pad_actions_to,
                None,
            ),
            (pad_word, pad_action, None),
        )
    )

    pl.seed_everything(seed)
    meta_module = TransformerImgLearner(
        3,
        args.patch_size,
        len(IDX2WORD),
        len(IDX2ACTION),
        args.hidden_size,
        args.dropout_p,
        args.nlayers,
        args.nhead,
        pad_word,
        pad_action,
        sos_action,
        eos_action,
        lr=args.lr,
        norm_first=args.norm_first,
        decay_power=args.decay_power,
        warmup_proportion=args.warmup_proportion,
    )
    print(meta_module)

    pl.seed_everything(0)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, pin_memory=True
    )

    check_val_opts = {}
    interval = args.check_val_every / len(train_dataloader)

    # Every check_val_interval steps, regardless of how large the training dataloader is
    if interval > 1.0:
        check_val_opts["check_val_every_n_epoch"] = math.floor(interval)
    else:
        check_val_opts["val_check_interval"] = interval

    checkpoint_cb = ModelCheckpoint(save_last=True, save_top_k=0)

    logs_root_dir = f"{args.log_dir}/{exp_name}/{model_name}/{dataset_name}/{seed}"
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
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=1 if torch.cuda.is_available() else 0,
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
                    Subset(
                        ImageRenderingDemonstrationsDataset(
                            demonstrations, WORD2IDX, color_dictionary, noun_dictionary
                        ),
                        np.random.permutation(len(demonstrations))[
                            : args.limit_val_size
                        ],
                    ),
                    (
                        args.pad_instructions_to,
                        args.pad_actions_to,
                        None,
                    ),
                    (pad_word, pad_action, None),
                ),
                batch_size=max([args.train_batch_size, args.valid_batch_size]),
                pin_memory=True,
            )
            for demonstrations in valid_demonstrations_dict.values()
        ],
        ckpt_path="last",
    )
    print(f"Done, saving {model_path}")
    trainer.save_checkpoint(f"{model_path}")


if __name__ == "__main__":
    main()
