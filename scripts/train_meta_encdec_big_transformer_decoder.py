import argparse
import copy
import os
import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import sys
from torch.utils.data import DataLoader, Subset
from positional_encodings.torch_encodings import PositionalEncoding1D
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam

from gscan_metaseq2seq.models.embedding import BOWEmbedding
from gscan_metaseq2seq.util.dataset import (
    PaddingDataset,
    ReshuffleOnIndexZeroDataset,
    MapDataset,
    ReorderSupportsByDistanceDataset,
)
from gscan_metaseq2seq.util.load_data import load_data, load_data_directories
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config
from gscan_metaseq2seq.util.padding import pad_to


def init_parameters(module, scale=1e-2):
    if type(module) in [nn.LayerNorm]:
        return

    if type(module) in [nn.MultiheadAttention]:
        torch.nn.init.normal_(module.in_proj_weight, 0, scale)
        return

    if type(module) in [nn.Conv2d]:
        return

    if getattr(module, "weight", None) is not None:
        torch.nn.init.normal_(module.weight, 0, scale)

    if getattr(module, "bias", None) is not None:
        torch.nn.init.zeros_(module.bias)


def compute_loss_and_accuracy(preds, targets, lengths, pad_target_idx):
    actions_mask = targets == pad_target_idx

    # Ultimately we care about the cross entropy loss
    # loss = F.cross_entropy(
    #    preds.flatten(0, -2),
    #    targets.flatten().long(),
    #    ignore_index=pad_target_idx,
    # )

    segment_loss_mask = F.one_hot(lengths, targets.shape[-1]).cumsum(dim=-1).bool()

    loss = F.cross_entropy(
        preds[segment_loss_mask],
        targets[segment_loss_mask].long(),
        ignore_index=pad_target_idx,
    )

    argmax_preds = preds.argmax(dim=-1)

    truncated_matches = [
        (ap[length:][~mask[length:]] == tgt[length:][~mask[length:]])
        for ap, length, mask, tgt in zip(argmax_preds, lengths, actions_mask, targets)
    ]
    acc = torch.cat(truncated_matches).float().mean()
    exacts = torch.tensor([tm.all() for tm in truncated_matches]).float().mean()

    return {
        "loss": loss,
        "acc": acc,
        "exacts": exacts,
    }


class DSTransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for i in range(num_layers)])

    def forward(self, src, src_key_padding_mask, is_causal=False):
        x = src
        for layer in self.layers:
            x = deepspeed.checkpointing.checkpoint(
                layer, x, None, src_key_padding_mask, is_causal
            )

        return x


class BigTransformerDecoderLearner(pl.LightningModule):
    def __init__(
        self,
        n_input_components,
        n_embeddings,
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
        metalearn_dropout_p=0.0,
        metalearn_include_permutations=False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_embeddings, embed_dim, padding_idx=pad_word_idx)
        self.in_embedding_project = nn.Linear(n_input_components * embed_dim, embed_dim)
        self.pos_encoding = PositionalEncoding1D(embed_dim)
        self.in_pos_project = nn.Linear(embed_dim * 2, embed_dim)
        self.out_pos_project = nn.Linear(embed_dim * 2, embed_dim)
        self.transformer = DSTransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dropout=dropout_p,
                norm_first=norm_first,
                activation=F.silu,
            ),
            num_layers=nlayers * 2,
        )
        self.out = nn.Linear(embed_dim, n_embeddings)
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
            optimizer_func=DeepSpeedCPUAdam,
        )

    def encode(self, support_state, x_supports, y_supports, queries):
        return self.encoder(support_state, x_supports, y_supports, queries)

    def decode_autoregressive(self, decoder_in, encoder_outputs, encoder_padding):
        return self.decoder(decoder_in, encoder_outputs, encoder_padding)

    def forward(self, decoder_in):
        bow_embeddings = self.in_embedding_project(
            self.embedding(decoder_in).flatten(-2, -1)
        )
        bow_embeddings_pos_encoded = self.in_pos_project(
            torch.cat([bow_embeddings, self.pos_encoding(bow_embeddings)], dim=-1)
        )

        # Regular old transformer decoder
        decoded_sequence = self.transformer(
            src=bow_embeddings_pos_encoded.transpose(0, 1),
            src_key_padding_mask=(decoder_in == self.pad_word_idx).all(dim=-1),
            is_causal=True,
        ).transpose(0, 1)

        return self.out(decoded_sequence)

    def training_step(self, x, idx):
        target, in_len = x
        decoder_in = torch.cat(
            [torch.ones_like(target)[:, :1] * self.sos_action_idx, target[:, :-1]],
            dim=1,
        )

        preds = self.forward(decoder_in)
        stats = compute_loss_and_accuracy(
            preds, target[..., 0], in_len, self.pad_action_idx
        )

        self.log("tloss", stats["loss"], prog_bar=True)
        self.log("tacc", stats["acc"], prog_bar=True)
        self.log("texact", stats["exacts"], prog_bar=True)

        return stats["loss"]

    def validation_step(self, x, idx, dataloader_idx=0):
        target, in_len = x
        decoder_in = torch.cat(
            [torch.ones_like(target)[:, :1] * self.sos_action_idx, target[:, :-1]],
            dim=1,
        )

        preds = self.forward(decoder_in)
        stats = compute_loss_and_accuracy(
            preds, target[..., 0], in_len, self.pad_action_idx
        )

        self.log("vloss", stats["loss"], prog_bar=True)
        self.log("vacc", stats["acc"], prog_bar=True)
        self.log("vexact", stats["exacts"], prog_bar=True)

        return stats["loss"]

    def predict_step(self, x, idx, dl_idx=0):
        (
            x_permutation,
            y_permutation,
            query_state,
            support_state,
            queries,
            targets,
            x_supports,
            y_supports,
        ) = x
        decoder_in = torch.ones_like(targets)[:, :1] * self.sos_action_idx
        support_mask = torch.zeros_like(x_supports[..., 0]).bool()

        if self.hparams.metalearn_include_permutations:
            # We concatenate the x and y permutations to the supports, this way
            # they get encoded and also go through the attention process. The
            # permutations are never masked.
            x_supports = torch.cat([x_permutation[..., None, :], x_supports], dim=1)
            y_supports = torch.cat([y_permutation[..., None, :], y_supports], dim=1)
            support_mask = torch.cat(
                [torch.zeros_like(x_permutation[..., :1]).bool(), support_mask], dim=1
            )

        padding = queries == self.pad_word_idx

        # We do autoregressive prediction, predict for as many steps
        # as there are targets, but don't use teacher forcing
        logits = []

        with torch.no_grad():
            encoded = self.encoder(
                query_state,
                # We expand the support_state if it was not already expanded.
                support_state
                if support_state.ndim == 4
                else support_state[:, None].expand(-1, x_supports.shape[1], -1, -1),
                x_supports,
                y_supports,
                support_mask,
                queries,
            )

            for i in range(targets.shape[1]):
                logits.append(
                    self.decode_autoregressive(decoder_in, encoded, padding)[:, -1]
                )
                decoder_out = logits[-1].argmax(dim=-1)
                decoder_in = torch.cat([decoder_in, decoder_out[:, None]], dim=1)

            decoded_eq_mask = (
                (decoder_in == self.eos_action_idx).int().cumsum(dim=-1).bool()[:, :-1]
            )
            decoded = decoder_in[:, 1:]
            decoded[decoded_eq_mask] = self.pad_action_idx
            logits = torch.stack(logits, dim=1)

        exacts = (decoded == targets).all(dim=-1)

        return decoded, logits, exacts


class PermuteActionsDataset(Dataset):
    def __init__(
        self,
        dataset,
        x_categories,
        y_categories,
        pad_word_idx,
        pad_action_idx,
        shuffle=True,
        seed=0,
    ):
        super().__init__()
        self.dataset = dataset
        self.x_categories = x_categories
        self.y_categories = y_categories
        self.pad_word_idx = pad_word_idx
        self.pad_action_idx = pad_action_idx
        self.shuffle = shuffle
        self.generator = np.random.default_rng(seed)

    def state_dict(self):
        return {"random_state": self.generator.__getstate__()}

    def load_state_dict(self, sd):
        if "random_state" in sd:
            self.generator.__setstate__(sd["random_state"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (
            query_state,
            support_state,
            queries,
            targets,
            x_supports,
            y_supports,
        ) = self.dataset[idx]

        x_permutation = np.arange(self.x_categories)
        y_permutation = np.arange(self.y_categories)

        # Compute permutations of outputs
        if self.shuffle:
            # Do the permutation
            x_permutation[0 : self.pad_word_idx] = x_permutation[0 : self.pad_word_idx][
                self.generator.permutation(self.pad_word_idx)
            ]
            y_permutation[0 : self.pad_action_idx] = y_permutation[
                0 : self.pad_action_idx
            ][self.generator.permutation(self.pad_action_idx)]

            # Only permute the outputs, not the inputs
            y_supports = [y_permutation[np.array(ys)] for ys in y_supports]
            targets = y_permutation[np.array(targets)]

        return (
            query_state,
            support_state,
            queries,
            targets,
            x_supports,
            y_supports,
        )


class ShuffleDemonstrationsDataset(Dataset):
    def __init__(self, dataset, active, seed=0):
        super().__init__()
        self.dataset = dataset
        self.active = active
        self.generator = np.random.default_rng(seed)

    def state_dict(self):
        return {"random_state": self.generator.__getstate__()}

    def load_state_dict(self, sd):
        if "random_state" in sd:
            self.generator.__setstate__(sd["random_state"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if not self.active:
            return self.dataset[idx]

        (
            query_state,
            support_state,
            queries,
            targets,
            x_supports,
            y_supports,
        ) = self.dataset[idx]
        x_supports = np.copy(x_supports)
        y_supports = np.copy(y_supports)

        support_permutation = self.generator.permutation(x_supports.shape[0])

        return (
            query_state,
            [support_state[i] for i in support_permutation]
            if isinstance(support_state, list)
            else support_state,
            queries,
            targets,
            x_supports[support_permutation],
            y_supports[support_permutation],
        )


class ApplyOffsetsDataset(Dataset):
    def __init__(self, dataset, offsets, pad_word, pad_action, pad_out=2048):
        """Used to apply offsets to each of the tokens.

        Enables the use of a single dictionary.
        """
        super().__init__()
        self.dataset = dataset
        # Reserve 8 tokens for use as padding etc
        self.offsets = np.cumsum(
            [0, 8] + offsets
        )  # [size, color, shape, agent, dictionary, x_pos, y_pos, words, actions]
        self.pad_out = pad_out
        self.pad_word = pad_word
        self.pad_action = pad_action

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (
            query_state,
            support_state,
            queries,
            targets,
            x_supports,
            y_supports,
        ) = self.dataset[idx]

        state_offsets = self.offsets[1:8]
        query_offsets = self.offsets[8]
        target_offsets = self.offsets[9]

        query_state_offseted = (
            query_state[~(query_state == 0).all(axis=-1)] + state_offsets
        )
        support_state_offsetted = (
            [s[~(s == 0).all(axis=-1)] + state_offsets for s in support_state]
            if isinstance(support_state, list)
            else ([support_state + state_offsets] * len(x_supports))
        )

        x_supports_offseted = [
            xs[xs != self.pad_word] + query_offsets for xs in x_supports
        ]
        y_supports_offseted = [
            ys[ys != self.pad_action] + target_offsets for ys in y_supports
        ]

        queries_offseted = queries[queries != self.pad_word] + query_offsets
        targets_offseted = targets[targets != self.pad_action] + target_offsets

        # Now we glue everything together
        context_in = np.concatenate(
            [
                np.concatenate(
                    [
                        np.array([2] * 7).astype(int)[None],
                        ss,
                        np.array([3] * 7).astype(int)[None],
                        np.concatenate(
                            [xs[:, None], np.tile(np.zeros_like(xs)[:, None], (1, 6))],
                            axis=-1,
                        ),
                        np.array([4] * 7).astype(int)[None],
                        np.concatenate(
                            [ys[:, None], np.tile(np.zeros_like(ys)[:, None], (1, 6))],
                            axis=-1,
                        ),
                    ],
                    axis=0,
                )
                for ss, xs, ys in zip(
                    support_state_offsetted, x_supports_offseted, y_supports_offseted
                )
            ]
            + [
                np.array([2] * 7).astype(int)[None],
                query_state_offseted,
                np.array([3] * 7).astype(int)[None],
                np.concatenate(
                    [
                        queries_offseted[:, None],
                        np.tile(np.zeros_like(queries_offseted)[:, None], (1, 6)),
                    ],
                    axis=-1,
                ),
                np.array([4] * 7).astype(int)[None],
            ],
            axis=0,
        )
        src_len = context_in.shape[0]
        context_in = np.concatenate(
            [
                context_in,
                np.concatenate(
                    [
                        targets_offseted[:, None],
                        np.tile(np.zeros_like(targets_offseted)[:, None], (1, 6)),
                    ],
                    axis=-1,
                ),
            ]
        )
        total_len = context_in.shape[0]

        if total_len > self.pad_out:
            raise RuntimeError(
                f"Got a sequence of len {total_len}, it is too long compared to {self.pad_out}"
            )

        context_in = np.concatenate(
            [
                context_in[: self.pad_out],
                np.tile(
                    np.array([0] * (max(0, self.pad_out - total_len)))[:, None], (1, 7)
                ),
            ]
        )

        return (context_in.astype(int), src_len)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-demonstrations", type=str, required=True)
    parser.add_argument("--valid-demonstrations-directory", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--valid-batch-size", type=int, default=128)
    parser.add_argument("--batch-size-mult", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--nlayers", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--norm-first", action="store_true")
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1)
    parser.add_argument("--decay-power", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=2500000)
    parser.add_argument("--disable-shuffle", action="store_true")
    parser.add_argument("--check-val-every", type=int, default=500)
    parser.add_argument("--limit-val-size", type=int, default=None)
    parser.add_argument("--enable-progress", action="store_true")
    parser.add_argument("--restore-from-checkpoint", action="store_true")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="gscan")
    parser.add_argument("--tag", type=str, default="none")
    parser.add_argument("--metalearn-dropout-p", type=float, default=0.0)
    parser.add_argument("--metalearn-demonstrations-limit", type=int, default=6)
    parser.add_argument("--metalearn-include-permutations", action="store_true")
    parser.add_argument("--pad-instructions-to", type=int, default=8)
    parser.add_argument("--pad-actions-to", type=int, default=128)
    parser.add_argument("--pad-state-to", type=int, default=36)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--limit-load", type=int, default=None)
    parser.add_argument("--dataloader-ncpus", type=int, default=1)
    parser.add_argument("--shuffle-demonstrations", action="store_true")
    args = parser.parse_args()

    exp_name = "meta_gscan"
    model_name = f"meta_encdec_big_transformer_decoder_l_{args.nlayers}_h_{args.nhead}_d_{args.hidden_size}"
    dataset_name = args.dataset_name
    effective_batch_size = args.train_batch_size * args.batch_size_mult
    exp_name = f"{exp_name}_s_{args.seed}_m_{model_name}_it_{args.iterations}_b_{effective_batch_size}_d_{dataset_name}_t_{args.tag}_drop_{args.dropout_p}_ml_d_limit_{args.metalearn_demonstrations_limit}"
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

    torch.set_float32_matmul_precision("medium")
    print("Flash attention:", torch.backends.cuda.flash_sdp_enabled())

    seed = args.seed
    iterations = args.iterations

    (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (meta_train_demonstrations, meta_valid_demonstrations_dict),
    ) = load_data_directories(
        args.train_demonstrations, args.dictionary, limit_load=args.limit_load
    )

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}
    IDX2ACTION = {i: w for w, i in ACTION2IDX.items()}

    pad_word = WORD2IDX["[pad]"]
    pad_action = ACTION2IDX["[pad]"]
    sos_action = ACTION2IDX["[sos]"]
    eos_action = ACTION2IDX["[eos]"]

    pl.seed_everything(0)
    meta_train_dataset = ReshuffleOnIndexZeroDataset(
        ApplyOffsetsDataset(
            PermuteActionsDataset(
                ShuffleDemonstrationsDataset(
                    ReorderSupportsByDistanceDataset(
                        MapDataset(
                            MapDataset(
                                meta_train_demonstrations,
                                lambda x: (x[2], x[3], x[0], x[1], x[4], x[5], x[6]),
                            ),
                            lambda x: (
                                x[0],
                                [x[1]] * len(x[-1])
                                if not isinstance(x[1][0], list)
                                else x[1],
                                x[2],
                                x[3],
                                x[4],
                                x[5],
                                x[6],
                            ),
                        ),
                        args.metalearn_demonstrations_limit,
                    ),
                    args.shuffle_demonstrations,
                ),
                len(WORD2IDX),
                len(ACTION2IDX),
                pad_word,
                pad_action,
                shuffle=not args.disable_shuffle,
            ),
            [8] * 7 + [len(WORD2IDX), len(ACTION2IDX)],
            pad_word,
            pad_action,
        )
    )

    pl.seed_everything(seed)
    meta_module = BigTransformerDecoderLearner(
        7,
        meta_train_dataset.dataset.offsets[-1],
        args.hidden_size,
        args.dropout_p,
        args.nlayers,
        args.nhead,
        0,
        0,
        1,
        eos_action + meta_train_dataset.dataset.offsets[-2],
        norm_first=args.norm_first,
        lr=args.lr,
        decay_power=args.decay_power,
        warmup_proportion=args.warmup_proportion,
        metalearn_dropout_p=args.metalearn_dropout_p,
        metalearn_include_permutations=args.metalearn_include_permutations,
    )
    print(meta_module)

    train_dataloader = DataLoader(
        meta_train_dataset,
        batch_size=args.train_batch_size,
        # num_workers=args.dataloader_ncpus
    )

    check_val_opts = {}
    interval = args.check_val_every / len(train_dataloader)

    # Every check_val_interval steps, regardless of how large the training dataloader is
    if interval > 1.0:
        check_val_opts["check_val_every_n_epoch"] = math.floor(interval)
    else:
        check_val_opts["val_check_interval"] = interval

    logs_root_dir = f"{args.log_dir}/{exp_name}/{model_name}/{dataset_name}/{seed}"
    most_recent_version = args.version

    deepspeed.ops.op_builder.CPUAdamBuilder().load()
    meta_trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(logs_root_dir, version=most_recent_version),
            LoadableCSVLogger(
                logs_root_dir, version=most_recent_version, flush_logs_every_n_steps=10
            ),
        ],
        callbacks=[
            pl.callbacks.LearningRateMonitor(),
            ModelCheckpoint(save_last=True, save_top_k=0),
        ],
        max_steps=iterations,
        num_sanity_val_steps=1,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        strategy="deepspeed_stage_3_offload",
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=args.batch_size_mult,
        enable_progress_bar=sys.stdout.isatty() or args.enable_progress,
        gradient_clip_val=1.0,
        **check_val_opts,
    )

    valid_dataloaders = [
        DataLoader(
            ApplyOffsetsDataset(
                ReorderSupportsByDistanceDataset(
                    MapDataset(
                        MapDataset(
                            Subset(
                                demonstrations,
                                np.random.permutation(len(demonstrations))[
                                    : args.limit_val_size
                                ],
                            ),
                            lambda x: (x[2], x[3], x[0], x[1], x[4], x[5], x[6]),
                        ),
                        lambda x: (
                            x[0],
                            [x[1]] * len(x[-1])
                            if not isinstance(x[1][0], list)
                            else x[1],
                            x[2],
                            x[3],
                            x[4],
                            x[5],
                            x[6],
                        ),
                    ),
                    args.metalearn_demonstrations_limit,
                ),
                [8] * 7 + [len(WORD2IDX), len(ACTION2IDX)],
                pad_word,
                pad_action,
            ),
            batch_size=max([args.train_batch_size, args.valid_batch_size]),
            pin_memory=True,
        )
        for demonstrations in meta_valid_demonstrations_dict.values()
    ]

    print(valid_dataloaders[-1].dataset[0])

    meta_trainer.fit(meta_module, train_dataloader, valid_dataloaders, ckpt_path="last")
    print(f"Done, saving {model_path}")
    meta_trainer.save_checkpoint(f"{model_path}")


if __name__ == "__main__":
    main()
