import argparse
import functools
import copy
import itertools
import os
import math
import torch
import torch.nn as nn
import numpy as np
import sys
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.auto import tqdm

import torch.nn as nn
import torch.nn.functional as F


from gscan_metaseq2seq.models.embedding import BOWEmbedding
from gscan_metaseq2seq.util.dataset import MapDataset, PaddingDataset, SampleSentencesByWordWeights
from gscan_metaseq2seq.util.load_data import load_data_directories
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config
from gscan_metaseq2seq.models.enc_dec_transformer.enc_dec_transformer_model import (
    TransformerLearner,
)
from train_transformer import determine_padding, determine_state_profile

class MonotonicRandomPositionEmbedding(nn.Module):
    def __init__(self, num_positions, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_positions, emb_dim)
        self.num_positions = num_positions

    def forward(self, x):
        permutation = (
            torch.from_numpy(
                np.sort(np.random.permutation(self.num_positions)[: x.shape[1]])
            )[None]
            .expand(x.shape[0], -1)
            .to(x.device)
        )

        return self.embedding(permutation)


class MonotonicPositionEncodingByMask(nn.Module):
    def __init__(self, num_positions, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_positions, emb_dim)
        self.num_positions = num_positions

    def forward(self, key_padding_mask):
        inv_key_padding_mask = ~key_padding_mask
        inv_key_padding_mask_cumsum = inv_key_padding_mask.to(torch.long).cumsum(dim=-1)

        return self.embedding(inv_key_padding_mask_cumsum)


class EncoderDecoderLanguageModel(pl.LightningModule):
    def __init__(
        self,
        num_words,
        num_positions,
        pad_word_idx,
        sos_word_idx,
        eos_word_idx,
        lr=0.0001,
        wd=1e-2,
        emb_dim=128,
        nlayers=8,
        nhead=4,
        dropout=0.1,
        norm_first=False,
        decay_power=-1,
        warmup_proportion=0.14,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.embedding_instructions = nn.Embedding(num_words, emb_dim)
        self.positional_encoding = MonotonicPositionEncodingByMask(
            num_positions, emb_dim
        )
        self.decoder_positional_encoding = MonotonicRandomPositionEmbedding(
            128, emb_dim
        )
        self.transformer = nn.Transformer(
            d_model=emb_dim,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            nhead=nhead,
            norm_first=norm_first,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
        )
        self.project = nn.Linear(emb_dim, num_words)
        self.sos_word_idx = sos_word_idx
        self.pad_word_idx = pad_word_idx
        self.eos_word_idx = eos_word_idx

    def configure_optimizers(self):
        return transformer_optimizer_config(
            self,
            self.hparams.lr,
            weight_decay=self.hparams.wd,
            decay_power=self.hparams.decay_power,
            warmup_proportion=self.hparams.warmup_proportion,
        )

    def forward(self, base_instruction, instruction_mask, right_shifted_instruction):
        masked_instruction = base_instruction.clone()

        encoded_instruction = self.embedding_instructions(
            masked_instruction
        ) + self.positional_encoding(instruction_mask)
        encoded_right_shifted_instruction = self.embedding_instructions(
            right_shifted_instruction
        )
        encoded_right_shifted_instruction = (
            encoded_right_shifted_instruction
            + self.decoder_positional_encoding(encoded_right_shifted_instruction)
        )

        decoded_instruction = self.transformer(
            src=encoded_instruction.transpose(0, 1),
            tgt=encoded_right_shifted_instruction.transpose(0, 1),
            src_key_padding_mask=instruction_mask,
            memory_key_padding_mask=instruction_mask,
            tgt_key_padding_mask=(right_shifted_instruction == self.pad_word_idx),
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(
                right_shifted_instruction.shape[1]
            ).to(self.device),
        ).transpose(0, 1)

        return self.project(decoded_instruction)

    def training_step(self, x, idx):
        (instruction,) = x

        right_shifted_instruction = torch.cat(
            [
                torch.ones_like(instruction[:, :1]) * self.sos_word_idx,
                instruction,
            ],
            dim=1,
        )[:, :-1]

        instruction_mask = torch.logical_or(
            torch.rand(instruction.shape, device=self.device)
            < np.random.uniform(0.1, 0.9),
            instruction == self.pad_word_idx,
        )

        # We always keep the very last pad bit unset, so that the entire
        # input is not padded
        instruction_mask[:, -1] = False

        logits = self(instruction, instruction_mask, right_shifted_instruction)

        loss = F.cross_entropy(logits.flatten(0, -2), instruction.flatten())
        self.log("loss", loss)

        return loss

    def validation_step(self, x, idx, dl_idx=0):
        (instruction,) = x

        right_shifted_instruction = torch.cat(
            [
                torch.ones_like(instruction[:, :1]) * self.sos_word_idx,
                instruction,
            ],
            dim=1,
        )[:, :-1]

        instruction_mask = torch.logical_or(
            torch.rand(instruction.shape, device=self.device) < 0.5,
            instruction == self.pad_word_idx,
        )

        # We always keep the very last pad bit unset, so that the entire
        # input is not padded
        instruction_mask[:, -1] = False

        logits = self(instruction, instruction_mask, right_shifted_instruction)

        loss = F.cross_entropy(logits.flatten(0, -2), instruction.flatten())
        self.log("vloss", loss)

        return loss


class StateEncoderDecoderLanguageModel(pl.LightningModule):
    def __init__(
        self,
        num_words,
        num_positions,
        pad_word_idx,
        sos_word_idx,
        state_feat_len,
        lr=0.0001,
        wd=1e-2,
        emb_dim=128,
        nlayers=8,
        nhead=4,
        dropout=0.1,
        norm_first=False,
        decay_power=-1,
        warmup_proportion=0.14,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.state_encoder = BOWEmbedding(64, state_feat_len, emb_dim)
        self.state_encoder_projection = nn.Linear(state_feat_len * emb_dim, emb_dim)
        self.embedding_instructions = nn.Embedding(num_words, emb_dim)
        self.positional_encoding = MonotonicPositionEncodingByMask(
            num_positions, emb_dim
        )
        self.decoder_positional_encoding = MonotonicRandomPositionEmbedding(
            128, emb_dim
        )
        self.norm_input = nn.LayerNorm(emb_dim)
        self.norm_decoded_output = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.transformer = nn.Transformer(
            d_model=emb_dim,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            nhead=nhead,
            norm_first=norm_first,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
        )
        self.project = nn.Linear(emb_dim, num_words)
        self.sos_word_idx = sos_word_idx
        self.pad_word_idx = pad_word_idx

    def configure_optimizers(self):
        return transformer_optimizer_config(
            self,
            self.hparams.lr,
            weight_decay=self.hparams.wd,
            decay_power=self.hparams.decay_power,
            warmup_proportion=self.hparams.warmup_proportion,
            optimizer_kwargs={"fused": True},
        )

    def encode(self, base_instruction, base_state, all_mask, instruction_mask):
        masked_instruction = base_instruction.clone()

        encoded_state = self.state_encoder_projection(self.state_encoder(base_state))
        encoded_instruction = self.embedding_instructions(
            masked_instruction
        ) + self.positional_encoding(instruction_mask)

        encoded_inp = torch.cat(
            [
                encoded_instruction,
                encoded_state,
            ],
            dim=-2,
        )
        encoded_inp = self.dropout(self.norm_input(encoded_inp))

        t_encoded_inp = self.transformer.encoder(
            encoded_inp.transpose(0, 1), src_key_padding_mask=all_mask
        )

        return t_encoded_inp.transpose(0, 1)

    def decode(self, encodings, encoding_mask, decoder_in):
        encoded_right_shifted_instruction = self.embedding_instructions(decoder_in)
        encoded_right_shifted_instruction = (
            encoded_right_shifted_instruction
            + self.decoder_positional_encoding(encoded_right_shifted_instruction)
        )
        encoded_right_shifted_instruction = self.dropout(
            self.norm_decoded_output(encoded_right_shifted_instruction)
        )

        return self.project(
            self.transformer.decoder(
                memory=encodings.transpose(0, 1),
                memory_key_padding_mask=encoding_mask,
                tgt=encoded_right_shifted_instruction.transpose(0, 1),
                tgt_key_padding_mask=(decoder_in == self.pad_word_idx),
                tgt_mask=nn.Transformer.generate_square_subsequent_mask(
                    decoder_in.shape[1]
                ).to(self.device),
            ).transpose(0, 1)
        )

    def forward(
        self,
        base_instruction,
        base_state,
        all_mask,
        instruction_mask,
        right_shifted_instruction,
    ):
        encodings = self.encode(
            base_instruction, base_state, all_mask, instruction_mask
        )
        return self.decode(encodings, all_mask, right_shifted_instruction)

    def training_step(self, x, idx):
        instruction, state = x

        right_shifted_instruction = torch.cat(
            [
                torch.ones_like(instruction[:, :1]) * self.sos_word_idx,
                instruction,
            ],
            dim=1,
        )[:, :-1]

        instruction_state_padding = torch.cat(
            [instruction == self.pad_word_idx, (state == 0).all(dim=-1)], dim=-1
        )

        all_mask = torch.logical_or(
            torch.rand(instruction_state_padding.shape, device=self.device)
            < np.random.uniform(0.0, 0.3),
            instruction_state_padding,
        )
        instruction_only_mask = all_mask[:, : instruction.shape[1]]

        # We always keep the very last pad bit unset, so that the entire
        # input is not padded
        all_mask[:, -1] = False
        instruction_only_mask[:, -1] = False

        logits = self(
            instruction,
            state,
            all_mask,
            instruction_only_mask,
            right_shifted_instruction,
        )

        loss = F.cross_entropy(logits.flatten(0, -2), instruction.flatten())
        self.log("tloss", loss, prog_bar=True)

        return loss

    def validation_step(self, x, idx):
        return self.training_step(x, idx)


def train_state_encoder_decoder(
    dataset,
    seed,
    mlm_iterations,
    pad_word,
    sos_word,
    state_feat_len,
    vocab_size,
    batch_size,
    device="cuda",
    nlayers=8,
    nhead=8,
    hidden_size=512,
    dropout_p=0.1,
    batch_size_mult=1,
    precision=32,
    lr=1e-4,
    decay_power=-1,
    warmup_proportion=0.1,
    load=None,
):
    dataset_name = "gscan"
    check_val_every = 8000

    exp_name = "enc_dec"
    model_name = f"transformer_l_{nlayers}_h_{nhead}_d_{hidden_size}"
    dataset_name = dataset_name
    effective_batch_size = batch_size * batch_size_mult
    exp_name = f"{exp_name}_s_{seed}_m_{model_name}_it_{mlm_iterations}_b_{effective_batch_size}_d_{dataset_name}_drop_{dropout_p}"
    model_dir = f"models/{exp_name}/{model_name}"
    model_path = f"{model_dir}/{exp_name}.pt"
    print(model_path)
    print(
        f"Batch size {batch_size}, mult {batch_size_mult}, total {batch_size * batch_size_mult}"
    )

    train_dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)

    logs_root_dir = f"logs/{exp_name}/{model_name}/{dataset_name}/{seed}"

    num_positions = 72

    model = StateEncoderDecoderLanguageModel(
        vocab_size,
        num_positions,
        pad_word,
        sos_word,
        state_feat_len,
        nlayers=nlayers,
        nhead=nhead,
        emb_dim=hidden_size,
        dropout=dropout_p,
        norm_first=True,
        lr=1e-4,
        decay_power=-1,
        warmup_proportion=0.1,
    )

    if load is not None:
        model.load_state_dict(torch.load(load))

    os.makedirs(logs_root_dir, exist_ok=True)
    trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(logs_root_dir),
        ],
        callbacks=[pl.callbacks.LearningRateMonitor()],
        max_steps=mlm_iterations,
        num_sanity_val_steps=10,
        accelerator="gpu",
        devices=1,
        precision=(
            "bf16-mixed" if (
                precision == 16 and
                torch.cuda.is_bf16_supported()
            ) else (
                "16-mixed" if precision == 16 and torch.cuda.is_available()
                else (
                    "32"
                )
            )
        ),
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=batch_size_mult,
        limit_val_batches=128,
        # gradient_clip_val=0.2,
    )

    trainer.fit(model, train_dataloader)

    # We validate on the train_dataloader, but its just a sanity check to make sure that
    # we loaded a valid model
    trainer.validate(model, train_dataloader)

    return model


def train_encoder_decoder(
    dataset,
    seed,
    mlm_train_iterations,
    pad_word,
    sos_word,
    eos_word,
    vocab_size,
    batch_size,
    nlayers=4,
    nhead=8,
    hidden_size=128,
    dropout_p=0.1,
    batch_size_mult=1,
    precision=32,
    lr=1e-4,
    decay_power=-1,
    warmup_proportion=0.1,
    device="cpu",
):
    dataset_name = "cogs"
    check_val_every = 8000

    exp_name = "enc_dec"
    model_name = f"transformer_l_{nlayers}_h_{nhead}_d_{hidden_size}"
    dataset_name = dataset_name
    effective_batch_size = train_batch_size * batch_size_mult
    exp_name = f"{exp_name}_s_{seed}_m_{model_name}_it_{mlm_train_iterations}_b_{effective_batch_size}_d_{dataset_name}_drop_{dropout_p}"
    model_dir = f"models/{exp_name}/{model_name}"
    model_path = f"{model_dir}/{exp_name}.pt"
    print(model_path)
    print(
        f"Batch size {batch_size}, mult {batch_size_mult}, total {batch_size * batch_size_mult}"
    )

    train_dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)

    logs_root_dir = f"logs/{exp_name}/{model_name}/{dataset_name}/{seed}"

    num_positions = 72

    model = EncoderDecoderLanguageModel(
        vocab_size,
        num_positions,
        pad_word,
        sos_word,
        eos_word,
        nlayers=nlayers,
        nhead=nhead,
        emb_dim=hidden_size,
        dropout=dropout_p,
        norm_first=True,
        lr=lr,
        decay_power=decay_power,
        warmup_proportion=warmup_proportion,
    )

    trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(logs_root_dir),
            LoadableCSVLogger(logs_root_dir, flush_logs_every_n_steps=10),
        ],
        callbacks=[pl.callbacks.LearningRateMonitor()],
        max_steps=mlm_train_iterations,
        num_sanity_val_steps=10,
        gpus=1 if device == "cuda" else 0,
        precision=(
            "bf16" if (
                precision == 16 and
                torch.cuda.is_bf16_supported()
            ) else (
                "16-mixed" if precision == 16 and torch.cuda.is_available()
                else (
                    "32"
                )
            )
        ),
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=batch_size_mult,
        # gradient_clip_val=0.2,
    )

    trainer.fit(model, train_dataloader)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-demonstrations", type=str, required=True)
    parser.add_argument("--valid-demonstrations-directory", type=str, required=True)
    parser.add_argument("--output-directory", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--batch-size-mult", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--nlayers", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--precision", type=int, choices=(16, 32), default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1)
    parser.add_argument("--decay-power", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--enable-progress", action="store_true")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--tag", type=str, default="none")
    parser.add_argument("--dataset-name", type=str, default="gscan")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--dataloader-ncpus", type=int, default=1)
    parser.add_argument("--limit-load", type=int)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    exp_name = "gscan"
    model_name = f"bart_l_{args.nlayers}_h_{args.nhead}_d_{args.hidden_size}"
    dataset_name = args.dataset_name
    effective_batch_size = args.train_batch_size * args.batch_size_mult
    exp_name = f"{exp_name}_s_{args.seed}_m_{model_name}_it_{args.iterations}_b_{effective_batch_size}_d_{dataset_name}_t_{args.tag}_drop_{args.dropout_p}"
    model_dir = f"{args.output_directory}/models/{exp_name}/{model_name}"
    model_path = f"{model_dir}/{exp_name}.pt"
    print(model_path)
    print(
        f"Batch size {args.train_batch_size}, mult {args.batch_size_mult}, total {args.train_batch_size * args.batch_size_mult}"
    )

    torch.set_float32_matmul_precision("medium")
    print("Flash attention:", torch.backends.cuda.flash_sdp_enabled())

    os.makedirs(model_dir, exist_ok=True)

    seed = args.seed
    iterations = args.iterations

    (
        dictionaries,
        (train_demonstrations, valid_demonstrations_dict),
    ) = load_data_directories(args.train_demonstrations, args.dictionary, limit_load=args.limit_load)

    WORD2IDX = dictionaries[0]
    ACTION2IDX = dictionaries[1]

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}
    IDX2ACTION = {i: w for w, i in ACTION2IDX.items()}

    pad_word = WORD2IDX["[pad]"]
    pad_action = ACTION2IDX["[pad]"]
    sos_action = ACTION2IDX["[sos]"]
    eos_action = ACTION2IDX["[eos]"]

    training_data_indices_by_command = {}
    for i in range(len(train_demonstrations)):
        # A workaround in the gSCAN case to not include stuff from split G
        if WORD2IDX.get("cautiously", -1) in train_demonstrations[i][0]:
            continue

        cmd = " ".join(map(lambda x: IDX2WORD[x], train_demonstrations[i][0]))
        if cmd not in training_data_indices_by_command:
            training_data_indices_by_command[cmd] = []
        training_data_indices_by_command[cmd].append(i)

    min_len = min(
        [(cmd, len(x)) for cmd, x in training_data_indices_by_command.items()],
        key=lambda x: x[-1],
    )[-1]

    sentence2idx = {s: i for i, s in enumerate(training_data_indices_by_command)}
    idx2sentence = [s for s in sentence2idx]

    state_component_max_len, state_feat_len = determine_state_profile(
        train_demonstrations,
        valid_demonstrations_dict
    )

    print("State component lengths", state_component_max_len)

    pad_instructions_to, pad_actions_to, pad_state_to = determine_padding(
        itertools.chain.from_iterable([
            train_demonstrations, *valid_demonstrations_dict.values()
        ])
    )

    print(f"Paddings instr: {pad_instructions_to} ({pad_word}) act: {pad_actions_to} ({pad_action}) state: {pad_state_to} (0)")

    pl.seed_everything(0)
    balanced_training_data_subset = SampleSentencesByWordWeights(
        {sentence2idx[s]: v for s, v in training_data_indices_by_command.items()},
        np.ones(len(sentence2idx)) / len(sentence2idx),
        MapDataset(
            PaddingDataset(
                train_demonstrations,
                (
                    pad_instructions_to,
                    pad_actions_to,
                    (pad_state_to, state_feat_len)
                ),
                (pad_word, pad_action, 0)
            ),
            lambda x: (x[0], x[2])
        ),
    )

    model = train_state_encoder_decoder(
        balanced_training_data_subset,
        args.seed,
        args.iterations,
        pad_word,
        WORD2IDX["[sos]"],
        state_feat_len,
        len(WORD2IDX),
        args.train_batch_size,
        device=args.device,
        nlayers=args.nlayers,
        hidden_size=args.hidden_size,
        nhead=args.nhead,
        dropout_p=args.dropout_p,
        lr=args.lr,
        decay_power=args.decay_power,
        warmup_proportion=args.warmup_proportion,
        load=None
    )

    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
