import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from positional_encodings.torch_encodings import PositionalEncoding1D

from gscan_metaseq2seq.models.embedding import BOWEmbedding
from gscan_metaseq2seq.util.dataset import (
    PaddingDataset,
    ReshuffleOnIndexZeroDataset,
    MapDataset,
)
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config


class InducingPointEncoder(nn.Module):
    def __init__(self, emb_dim, nhead, nlayers, norm_first=False, dropout=0.1):
        super().__init__()
        self.inducing_point = nn.Parameter(torch.randn(emb_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_dim,
                dim_feedforward=emb_dim * 4,
                nhead=nhead,
                norm_first=norm_first,
                dropout=dropout,
            ),
            num_layers=nlayers,
        )

    def forward(self, x, key_padding_mask):
        return self.transformer(
            torch.cat(
                [x, self.inducing_point[None, None].expand(x.shape[0], 1, -1)], dim=1
            ).transpose(0, 1),
            src_key_padding_mask=torch.cat(
                [key_padding_mask, torch.zeros_like(key_padding_mask[:, :1])], dim=-1
            ),
        )[-1]


class InstructionCLIPBCELearner(pl.LightningModule):
    def __init__(
        self,
        num_words,
        pad_word_idx,
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
        self.positional_encoding = PositionalEncoding1D(emb_dim)
        self.state_encoder = BOWEmbedding(64, 7, emb_dim)
        self.state_encoder_projection = nn.Linear(7 * emb_dim, emb_dim)
        self.transformer_encoder_instruction = InducingPointEncoder(
            emb_dim, nhead, nlayers, norm_first=norm_first, dropout=dropout
        )
        self.transformer_encoder_state = InducingPointEncoder(
            emb_dim, nhead, nlayers, norm_first=norm_first, dropout=dropout
        )
        self.pad_word_idx = pad_word_idx

    def configure_optimizers(self):
        return transformer_optimizer_config(
            self,
            self.hparams.lr,
            weight_decay=self.hparams.wd,
            decay_power=self.hparams.decay_power,
            warmup_proportion=self.hparams.warmup_proportion,
        )

    def forward(self, x):
        instruction, state, instruction_padding, state_padding = x

        encoded_state = self.state_encoder(state)
        projected_state = self.state_encoder_projection(encoded_state)
        encoded_instruction = self.embedding_instructions(instruction)
        encoded_instruction = encoded_instruction + self.positional_encoding(
            encoded_instruction
        )

        return (
            self.transformer_encoder_instruction(
                encoded_instruction, instruction_padding
            )
            @ self.transformer_encoder_state(projected_state, state_padding).T
        )

    def training_step(self, x, idx):
        instruction, state = x

        instruction_pad = instruction == self.pad_word_idx
        state_pad = torch.zeros_like(state[..., 0])
        # Lets do input dropout by randomly masking parts of the state
        state_pad[
            :,
            torch.randperm(state_pad.shape[1], device=state_pad.device)
            < (state_pad.shape[1] * 0.2),
        ] = 1
        state_pad = state_pad.to(torch.bool)

        outer_product = self.forward((instruction, state, instruction_pad, state_pad))

        matching_instruction = (
            (
                instruction[None, :].expand(instruction.shape[0], -1, -1)
                == instruction[:, None].expand(-1, instruction.shape[0], -1)
            )
            .all(dim=-1)
            .float()
        )

        loss = F.binary_cross_entropy_with_logits(
            outer_product,
            matching_instruction,
            pos_weight=torch.tensor(
                matching_instruction.flatten().shape[0]
                / (matching_instruction[matching_instruction == 1].shape[0]),
                device=self.device,
                dtype=torch.float,
            ),
        )
        self.log("loss", loss)

        return loss

    def predict_step(self, x, idx):
        instruction, state = x

        instruction_pad = instruction == self.pad_word_idx
        state_pad = torch.zeros_like(state[..., 0], dtype=torch.bool)

        outer_product = self.forward((instruction, state, instruction_pad, state_pad))

        return torch.diag(outer_product)


def train_clip(
    balanced_training_data,
    valid_demonstrations_dict,
    seed,
    clip_iterations,
    pad_word,
    vocab_size,
    device="cpu",
):
    nlayers = 4
    nhead = 8
    hidden_size = 128
    dropout_p = 0.1
    train_batch_size = 128
    batch_size_mult = 1
    dataset_name = "gscan_normal"
    check_val_every = 500

    clip_train_dataloader = DataLoader(
        ReshuffleOnIndexZeroDataset(
            MapDataset(balanced_training_data, lambda x: (x[0][1], x[0][0])),
        ),
        batch_size=train_batch_size,
        pin_memory=True,
        shuffle=True,
    )

    clip_valid_dataloaders = [
        DataLoader(
            MapDataset(data, lambda x: (x[0][1], x[0][0])),
            batch_size=train_batch_size,
            pin_memory=True,
        )
        for data in valid_demonstrations_dict.values()
    ]

    pl.seed_everything(seed)
    instruction_clip = InstructionCLIPBCELearner(
        vocab_size,
        pad_word,
        nlayers=4,
        nhead=8,
        emb_dim=512,
        dropout=dropout_p,
        norm_first=False,
        lr=1e-4,
        decay_power=-1,
        warmup_proportion=0.1,
    )

    exp_name = "clip"
    model_name = "transformer"
    dataset_name = "gscan_normal"

    logs_root_dir = f"logs/{exp_name}/{model_name}/{dataset_name}/{seed}"

    check_val_opts = {}
    interval = check_val_every / len(clip_train_dataloader)

    # Every check_val_interval steps, regardless of how large the training dataloader is
    if interval > 1.0:
        check_val_opts["check_val_every_n_epoch"] = math.floor(interval)
    else:
        check_val_opts["val_check_interval"] = interval

    instruction_clip_trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(logs_root_dir),
            LoadableCSVLogger(logs_root_dir, flush_logs_every_n_steps=10),
        ],
        callbacks=[pl.callbacks.LearningRateMonitor()],
        max_steps=clip_iterations,
        num_sanity_val_steps=10,
        gpus=1 if device == "cuda" else 0,
        precision=16 if device == "cuda" else 32,
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=batch_size_mult,
        gradient_clip_val=0.2,
        **check_val_opts,
    )

    instruction_clip_trainer.fit(
        instruction_clip, clip_train_dataloader, clip_valid_dataloaders
    )

    return instruction_clip
