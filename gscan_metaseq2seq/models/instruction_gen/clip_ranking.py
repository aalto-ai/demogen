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
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(p=dropout)
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
        inputs = self.dropout(self.norm(torch.cat(
            [x, self.inducing_point[None, None].expand(x.shape[0], 1, -1)], dim=1
        )))
        return self.transformer(
            inputs.transpose(0, 1),
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
        self.temp = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def configure_optimizers(self):
        return transformer_optimizer_config(
            self,
            self.hparams.lr,
            weight_decay=self.hparams.wd,
            decay_power=self.hparams.decay_power,
            warmup_proportion=self.hparams.warmup_proportion,
            optimizer_kwargs={"fused": True}
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
            F.normalize(self.transformer_encoder_instruction(
                encoded_instruction, instruction_padding
            ), dim=-1)
            @ F.normalize(self.transformer_encoder_state(projected_state, state_padding), dim=-1).T
        ) * self.temp.exp() + self.bias

    def training_step(self, x, idx):
        instruction, state = x

        instruction_pad = instruction == self.pad_word_idx
        state_pad = torch.rand_like(state[..., 0].float()) < 0.2
        state_pad |= (state == 0).all(dim=-1)

        outer_product = self.forward((instruction, state, instruction_pad, state_pad))

        labels = torch.arange(instruction_pad.shape[0], dtype=torch.long).to(instruction_pad.device)
        loss = (F.cross_entropy(outer_product, labels) + F.cross_entropy(outer_product.T, labels)) / 2.0
        self.log("tloss", loss, prog_bar=True)

        return loss

    def validation_step(self, x, idx):
        return self.training_step(x, idx)

    def predict_step(self, x, idx):
        instruction, state = x

        instruction_pad = instruction == self.pad_word_idx
        state_pad = (state == 0).all(dim=-1)

        outer_product = self.forward((instruction, state, instruction_pad, state_pad))

        return torch.diag(outer_product)


def train_clip(
    balanced_training_data,
    seed,
    clip_iterations,
    pad_word,
    vocab_size,
    device="cpu",
    load=None
):
    nlayers = 8
    nhead = 8
    hidden_size = 512
    dropout_p = 0.1
    train_batch_size = 128
    batch_size_mult = 1
    dataset_name = "gscan_clip"
    check_val_every = 500

    clip_train_dataloader = DataLoader(
        balanced_training_data,
        batch_size=train_batch_size,
        pin_memory=True
    )

    pl.seed_everything(seed)
    instruction_clip = InstructionCLIPBCELearner(
        vocab_size,
        pad_word,
        nlayers=nlayers,
        nhead=nhead,
        emb_dim=hidden_size,
        dropout=dropout_p,
        norm_first=False,
        lr=1e-4,
        decay_power=-1,
        warmup_proportion=0.1
    )
    if load is not None:
        instruction_clip.load_state_dict(torch.load(load))
    print(instruction_clip)

    exp_name = "clip"
    model_name = "transformer"
    dataset_name = "gscan_normal"

    logs_root_dir = f"logs/{exp_name}/{model_name}/{dataset_name}/{seed}"

    check_val_opts = {}

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
        instruction_clip, clip_train_dataloader
    )

    return instruction_clip
