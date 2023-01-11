import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from gscan_metaseq2seq.util.dataset import (
    PaddingDataset,
    ReshuffleOnIndexZeroDataset,
    MapDataset,
)
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config


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


def compute_mlm_model_loss(model, input_sequence, eos_idx):
    input_sequence_mask = (
        torch.stack(
            [
                torch.randperm(input_sequence.shape[-1])
                for _ in range(input_sequence.shape[0])
            ]
        )
        < (input_sequence.shape[-1] * np.random.uniform(0.0, 0.7))
    ).to(model.device)
    logits = model(input_sequence, input_sequence)

    # We only consider the CE loss for the masked tokens
    # but not the unmasked ones
    target_instruction = input_sequence.clone()
    target_instruction[~input_sequence_mask] = eos_idx

    loss = F.cross_entropy(
        logits.flatten(0, -2),
        target_instruction.flatten(),
        ignore_index=eos_idx,
    )

    return {"loss": loss}


class MaskedLanguageModelLearner(pl.LightningModule):
    def __init__(
        self,
        num_words,
        pad_word_idx,
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
        self.positional_encoding = MonotonicRandomPositionEmbedding(16, emb_dim)
        self.transformer_encoder_instruction = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_dim,
                dim_feedforward=emb_dim * 4,
                dropout=dropout,
                nhead=nhead,
                norm_first=norm_first,
            ),
            num_layers=nlayers,
        )
        self.project = nn.Linear(emb_dim, num_words)
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

    def forward(self, base_instruction, instruction_mask):
        masked_instruction = base_instruction.clone()
        masked_instruction[instruction_mask] = self.pad_word_idx

        encoded_instruction = self.embedding_instructions(
            masked_instruction
        ) + self.positional_encoding(masked_instruction)
        decoded_instruction = self.transformer_encoder_instruction(
            encoded_instruction.transpose(0, 1)
        ).transpose(0, 1)

        return self.project(decoded_instruction)

    def training_step(self, x, idx):
        (instruction,) = x
        stats = compute_mlm_model_loss(self, instruction, self.eos_word_idx)
        self.log("loss", stats["loss"])

        return stats["loss"]

    def validation_step(self, x, idx, dl_idx):
        (instruction,) = x
        stats = compute_mlm_model_loss(self, instruction, self.eos_word_idx)
        self.log("vloss", stats["loss"])

        return stats["loss"]


def sample_from_mlm(model, instruction, sample_n, noise_level=0.2, device="cpu"):
    model.eval()
    model.to(device)

    with torch.inference_mode():
        expanded_instruction = (
            instruction[:, None].expand(-1, sample_n, -1).flatten(0, 1)
        )
        expanded_instruction = expanded_instruction.to(device)

        instruction_mask = (
            torch.stack(
                [
                    torch.randperm(expanded_instruction.shape[-1])
                    for _ in range(expanded_instruction.shape[0])
                ]
            )
            < (expanded_instruction.shape[-1] * noise_level)
        ).to(device)
        logits = model(expanded_instruction, instruction_mask)

        samples = torch.distributions.Categorical(logits=logits).sample()

        resampled = (
            samples * instruction_mask + expanded_instruction * ~instruction_mask
        )

        return (
            instruction.cpu(),
            resampled.view(instruction.shape[0], sample_n, -1).cpu(),
            instruction_mask.view(instruction.shape[0], sample_n, -1).cpu(),
        )


def train_mlm(
    balanced_training_data,
    valid_demonstrations_dict,
    seed,
    mlm_iterations,
    pad_word,
    sos_word,
    vocab_size,
    device="cpu",
):
    dataset = ReshuffleOnIndexZeroDataset(data)

    nlayers = 4
    nhead = 8
    hidden_size = 128
    dropout_p = 0.1
    train_batch_size = 128
    batch_size_mult = 1
    dataset_name = "gscan_normal"
    check_val_every = 500

    exp_name = "gscan_mlm"
    model_name = f"transformer_l_{nlayers}_h_{nhead}_d_{hidden_size}"
    dataset_name = dataset_name
    effective_batch_size = train_batch_size * batch_size_mult
    exp_name = f"{exp_name}_s_{seed}_m_{model_name}_it_{mlm_iterations}_b_{effective_batch_size}_d_{dataset_name}_drop_{dropout_p}"
    model_dir = f"models/{exp_name}/{model_name}"
    model_path = f"{model_dir}/{exp_name}.pt"
    print(model_path)
    print(
        f"Batch size {train_batch_size}, mult {batch_size_mult}, total {train_batch_size * batch_size_mult}"
    )

    train_dataloader = DataLoader(
        dataset, batch_size=train_batch_size, pin_memory=True, shuffle=True
    )

    valid_dataloaders = [
        DataLoader(
            Subset(
                data,
                np.random.permutation(len(data))[:512],
            ),
            batch_size=train_batch_size,
            pin_memory=True,
        )
        for data in valid_demonstrations_dict.values()
    ]

    check_val_opts = {}
    interval = check_val_every / len(train_dataloader)

    # Every check_val_interval steps, regardless of how large the training dataloader is
    if interval > 1.0:
        check_val_opts["check_val_every_n_epoch"] = math.floor(interval)
    else:
        check_val_opts["val_check_interval"] = interval

    logs_root_dir = f"logs/{exp_name}/{model_name}/{dataset_name}/{seed}"

    model = MaskedLanguageModelLearner(
        vocab_size,
        pad_word,
        sos_word,
        nlayers=nlayers,
        nhead=nhead,
        emb_dim=hidden_size,
        dropout=dropout_p,
        norm_first=True,
        lr=1e-4,
        decay_power=-1,
        warmup_proportion=0.1,
    )

    trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(logs_root_dir),
            LoadableCSVLogger(logs_root_dir, flush_logs_every_n_steps=10),
        ],
        callbacks=[pl.callbacks.LearningRateMonitor()],
        max_steps=mlm_iterations,
        num_sanity_val_steps=10,
        gpus=1 if device == "cuda" else 0,
        precision=16 if device == "cuda" else 32,
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=batch_size_mult,
        gradient_clip_val=0.2,
        **check_val_opts,
    )

    trainer.fit(model, train_dataloader, valid_dataloaders)

    return model
