import argparse
import itertools
import os
import math
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from positional_encodings.torch_encodings import PositionalEncoding1D
from collections import defaultdict

from gscan_metaseq2seq.models.embedding import BOWEmbedding
from gscan_metaseq2seq.util.dataset import (
    PaddingDataset,
    ReshuffleOnIndexZeroDataset,
    MapDataset,
)
from gscan_metaseq2seq.util.load_data import load_data
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config
from gscan_metaseq2seq.models.enc_dec_transformer.enc_dec_transformer_model import (
    TransformerLearner,
    autoregressive_model_unroll_predictions
)

from tqdm.auto import tqdm


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


class MaskedLanguageModel(pl.LightningModule):
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

        instruction_mask = (
            torch.stack(
                [
                    torch.randperm(instruction.shape[-1])
                    for _ in range(instruction.shape[0])
                ]
            )
            < (instruction.shape[-1] * np.random.uniform(0.0, 0.7))
        ).to(self.device)
        logits = self(instruction, instruction_mask)

        # We only consider the CE loss for the masked tokens
        # but not the unmasked ones
        target_instruction = instruction.clone()
        target_instruction[~instruction_mask] = self.eos_word_idx

        loss = F.cross_entropy(
            logits.flatten(0, -2),
            target_instruction.flatten(),
            ignore_index=self.eos_word_idx,
        )
        self.log("loss", loss)

        return loss

    def validation_step(self, x, idx, dl_idx):
        (instruction,) = x

        instruction_mask = (
            torch.stack(
                [
                    torch.randperm(instruction.shape[-1])
                    for _ in range(instruction.shape[0])
                ]
            )
            < (instruction.shape[-1] * 0.2)
        ).to(self.device)
        logits = self(instruction, instruction_mask)

        # We only consider the CE loss for the masked tokens
        # but not the unmasked ones
        target_instruction = instruction.clone()
        target_instruction[~instruction_mask] = self.eos_word_idx

        loss = F.cross_entropy(
            logits.flatten(0, -2),
            target_instruction.flatten(),
            ignore_index=self.eos_word_idx,
        )
        self.log("vloss", loss, prog_bar=True)

        return loss


def sample_from_model(model, instruction, sample_n, noise_level=0.2, device="cpu"):
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


class InstructionCLIPBCE(pl.LightningModule):
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


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))

        if not batch:
            break

        yield batch


def transformer_predict(transformer_learner, state, instruction, decode_len):
    state = state.to(transformer_learner.device)
    instruction = instruction.to(transformer_learner.device)
    encodings, key_padding_mask = transformer_learner.encode(state, instruction)
    dummy_targets = torch.zeros(instruction.shape[0], decode_len, dtype=torch.long, device=transformer_learner.device)

    decoded, logits, exacts, _ = autoregressive_model_unroll_predictions(
        transformer_learner,
        (state, instruction),
        dummy_targets,
        transformer_learner.sos_action_idx,
        transformer_learner.eos_action_idx,
        transformer_learner.pad_action_idx
    )

    return decoded, logits


def generate_instructions_and_rank(
    instruction_gen_model,
    instruction_clip_model,
    transformer_prediction_model,
    dataloader,
    sample_n,
    batch_size,
    noise_level,
    decode_len,
    pad_word_idx,
    device="cpu",
):
    instruction_gen_model.to(device)
    instruction_gen_model.eval()
    instruction_clip_model.to(device)
    instruction_clip_model.eval()
    transformer_prediction_model.to(device)
    transformer_prediction_model.eval()

    instruction_clip_model.positional_encoding.cached_penc = None

    for batch in dataloader:
        instruction, targets, state = batch

        result_instrs, result_samples, result_samples_mask = sample_from_model(
            instruction_gen_model,
            instruction,
            sample_n,
            noise_level=noise_level,
            device=device,
        )

        # Now for each element in the batch, we take
        # the set (involves a conversion back to tuples
        # but its fine)
        result_sets = [
            set([tuple(x.tolist()) for x in result_sample])
            for result_sample in result_samples
        ]
        result_set_ids = np.concatenate(
            [np.array([i] * len(s)) for i, s in enumerate(result_sets)]
        )
        result_set_states = np.concatenate(
            [
                np.repeat(s[None], len(result_set), axis=0)
                for s, result_set in zip(state, result_sets)
            ]
        )

        result_sets_concat = np.concatenate(
            [
                np.stack([np.array(x) for x in result_sample_set])
                for result_sample_set in result_sets
            ]
        )

        per_id_results = defaultdict(list)

        # Now we take the result sets and we split into batches
        # of size 128 each and use that with the CLIP model
        for i in range(result_sets_concat.shape[0] // batch_size + 1):
            result_set_ids_batch = result_set_ids[i * batch_size : (i + 1) * batch_size]
            result_set_states_batch = result_set_states[
                i * batch_size : (i + 1) * batch_size
            ]
            result_set_batch = result_sets_concat[i * batch_size : (i + 1) * batch_size]

            result_set_ids_batch = torch.from_numpy(result_set_ids_batch).to(device)
            result_set_states_batch = torch.from_numpy(result_set_states_batch).to(
                device
            )
            result_set_batch = torch.from_numpy(result_set_batch).to(device)

            instruction_pad = result_set_batch == pad_word_idx
            state_pad = torch.zeros_like(result_set_states_batch[..., 0])
            state_pad = state_pad.to(torch.bool)

            with torch.inference_mode():
                encoded_state = instruction_clip_model.state_encoder(
                    result_set_states_batch
                )
                projected_state = instruction_clip_model.state_encoder_projection(
                    encoded_state
                )
                encoded_instruction = instruction_clip_model.embedding_instructions(
                    result_set_batch
                )
                encoded_instruction = (
                    encoded_instruction
                    + instruction_clip_model.positional_encoding(encoded_instruction)
                )

                decoded_instruction = (
                    instruction_clip_model.transformer_encoder_instruction(
                        encoded_instruction, instruction_pad
                    )
                )
                decoded_state = instruction_clip_model.transformer_encoder_state(
                    projected_state, state_pad
                )

                # Take the componentwise product
                scores = (decoded_instruction * decoded_state).sum(dim=-1)

            predicted_targets, logits = transformer_predict(
                transformer_prediction_model,
                result_set_states_batch,
                result_set_batch,
                decode_len,
            )

            # Now we populate per_id_results with the score
            # and the result_set_ids
            for batch_id, sample_result, predicted_target, score in zip(
                result_set_ids_batch.cpu(),
                result_set_batch.cpu(),
                predicted_targets.cpu(),
                scores.cpu(),
            ):
                batch_id = batch_id.item()
                if not (sample_result == instruction[batch_id]).all():
                    per_id_results[batch_id].append(
                        (sample_result.numpy(), predicted_target.numpy(), score.item())
                    )

        # Now we sort the per_id_results ascending by score
        per_id_results = {
            i: sorted(results, key=lambda x: -x[-1])
            for i, results in per_id_results.items()
        }

        # Now yield per_id, the state, the query instruction and all supports
        # and their scores
        for batch_id, sample_scores in sorted(
            per_id_results.items(), key=lambda x: x[0]
        ):
            yield (
                instruction[batch_id].numpy(),
                targets[batch_id].numpy(),
                state[batch_id].numpy(),
                state[batch_id].numpy(),
                [s[0] for s in sample_scores],
                [s[1] for s in sample_scores],
                [s[2] for s in sample_scores],
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
    dataset = ReshuffleOnIndexZeroDataset(
        PaddingDataset(
            MapDataset(balanced_training_data, lambda x: (x[0],)), (8,), (pad_word,)
        )
    )

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
                PaddingDataset(MapDataset(data, lambda x: (x[0],)), (8,), (pad_word,)),
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

    model = MaskedLanguageModel(
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
            PaddingDataset(
                MapDataset(balanced_training_data, lambda x: (x[0], x[2])),
                (8, None),
                (pad_word, None),
            )
        ),
        batch_size=train_batch_size,
        pin_memory=True,
        shuffle=True,
    )

    clip_valid_dataloaders = [
        DataLoader(
            PaddingDataset(
                MapDataset(data, lambda x: (x[0], x[2])), (8, None), (pad_word, None)
            ),
            batch_size=16,
            pin_memory=True,
        )
        for data in valid_demonstrations_dict.values()
    ]

    pl.seed_everything(seed)
    instruction_clip = InstructionCLIPBCE(
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", type=str, required=True)
    parser.add_argument("--validation-data-directory", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mlm-train-iterations", type=int, default=100000)
    parser.add_argument("--clip-train-iterations", type=int, default=100000)
    parser.add_argument("--load-mlm-model", type=str)
    parser.add_argument("--save-mlm-model", type=str)
    parser.add_argument("--load-transformer-model", type=str, required=True)
    parser.add_argument("--load-clip-model", type=str)
    parser.add_argument("--save-mlm-model", type=str)
    parser.add_argument("--data-output-directory", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--only-splits", nargs="*", description="Which splits to include"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    seed = args.seed
    mlm_iterations = args.mlm_train_iterations
    clip_iterations = args.clip_train_iterations

    (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (train_demonstrations, valid_demonstrations_dict),
    ) = load_data(args.training_data, args.validation_data_directory, args.dictionary)

    pad_action = ACTION2IDX["[pad]"]
    pad_word = WORD2IDX["[pad]"]

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}

    training_data_indices_by_command = {}
    for i in range(len(train_demonstrations)):
        if WORD2IDX["cautiously"] in train_demonstrations[i][0]:
            continue

        cmd = " ".join(map(lambda x: IDX2WORD[x], train_demonstrations[i][0]))
        if cmd not in training_data_indices_by_command:
            training_data_indices_by_command[cmd] = []
        training_data_indices_by_command[cmd].append(i)

    min_len = min(
        [(cmd, len(x)) for cmd, x in training_data_indices_by_command.items()],
        key=lambda x: x[-1],
    )[-1]
    balanced_training_data = list(
        itertools.chain.from_iterable(
            [
                [train_demonstrations[i] for i in x[:min_len]]
                for x in training_data_indices_by_command.values()
            ]
        )
    )

    model = train_mlm(
        balanced_training_data,
        valid_demonstrations_dict,
        seed,
        0 if args.load_mlm else mlm_iterations,
        pad_word,
        WORD2IDX["[sos]"],
        len(WORD2IDX),
        device=args.device,
    )

    if args.load_mlm:
        model.load_state_dict(torch.load(args.load_mlm))

    if args.save_mlm:
        torch.save(model.state_dict(), args.save_mlm)

    instruction_clip = train_clip(
        balanced_training_data,
        valid_demonstrations_dict,
        seed,
        0 if args.load_clip else clip_iterations,
        pad_word,
        len(WORD2IDX),
        device=args.device,
    )

    if args.load_clip:
        instruction_clip.load_state_dict(torch.load(args.load_clip))

    if args.save_clip:
        torch.save(instruction_clip.state_dict(), args.save_clip)

    instruction_clip.positional_encoding.cached_penc = None

    os.makedirs(os.path.join(args.data_output_directory), exist_ok=True)

    transformer_model_weights = torch.load(args.load_transformer_model)
    transformer_model = TransformerLearner(
        **transformer_model_weights["hyper_parameters"]
    )
    transformer_model.load_state_dict(transformer_model_weights["state_dict"])

    transformer_model_trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16 if torch.cuda.is_available() else None,
    )
    transformer_model_trainer.validate(
        transformer_model,
        [
            DataLoader(
                Subset(
                    PaddingDataset(data, (8, 128, None), (pad_word, pad_action, None)),
                    np.random.permutation(512),
                ),
                batch_size=16,
                pin_memory=True,
            )
            for data in valid_demonstrations_dict.values()
        ],
    )

    dataloader_splits = {
        split: DataLoader(
            PaddingDataset(demos, (8, 128, None), (pad_word, pad_action, None)),
            batch_size=16,
            pin_memory=True,
        )
        for split, demos in zip(
            itertools.chain.from_iterable([valid_demonstrations_dict.keys(), "train"]),
            itertools.chain.from_iterable(
                [valid_demonstrations_dict.values(), train_demonstrations]
            ),
        )
        if not args.only_splits or split in args.only_splits
    }

    for split, dataloader in tqdm(dataloader_splits):
        os.makedirs(os.path.join(args.data_output_directory, split), exist_ok=True)

        for i, batch in enumerate(
            batched(
                generate_instructions_and_rank(
                    model,
                    instruction_clip,
                    transformer_model,
                    tqdm(dataloader),
                    256,
                    batch_size=args.batch_size,
                    noise_level=0.1,
                    decode_len=128,
                    pad_word_idx=pad_word,
                    device=args.device,
                ),
                1000,
            )
        ):
            with open(
                os.path.join(args.data_output_directory, split, f"{i}.pb"), "wb"
            ) as f:
                pickle.dump(batch, f)
