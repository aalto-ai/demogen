import argparse
import itertools
import operator
import os
import math
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from collections import Counter, defaultdict

from gscan_metaseq2seq.models.embedding import BOWEmbedding
from gscan_metaseq2seq.util.dataset import (
    PaddingDataset,
    ReshuffleOnIndexZeroDataset,
    MapDataset,
)
from gscan_metaseq2seq.util.load_data import load_data_directories
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config
from gscan_metaseq2seq.models.enc_dec_transformer.enc_dec_transformer_model import (
    SequenceTransformerLearner,
    TransformerLearner,
    autoregressive_model_unroll_predictions,
)
from gscan_metaseq2seq.models.instruction_gen.masked_language_model import (
    sample_from_mlm,
    train_mlm,
)
from gscan_metaseq2seq.models.instruction_gen.clip_ranking import (
    train_clip,
)

from tqdm.auto import tqdm, trange


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
    dummy_targets = torch.zeros(
        instruction.shape[0],
        decode_len,
        dtype=torch.long,
        device=transformer_learner.device,
    )

    decoded, logits, exacts, _ = autoregressive_model_unroll_predictions(
        transformer_learner,
        (state, instruction),
        dummy_targets,
        transformer_learner.sos_action_idx,
        transformer_learner.eos_action_idx,
        transformer_learner.pad_action_idx,
    )

    return decoded, logits


def make_gscan_instruction_gen_closure(
    instruction_gen_model, pad_target_idx, eos_tgt_idx, device="cpu", noise_level=0.2
):
    instruction_gen_model.to(device)
    instruction_gen_model.eval()

    def generate_instruction(inputs, sample_n):
        state, instruction = inputs
        (
            result_instrs,
            result_samples,
            result_samples_mask,
        ) = sample_from_state_encoder_decoder_model(
            instruction_gen_model,
            instruction,
            state,
            sample_n,
            pad_target_idx,
            eos_tgt_idx,
            noise_level=noise_level,
            device=device,
        )

        is_same_mask = (result_samples == instruction[:, None]).all(dim=-1)

        return [
            s[~m].reshape(-1, s.shape[-1]) for s, m in zip(result_samples, is_same_mask)
        ]

    return generate_instruction


def make_gscan_clip_ranking_closure(clip_ranking_model, pad_word_idx, device="cpu"):
    clip_ranking_model.to(device)
    clip_ranking_model.eval()

    clip_ranking_model.positional_encoding.cached_penc = None

    def compute_scores(instructions, inputs):
        states, query_instructions = inputs

        states = states.to(device)
        instructions = instructions.to(device)

        instruction_pad = instructions == pad_word_idx
        state_pad = torch.zeros_like(states[..., 0])
        state_pad = state_pad.to(torch.bool)

        with torch.inference_mode(), torch.autocast(
            device_type=device, dtype=torch.float16, enabled=True
        ):
            encoded_state = clip_ranking_model.state_encoder(states)
            projected_state = clip_ranking_model.state_encoder_projection(encoded_state)
            encoded_instruction = clip_ranking_model.embedding_instructions(
                instructions
            )
            encoded_instruction = (
                encoded_instruction
                + clip_ranking_model.positional_encoding(encoded_instruction)
            )

            decoded_instruction = clip_ranking_model.transformer_encoder_instruction(
                encoded_instruction, instruction_pad
            )
            decoded_state = clip_ranking_model.transformer_encoder_state(
                projected_state, state_pad
            )

            # Take the componentwise product
            scores = (decoded_instruction * decoded_state).sum(dim=-1)

            return scores

    return compute_scores


def make_gscan_seq2seq_ranking_closure(model, pad_word_idx, sos_word_idx, device="cpu"):
    model.eval()

    def compute_scores(instructions, inputs):
        states, query_instructions = inputs

        states = states.to(device)
        instructions = instructions.to(device)

        instruction_pad = instructions == pad_word_idx
        query_instruction_pad = query_instructions == pad_word_idx
        state_pad = (states == 0).all(dim=-1).bool()

        with torch.inference_mode(), torch.autocast(
            device_type=device, dtype=torch.float16, enabled=True
        ):
            logits = model(
                query_instructions,
                states,
                torch.cat([query_instruction_pad, state_pad], dim=-1),
                query_instruction_pad,
                torch.cat(
                    [
                        torch.ones_like(instructions[:, :1]) * sos_word_idx,
                        instructions[:, :-1],
                    ],
                    dim=-1,
                ),
            ).log_softmax(dim=-1)
            selected_logits = logits.gather(-1, instructions.unsqueeze(-1)).squeeze(-1)
            selected_logits[instruction_pad] = 0.0
            scores = selected_logits.sum(dim=-1) / (~instruction_pad).float().sum(
                dim=-1
            )

            return scores

    return compute_scores


def make_gscan_generate_targets_closure(
    transformer_model, pad_word_idx, pad_action_idx, device="cpu"
):
    transformer_model.to(device)
    transformer_model.eval()

    transformer_model.encoder.pos_encoding.cached_penc = None
    transformer_model.decoder.pos_encoding.cached_penc = None

    def compute_targets(instructions, inputs, decode_len):
        states, query_instructions = inputs

        predicted_targets, logits = transformer_predict(
            transformer_model,
            states,
            instructions,
            decode_len,
        )

        return predicted_targets

    return compute_targets


def make_gscan_format_output_closure():
    def format_output(inputs, targets, sample_scores):
        (generated_instructions, generated_targets, scores) = list(zip(*sample_scores))
        query_state, query_instruction = inputs

        return (
            query_instruction.numpy(),
            targets.numpy(),
            query_state.numpy(),
            query_state.numpy(),
            generated_instructions,
            generated_targets,
            scores,
        )

    return format_output


class SamplingError(Exception):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "SamplingError"


def try_gen_instructions(
    inputs,
    targets,
    instruction_gen_closure,
    instruction_ranking_closure,
    target_gen_closure,
    format_output_closure,
    sample_n,
    batch_size,
    decode_len,
    dictionaries,
    device="cpu",
):
    sampled_instructions = instruction_gen_closure(inputs, sample_n)

    # Now for each element in the batch, we take
    # the set (involves a conversion back to tuples
    # but its fine)
    sampled_instructions_sets = [
        set([tuple(x.tolist()) for x in sampled_instruction])
        for sampled_instruction in sampled_instructions
    ]
    sampled_instruction_set_ids = np.concatenate(
        [np.array([i] * len(s)) for i, s in enumerate(sampled_instructions_sets)]
    )
    # We want batches of inputs, so we transpose here
    sampled_instruction_set_inputs = [
        np.concatenate(
            [
                np.repeat(
                    input_element[None].numpy(),
                    len(sampled_instruction_set),
                    axis=0,
                )
                for input_element, sampled_instruction_set in zip(
                    input_elements_batch, sampled_instructions_sets
                )
            ],
            axis=0,
        )
        for input_elements_batch in inputs
    ]
    try:
        concat_sampled_instruction_set_inputs = np.concatenate(
            [
                np.stack([np.array(x) for x in sampled_instruction_set])
                for sampled_instruction_set in sampled_instructions_sets
            ]
        )
    except ValueError:
        raise SamplingError()

    per_id_results = defaultdict(list)

    for i in trange(
        concat_sampled_instruction_set_inputs.shape[0] // batch_size + 1,
        desc="Score/pred batch",
    ):
        first_index = i * batch_size
        last_index = (i + 1) * batch_size

        # Skip if this batch would have a size of zero
        if first_index >= concat_sampled_instruction_set_inputs.shape[0]:
            continue

        sampled_instruction_set_ids_batch = torch.from_numpy(
            sampled_instruction_set_ids[first_index:last_index]
        ).to(device)
        sampled_instruction_set_inputs_batch = tuple(
            [
                torch.from_numpy(
                    sampled_instruction_set_inputs_elements[first_index:last_index]
                ).to(device)
                for sampled_instruction_set_inputs_elements in sampled_instruction_set_inputs
            ]
        )
        sampled_instruction_set_batch = torch.from_numpy(
            concat_sampled_instruction_set_inputs[first_index:last_index]
        ).to(device)
        sampled_instruction_set_scores = instruction_ranking_closure(
            sampled_instruction_set_batch, sampled_instruction_set_inputs_batch
        )
        sampled_instruction_set_targets = target_gen_closure(
            sampled_instruction_set_batch,
            sampled_instruction_set_inputs_batch,
            decode_len,
        )

        # Now we populate per_id_results with the score
        # and the result_set_ids
        for batch_id, sample_result, predicted_target, score in zip(
            sampled_instruction_set_ids_batch.cpu(),
            sampled_instruction_set_batch.cpu(),
            sampled_instruction_set_targets.cpu(),
            sampled_instruction_set_scores.cpu(),
        ):
            batch_id = batch_id.item()

            # Filter out demonstrations that do not give any new information
            if (targets[batch_id] == predicted_target).all(axis=-1):
                continue

            per_id_results[batch_id].append(
                (sample_result.numpy(), predicted_target.numpy(), score.item())
            )

    # Now we sort the per_id_results ascending by score
    per_id_results = {
        i: sorted(results, key=lambda x: -x[-1])
        for i, results in per_id_results.items()
    }

    if any([not v for v in per_id_results.items()]):
        raise SamplingError()

    # Now yield per_id, the state, the query instruction and all supports
    # and their scores
    return [
        format_output_closure(
            tuple([i[batch_id] for i in inputs]), targets[batch_id], sample_scores
        )
        for batch_id, sample_scores in sorted(
            per_id_results.items(), key=lambda x: x[0]
        )
    ]


def generate_instructions_and_rank(
    instruction_gen_closure,
    instruction_ranking_closure,
    target_gen_closure,
    format_output_closure,
    dataloader,
    sample_n,
    batch_size,
    decode_len,
    dictionaries,
    device="cpu",
):
    for batch in dataloader:
        inputs, targets = batch

        while True:
            try:
                yield from try_gen_instructions(
                    inputs,
                    targets,
                    instruction_gen_closure,
                    instruction_ranking_closure,
                    target_gen_closure,
                    format_output_closure,
                    sample_n,
                    batch_size,
                    decode_len,
                    dictionaries,
                    device=device,
                )
                break
            except SamplingError:
                continue


class StateEncoderDecoderLanguageModel(pl.LightningModule):
    def __init__(
        self,
        num_words,
        num_positions,
        pad_word_idx,
        sos_word_idx,
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
        self.state_encoder = BOWEmbedding(64, 7, emb_dim)
        self.state_encoder_projection = nn.Linear(7 * emb_dim, emb_dim)
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


def sample_from_state_encoder_decoder_model_with_mask(
    model,
    expanded_instruction,
    expanded_state,
    all_mask,
    instruction_mask,
    pad_tgt_idx,
    eos_tgt_idx,
    noise_level=0.2,
    device="cpu",
    deterministic=False,
):
    model.eval()
    model.to(device)

    unroll_length = expanded_instruction.shape[1]

    with torch.inference_mode(), torch.autocast(
        device_type=device, dtype=torch.float16, enabled=True
    ):
        decoded_instruction = (
            torch.ones_like(expanded_instruction[:, :1]) * model.sos_word_idx
        )

        # We always keep the very last pad bit unset, so that the entire
        # input is not padded
        instruction_mask[:, -1] = False

        encodings = model.encode(
            expanded_instruction, expanded_state, all_mask, instruction_mask
        )

        for i in trange(unroll_length, desc="Gen instrs"):
            logits = model.decode(
                encodings,
                all_mask,
                decoded_instruction,
            )
            if deterministic:
                samples = logits[:, -1].argmax(dim=-1)
            else:
                samples = torch.distributions.Categorical(
                    logits=logits[:, -1] + 10
                ).sample()
            decoded_instruction = torch.cat(
                [decoded_instruction, samples[:, None]], dim=1
            )

            # Set the decoded-so-far instruction to be pad_tgt_idx if we hit pad_tgt_idx once
            decoded_instruction[
                (decoded_instruction == pad_tgt_idx).cumsum(dim=-1).bool()
            ] = pad_tgt_idx

        return (
            expanded_instruction.cpu(),
            decoded_instruction[:, 1:].view(expanded_instruction.shape[0], -1).cpu(),
            instruction_mask.view(expanded_instruction.shape[0], -1).cpu(),
        )


def sample_from_state_encoder_decoder_model(
    model,
    instruction,
    state,
    sample_n,
    pad_target_idx,
    eos_target_idx,
    noise_level=0.2,
    device="cpu",
    deterministic=False,
):
    model.eval()
    model.to(device)

    unroll_length = instruction.shape[1]

    expanded_instruction = (
        instruction[:, None].expand(-1, sample_n, -1).flatten(0, 1).clone()
    )
    expanded_instruction = expanded_instruction.to(device)

    expanded_state = state[:, None].expand(-1, sample_n, -1, -1).flatten(0, 1).clone()
    expanded_state = expanded_state.to(device)

    instruction_state_padding = torch.cat(
        [expanded_instruction == pad_target_idx, (expanded_state == 0).all(dim=-1)],
        dim=-1,
    )

    all_mask = torch.logical_or(
        torch.rand(instruction_state_padding.shape, device=device) < noise_level,
        instruction_state_padding,
    )
    instruction_only_mask = all_mask[:, : expanded_instruction.shape[1]]

    # We always keep the very last pad bit unset, so that the entire
    # input is not padded
    all_mask[:, -1] = False
    instruction_only_mask[:, -1] = False

    (
        expanded_instruction,
        decoded_instruction,
        instruction_mask,
    ) = sample_from_state_encoder_decoder_model_with_mask(
        model,
        expanded_instruction,
        expanded_state,
        all_mask,
        instruction_only_mask,
        pad_target_idx,
        eos_target_idx,
        noise_level=noise_level,
        device=device,
        deterministic=deterministic,
    )

    return (
        instruction,
        decoded_instruction.view(instruction.shape[0], sample_n, -1),
        instruction_mask.view(instruction.shape[0], sample_n, -1),
    )


def train_state_encoder_decoder(
    dataset,
    seed,
    mlm_iterations,
    pad_word,
    sos_word,
    vocab_size,
    batch_size,
    device="cuda",
    load=None,
):
    nlayers = 8
    nhead = 8
    hidden_size = 512
    dropout_p = 0.1
    train_batch_size = batch_size
    batch_size_mult = 1
    dataset_name = "gscan"
    check_val_every = 8000

    exp_name = "enc_dec"
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

    train_dataloader = DataLoader(dataset, batch_size=train_batch_size, pin_memory=True)

    logs_root_dir = f"logs/{exp_name}/{model_name}/{dataset_name}/{seed}"

    num_positions = 72

    model = StateEncoderDecoderLanguageModel(
        vocab_size,
        num_positions,
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
        precision="16-mixed" if device == "cuda" else 32,
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


def make_inv_counts_dist(counts_dictionary):
    counts_array = np.zeros(max(counts_dictionary.keys()) + 1)
    for word, count in counts_dictionary.items():
        counts_array[word] = count

    inv_counts = 1 / counts_array
    inv_counts[counts_array == 0] = 0
    inv_counts_dist = inv_counts / inv_counts.sum()

    return inv_counts_dist


class SampleSentencesByWordWeights(IterableDataset):
    def __init__(self, train_data_indices_by_word_idx, word_weights, dataset):
        super().__init__()
        self.train_data_indices_by_word_idx = train_data_indices_by_word_idx
        self.word_weights = word_weights
        self.words_array = np.arange(len(word_weights))
        self.dataset = dataset

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            idx = np.random.choice(self.words_array, replace=True, p=self.word_weights)
            if not self.train_data_indices_by_word_idx[idx]:
                continue

            break

        return self.dataset[
            np.random.choice(self.train_data_indices_by_word_idx[idx], replace=True)
        ]


def gscan_make_closures(args, dictionaries, datasets, extra_data):
    WORD2IDX, ACTION2IDX = dictionaries

    pad_action = ACTION2IDX["[pad]"]
    pad_word = WORD2IDX["[pad]"]

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}

    # Punching through the abstraction a bit, we reach
    # through the MapDataset and PaddingDataset to get the underlying
    # demonstrations without any padding
    train_demonstrations = datasets["train"].dataset.dataset

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

    sentence2idx = {s: i for i, s in enumerate(training_data_indices_by_command)}
    idx2sentence = [s for s in sentence2idx]

    balanced_training_data_subset = SampleSentencesByWordWeights(
        {sentence2idx[s]: v for s, v in training_data_indices_by_command.items()},
        np.ones(len(sentence2idx)) / len(sentence2idx),
        MapDataset(datasets["train"], lambda x: (x[0][1], x[0][0])),
    )

    model = train_state_encoder_decoder(
        balanced_training_data_subset,
        args.seed,
        0 if args.load_mlm_model else args.mlm_train_iterations,
        pad_word,
        WORD2IDX["[sos]"],
        len(WORD2IDX),
        args.batch_size,
        device=args.device,
        load=args.load_mlm_model,
    )

    if args.save_mlm_model:
        torch.save(model.state_dict(), args.save_mlm_model)

    if False:
        instruction_clip = train_clip(
            balanced_training_data_subset,
            args.seed,
            0 if args.load_clip_model else args.clip_train_iterations,
            pad_word,
            len(WORD2IDX),
            device=args.device,
            load=args.load_clip_model,
            dictionaries=dictionaries,
        )

        if args.save_clip_model:
            torch.save(instruction_clip.state_dict(), args.save_clip_model)

        instruction_clip.positional_encoding.cached_penc = None

    os.makedirs(os.path.join(args.data_output_directory), exist_ok=True)

    transformer_model_weights = torch.load(args.load_transformer_model)
    transformer_model = TransformerLearner(
        **transformer_model_weights["hyper_parameters"]
    )
    transformer_model.load_state_dict(transformer_model_weights["state_dict"])

    transformer_model_trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=1 if torch.cuda.is_available() else 0,
        precision="16-mixed" if torch.cuda.is_available() else None,
    )
    transformer_model_trainer.validate(
        transformer_model,
        [
            DataLoader(
                MapDataset(
                    Subset(data, np.random.permutation(512)),
                    lambda x: (x[0][1], x[1], x[0][0]),
                ),
                batch_size=args.batch_size,
                pin_memory=True,
            )
            for data in {k: v for k, v in datasets.items() if k != "train"}.values()
        ],
    )

    return (
        make_gscan_instruction_gen_closure(
            model, pad_word, None, device=args.device, noise_level=0.2
        ),
        make_gscan_seq2seq_ranking_closure(
            model, pad_word, WORD2IDX["[sos]"], device=args.device
        ),
        make_gscan_generate_targets_closure(
            transformer_model, pad_word, pad_action, device=args.device
        ),
        make_gscan_format_output_closure(),
    )


def gscan_load_data(args):
    (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (train_demonstrations, valid_demonstrations_dict),
    ) = load_data_directories(
        args.data_directory, args.dictionary, limit_load=args.limit_load
    )

    dataset_splits = {
        split: MapDataset(
            PaddingDataset(
                demos,
                (32, 128, (36, 7)),
                (WORD2IDX["[pad]"], ACTION2IDX["[pad]"], 0),
            ),
            lambda x: ((x[2], x[0]), x[1]),
        )
        for split, demos in zip(
            itertools.chain.from_iterable(
                [valid_demonstrations_dict.keys(), ["train"]]
            ),
            itertools.chain.from_iterable(
                [valid_demonstrations_dict.values(), [train_demonstrations]]
            ),
        )
    }

    return ((WORD2IDX, ACTION2IDX), dataset_splits, (color_dictionary, noun_dictionary))


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


def train_encoder_decoder(
    dataset,
    seed,
    mlm_train_iterations,
    pad_word,
    sos_word,
    eos_word,
    vocab_size,
    batch_size,
    device="cpu",
):
    nlayers = 4
    nhead = 8
    hidden_size = 128
    dropout_p = 0.1
    train_batch_size = batch_size
    batch_size_mult = 1
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
        f"Batch size {train_batch_size}, mult {batch_size_mult}, total {train_batch_size * batch_size_mult}"
    )

    train_dataloader = DataLoader(dataset, batch_size=train_batch_size, pin_memory=True)

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
        max_steps=mlm_train_iterations,
        num_sanity_val_steps=10,
        gpus=1 if device == "cuda" else 0,
        precision=16 if device == "cuda" else 32,
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=batch_size_mult,
        # gradient_clip_val=0.2,
    )

    trainer.fit(model, train_dataloader)

    return model


def sample_from_encoder_decoder_model_with_mask(
    model,
    expanded_instruction,
    instruction_mask,
    eos_target_idx,
    pad_target_idx,
    noise_level=0.2,
    device="cpu",
    deterministic=False,
):
    model.eval()
    model.to(device)

    unroll_length = expanded_instruction.shape[1]

    with torch.inference_mode(), torch.autocast(
        device_type=device, dtype=torch.float16, enabled=True
    ):
        decoded_instruction = (
            torch.ones_like(expanded_instruction[:, :1]) * model.sos_word_idx
        )

        # We always keep the very last pad bit unset, so that the entire
        # input is not padded
        instruction_mask[:, -1] = False

        for i in range(unroll_length):
            logits = model(expanded_instruction, instruction_mask, decoded_instruction)
            if deterministic:
                samples = logits[:, -1].argmax(dim=-1)
            else:
                samples = torch.distributions.Categorical(
                    logits=logits[:, -1] + 10
                ).sample()
            decoded_instruction = torch.cat(
                [decoded_instruction, samples[:, None]], dim=1
            )

        # Set anything past EOS to be a PAD token
        decoded_eq_mask = (
            (decoded_instruction == eos_target_idx).int().cumsum(dim=-1).bool()[:, :-1]
        )
        decoded_eq_mask = torch.cat(
            [torch.zeros_like(decoded_eq_mask[:, :1]), decoded_eq_mask], dim=-1
        )
        decoded_instruction
        decoded_instruction[decoded_eq_mask] = pad_target_idx

        return (
            expanded_instruction.cpu(),
            decoded_instruction[:, 1:].view(expanded_instruction.shape[0], -1).cpu(),
            instruction_mask.view(expanded_instruction.shape[0], -1).cpu(),
        )


def sample_from_encoder_decoder_model(
    model,
    instruction,
    eos_target_idx,
    pad_target_idx,
    sample_n,
    noise_level=0.2,
    device="cpu",
    deterministic=False,
):
    unroll_length = instruction.shape[1]

    expanded_instruction = instruction[:, None].expand(-1, sample_n, -1).flatten(0, 1)
    expanded_instruction = expanded_instruction.to(device)

    instruction_padding = expanded_instruction == pad_target_idx

    instruction_mask = torch.logical_or(
        torch.rand(instruction_padding.shape, device=device) < noise_level,
        instruction_padding,
    )

    (
        expanded_instruction,
        decoded_instruction,
        instruction_mask,
    ) = sample_from_encoder_decoder_model_with_mask(
        model,
        expanded_instruction,
        instruction_mask,
        eos_target_idx,
        pad_target_idx,
        noise_level=noise_level,
        device=device,
        deterministic=deterministic,
    )

    return (
        instruction,
        decoded_instruction.view(instruction.shape[0], sample_n, -1),
        instruction_mask.view(instruction.shape[0], sample_n, -1),
        [
            [i[m] for i, m in zip(i_batch, m_batch)]
            for i_batch, m_batch in zip(
                expanded_instruction.view(instruction.shape[0], sample_n, -1),
                (~instruction_mask).view(instruction.shape[0], sample_n, -1),
            )
        ],
    )


def sample_from_encoder_decoder_model_deterministic(
    model,
    instruction,
    eos_target_idx,
    pad_target_idx,
    noise_level=0.2,
    device="cpu",
    deterministic=False,
):
    unroll_length = instruction.shape[1]

    expanded_instruction = (
        instruction[:, None].expand(-1, instruction.shape[-1], -1).flatten(0, 1)
    )
    expanded_instruction = expanded_instruction.to(device)

    # We'll make a mask of 1-grams, 2-grams
    mask_lowerbound = (
        torch.arange(expanded_instruction.shape[0], device=device)
        % (expanded_instruction.shape[-1] + 1)
    ) - 1
    mask_upperbound = mask_lowerbound + 1
    instruction_mask_indices = torch.arange(
        expanded_instruction.shape[1], device=device
    )[None].expand(expanded_instruction.shape[0], -1)
    instruction_mask = ~torch.bitwise_and(
        instruction_mask_indices >= mask_lowerbound[:, None],
        instruction_mask_indices < mask_upperbound[:, None],
    )

    (
        expanded_instruction,
        decoded_instruction,
        instruction_mask,
    ) = sample_from_encoder_decoder_model_with_mask(
        model,
        expanded_instruction,
        instruction_mask,
        eos_target_idx,
        pad_target_idx,
        noise_level=noise_level,
        device=device,
        deterministic=deterministic,
    )

    return (
        instruction,
        decoded_instruction.view(instruction.shape[0], instruction.shape[-1], -1),
        instruction_mask.view(instruction.shape[0], instruction.shape[-1], -1),
        [
            [i[m] for i, m in zip(i_batch, m_batch)]
            for i_batch, m_batch in zip(
                expanded_instruction.view(
                    instruction.shape[0], instruction.shape[-1], -1
                ),
                (~instruction_mask).view(
                    instruction.shape[0], instruction.shape[-1], -1
                ),
            )
        ],
    )


def make_cogs_instruction_gen_closure(
    encoder_decoder_model, eos_target_idx, pad_target_idx, device="cpu", noise_level=0.2
):
    encoder_decoder_model.to(device)
    encoder_decoder_model.eval()

    def generate_instruction(inputs, sample_n):
        (query_instructions,) = inputs
        (
            result_instrs,
            result_samples,
            result_samples_mask,
            result_masked_instruction,
        ) = sample_from_encoder_decoder_model(
            encoder_decoder_model,
            query_instructions,
            eos_target_idx,
            pad_target_idx,
            sample_n,
            noise_level=noise_level,
            device="cuda",
            deterministic=False,
        )

        is_same_mask = (result_samples == query_instructions[:, None]).all(dim=-1)

        return [
            s[~m].reshape(-1, s.shape[-1]) for s, m in zip(result_samples, is_same_mask)
        ]

    return generate_instruction


def compute_ll_from_encoder_decoder_model(model, query, samples, device="cuda"):
    samples = samples.to(device)
    query = query.to(device)

    flat_query = query
    flat_samples = samples
    encoder_mask = flat_query == model.pad_word_idx
    decoded_instruction = torch.cat(
        [torch.ones_like(flat_samples[:, :1]) * model.sos_word_idx, flat_samples],
        dim=-1,
    )[:, :-1]

    with torch.inference_mode():
        logits = model(flat_query, encoder_mask, decoded_instruction)
        logprobs = torch.gather(logits, -1, flat_samples[..., None]).squeeze(-1)
        total_logprobs = logprobs.sum(dim=-1)

    return total_logprobs.detach().cpu()


def make_cogs_instruction_ranking_closure(encoder_decoder_model, device="cpu"):
    encoder_decoder_model.to(device)
    encoder_decoder_model.eval()

    def score_instructions(sampled, inputs):
        generated_instructions = sampled
        (query_instructions,) = inputs
        return compute_ll_from_encoder_decoder_model(
            encoder_decoder_model,
            query_instructions,
            generated_instructions,
            device=device,
        )

    return score_instructions


def make_cogs_target_generation_closure(
    transformer_encoder_decoder_model, device="cpu"
):
    transformer_encoder_decoder_model.to(device)
    transformer_encoder_decoder_model.eval()

    transformer_encoder_decoder_model.encoder.pos_encoding.cached_penc = None
    transformer_encoder_decoder_model.decoder.pos_encoding.cached_penc = None

    def compute_targets(instructions, inputs, decode_len):
        (query_instructions,) = inputs

        dummy_targets = torch.zeros(
            query_instructions.shape[0], decode_len, dtype=torch.long, device=device
        )
        instructions = instructions.to(device)

        decoded, logits, exacts, _ = autoregressive_model_unroll_predictions(
            transformer_encoder_decoder_model,
            (instructions,),
            dummy_targets,
            transformer_encoder_decoder_model.sos_action_idx,
            transformer_encoder_decoder_model.eos_action_idx,
            transformer_encoder_decoder_model.pad_action_idx,
        )

        return decoded

    return compute_targets


def make_cogs_format_output_closure():
    def format_output(inputs, targets, sample_scores):
        (generated_instructions, generated_targets, scores) = list(zip(*sample_scores))
        (query_instruction,) = inputs

        return (
            query_instruction.numpy(),
            targets.numpy(),
            generated_instructions,
            generated_targets,
            scores,
        )

    return format_output


def train_encoder_decoder_transformer(
    dataset,
    valid_dataset_dict,
    seed,
    iterations,
    pad_word,
    pad_action,
    sos_action,
    eos_action,
    query_vocab_size,
    output_vocab_size,
    batch_size,
    device="cpu",
):
    nlayers = 4
    nhead = 8
    hidden_size = 128
    dropout_p = 0.1
    train_batch_size = batch_size
    batch_size_mult = 1
    dataset_name = "cogs"
    check_val_every = 8000

    exp_name = "cogs_enc_dec_transformer"
    model_name = f"transformer_l_{nlayers}_h_{nhead}_d_{hidden_size}"
    dataset_name = dataset_name
    effective_batch_size = train_batch_size * batch_size_mult
    exp_name = f"{exp_name}_s_{seed}_m_{model_name}_it_{iterations}_b_{effective_batch_size}_d_{dataset_name}_drop_{dropout_p}"
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
            Subset(ds, np.random.permutation(len(ds))[:128]),
            batch_size=train_batch_size,
            pin_memory=True,
        )
        for ds in valid_dataset_dict.values()
    ]

    logs_root_dir = f"logs/{exp_name}/{model_name}/{dataset_name}/{seed}"

    model = SequenceTransformerLearner(
        query_vocab_size,
        output_vocab_size,
        hidden_size,
        dropout_p,
        nlayers,
        nhead,
        pad_word,
        pad_action,
        sos_action,
        eos_action,
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
        max_steps=iterations,
        num_sanity_val_steps=10,
        gpus=1 if device == "cuda" else 0,
        precision=16 if device == "cuda" else 32,
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=batch_size_mult,
        gradient_clip_val=0.2,
        check_val_every_n_epoch=int(len(train_dataloader) / 5000) + 1,
    )

    trainer.fit(model, train_dataloader, valid_dataloaders)

    return model


def cogs_make_closures(args, dictionaries, datasets, extra_data):
    in_word2idx, out_word2idx = dictionaries

    train_inputs = MapDataset(datasets["train"], lambda x: x[0])
    train_data_indices_by_word_idx = defaultdict(list)

    for i, train_sentence in enumerate(train_inputs):
        for word in train_sentence[0][train_sentence[0] != in_word2idx["[pad]"]]:
            train_data_indices_by_word_idx[word].append(i)

    inv_counts = make_inv_counts_dist(
        {k: len(v) for k, v in train_data_indices_by_word_idx.items()}
    )

    sample_dataset = SampleSentencesByWordWeights(
        train_data_indices_by_word_idx, inv_counts, train_inputs
    )

    encoder_decoder_lm = train_encoder_decoder(
        sample_dataset,
        args.seed,
        args.mlm_train_iterations if not args.load_mlm_model else 0,
        in_word2idx["[pad]"],
        in_word2idx["[sos]"],
        in_word2idx["[eos]"],
        len(in_word2idx),
        args.batch_size,
        device=args.device,
    )

    if args.load_mlm_model:
        encoder_decoder_lm.load_state_dict(torch.load(args.load_mlm_model))

    if args.save_mlm_model:
        torch.save(encoder_decoder_lm.state_dict(), args.save_mlm_model)

    encoder_decoder_transformer = train_encoder_decoder_transformer(
        MapDataset(datasets["train"], lambda x: (x[0][0], x[1])),
        {
            k: MapDataset(v, lambda x: (x[0][0], x[1]))
            for k, v in datasets.items()
            if k != "train"
        },
        args.seed,
        args.transformer_train_iterations if not args.load_transformer_model else 0,
        in_word2idx["[pad]"],
        out_word2idx["[pad]"],
        out_word2idx["[sos]"],
        out_word2idx["[eos]"],
        len(in_word2idx),
        len(out_word2idx),
        args.batch_size,
        device=args.device,
    )

    if args.load_transformer_model:
        encoder_decoder_transformer.load_state_dict(
            torch.load(args.load_transformer_model)
        )

    if args.save_transformer_model:
        torch.save(
            encoder_decoder_transformer.state_dict(), args.save_transformer_model
        )

    return (
        make_cogs_instruction_gen_closure(
            encoder_decoder_lm,
            in_word2idx["[eos]"],
            in_word2idx["[pad]"],
            device=args.device,
            noise_level=0.2,
        ),
        make_cogs_instruction_ranking_closure(encoder_decoder_lm, device=args.device),
        make_cogs_target_generation_closure(
            encoder_decoder_transformer, device=args.device
        ),
        make_cogs_format_output_closure(),
    )


def cogs_parse_task(task):
    input_statement, output_statement, category = task

    return (input_statement.split(), output_statement.split(), category)


def cogs_make_vocab(examples):
    idx2word = sorted(list(set(itertools.chain.from_iterable(examples))))
    idx2word.append("[sos]")
    idx2word.append("[eos]")
    idx2word.append("[pad]")
    word2idx = {w: i for i, w in enumerate(idx2word)}

    return idx2word, word2idx


def cogs_load_data(args, filter_input_length=32, filter_output_length=128):
    train_df = pd.read_csv(
        os.path.join(args.data_directory, "train_100.tsv"), sep="\t", header=None
    )
    test_df = pd.concat(
        [
            pd.read_csv(
                os.path.join(args.data_directory, "test.tsv"), sep="\t", header=None
            ),
            pd.read_csv(
                os.path.join(args.data_directory, "gen.tsv"), sep="\t", header=None
            ),
        ],
        axis=0,
    )

    train_df = train_df.sort_values(2, axis=0)
    test_df = test_df.sort_values(2, axis=0)

    cogs_train = list(map(cogs_parse_task, train_df.values))
    cogs_test = list(map(cogs_parse_task, test_df.values))

    in_idx2word, in_word2idx = cogs_make_vocab(
        map(
            operator.itemgetter(0),
            itertools.chain.from_iterable([cogs_train, cogs_test]),
        )
    )
    out_idx2word, out_word2idx = cogs_make_vocab(
        map(
            operator.itemgetter(1),
            itertools.chain.from_iterable([cogs_train, cogs_test]),
        )
    )

    cogs_train_idx_inputs, cogs_train_idx_outputs = list(
        zip(
            *[
                (
                    [in_word2idx[w] for w in in_sentence] + [in_word2idx["[eos]"]],
                    [out_word2idx[w] for w in out_sentence] + [out_word2idx["[eos]"]],
                )
                for in_sentence, out_sentence, category in cogs_train
            ]
        )
    )

    cogs_test_idx_inputs, cogs_test_idx_outputs = list(
        zip(
            *[
                (
                    [in_word2idx[w] for w in in_sentence] + [in_word2idx["[eos]"]],
                    [out_word2idx[w] for w in out_sentence] + [out_word2idx["[eos]"]],
                )
                for in_sentence, out_sentence, category in cogs_test
            ]
        )
    )

    cogs_grouped_train_data = {
        "train": list(zip(cogs_train_idx_inputs, cogs_train_idx_outputs))
    }
    cogs_grouped_test_data = {
        k: list(map(lambda x: (x[1], x[2]), v))
        for k, v in itertools.groupby(
            zip(cogs_test, cogs_test_idx_inputs, cogs_test_idx_outputs),
            key=lambda x: x[0][-1],
        )
    }

    cogs_grouped_train_datasets = {
        k: MapDataset(
            PaddingDataset(
                MapDataset(
                    list(
                        filter(
                            lambda x: len(x[1]) <= filter_output_length
                            or len(x[0]) <= filter_input_length,
                            v,
                        )
                    ),
                    lambda x: (np.array(x[0]), np.array(x[1])),
                ),
                (filter_input_length, filter_output_length),
                (in_word2idx["[pad]"], out_word2idx["[pad]"]),
            ),
            lambda x: ((x[0],), x[1]),
        )
        for k, v in itertools.chain.from_iterable(
            [cogs_grouped_train_data.items(), cogs_grouped_test_data.items()]
        )
    }

    return ((in_word2idx, out_word2idx), cogs_grouped_train_datasets, None)


def gscan_add_subparser(subparsers):
    gscan_parser = subparsers.add_parser("gscan", help="gscan generation help")
    gscan_parser.add_argument("--dictionary", type=str, required=True)
    gscan_parser.add_argument("--mlm-train-iterations", type=int, default=100000)
    gscan_parser.add_argument("--clip-train-iterations", type=int, default=100000)
    gscan_parser.add_argument("--load-mlm-model", type=str)
    gscan_parser.add_argument("--save-mlm-model", type=str)
    gscan_parser.add_argument("--load-transformer-model", type=str, required=True)
    gscan_parser.add_argument("--load-clip-model", type=str)
    gscan_parser.add_argument("--save-clip-model", type=str)
    gscan_parser.add_argument("--limit-load", type=int, default=None)


def cogs_add_subparser(subparsers):
    gscan_parser = subparsers.add_parser("cogs", help="cogs generation help")
    gscan_parser.add_argument("--mlm-train-iterations", type=int, default=100000)
    gscan_parser.add_argument(
        "--transformer-train-iterations", type=int, default=100000
    )
    gscan_parser.add_argument("--load-mlm-model", type=str)
    gscan_parser.add_argument("--save-mlm-model", type=str)
    gscan_parser.add_argument("--load-transformer-model", type=str)
    gscan_parser.add_argument("--save-transformer-model", type=str)
    gscan_parser.add_argument("--limit-load", type=int, default=None)


DATASET_CONFIGS = {
    "gscan": {
        "add_subparser": gscan_add_subparser,
        "load_data": gscan_load_data,
        "make_closures": gscan_make_closures,
    },
    "cogs": {
        "add_subparser": cogs_add_subparser,
        "load_data": cogs_load_data,
        "make_closures": cogs_make_closures,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", type=str, required=True)
    parser.add_argument("--data-output-directory", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gen-batch-size", type=int, default=16)
    parser.add_argument("--gen-sample-n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only-splits", nargs="*", help="Which splits to include")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    subparsers = parser.add_subparsers(dest="dataset")

    for config_name, config_values in DATASET_CONFIGS.items():
        config_values["add_subparser"](subparsers)

    args = parser.parse_args()

    dictionaries, datasets, extra_data = DATASET_CONFIGS[args.dataset]["load_data"](
        args
    )

    print(args.offset, args.offset + (args.limit or 0))

    dataloader_splits = {
        split: DataLoader(
            Subset(
                dataset,
                np.arange(
                    min(args.offset, len(dataset)),
                    min(
                        args.offset + (len(dataset) if not args.limit else args.limit),
                        len(dataset),
                    ),
                ),
            ),
            batch_size=args.gen_batch_size,
            pin_memory=True,
        )
        for split, dataset in datasets.items()
        if not args.only_splits or split in args.only_splits
    }
    (
        instruction_gen_closure,
        ranking_closure,
        generate_targets_closure,
        format_output_closure,
    ) = DATASET_CONFIGS[args.dataset]["make_closures"](
        args, dictionaries, datasets, extra_data
    )

    os.makedirs(os.path.join(args.data_output_directory), exist_ok=True)
    with open(os.path.join(args.data_output_directory, "dictionary.pb"), "wb") as f:
        pickle.dump(dictionaries, f)

    for split, dataloader in tqdm(dataloader_splits.items()):
        os.makedirs(os.path.join(args.data_output_directory, split), exist_ok=True)

        for i, batch in enumerate(
            batched(
                generate_instructions_and_rank(
                    instruction_gen_closure,
                    ranking_closure,
                    generate_targets_closure,
                    format_output_closure,
                    tqdm(dataloader),
                    args.gen_sample_n,
                    batch_size=args.batch_size,
                    decode_len=128,
                ),
                1000,
            )
        ):
            with open(
                os.path.join(args.data_output_directory, split, f"{i}.pb"), "wb"
            ) as f:
                pickle.dump(batch, f)


if __name__ == "__main__":
    main()
