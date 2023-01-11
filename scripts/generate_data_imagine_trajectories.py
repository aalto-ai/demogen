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
from collections import defaultdict

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

from tqdm.auto import tqdm


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

        result_instrs, result_samples, result_samples_mask = sample_from_mlm(
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
