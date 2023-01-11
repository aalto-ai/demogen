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


def make_gscan_instruction_gen_closure(
    instruction_gen_model, device="cpu", noise_level=0.2
):
    instruction_gen_model.to(device)
    instruction_gen_model.eval()

    def generate_instruction(inputs, sample_n):
        state, instruction = inputs
        result_instrs, result_samples, result_samples_mask = sample_from_mlm(
            instruction_gen_model,
            instruction,
            sample_n,
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

        with torch.inference_mode():
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


def generate_instructions_and_rank(
    instruction_gen_closure,
    instruction_ranking_closure,
    target_gen_closure,
    format_output_closure,
    dataloader,
    sample_n,
    batch_size,
    decode_len,
    device="cpu",
):
    for batch in dataloader:
        inputs, targets = batch

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
        concat_sampled_instruction_set_inputs = np.concatenate(
            [
                np.stack([np.array(x) for x in sampled_instruction_set])
                for sampled_instruction_set in sampled_instructions_sets
            ]
        )

        per_id_results = defaultdict(list)

        for i in range(
            concat_sampled_instruction_set_inputs.shape[0] // batch_size + 1
        ):
            first_index = i * batch_size
            last_index = (i + 1) * batch_size
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
            yield format_output_closure(
                tuple([i[batch_id] for i in inputs]), targets[batch_id], sample_scores
            )


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
    balanced_training_data_indices = np.array(
        list(
            itertools.chain.from_iterable(
                [x[:min_len] for x in training_data_indices_by_command.values()]
            )
        )
    )

    balanced_training_data_subset = Subset(
        datasets["train"], balanced_training_data_indices
    )

    model = train_mlm(
        MapDataset(balanced_training_data_subset, lambda x: (x[0][1],)),
        {
            k: MapDataset(v, lambda x: (x[0][1],))
            for k, v in datasets.items() if k != "train"
        },
        args.seed,
        0 if args.load_mlm_model else args.mlm_iterations,
        pad_word,
        WORD2IDX["[sos]"],
        len(WORD2IDX),
        device=args.device,
    )

    if args.load_mlm_model:
        model.load_state_dict(torch.load(args.load_mlm_model))

    if args.save_mlm_model:
        torch.save(model.state_dict(), args.save_mlm_model)

    instruction_clip = train_clip(
        MapDataset(balanced_training_data_subset, lambda x: (x[0][1], x[0][0])),
        {
            k: MapDataset(v, lambda x: (x[0][1], x[0][0]))
            for k, v in datasets.items() if k != "train"
        },
        {k: v for k, v in datasets.items() if k != "train"},
        args.seed,
        0 if args.load_clip_model else args.clip_iterations,
        pad_word,
        len(WORD2IDX),
        device=args.device,
    )

    if args.load_clip_model:
        instruction_clip.load_state_dict(torch.load(args.load_clip_model))

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
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16 if torch.cuda.is_available() else None,
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
        make_gscan_instruction_gen_closure(model, device=args.device, noise_level=0.1),
        make_gscan_clip_ranking_closure(instruction_clip, pad_word, device=args.device),
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
                (8, 128, (36, 7)),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", type=str, required=True)
    parser.add_argument("--data-output-directory", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only-splits", nargs="*", help="Which splits to include")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    subparsers = parser.add_subparsers()

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
    args = parser.parse_args()

    dictionaries, datasets, extra_data = gscan_load_data(args)

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
            batch_size=16,
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
    ) = gscan_make_closures(args, dictionaries, datasets, extra_data)

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
                    256,
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
