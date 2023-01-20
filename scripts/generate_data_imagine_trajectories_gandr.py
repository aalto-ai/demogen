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

import faiss
from sklearn.feature_extraction.text import TfidfTransformer
from pytorch_lightning.loggers import TensorBoardLogger
from positional_encodings.torch_encodings import PositionalEncoding1D
from collections import defaultdict

from gscan_metaseq2seq.models.embedding import BOWEmbedding
from gscan_metaseq2seq.util.dataset import (
    PaddingDataset,
    MapDataset,
)
from gscan_metaseq2seq.util.load_data import load_data_directories
from gscan_metaseq2seq.util.logging import LoadableCSVLogger
from gscan_metaseq2seq.util.scheduler import transformer_optimizer_config
from gscan_metaseq2seq.models.enc_dec_transformer.enc_dec_transformer_model import (
    TransformerLearner,
    autoregressive_model_unroll_predictions,
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


def train_transformer(
    weights_path,
    dataset,
    validation_datasets,
    seed,
    transformer_train_iterations,
    batch_size,
    x_categories,
    y_categories,
    pad_word_idx,
    pad_action_idx,
    sos_action_idx,
    eos_action_idx,
    hidden_size=128,
    nlayers=8,
    nhead=8,
    device="cpu",
):
    dropout_p = 0.0
    train_batch_size = batch_size
    batch_size_mult = 1
    dataset_name = "gscan"
    check_val_every = 8000

    exp_name = "transformer"
    model_name = f"transformer_l_{nlayers}_h_{nhead}_d_{hidden_size}"
    dataset_name = dataset_name
    effective_batch_size = train_batch_size * batch_size_mult
    exp_name = f"{exp_name}_s_{seed}_m_{model_name}_it_{transformer_train_iterations}_b_{effective_batch_size}_d_{dataset_name}_drop_{dropout_p}"
    model_dir = f"models/{exp_name}/{model_name}"
    model_path = f"{model_dir}/{exp_name}.pt"
    print(model_path)
    print(
        f"Batch size {train_batch_size}, mult {batch_size_mult}, total {train_batch_size * batch_size_mult}"
    )

    pl.seed_everything(seed)
    train_dataloader = DataLoader(
        dataset, batch_size=train_batch_size, pin_memory=True, shuffle=True
    )
    validation_dataloaders = [
        DataLoader(
            ds,
            batch_size=train_batch_size,
            pin_memory=True,
        )
        for ds in validation_datasets
    ]

    logs_root_dir = f"logs/{exp_name}/{model_name}/{dataset_name}/{seed}"

    model = TransformerLearner(
        7,
        x_categories,
        y_categories,
        embed_dim=hidden_size,
        dropout_p=dropout_p,
        nlayers=nlayers,
        nhead=nhead,
        pad_word_idx=pad_word_idx,
        pad_action_idx=pad_action_idx,
        sos_action_idx=sos_action_idx,
        eos_action_idx=eos_action_idx,
        wd=1e-2,
        lr=1e-4,
        decay_power=-1,
        warmup_proportion=0.1,
    )
    print(model)

    if weights_path:
        model.load_state_dict(torch.load(weights_path))

    pl.seed_everything(seed)
    trainer = pl.Trainer(
        logger=[
            TensorBoardLogger(logs_root_dir),
            LoadableCSVLogger(logs_root_dir, flush_logs_every_n_steps=10),
        ],
        callbacks=[pl.callbacks.LearningRateMonitor()],
        max_steps=transformer_train_iterations,
        num_sanity_val_steps=10,
        gpus=1 if device == "cuda" else 0,
        precision=16 if device == "cuda" else 32,
        default_root_dir=logs_root_dir,
        accumulate_grad_batches=batch_size_mult,
        gradient_clip_val=0.2,
    )

    trainer.fit(model, train_dataloader, validation_dataloaders)

    return model


def to_count_matrix(action_word_arrays, word_vocab_size, action_vocab_size):
    count_matrix = np.zeros(
        (len(action_word_arrays), word_vocab_size + action_vocab_size)
    )

    for i, (word_array, action_array) in enumerate(action_word_arrays):
        for element in word_array:
            count_matrix[i, element] += 1
        for element in action_array:
            count_matrix[i, element + word_vocab_size] += 1

    return count_matrix


def to_tfidf(tfidf_transformer, count_matrix):
    return tfidf_transformer.transform(count_matrix).todense().astype("float32")


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


def gandr_like_search(
    transformer_prediction_model,
    index,
    train_dataset,
    tfidf_transformer,
    dataloader,
    sample_n,
    decode_len,
    pad_word_idx,
    pad_action_idx,
    word_vocab_size,
    action_vocab_size,
    device="cpu",
):
    transformer_prediction_model.to(device)
    transformer_prediction_model.eval()

    for batch in dataloader:
        instruction, targets, state = batch

        predicted_targets, logits = transformer_predict(
            transformer_prediction_model,
            state,
            instruction,
            decode_len,
        )

        near_neighbour_distances_batch, near_neighbour_indices_batch = index.search(
            to_tfidf(
                tfidf_transformer,
                to_count_matrix(
                    [
                        (i[i != pad_word_idx], t[t != pad_action_idx])
                        for i, t in zip(instruction, predicted_targets)
                    ],
                    word_vocab_size,
                    action_vocab_size,
                ),
            ),
            sample_n,
        )

        near_neighbour_supports_batch = [
            (
                [train_dataset[i] for i in near_neighbour_indices],
                near_neighbour_distances,
            )
            for near_neighbour_indices, near_neighbour_distances in zip(
                near_neighbour_indices_batch, near_neighbour_distances_batch
            )
        ]

        for i, (supports, distances) in enumerate(near_neighbour_supports_batch):
            yield (
                instruction[i].numpy(),
                targets[i].numpy(),
                state[i].numpy(),
                [s[-1] for s in supports],  # support state
                [s[-3] for s in supports],  # support instruction
                [s[-2] for s in supports],  # support actions
                distances,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", type=str, required=True)
    parser.add_argument("--validation-data-directory", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load-transformer-model", type=str, required=True)
    parser.add_argument("--data-output-directory", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--only-splits", nargs="*", help="Which splits to include")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (train_demonstrations, valid_demonstrations_dict),
    ) = load_data_directories(args.training_data, args.dictionary)

    pad_action = ACTION2IDX["[pad]"]
    pad_word = WORD2IDX["[pad]"]

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}

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
                    PaddingDataset(data, (8, 128, (36, 7)), (pad_word, pad_action, 0)),
                    np.random.permutation(512),
                ),
                batch_size=16,
                pin_memory=True,
            )
            for data in valid_demonstrations_dict.values()
        ],
    )

    print(args.offset, args.offset + args.limit)

    # Make an index from the training data
    np.random.seed(args.seed)
    index = faiss.IndexFlatL2(len(WORD2IDX) + len(ACTION2IDX))
    count_matrix = to_count_matrix(
        [
            (instruction, actions)
            for instruction, actions, state in train_demonstrations
        ],
        len(WORD2IDX),
        len(ACTION2IDX),
    )
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(count_matrix)
    index.add(tfidf_transformer.transform(count_matrix).todense().astype("float32"))

    dataloader_splits = {
        split: DataLoader(
            PaddingDataset(
                Subset(
                    demos,
                    np.arange(
                        min(args.offset, len(demos)),
                        min(
                            args.offset
                            + (len(demos) if not args.limit else args.limit),
                            len(demos),
                        ),
                    ),
                ),
                (8, 128, (36, 7)),
                (pad_word, pad_action, 0),
            ),
            batch_size=args.batch_size,
            pin_memory=True,
        )
        for split, demos in zip(
            itertools.chain.from_iterable(
                [valid_demonstrations_dict.keys(), ["train"]]
            ),
            itertools.chain.from_iterable(
                [valid_demonstrations_dict.values(), [train_demonstrations]]
            ),
        )
        if not args.only_splits or split in args.only_splits
    }

    for split, dataloader in tqdm(dataloader_splits.items()):
        os.makedirs(os.path.join(args.data_output_directory, split), exist_ok=True)

        for i, batch in enumerate(
            batched(
                gandr_like_search(
                    transformer_model,
                    index,
                    train_demonstrations,
                    tfidf_transformer,
                    tqdm(dataloader),
                    16,
                    decode_len=128,
                    pad_word_idx=pad_word,
                    pad_action_idx=pad_action,
                    word_vocab_size=len(WORD2IDX),
                    action_vocab_size=len(ACTION2IDX),
                    device=args.device,
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
