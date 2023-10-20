import argparse
from bisect import bisect_left
import ujson as json
import itertools
import numpy as np
from typing import List
import os
import pickle
import multiprocessing
import faiss
from sklearn.preprocessing import normalize
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.auto import tqdm, trange

from gscan_metaseq2seq.gscan.world import Situation
from gscan_metaseq2seq.util.solver import (
    create_vocabulary,
    create_world
)

from gscan_metaseq2seq.util.dataset import PaddingIterableDataset

DIR_TO_INT = {"west": 3, "east": 1, "north": 2, "south": 0}
INT_TO_DIR = {
    direction_int: direction for direction, direction_int in DIR_TO_INT.items()
}


def parse_command_repr(command_repr: str) -> List[str]:
    return command_repr.split(",")


def parse_sparse_situation(
    situation_representation: dict,
    grid_size: int,
    color2idx,
    noun2idx,
    world_encoding_scheme,
    reascan_boxes,
) -> np.ndarray:
    """
    Each grid cell in a situation is fully specified by a vector:
    [_ _ _ _ _ _ _   _       _      _       _   _ _ _ _]
     1 2 3 4 r g b circle square cylinder agent E S W N
     _______ _____ ______________________ _____ _______
       size  color        shape           agent agent dir.
    :param situation_representation: data from dataset.txt at key "situation".
    :param grid_size: int determining row/column number.
    :return: grid to be parsed by computational models.
    """
    # Place the agent.
    agent_row = int(situation_representation["agent_position"].row)
    agent_column = int(situation_representation["agent_position"].column)
    agent_direction = DIR_TO_INT[situation_representation["agent_direction"].name]

    grid = None

    if world_encoding_scheme == "sequence":
        # attribute bits + agent + agent direction + location
        num_grid_channels = 7 + (3 if reascan_boxes else 0)
        grid = []
        agent_representation = np.zeros([num_grid_channels], dtype=np.int32)
        agent_representation[3] = 1
        agent_representation[4] = agent_direction
        agent_representation[5] = agent_row
        agent_representation[6] = agent_column
        grid.append(agent_representation)

        for placed_object in situation_representation["objects"]:
            object_row = int(placed_object.position.row)
            object_column = int(placed_object.position.column)
            if placed_object.object.shape != "box":
                grid.append(
                    np.array(
                        [
                            int(placed_object.object.size),
                            int(color2idx[placed_object.object.color]),
                            int(noun2idx[placed_object.object.shape]),
                            0,
                            0,
                            object_row,
                            object_column,
                        ]
                        + ([0] * 3 if reascan_boxes else [])
                    )
                )
            elif reascan_boxes:
                grid.append(
                    np.array(
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                            object_row,
                            object_column,
                            int(placed_object.object.size),
                            int(color2idx[placed_object.object.color]),
                            1,
                        ]
                    )
                )

    elif world_encoding_scheme == "all":
        # attribute bits + agent + agent direction
        num_grid_channels = 5 + (3 if reascan_boxes else 0)
        grid = np.zeros([grid_size, grid_size, num_grid_channels], dtype=int)
        agent_representation = np.zeros([num_grid_channels], dtype=np.int32)
        agent_representation[-2] = 1
        agent_representation[-1] = agent_direction

        grid[agent_row, agent_column, :] = agent_representation

        # Loop over the objects in the world and place them.
        for placed_object in situation_representation["objects"]:
            object_row = int(placed_object.position.row)
            object_column = int(placed_object.position.column)

            if placed_object.object.shape != "box":
                grid[object_row, object_column, 0] = int(placed_object.object.size)
                grid[object_row, object_column, 1] = int(
                    color2idx[placed_object.object.color]
                )
                grid[object_row, object_column, 2] = int(
                    noun2idx[placed_object.object.shape]
                )

            if reascan_boxes and placed_object.object.shape == "box":
                grid[object_row, object_column, 5] = int(placed_object.object.size)
                grid[object_row, object_column, 6] = int(
                    color2idx[placed_object.object.color]
                )
                grid[object_row, object_column, 7] = 1

        grid = add_positional_information_to_grid(grid)

    return grid


def add_positional_information_to_grid(grid):
    grid_pos = np.concatenate(
        [
            grid,
            np.ones_like(grid[..., 0]).cumsum(axis=0)[..., None],
            np.ones_like(grid[..., 0]).cumsum(axis=1)[..., None],
        ],
        axis=-1,
    )
    grid_pos = grid_pos.reshape(-1, grid_pos.shape[-1])

    return grid_pos


def add_positional_information_to_observation(observations):
    observations_pos = np.concatenate(
        [
            observations[0],
            np.ones_like(observations[0][..., 0]).cumsum(axis=0)[..., None],
            np.ones_like(observations[0][..., 0]).cumsum(axis=1)[..., None],
        ],
        axis=-1,
    )
    observations_pos = observations_pos.reshape(-1, observations_pos.shape[-1])

    return observations_pos


def add_eos_to_actions(actions, eos_token_idx):
    actions_pos = np.concatenate(
        [actions, np.ones_like(actions[:1]) * eos_token_idx], axis=-1
    )

    return actions_pos


def vectorize_state(situation, grid_size, color2dix, noun2idx, encoding_scheme, reascan_boxes):
    sparse_situation = parse_sparse_situation(
        Situation.from_representation(situation).to_dict(), 6, color2dix, noun2idx, "all", False
    )
    vectorized_situation = (
        sparse_situation[:, :-2, None] == np.arange(5, dtype=np.int32)[None, None]
    ).reshape(-1).astype(np.float32)
    return sparse_situation, vectorized_situation


def vectorize_state_star(args):
    return vectorize_state(*args)


def get_nearest_neighbour_statistics(dataset, vocabulary, splits, split_names, limit=None):
    colors = sorted(vocabulary.get_color_adjectives())
    COLOR2IDX = {c: i + 1 for i, c in enumerate(colors)}

    nouns = sorted(vocabulary.get_nouns())
    NOUN2IDX = {n: i + 1 for i, n in enumerate(nouns)}

    train_sparse_situations, train_state_vectors = (
        map(lambda x: np.stack(x), list(zip(*
            map(
                vectorize_state_star,
                map(
                    lambda e: (
                        e["situation"],
                        6,
                        COLOR2IDX,
                        NOUN2IDX,
                        "all",
                        False
                    ),
                    tqdm(dataset["examples"]["train"])
                )
            )
        )))
    )

    # Used for sanity checks below, but here so that everything is together
    normalized_train_state_vectors = normalize(train_state_vectors, axis=1)

    # Voronoi index to improve retrieval speed
    base_index = faiss.IndexFlatIP(normalized_train_state_vectors.shape[-1])
    index = faiss.IndexIVFFlat(base_index, normalized_train_state_vectors.shape[-1], 512)
    index.train(normalized_train_state_vectors[np.random.permutation(normalized_train_state_vectors.shape[0])[:512 * 40]])
    index.nprobe = 10
    index.add(normalized_train_state_vectors)

    splits_and_distances = []

    for split in tqdm(splits):
        split_sparse_situations, split_state_vectors = (
            map(lambda x: np.stack(x), list(zip(*
                map(
                    vectorize_state_star,
                    map(
                        lambda e: (
                            e["situation"],
                            6,
                            COLOR2IDX,
                            NOUN2IDX,
                            "all",
                            False
                        ),
                        tqdm(dataset["examples"][split])
                    )
                )
            )))
        )
        split_state_vectors = normalize(split_state_vectors, axis=1)

        labels = (2 ** np.arange(14)).astype(np.int32)
        split_distances = np.concatenate([
            np.stack([
                (v[None] == train_sparse_situations[indices, None]).all(axis=-1).astype(np.float32).mean(axis=-1)
                for v, indices in zip(
                    split_sparse_situations[b * 128:(b + 1) * 128],
                    index.search(split_state_vectors[b * 128:(b + 1) * 128], int(labels[-1] + 1))[1][:, labels - 1]
                )
            ])
            for b in trange(
                (min(limit or split_state_vectors.shape[0], split_state_vectors.shape[0])) // 128 + 1,
                desc=f"Finding near neighbours for split {split}"
            )
        ], axis=0)[..., 0]
        df = pd.DataFrame(split_distances, columns=labels)
        df = pd.concat([
            pd.Series([split_names.get(split, split)] * len(split_distances), name="split").to_frame(),
            df
        ], axis=1)
        splits_and_distances.append(df)

    splits_and_distances = pd.concat(splits_and_distances, axis=0)

    return splits_and_distances


SPLITS_NAMES_MAP = {
    "train": "train",
    "test": "a",
    "visual_easier": "b",
    "visual": "c",
    "situational_1": "d",
    "situational_2": "e",
    "contextual": "f",
    "adverb_2": "h",
    "adverb_1": "g",
}


def main():
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gscan-dataset", type=str, required=True)
    parser.add_argument("--output-directory", type=str, required=True)
    parser.add_argument("--only-splits", nargs="*", type=str)
    parser.add_argument("--reascan-boxes", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    with open(args.gscan_dataset, "r") as f:
        d = json.load(f)

    vocabulary = create_vocabulary()
    splits = {s: SPLITS_NAMES_MAP.get(s, s) for s in list(d["examples"].keys())}

    if args.only_splits:
        splits = {k: v for k, v in splits.items() if v in args.only_splits}

    splits_and_distances = get_nearest_neighbour_statistics(d, vocabulary, splits, SPLITS_NAMES_MAP, args.limit)

    splits_names = ['train', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    ax = ax.ravel()
    for split, axis in zip(splits_names, ax):
        sns.lineplot(ax=axis,
                        data=splits_and_distances[splits_and_distances.split == split].melt(
                            id_vars=["split"]
                        ),
                        x="variable",
                        y="value")
        axis.set_xscale('log')
        axis.set_ylim(0, 1.0)
        axis.set_ylabel('Hamming Similarity')
        axis.set_xlabel('Nth nearest-neighbour')
        axis.set_title(f"Split {split[:1].upper() + split[1:]}")
    fig.tight_layout()

    os.makedirs(args.output_directory)
    plt.savefig(os.path.join(args.output_directory, "decay_nn.pdf"))


if __name__ == "__main__":
    main()
