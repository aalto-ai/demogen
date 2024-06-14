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
import functools

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.auto import tqdm, trange

from gscan_metaseq2seq.gscan.world import Situation
from gscan_metaseq2seq.util.solver import (
    create_vocabulary,
    create_world
)

from gscan_metaseq2seq.util.dataset import PaddingIterableDataset
from generate_data_retrieval import vectorize_all_example_situations
from gscan_metaseq2seq.util.load_data import load_data_directories


def situation_to_dense(situation, grid_size=6):
    grid = np.zeros((grid_size, grid_size, situation[0].shape[-1] - 2), dtype=np.int8)

    for x in situation:
        grid[x[-2], x[-1]] = x[:-2].astype(np.int8)

    return grid.reshape(-1, grid.shape[-1])


def situations_to_dense_situations(situations, grid_size=6):
    # We take a situation which is encoded as a series of vectors
    # and project to the grid based on the positional encoding
    return np.stack([
        situation_to_dense(x) for x in situations
    ])



def get_nearest_neighbour_statistics(valid_demonstrations_dict, limit=None):
    max_state_component_lengths = functools.reduce(
        lambda x, o: np.stack([x, o]).max(axis=0),
        map(lambda x: np.stack(x[-1]).max(axis=0), itertools.chain.from_iterable([
            demonstrations for split, demonstrations in valid_demonstrations_dict.items()
        ]))
    ) + 1

    train_state_vectors = vectorize_all_example_situations(valid_demonstrations_dict["train"], max_state_component_lengths)
    train_sparse_situations = situations_to_dense_situations(map(lambda x: x[-1], valid_demonstrations_dict["train"]))

    # Used for sanity checks below, but here so that everything is together
    normalized_train_state_vectors = normalize(train_state_vectors, axis=1)

    # Voronoi index to improve retrieval speed
    base_index = faiss.IndexFlatIP(normalized_train_state_vectors.shape[-1])
    index = faiss.IndexIVFFlat(base_index, normalized_train_state_vectors.shape[-1], 512)
    index.train(normalized_train_state_vectors[np.random.permutation(normalized_train_state_vectors.shape[0])[:512 * 40]])
    index.nprobe = 10
    index.add(normalized_train_state_vectors)

    splits_and_distances = []

    for split in tqdm(valid_demonstrations_dict.keys()):
        if len(valid_demonstrations_dict[split]) == 0:
            continue

        split_state_vectors = vectorize_all_example_situations(valid_demonstrations_dict[split], max_state_component_lengths) if split != "train" else train_state_vectors
        split_sparse_situations = situations_to_dense_situations(map(lambda x: x[-1], valid_demonstrations_dict[split])) if split != "train" else train_sparse_situations
        normalized_split_state_vectors = normalize(split_state_vectors, axis=1)
        split_search_indices = np.random.permutation(normalized_split_state_vectors.shape[0])

        labels = (2 ** np.arange(14)).astype(np.int32)
        split_distances = np.concatenate([
            np.stack([
                (v[None] == train_sparse_situations[indices, None]).all(axis=-1).astype(np.float32).mean(axis=-1)
                for v, indices in zip(
                    split_sparse_situations[split_search_indices[b * 128:(b + 1) * 128]],
                    index.search(normalized_split_state_vectors[split_search_indices[b * 128:(b + 1) * 128]], int(labels[-1] + 1))[1][:, labels - 1]
                )
            ])
            for b in trange(
                (min(limit or normalized_split_state_vectors.shape[0], normalized_split_state_vectors.shape[0])) // 128 + 1,
                desc=f"Finding near neighbours for split {split}"
            )
        ], axis=0)[..., 0]
        df = pd.DataFrame(split_distances, columns=labels)
        df = pd.concat([
            pd.Series([split] * len(split_distances), name="split").to_frame(),
            df
        ], axis=1)
        splits_and_distances.append(df)

    splits_and_distances = pd.concat(splits_and_distances, axis=0)

    return splits_and_distances


def main():
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--only-splits", nargs="*", type=str)
    parser.add_argument("--limit-load", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    dictionaries, (train_dataset, examples) = load_data_directories(
        args.dataset,
        os.path.join(args.dataset, "dictionary.pb"),
        limit_load=args.limit_load,
        only_splits=args.only_splits
    )

    splits_and_distances = get_nearest_neighbour_statistics({**examples, "train": train_dataset}, args.limit)

    print(splits_and_distances.groupby("split").mean().to_latex(float_format='%.2f'))


if __name__ == "__main__":
    main()
