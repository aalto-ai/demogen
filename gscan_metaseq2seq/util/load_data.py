import fnmatch
import itertools
import os
import pickle
import numpy as np


def load_pickle_file(path):
    with open(path, "rb") as f:
        print(f"Loading {path}")
        return pickle.load(f)


def load_concat_pickle_files_from_directory(directory_path, limit_load=None):
    return list(
        itertools.chain.from_iterable(
            [
                load_pickle_file(os.path.join(directory_path, filename))
                for filename in sorted(
                    fnmatch.filter(os.listdir(directory_path), "*.pb"),
                    key=lambda k: int(os.path.splitext(k)[0]),
                )[:limit_load]
            ]
        )
    )


def load_data(
    train_meta_trajectories_path, valid_trajectories_directory, dictionary_path
):
    with open(dictionary_path, "rb") as f:
        WORD2IDX, ACTION2IDX, color_dictionary, noun_dictionary = pickle.load(f)

    meta_train_demonstrations = load_pickle_file(train_meta_trajectories_path)

    valid_trajectories_dict = (
        {
            os.path.splitext(fname)[0]: load_pickle_file(
                os.path.join(valid_trajectories_directory, fname)
            )
            for fname in sorted(os.listdir(valid_trajectories_directory))
        }
        if valid_trajectories_directory
        else {}
    )

    return (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (meta_train_demonstrations, valid_trajectories_dict),
    )


def load_data_directories(data_directory, dictionary_path, limit_load=None):
    assert os.path.isdir(os.path.join(data_directory, "train"))

    meta_train_demonstrations = load_concat_pickle_files_from_directory(
        os.path.join(data_directory, "train"), limit_load=limit_load
    )
    valid_trajectories_dict = {
        fname: load_concat_pickle_files_from_directory(
            os.path.join(data_directory, fname), limit_load=limit_load
        )
        for fname in sorted(os.listdir(data_directory))
        if os.path.isdir(os.path.join(data_directory, fname))
        and fname not in ("train", "valid")
    }

    with open(dictionary_path, "rb") as f:
        WORD2IDX, ACTION2IDX, color_dictionary, noun_dictionary = pickle.load(f)

    return (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (meta_train_demonstrations, valid_trajectories_dict),
    )


def split_dataset(dataset, pct=0.01):
    indices = np.arange(len(dataset), dtype=int)
    np.random.shuffle(indices)
    train = indices[: -int(len(dataset) * pct)]
    test = indices[-int(len(dataset) * pct) :]
    return [dataset[x] for x in train], [dataset[x] for x in test]
