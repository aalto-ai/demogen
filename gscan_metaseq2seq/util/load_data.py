import os
import pickle
import numpy as np


def load_pickle_file(path):
    with open(path, "rb") as f:
        print(f"Loading {path}")
        return pickle.load(f)


def load_data(
    train_meta_trajectories_path, valid_trajectories_directory, dictionary_path
):
    meta_train_demonstrations = load_pickle_file(train_meta_trajectories_path)
    np.random.shuffle(meta_train_demonstrations)

    valid_trajectories_dict = {
        os.path.splitext(fname)[0]: load_pickle_file(
            os.path.join(valid_trajectories_directory, fname)
        )
        for fname in sorted(os.listdir(valid_trajectories_directory))
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
