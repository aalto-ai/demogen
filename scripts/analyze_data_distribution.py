import argparse
import os
import numpy as np
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm

from gscan_metaseq2seq.util.load_data import load_data_directories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--data-directory", type=str, required=True)
    parser.add_argument("--only-splits", type=str, nargs="*")
    parser.add_argument("--limit-load", type=int, default=None)
    args = parser.parse_args()

    (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (train_demonstrations, valid_demonstrations_dict),
    ) = load_data_directories(
        args.data_directory, args.dictionary, limit_load=args.limit_load, only_splits=args.only_splits
    )

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}
    IDX2ACTION = {i: w for w, i in ACTION2IDX.items()}

    pad_word = WORD2IDX["[pad]"]
    pad_action = ACTION2IDX["[pad]"]
    pad_state = 0
    sos_action = ACTION2IDX["[sos]"]
    eos_action = ACTION2IDX["[eos]"]

    len_dicts = {
        k: [len(x[1]) for x in v]
        for k, v in {
            **valid_demonstrations_dict,
            "train": train_demonstrations
        }.items()
        if k not in ["dev"]
    }

    print(pd.DataFrame.from_dict({
        k: [int(len(v)), np.mean(v), np.std(v), np.max(v)]
        for k, v in len_dicts.items()
        if v
    }, orient="index", columns=["len", "mean", "std", "max"]).to_latex(float_format='%.1f'))


if __name__ == "__main__":
    main()