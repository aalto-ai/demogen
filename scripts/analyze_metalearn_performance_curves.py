import argparse
import numpy as np
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm

from gscan_metaseq2seq.util.load_data import load_data_directories
from gscan_metaseq2seq.util.dataset import (
    PaddingDataset,
    ReorderSupportsByDistanceDataset,
    MapDataset,
)
from train_meta_encdec_big_symbol_transformer import (
    BigSymbolTransformerLearner,
)

from analyze_failure_cases import get_metaseq2seq_predictions

def mean_std(array):
    return [array.mean(), array.std()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--data-directory", type=str, required=True)
    parser.add_argument("--transformer-checkpoint", type=str, required=True)
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--limit-load", type=int, default=None)
    parser.add_argument("--pad-instructions-to", type=int, default=8)
    parser.add_argument("--pad-actions-to", type=int, default=128)
    parser.add_argument("--pad-state-to", type=int, default=36)
    parser.add_argument("--metalearn-demonstrations-limits", nargs="+", type=int, default=[4, 8, 12, 16])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--only-splits", type=str, nargs="*")
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

    per_n_per_split_exacts = list(itertools.chain.from_iterable([
        list(itertools.chain.from_iterable([
            list(map(lambda x: (n, k, x), get_metaseq2seq_predictions(
                args.transformer_checkpoint,
                PaddingDataset(
                    ReorderSupportsByDistanceDataset(
                        MapDataset(
                            MapDataset(
                                demonstrations,
                                lambda x: (x[2], x[3], x[0], x[1], x[4], x[5], x[6]),
                            ),
                            lambda x: (
                                x[0],
                                [x[1]] * len(x[-1])
                                if not isinstance(x[1][0], list)
                                else x[1],
                                x[2],
                                x[3],
                                x[4],
                                x[5],
                                x[6],
                            ),
                        ),
                        n,
                    ),
                    (
                        (args.pad_state_to, None),
                        (n, args.pad_state_to, None),
                        32,
                        128,
                        (n, 32),
                        (n, 128),
                    ),
                    (pad_state, pad_state, pad_word, pad_action, pad_word, pad_action),
                ),
                not args.disable_cuda,
                batch_size=args.batch_size,
                only_exacts=True
            ).cpu().numpy().astype(np.float32)))
            for k, demonstrations in tqdm(valid_demonstrations_dict.items())
        ]))
        for n in tqdm(args.metalearn_demonstrations_limits)
    ]))

    plot_data = pd.DataFrame(per_n_per_split_exacts, columns=["Number of Demonstrations", "Split", "Exact Match Fraction"])

    sns.lineplot(data=plot_data, x="Number of Demonstrations", y="Exact Match Fraction", hue="Split")
    plt.ylim(0, 1.1)
    plt.savefig("demonstrations-efficiency.pdf")

    for split in valid_demonstrations_dict.keys():
        plt.clf()
        sns.lineplot(data=plot_data.loc[plot_data["Split"] == split],
                     x="Number of Demonstrations",
                     y="Exact Match Fraction",
                     hue="Split")
        plt.ylim(0, 1.1)
        plt.savefig(f"demonstrations-efficiency-{split}.pdf")


if __name__ == "__main__":
    main()