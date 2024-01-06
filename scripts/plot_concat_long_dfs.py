import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs="+", type=str)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()

    df = pd.concat([
        pd.read_csv(csv)
        for csv in tqdm(args.csvs)
    ]).reset_index(drop=True).drop("Unnamed: 0", axis=1)

    sns.lineplot(data=df, x=df.columns[0], y=df.columns[2], hue=df.columns[1], errorbar="ci")
    plt.ylim(0, 1.1)
    plt.savefig(args.output_file)


if __name__ == "__main__":
    main()