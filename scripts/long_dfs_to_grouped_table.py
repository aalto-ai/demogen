import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs="+", type=str)
    args = parser.parse_args()

    df = pd.concat([
        pd.read_csv(csv)
        for csv in tqdm(args.csvs)
    ]).reset_index(drop=True).drop("Unnamed: 0", axis=1)
    df = df.loc[df[df.columns[1]] == "h"]
    mean = df.groupby(df.columns[0]).agg({
        df.columns[1]: "first",
        df.columns[2]: "mean"
    }).round(2).astype(str)
    std = df.groupby(df.columns[0]).agg({
        df.columns[1]: "first",
        df.columns[2]: "std"
    }).round(2).astype(str)

    print(mean.to_latex())


if __name__ == "__main__":
    main()