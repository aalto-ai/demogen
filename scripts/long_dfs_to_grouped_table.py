import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm

def concat_seed(df, i):
    df['seed'] = i
    print(df.columns)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs="+", type=str)
    args = parser.parse_args()

    df = pd.concat([
        concat_seed(pd.read_csv(csv), i)
        for i, csv in enumerate(tqdm(args.csvs))
    ]).reset_index(drop=True).drop("Unnamed: 0", axis=1)
    df = df.loc[df[df.columns[1]] == "h"]
    means = df.groupby(['seed', df.columns[0]]).agg({
        df.columns[0]: "first",
        df.columns[1]: "first",
        df.columns[2]: "mean"
    }).reset_index(drop=True)
    mean = means.groupby(df.columns[0]).agg({
        df.columns[0]: "first",
        df.columns[1]: "first",
        df.columns[2]: "mean"
    }).reset_index(drop=True).round(2).astype(str)
    std = means.groupby(df.columns[0]).agg({
        df.columns[0]: "first",
        df.columns[1]: "first",
        df.columns[2]: "std"
    }).reset_index(drop=True).round(2).astype(str)

    mean[mean.columns[2]] = mean[mean.columns[2]] + " ± " + std[std.columns[2]]

    print(mean.to_latex())


if __name__ == "__main__":
    main()