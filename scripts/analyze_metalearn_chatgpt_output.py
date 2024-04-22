import argparse
import json
import pandas
import os
import itertools
import numpy as np
import editdistance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", type=str, help="The input json file")
    args = parser.parse_args()

    with open(args.json, "r") as f:
        results = json.load(f)

    grouped_results = dict(map(lambda x: (x[0], list(x[1])), itertools.groupby(
        results,
        key=lambda x: x[0]
    )))
    group_results_matches = {
        split: [
            any([
                reference.replace("[eos]", "").strip() == option.replace("[eos]", "").strip()
                for option in options
            ]) for s, options, reference in values
        ]
        for split, values in grouped_results.items()
    }
    group_results_edit_dists = {
        split: [
            min([
                editdistance.eval(
                    option.split(), reference.split()
                ) / (max(len(option.split()), len(reference.split())))
                for option in options
            ]) for s, options, reference in values
        ]
        for split, values in grouped_results.items()
    }
    print({
        s: np.array(values).astype(np.float32).mean()
        for s, values in group_results_matches.items()
    })
    print({
        s: np.array(values).astype(np.float32).mean()
        for s, values in group_results_edit_dists.items()
    })


if __name__ == "__main__":
    main()