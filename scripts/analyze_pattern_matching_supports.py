import argparse
import os

from gscan_metaseq2seq.util.load_data import load_data, load_data_directories
from tqdm.auto import tqdm
import re

import pandas as pd


def matches(regex, example, ACTION2IDX):
    main_demo = "".join(map(str, example[1][example[1] != ACTION2IDX['[pad]']].tolist()))
    support_demos = [
        "".join(map(str, e[e != ACTION2IDX['[pad]']].tolist()))
        for e in example[5]
        if not (e == ACTION2IDX['[pad]']).all()
    ]

    main_demo_matches = re.match(regex, main_demo) is not None
    all_matches = [
        (re.match(regex, e))
        for e in support_demos
    ]
    
    return main_demo_matches, sum(map(lambda m: m != None, all_matches)), len(support_demos)


def matches_pattern_on_sets_statistics(pattern, meta_train_demonstrations, meta_valid_demonstrations_dict, ACTION2IDX):
    match_pull_lturn_pattern_train = [
        matches(pattern, example, ACTION2IDX)
        for example in tqdm(meta_train_demonstrations)
    ]
    match_pull_lturn_pattern_valid = {
        s: [
            matches(pattern, example, ACTION2IDX)
            for example in tqdm(demos)
        ]
        for s, demos in meta_valid_demonstrations_dict.items()
    }

    stats_dict = {
        s: [len(matches)] + list(map(sum, zip(*matches)))
        for s, matches in match_pull_lturn_pattern_valid.items()
    }
    stats_dict["train"] = [len(meta_train_demonstrations)] + list(map(sum, zip(*match_pull_lturn_pattern_train)))
    
    return {
        k: {
            "Target": (v[1] / v[0]),
            "Supports": (v[2] / v[3])
        }
        for k, v in stats_dict.items()
    }


_RE_CAPTION_TABLE = [
    [r'(?:(.)\1\1\1[5])(?:(.)\1\1\1[5])+', 'Matching pattern in Split H'],
    [r'(.)\1(?:(?!\1).)+\1(?:(?!\1).)+', 'Matching pattern in Split D'],
    [r'.*(?:(.)(?!\1)(.)\2\1(?!\1).*?5)(?:(.)(?!\1)(.)\2\1(?!\1).*?5)+.*', 'Matching pattern in Split G']
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-demonstrations", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--valid-demonstrations-directory", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (meta_train_demonstrations, meta_valid_demonstrations_dict),
    ) = load_data_directories(
        args.valid_demonstrations_directory,
        args.dictionary,
        limit_load=args.limit
    )

    for regex, caption in _RE_CAPTION_TABLE:
        print(
            pd.DataFrame.from_dict(
                matches_pattern_on_sets_statistics(
                    regex,
                    meta_train_demonstrations,
                    meta_valid_demonstrations_dict,
                    ACTION2IDX
                )
            ).T.to_latex(
                float_format='%.2g',
                caption=caption
            )
        )



if __name__ == "__main__":
    main()