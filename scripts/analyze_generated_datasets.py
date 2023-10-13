import argparse
import os
import itertools
import pandas as pd
import multiprocessing
import numpy as np
import re
import pickle
from tqdm.auto import tqdm

from gscan_metaseq2seq.util.solver import (
    segment_instruction,
    find_agent_position,
    find_target_object,
)
from gscan_metaseq2seq.util.load_data import load_data_directories


def compute_statistics_for_example(example, word2idx, colors, nouns, limit_demos=None):
    idx2word = [w for w in word2idx]
    idx2color = [c for c in colors]
    idx2noun = [n for n in nouns]

    (
        query,
        target,
        state,
        support_state,
        support_query,
        support_target,
        ranking,
    ) = example
    query_verb, query_adverb, query_size, query_color, query_noun = segment_instruction(
        query, word2idx, colors, nouns
    )
    segmented_support_queries = [
        segment_instruction(support_instruction, word2idx, colors, nouns)
        for support_instruction in support_query[:limit_demos]
    ]
    support_state = (
        [support_state] * len(support_query)
        if isinstance(support_state[0], np.ndarray)
        else support_state
    )[:limit_demos]

    query_agent_position = find_agent_position(state)
    query_target_object = find_target_object(
        state, query_size, query_color, query_noun, idx2word, idx2color, idx2noun
    )
    support_agent_positions = [find_agent_position(state) for state in support_state]
    support_target_objects = [
        find_target_object(
            state,
            support_size,
            support_color,
            support_noun,
            idx2word,
            idx2color,
            idx2noun,
        )
        for state, (
            support_verb,
            support_adverb,
            support_size,
            support_color,
            support_noun,
        ) in zip(support_state, segmented_support_queries)
    ]

    matches = np.array(
        [
            (
                query_verb == support_verb,
                query_adverb == support_adverb,
                query_size + query_color + query_noun
                == support_size + support_color + support_noun,
                (query_agent_position[-2:] == support_agent_pos[-2:]).all(),
                support_target_object is not None
                and (query_target_object[-2:] == support_target_object[-2:]).all(),
                support_target_object is not None
                and (
                    (query_target_object[-2:] - query_agent_position[-2:])
                    == (support_target_object[-2:] - support_agent_pos[-2:])
                ).all(),
                support_target_object is not None
                and (query_target_object[:3] == support_target_object[:3]).all(),
                support_target_object is not None,
            )
            for (
                support_verb,
                support_adverb,
                support_size,
                support_color,
                support_noun,
            ), support_agent_pos, support_target_object in zip(
                segmented_support_queries,
                support_agent_positions,
                support_target_objects,
            )
        ]
    )

    return matches


def summarize_by_dividing_out_count(sum_summaries):
    return np.nan_to_num(
        sum_summaries[..., :-1].sum(axis=0) / sum_summaries[..., -1].sum(), 0.0
    )


def summarize_hits(hits):
    if not len(hits.shape) or any([s == 0 for s in hits.shape]):
        return np.zeros(13, dtype=float)

    right_verb = hits[..., 0]
    right_adverb = hits[..., 1]
    right_object = hits[..., 2]
    right_agent_position = hits[..., 3]
    right_target_position = hits[..., 4]
    right_distance = hits[..., 5]
    right_target = hits[..., 6]
    support_target_object_exists = hits[..., 7]

    right_verb_and_object = right_target & right_verb
    right_adverb_and_object = right_target & right_adverb

    right_verb_and_object_at_least_once = right_verb_and_object.any()
    right_adverb_and_object_at_least_once = right_adverb_and_object.any()
    right_verb_and_object_and_right_adverb_and_object = (
        right_verb_and_object_at_least_once and right_adverb_and_object_at_least_once
    )

    right_verb_and_object_and_distance = right_verb_and_object & right_distance
    right_adverb_and_object_and_distance = right_adverb_and_object & right_distance

    right_verb_and_object_and_distance_at_least_once = (
        right_verb_and_object_and_distance.any()
    )
    right_adverb_and_object_and_distance_at_least_once = (
        right_adverb_and_object_and_distance.any()
    )
    right_verb_and_object_and_distance_and_right_adverb_and_object_and_distance = (
        right_verb_and_object_and_distance_at_least_once
        and right_adverb_and_object_and_distance_at_least_once
    )

    # (0) - query_verb == support_verb,
    # (1) - query_adverb == support_adverb,
    # (2) - query_size + query_color + query_noun == support_size + support_color + support_noun,
    # (3) - (query_agent_position[-2:] == support_agent_pos[-2:]).all(),
    # (4) - support_target_object is not None and (
    #   query_target_object[-2:] == support_target_object[-2:]
    # ).all(),
    # (5) - support_target_object is not None and (
    #       (query_target_object[-2:] - query_agent_position[-2:])
    #        == (support_target_object[-2:] - support_agent_pos[-2:])
    #       ).all(),
    # (6) - support_target_object is not None
    #       and (query_target_object[:3] == support_target_object[:3]).all(),
    # (7) support_target_object is not None,
    sums = hits[..., [2, 3, 4, 5, 6, 7]].sum(axis=0)

    return np.concatenate(
        [
            # Desc Obj (2) -> 0
            # Agent Pos (3) -> 1
            # Tgt Pos (4) -> 2
            # Correct Dist (5) -> 3
            # Tgt Obj (6) -> 4
            # Has tgt (7) -> 5
            sums,
            # Verb & Desc Obj -> 6
            # Adverb & Desc Obj -> 7
            # 6 & 7 -> 8
            # 6 & 5 -> 9
            # 7 & 5 -> 10
            # 9 & 10 -> 11
            np.array(
                [
                    right_verb_and_object_at_least_once * hits.shape[0],
                    right_adverb_and_object_at_least_once * hits.shape[0],
                    right_verb_and_object_and_right_adverb_and_object * hits.shape[0],
                    right_verb_and_object_and_distance_at_least_once * hits.shape[0],
                    right_adverb_and_object_and_distance_at_least_once * hits.shape[0],
                    right_verb_and_object_and_distance_and_right_adverb_and_object_and_distance
                    * hits.shape[0],
                    hits.shape[0],
                ]
            ),
        ]
    )


def compute_statistics_and_summarize(
    example, word2idx, color_dictionary, noun_dictionary, limit_demos
):
    return summarize_hits(
        compute_statistics_for_example(
            example,
            word2idx,
            color_dictionary,
            noun_dictionary,
            limit_demos=limit_demos,
        )
    )


def compute_statistics_and_summarize_star(args):
    return compute_statistics_and_summarize(*args)


def analyze_all_examples(
    examples, word2idx, color_dictionary, noun_dictionary, limit_demos=None, num_procs=8
):
    with multiprocessing.Pool(num_procs) as pool:
        yield from pool.imap_unordered(
            compute_statistics_and_summarize_star,
            map(
                lambda x: (x, word2idx, color_dictionary, noun_dictionary, limit_demos),
                tqdm(examples),
            ),
            chunksize=100,
        )


def load_data_and_make_hit_results(
    data_directory, limit_load=None, limit_demos=None, splits=None
):
    (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (meta_train_demonstrations, meta_valid_demonstrations_dict),
    ) = load_data_directories(
        data_directory,
        os.path.join(data_directory, "dictionary.pb"),
        limit_load=limit_load,
        only_splits=splits
    )

    color_dictionary = sorted(color_dictionary)
    noun_dictionary = sorted(noun_dictionary)

    return {
        split: summarize_by_dividing_out_count(
            np.stack(
                list(
                    analyze_all_examples(
                        tqdm(examples, desc=f"Split {split}"),
                        WORD2IDX,
                        color_dictionary,
                        noun_dictionary,
                        limit_demos=limit_demos,
                    )
                )
            )
        )
        for split, examples in tqdm(
            itertools.chain.from_iterable(
                [
                    [["train", meta_train_demonstrations]],
                    meta_valid_demonstrations_dict.items(),
                ]
            ),
            total=len(meta_valid_demonstrations_dict) + 1,
        )
        if not splits or split in splits
    }


NAME_MAP = {
    "i2g_seq2seq_model_score": "DemoGen",
    "gandr": "GandR",
    "metalearn_allow_any": "Expert",
    "metalearn_find_matching_instruction_demos_allow_any": "Retrieval",
    "metalearn_random_instructions_same_layout_allow_any": "Random",
}


def extract_split_to_table(dataset_summaries, split):
    df = pd.DataFrame.from_dict(
        {k: v[split] for k, v in dataset_summaries.items()}, orient="columns"
    )

    # Desc Obj (2) -> 0
    # Agent Pos (3) -> 1
    # Tgt Pos (4) -> 2
    # Correct Dist (5) -> 3
    # Tgt Obj (6) -> 4
    # Verb & Desc Obj -> 6
    # Adverb & Desc Obj -> 7
    # 6 & 7 -> 8
    # (6 & 5) & (7 & 5) -> 11
    df = df.drop([5, 9, 10], axis="index")
    df.index = INDEX_COLS
    df.columns = [NAME_MAP.get(n, n) for n in df.columns]

    return df


INDEX_COLS = [
    f"\\footnotesize{{({i + 1}) {s}}}"
    for i, s in enumerate(
        [
            "Desc. Obj.",
            "Agent Pos.",
            "Tgt. Pos.",
            "Same Diff.",
            "Tgt. Obj.",
            "Verb \\& (5)",
            "Advb \\& (5)",
            "(6) \\& (7)",
            "(4) \\& (8)",
        ]
    )
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", required=True)
    parser.add_argument("--limit-load", type=int, default=None)
    parser.add_argument("--limit-demos", type=int, default=16)
    parser.add_argument("--load-analyzed", type=str)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "a", "b", "c", "d", "e", "f", "g", "h"]
    )
    args = parser.parse_args()

    if args.load_analyzed:
        with open(args.load_analyzed, "rb") as f:
            dataset_summaries = pickle.load(f)
    else:
        dataset_summaries = {
            dataset: load_data_and_make_hit_results(
                os.path.join(args.data_directory, dataset),
                limit_load=args.limit_load,
                limit_demos=args.limit_demos,
                splits=args.splits,
            )
            for dataset in args.datasets
        }

    print("\\begin{table*}[ht]")
    print("\\centering")
    for i, split in enumerate(args.splits):
        print("% {split}")
        print("\\subfloat[]{Split " + split.upper() + "}{")
        print("\\centering")
        print("\\resizebox{0.4\\linewidth}{!}{")
        print(
            extract_split_to_table(dataset_summaries, split)[
                [NAME_MAP.get(n, n) for n in args.datasets]
            ]
            .loc[INDEX_COLS]
            .to_latex(
                float_format="%.2f",
                formatters={
                    name: (lambda x: f"\\footnotesize{{{x:.2f}}}")
                    for name in [NAME_MAP.get(n, n) for n in args.datasets]
                },
                escape=False
            )
        )
        print("}")
        print("}")
        print("\\qquad" if i % 2 == 0 else "\\vskip 10mm")
    print("\\caption{Property statistics on all gSCAN test splits}")
    print("\\label{tab:gscan_split_properties}")
    print("\\end{table*}")


if __name__ == "__main__":
    main()
