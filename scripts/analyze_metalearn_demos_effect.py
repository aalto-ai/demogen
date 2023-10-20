import argparse
import os
import numpy as np
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset, IterableDataset
from tqdm.auto import tqdm

from gscan_metaseq2seq.util.load_data import load_data_directories
from gscan_metaseq2seq.util.dataset import (
    PaddingDataset,
    PaddingIterableDataset,
    ReorderSupportsByDistanceDataset,
    MapDataset,
)
from gscan_metaseq2seq.util.solver import (
    segment_instruction,
    find_agent_position,
    find_target_object,
)
from train_meta_encdec_big_symbol_transformer import (
    BigSymbolTransformerLearner,
)

from analyze_failure_cases import get_metaseq2seq_predictions

def mean_std(array):
    return [array.mean(), array.std()]


CRITERIAS = (
    "Remove Same Object",
    "Remove Same Verb",
    "Remove Same Adverb"
)


def flatten(x):
    return list(itertools.chain.from_iterable(x))


def drop_supports_matching_criteria(criteria, word2idx, colors, nouns):
    def func(example):
        (
            query_state,
            support_states,
            query_instruction,
            query_targets,
            support_instructions,
            support_targets
        ) = example

        query_verb, query_adverb, query_size, query_color, query_noun = segment_instruction(
            query_instruction, word2idx, colors, nouns
        )
        segmented_support_queries = [
            segment_instruction(support_instruction, word2idx, colors, nouns)
            for support_instruction in support_instructions
        ]

        if criteria == "Remove Same Object":
            indices = list(filter(lambda x: (
                bool(set(query_size) - set(segmented_support_queries[x][2])) or
                bool(set(query_color) - set(segmented_support_queries[x][3])) or
                bool(set(query_noun) - set(segmented_support_queries[x][4]))
            ), range(len(segmented_support_queries))))
        elif criteria == "Remove Same Verb":
            indices = list(filter(lambda x: (
                bool(set(flatten(query_verb)) - set(flatten(segmented_support_queries[x][0])))
            ), range(len(segmented_support_queries))))
        elif criteria == "Remove Same Adverb":
            indices = list(filter(lambda x: (
                bool(set(flatten(query_adverb)) - set(flatten(segmented_support_queries[x][1])))
            ), range(len(segmented_support_queries))))

        return (
            query_state,
            [support_states[i] for i in indices],
            query_instruction,
            query_targets,
            [support_instructions[i] for i in indices],
            [support_targets[i] for i in indices],
        )

    return func


class FilterDataset(IterableDataset):
    def __init__(self, dataset, func):
        super().__init__()
        self.dataset = dataset
        self.dataset_len = len(self.dataset)
        self.func = func
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.dataset_len:
            raise StopIteration

        while not self.func(self.dataset[self.index]):
            self.index += 1

            if self.index >= self.dataset_len:
                raise StopIteration

        item = self.dataset[self.index]
        self.index += 1
        return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--data-directory", type=str, required=True)
    parser.add_argument("--output-directory", type=str, required=True)
    parser.add_argument("--transformer-checkpoint", type=str, required=True)
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--limit-load", type=int, default=None)
    parser.add_argument("--metalearn-demonstrations-limit", type=int, default=16)
    parser.add_argument("--pad-instructions-to", type=int, default=8)
    parser.add_argument("--pad-actions-to", type=int, default=128)
    parser.add_argument("--pad-state-to", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--only-splits", type=str, nargs="*")
    args = parser.parse_args()

    os.makedirs(args.output_directory, exist_ok=True)

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

    per_criteria_exacts = list(itertools.chain.from_iterable([
        list(itertools.chain.from_iterable([
            list(map(lambda x: (criteria, k, x), get_metaseq2seq_predictions(
                args.transformer_checkpoint,
                PaddingIterableDataset(
                    FilterDataset(
                        MapDataset(
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
                                None,
                            ),
                            drop_supports_matching_criteria(criteria, WORD2IDX, color_dictionary, noun_dictionary)
                        ),
                        lambda x: len(x[1])
                    ),
                    (
                        (args.pad_state_to, None),
                        (args.metalearn_demonstrations_limit, args.pad_state_to, None),
                        32,
                        128,
                        (args.metalearn_demonstrations_limit, 32),
                        (args.metalearn_demonstrations_limit, 128),
                    ),
                    (pad_state, pad_state, pad_word, pad_action, pad_word, pad_action),
                ),
                not args.disable_cuda,
                batch_size=args.batch_size,
                only_exacts=True,
                validate_first=False
            ).cpu().numpy().astype(np.float32)))
            for k, demonstrations in tqdm(valid_demonstrations_dict.items())
        ]))
        for criteria in tqdm(CRITERIAS)
    ]))

    plot_data = pd.DataFrame(per_criteria_exacts, columns=["Criteria", "Split", "Exact Match Fraction"])
    plot_data.to_csv(os.path.join(args.output_directory, "results.csv"))

if __name__ == "__main__":
    main()