import argparse
from collections import defaultdict, Counter, deque
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import pickle
import json
import Levenshtein

from tqdm.auto import tqdm

from gscan_metaseq2seq.util.dataset import PaddingDataset
from train_meta_seq2seq_transformer import ImaginationMetaLearner
from train_transformer import TransformerLearner


def softmax(logits):
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1)[:, None]


def entropy_from_logits(logits):
    p = softmax(logits)
    logp = np.log(p)

    return -(p * logp).sum(axis=1)


def make_histplot_dataframe(values, name, hue):
    df = pd.Series(values, name=name).to_frame()
    df["Model"] = hue

    return df


def make_violinplot_dataframe(x_values, y_values, xlabel, ylabel, hue):
    df = pd.DataFrame(np.vstack([x_values, y_values]).T, columns=[xlabel, ylabel])
    df["Model"] = hue

    return df


def make_scatterplot_dataframe(x_values, y_values, xlabel, ylabel, hue):
    df = pd.DataFrame(np.vstack([x_values, y_values]).T, columns=[xlabel, ylabel])
    df["Model"] = hue

    return df


# Frequency counting conditional probabilities
def sliding_window(iterable, n):
    # sliding_window('ABCDEFG', 4) -> ABCD BCDE CDEF DEFG
    it = iter(iterable)
    window = deque(itertools.islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)


def make_target_commands_frequency_table(examples, actions):
    frequency_counts = defaultdict(Counter)
    for act in filter(lambda x: not "[" in x, actions):
        for act2 in filter(lambda x: not "[" in x, actions):
            frequency_counts[act][act2] = 0

    for example in examples:
        for target_prev, target_next in sliding_window(
            example["target_commands"].split(","), 2
        ):
            frequency_counts[target_next][target_prev] += 1

    frequency_counts_df = pd.DataFrame.from_dict(
        dict({k: dict(v) for k, v in frequency_counts.items()}), orient="index"
    ).fillna(0)
    frequency_counts_df = (frequency_counts_df.T / frequency_counts_df.sum(axis=1)).T
    frequency_counts_df = frequency_counts_df.fillna(0)
    frequency_counts_df = frequency_counts_df.reindex(
        sorted(frequency_counts_df.columns), axis=1
    )
    frequency_counts_df = frequency_counts_df.reindex(
        sorted(frequency_counts_df.index), axis=0
    )
    frequency_counts_df.columns = [
        r"\texttt{PULL}",
        r"\texttt{PUSH}",
        r"\texttt{STAY}",
        r"\texttt{LTURN}",
        r"\textt{RTURN}",
        r"\texttt{WALK}",
    ]
    frequency_counts_df.index = [
        r"\texttt{PULL}",
        r"\texttt{PUSH}",
        r"\texttt{STAY}",
        r"\texttt{LTURN}",
        r"\textt{RTURN}",
        r"\texttt{WALK}",
    ]

    return frequency_counts_df


def get_metaseq2seq_predictions(meta_seq2seq_checkpoint, dataset, use_cuda=True):
    module = ImaginationMetaLearner.load_from_checkpoint(meta_seq2seq_checkpoint)
    trainer = pl.Trainer(accelerator="gpu" if use_cuda else None, devices=1)
    preds = trainer.predict(module, DataLoader(dataset, batch_size=64))

    predicted_targets_stacked, logits_stacked, exacts_stacked = list(
        map(torch.cat, zip(*preds))
    )

    return (predicted_targets_stacked, logits_stacked, exacts_stacked)


def get_transformer_predictions(
    transformer_checkpoint, transformer_dataset, use_cuda=True
):
    transformer_module = TransformerLearner.load_from_checkpoint(transformer_checkpoint)

    # Sanity check - does this transformer perform well?
    trainer = pl.Trainer(accelerator="gpu" if use_cuda else None, devices=1)
    trainer.validate(
        transformer_module,
        DataLoader(Subset(transformer_dataset, torch.arange(1024)), batch_size=64),
    )
    transformer_preds = trainer.predict(
        transformer_module, DataLoader(transformer_dataset, batch_size=64)
    )

    (
        _,
        __,
        transformer_predicted_targets_stacked,
        transformer_logits_stacked,
        transformer_exacts_stacked,
        ___,
    ) = list(
        map(lambda x: list(itertools.chain.from_iterable(x)), zip(*transformer_preds))
    )

    return (
        transformer_predicted_targets_stacked,
        transformer_logits_stacked,
        transformer_exacts_stacked,
    )


def classify_error_types(
    gscan_dataset,
    dataset,
    predicted_targets_stacked,
    exacts_stacked,
    ACTION2IDX,
    IDX2ACTION,
):
    error_classifications = {
        "turn_failure": (("turn left", "turn right"), "walk"),
        "spurious_pull": (("walk",), "pull"),
        "missed_pull": (("pull",), "[eos]"),
        "missed_spin": (("turn left",), "pull"),
    }

    error_classifications_indices = {
        "turn_failure": [],
        "spurious_pull": [],
        "missed_pull": [],
        "missed_spin": [],
        "other": [],
    }

    for index, is_exact_match in zip(range(len(exacts_stacked)), exacts_stacked):
        if is_exact_match:
            continue

        example = gscan_dataset["examples"]["adverb_2"][index]

        errors = [
            (token_index, w1, w2)
            for (token_index, w1, w2, correct) in zip(
                np.arange(predicted_targets_stacked[index].shape[0]),
                example["target_commands"].split(","),
                predicted_targets_stacked[index].numpy(),
                predicted_targets_stacked[index].numpy() == dataset[index][3],
            )
            if not correct
        ]

        hit_any = False

        for error_classification_key, (
            mistake_srcs,
            mistake_tgt,
        ) in error_classifications.items():
            if any(
                [
                    e[1] in mistake_srcs and e[2] == ACTION2IDX[mistake_tgt]
                    for e in errors
                ]
            ):
                error_classifications_indices[error_classification_key].append(index)
                hit_any = True

        if not hit_any:
            error_classifications_indices["other"].append(index)

        print(index)
        print(
            "Errors:",
            " ".join(
                [
                    f"{token_index} - {w1} -> {IDX2ACTION[w2]}"
                    for (token_index, w1, w2) in errors
                ]
            ),
        )

    print(
        {
            k: len(v) / len(exacts_stacked[exacts_stacked == False])
            for k, v in error_classifications_indices.items()
        }
    )

    return error_classifications_indices


def compute_turn_entropies(
    error_classifications_indices,
    dataset,
    logits_stacked,
    predicted_targets_stacked,
    ACTION2IDX,
):
    logits_from_missed_turns = np.concatenate(
        [
            logits_stacked[idx].numpy()[
                np.logical_and(
                    (predicted_targets_stacked[idx].numpy() == ACTION2IDX["walk"]),
                    np.logical_or(
                        dataset[idx][3] == ACTION2IDX["turn left"],
                        dataset[idx][3] == ACTION2IDX["turn right"],
                    ),
                )
            ][:1]
            for idx in error_classifications_indices["turn_failure"]
        ]
    )

    logits_from_missed_turns_entropies = entropy_from_logits(
        logits_from_missed_turns[:, [3, 4, 5]]
    )

    print("Missed turns entropies")
    print(
        logits_from_missed_turns_entropies.mean(),
        logits_from_missed_turns_entropies.std(),
    )


def generate_edit_distance_plots(
    dataset,
    gscan_split_h_demonstrations,
    predicted_targets_stacked,
    exacts_stacked,
    transformer_predicted_targets_stacked,
    transformer_exacts_stacked,
    ACTION2IDX,
):
    meta_seq2seq_split_h_levenshtein_distances = np.array(
        [
            Levenshtein.distance(
                predicted_targets_stacked[idx].tolist(),
                dataset[idx][3].tolist(),
            )
            for idx, exact in zip(range(len(exacts_stacked)), tqdm(exacts_stacked))
            if not exact
        ]
    )
    transformer_split_h_levenshtein_distances = np.array(
        [
            Levenshtein.distance(
                transformer_predicted_targets_stacked[idx].tolist(),
                gscan_split_h_demonstrations[idx][1].tolist(),
            )
            for idx, exact in zip(
                range(len(transformer_exacts_stacked)),
                tqdm(transformer_exacts_stacked),
            )
            if not exact
        ]
    )

    sns.histplot(
        pd.concat(
            [
                make_histplot_dataframe(
                    meta_seq2seq_split_h_levenshtein_distances,
                    "Edit Distance",
                    "meta_seq2seq",
                ),
                make_histplot_dataframe(
                    transformer_split_h_levenshtein_distances,
                    "Edit Distance",
                    "transformer",
                ),
            ]
        ).reset_index(),
        x="Edit Distance",
        hue="Model",
        binwidth=2,
    )
    plt.savefig("comparison_edit_distance_mistakes.pdf")
    plt.clf()

    transformer_num_pulls_split_h = np.array(
        [
            (example[1] == ACTION2IDX["pull"]).sum(axis=0)
            for exact, example in zip(
                transformer_exacts_stacked, tqdm(gscan_split_h_demonstrations)
            )
            if not exact
        ]
    )

    meta_seq2seq_num_pulls_split_h = np.array(
        [
            (example[3] == ACTION2IDX["pull"]).sum(axis=0)
            for exact, example in zip(exacts_stacked, tqdm(dataset))
            if not exact
        ]
    )

    sns.kdeplot(
        data=pd.concat(
            [
                make_scatterplot_dataframe(
                    meta_seq2seq_num_pulls_split_h,
                    meta_seq2seq_split_h_levenshtein_distances,
                    "Number of pulls in target",
                    "Edit Distance",
                    "meta_seq2seq",
                ),
                make_scatterplot_dataframe(
                    transformer_num_pulls_split_h,
                    transformer_split_h_levenshtein_distances,
                    "Number of pulls in target",
                    "Edit Distance",
                    "transformer",
                ),
            ]
        ).reset_index(),
        x="Number of pulls in target",
        y="Edit Distance",
        hue="Model",
        fill=True,
    )
    plt.savefig("num_pulls_vs_edit_distance.pdf")
    plt.clf()

    violinplot_df = pd.concat(
        [
            make_scatterplot_dataframe(
                meta_seq2seq_num_pulls_split_h,
                meta_seq2seq_split_h_levenshtein_distances,
                "Number of pulls in target",
                "Edit Distance",
                "meta_seq2seq",
            ),
            make_scatterplot_dataframe(
                transformer_num_pulls_split_h,
                transformer_split_h_levenshtein_distances,
                "Number of pulls in target",
                "Edit Distance",
                "transformer",
            ),
        ]
    ).reset_index()
    violinplot_df = violinplot_df[violinplot_df["Number of pulls in target"] < 10][
        violinplot_df["Edit Distance"] < 80
    ]
    violinplot_df["Number of pulls in target"] = pd.cut(
        violinplot_df["Number of pulls in target"], [0, 2, 8, 16]
    )
    sns.violinplot(
        data=violinplot_df,
        y="Number of pulls in target",
        x="Edit Distance",
        hue="Model",
        split=True,
        scale="width",
        inner=None,
    )
    plt.savefig("pulls_vs_edit_distance_violinplot.pdf")
    plt.clf()

    sns.histplot(
        violinplot_df.loc[
            violinplot_df["Number of pulls in target"]
            == pd.Interval(0, 2, closed="right")
        ],
        x="Edit Distance",
        hue="Model",
        binwidth=2,
    )
    plt.savefig("edit_distance_vs_pulls_0_2.pdf")
    plt.clf()

    sns.histplot(
        violinplot_df.loc[
            violinplot_df["Number of pulls in target"]
            == pd.Interval(2, 8, closed="right")
        ],
        x="Edit Distance",
        hue="Model",
        binwidth=2,
    )
    plt.savefig("edit_distance_vs_pulls_2_8.pdf")
    plt.clf()

    sns.histplot(
        violinplot_df.loc[
            violinplot_df["Number of pulls in target"]
            == pd.Interval(8, 16, closed="right")
        ],
        x="Edit Distance",
        hue="Model",
        binwidth=2,
    )
    plt.savefig("edit_distance_vs_pulls_8_16.pdf")
    plt.clf()


def frequency_count_conditional_probabilities(gscan_dataset, ACTION2IDX):
    print("Conditional probabilities frequency counts")
    print(
        make_target_commands_frequency_table(
            gscan_dataset["examples"]["train"], list(ACTION2IDX.keys())
        ).to_latex(
            float_format="%.2f",
            caption="One-step ahead conditional probability table, training set",
            escape=False,
        )
    )
    print(
        make_target_commands_frequency_table(
            gscan_dataset["examples"]["test"], list(ACTION2IDX.keys())
        ).to_latex(
            float_format="%.2f",
            caption="One-step ahead conditional probability table, Split A",
            escape=False,
        )
    )
    print(
        make_target_commands_frequency_table(
            gscan_dataset["examples"]["adverb_2"], list(ACTION2IDX.keys())
        ).to_latex(
            float_format="%.2f",
            caption="One-step ahead conditional probability table, Split H",
            escape=False,
        )
    )
    print(
        make_target_commands_frequency_table(
            gscan_dataset["examples"]["visual"], list(ACTION2IDX.keys())
        ).to_latex(
            float_format="%.2f",
            caption="One-step ahead conditional probability table, Split C",
            escape=False,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compositional-splits", type=str, required=True)
    parser.add_argument("--metalearn-data-directory", type=str, required=True)
    parser.add_argument("--baseline-data-directory", type=str, required=True)
    parser.add_argument("--meta-seq2seq-checkpoint", type=str, required=True)
    parser.add_argument("--transformer-checkpoint", type=str, required=True)
    parser.add_argument("--disable-cuda", action="store_false")
    args = parser.parse_args()

    with open(f"{args.baseline_data_directory}/dictionary.pb", "rb") as f:
        WORD2IDX, ACTION2IDX, color_dictionary, noun_dictionary = pickle.load(f)

    IDX2WORD = {i: w for w, i in WORD2IDX.items()}
    IDX2ACTION = {i: w for w, i in ACTION2IDX.items()}

    pad_word = WORD2IDX["[pad]"]
    pad_action = ACTION2IDX["[pad]"]
    sos_action = ACTION2IDX["[sos]"]
    eos_action = ACTION2IDX["[eos]"]

    with open(f"{args.compositional_splits}", "r") as f:
        gscan_dataset = json.load(f)

    with open(f"{args.metalearn_data_directory}/valid/h.pb", "rb") as f:
        gscan_metalearn_split_h_demonstrations = pickle.load(f)

    with open(f"{args.baseline_data_directory}/valid/h.pb", "rb") as f:
        gscan_split_h_demonstrations = pickle.load(f)

    dataset = PaddingDataset(
        gscan_metalearn_split_h_demonstrations,
        (None, None, 8, 72, (8, 8), (8, 72)),
        (None, None, pad_word, pad_action, pad_word, pad_action),
    )
    transformer_dataset = PaddingDataset(
        gscan_split_h_demonstrations,
        (8, 72, None),
        (pad_word, pad_action, None),
    )

    (
        predicted_targets_stacked,
        logits_stacked,
        exacts_stacked,
    ) = get_metaseq2seq_predictions(
        args.meta_seq2seq_checkpoint, dataset, not args.disable_cuda
    )
    (
        transformer_predicted_targets_stacked,
        transformer_logits_stacked,
        transformer_exacts_stacked,
    ) = get_transformer_predictions(
        args.transformer_checkpoint, transformer_dataset, not args.disable_cuda
    )

    print("Exact match accurracy - transformer")
    print(np.array(transformer_exacts_stacked).astype(np.float).mean())

    print("Exact match accurracy - meta-seq2seq")
    print(np.array(exacts_stacked).astype(np.float).mean())

    error_classifications_indices = classify_error_types(
        gscan_dataset,
        dataset,
        predicted_targets_stacked,
        exacts_stacked,
        ACTION2IDX,
        IDX2ACTION,
    )

    # Compute entropy value for turns
    compute_turn_entropies(
        error_classifications_indices,
        dataset,
        logits_stacked,
        predicted_targets_stacked,
        ACTION2IDX,
    )

    # Measure edit distance between predicted targets and inputs
    generate_edit_distance_plots(
        dataset,
        gscan_split_h_demonstrations,
        predicted_targets_stacked,
        exacts_stacked,
        transformer_predicted_targets_stacked,
        transformer_exacts_stacked,
        ACTION2IDX,
    )

    # Frequency-counting conditional probabilities
    frequency_count_conditional_probabilities(gscan_dataset, ACTION2IDX)


if __name__ == "__main__":
    main()
