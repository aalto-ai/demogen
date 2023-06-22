import argparse
import os
import json
import itertools
from collections import Counter
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import special
import spacy
from tqdm.auto import tqdm
from scipy.stats import chisquare

def extract_dataset_instruction_set_and_targets(p):
    with open(p, "r") as f:
        return {
            split: {
                "targets": [e["situation"]["target_object"]["object"] for e in examples],
                "commands": [e["command"] for e in examples]
            }
            for split, examples in json.load(f)["examples"].items()
        }


def make_word_counts(instructions, splits=["train"]):
    return Counter(
        itertools.chain.from_iterable(
            map(
                lambda x: itertools.chain.from_iterable(
                    map(lambda y: itertools.chain.from_iterable(map(lambda z: z.split(), y.split(","))), x)
                ),
                map(lambda x: x[1]["commands"], filter(lambda y: splits is None or y[0] in splits, instructions.items()))
            )
        )
    )


def zipf_a_mle(word_counts):
    x_min = min(word_counts)
    a = 1 + len(word_counts) * (np.log(np.array(word_counts) / x_min).sum()) ** -1
    
    return a


def classic_zipf(N, k, s=1.5):
    return (1/(k + 1)**s)/(np.sum(1/(np.arange(1, N+1)**s)))


def zipf_stats(word_counts):
    # We discard anything with count < 5, since that would make the chisquared
    # test invalid
    word_counts = list(filter(lambda x: x >= 5, word_counts))
    
    a = zipf_a_mle(list(word_counts))
    table = np.stack([
        list(reversed(sorted([c / sum(word_counts) for c in word_counts]))),
        classic_zipf(
            len(word_counts),
            np.arange(len(word_counts)),
            s=a
        )
    ]).T

    chisquared_table = (sum(word_counts) * table).round().astype(int)
    
    return {
        "uniq": len(word_counts),
        "zipf_a": a,
        "rmse": np.sqrt(((table[:, 0] - table[:, 1]) ** 2).mean())
    }


def instruction_contains_target(instruction, target):
    return (
        (target["size"] == 1 and "small" in instruction) or
        (target["size"] == 4 and "big" in instruction),
        target["color"] in instruction,
        target["shape"] in instruction
    )


def plot_bar_pairs(first, second, index, labels, save=None):
    sns.barplot(
        data=pd.DataFrame(np.stack([
            first,
            second
        ]).T, index=index, columns=labels).melt(ignore_index=False).reset_index(),
        x="index",
        y="value",
        hue="variable"
    )
    plt.gca().set_yscale('log')
    plt.xticks(rotation = 45)
    plt.ylabel("p(word) (log scale)")
    plt.xlabel("Word")
    plt.gca().tick_params(axis='x', which='major', labelsize=6)
    
    if save:
        plt.savefig(save)
    
    plt.show()


def plot_bars_with_zipf(first, second, labels, save=None):
    second = second
    
    plt.bar(np.arange(first.shape[0]), first)
    plt.bar(np.arange(second.shape[0]), second)
    plt.gca().set_yscale('log')
    plt.xticks(rotation = 45)
    plt.ylabel("p(word) (log scale)")
    plt.xlabel("Word")
    
    x = np.arange(0, max(len(first), len(second)))
    plt.plot(
        x,
        classic_zipf(
            x.shape[0],
            x,
            zipf_a_mle(first)
        ), linewidth=2, color='r'
    )
    
    if save:
        plt.savefig(save)
    
    plt.show()

def make_word_counts_comparison_plots(
    synthetic_word_counts,
    paraphrased_word_counts,
    save_comparison=None,
    save_zipf=None
):
    most_common_paraphrased_word_counts = paraphrased_word_counts.most_common(35)
    most_common_paraphrased_words_positions = {
        word: i for i, word in enumerate(map(lambda x: x[0], most_common_paraphrased_word_counts))
    }
    most_common_synthetic_words_in_same_positions = {
        k: most_common_paraphrased_words_positions[k]
        for k in synthetic_word_counts.keys()
        if k in most_common_paraphrased_words_positions.keys()
    }
    
    parapharsed_words_bars = np.array(list(map(lambda x: x[1], most_common_paraphrased_word_counts)))
    synthetic_words_bars = np.zeros_like(parapharsed_words_bars)
    for w, i in most_common_synthetic_words_in_same_positions.items():
        synthetic_words_bars[i] = synthetic_word_counts[w]

    plot_bar_pairs(
        parapharsed_words_bars / sum(paraphrased_word_counts.values()),
        synthetic_words_bars  / sum(synthetic_word_counts.values()),
        index=most_common_paraphrased_words_positions.keys(),
        labels=["Paraphrased", "gSCAN"],
        save=save_comparison
    )
    plot_bars_with_zipf(
        np.array(list(reversed(sorted(paraphrased_word_counts.values())))) / sum(paraphrased_word_counts.values()),
        np.array(list(reversed(sorted(synthetic_word_counts.values())))) / sum(synthetic_word_counts.values()),
        labels=["Paraphrased", "gSCAN"],
        save=save_zipf
    )


def compute_unique_parses_with_spacy(instructions):
    return set([
        " ".join(list(map(lambda t: t.dep_, nlp(" ".join(e.split(","))))))
        for e in itertools.chain.from_iterable(map(lambda x: tqdm(x), tqdm(instructions.values())))
    ])


    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dataset-path", type=str, required=True)
    parser.add_argument("--paraphrased-dataset-path", type=str, required=True)
    parser.add_argument("--output-directory", type=str, required=True)
    parser.add_argument("--dataset-name", type=str)
    args = parser.parse_args()

    os.makedirs(args.output_directory, exist_ok=True)

    gscan_instructions = extract_dataset_instruction_set_and_targets(args.original_dataset_path)
    gscan_paraphrased_instructions = extract_dataset_instruction_set_and_targets(args.paraphrased_dataset_path)
    gscan_word_counts = make_word_counts(gscan_instructions)
    gscan_paraphrased_word_counts = make_word_counts(gscan_paraphrased_instructions)
    gscan_zipf_stats = zipf_stats(gscan_word_counts.values())
    gscan_paraphrased_zipf_stats = zipf_stats(gscan_paraphrased_word_counts.values())

    zipf_stats = {
        "original": zipf_stats(gscan_word_counts.values()),
        "paraphrased": zipf_stats(gscan_paraphrased_word_counts.values()),
    }

    gscan_instructions_contain_targets = np.array(list(map(
        lambda x: instruction_contains_target(*x),
        zip(
            gscan_instructions["train"]["commands"],
            gscan_instructions["train"]["targets"]
        )
    )))

    gscan_paraphrased_instructions_contain_targets = np.array(list(map(
        lambda x: instruction_contains_target(*x),
        zip(
            gscan_paraphrased_instructions["train"]["commands"],
            gscan_paraphrased_instructions["train"]["targets"]
        )
    )))

    gscan_vs_paraphrased_contains_targets = np.stack([
        gscan_instructions_contain_targets,
        gscan_paraphrased_instructions_contain_targets
    ], axis=-1)

    print(pd.DataFrame(np.stack([
        (gscan_vs_paraphrased_contains_targets[..., 0] == gscan_vs_paraphrased_contains_targets[..., 1]).astype(float).mean(axis=0)
    ]), index=[args.dataset_name], columns=["Size", "Color", "Object"]).to_latex(float_format='%.4f'))

    make_word_counts_comparison_plots(
        gscan_word_counts,
        gscan_paraphrased_word_counts,
        save_comparison=f"{args.output_directory}/{args.dataset_name}_paraphrase_comparison.pdf",
        save_zipf=f"{args.output_directory}/{args.dataset_name}_paraphrased_zipf.pdf"
    )

    gscan_unique_parses = compute_unique_parses_with_spacy(gscan_instructions)
    gscan_paraphrased_unique_parses = compute_unique_parses_with_spacy(gscan_paraphrased_instructions)

    unique_parses = {
        f"{args.dataset_name}": gscan_unique_parses,
        f"{args.dataset_name}_paraphrased": gscan_paraphrased_unique_parses
    }

    print(pd.DataFrame.from_dict({
        k: {
            "parses": len(unique_parses[k]),
            "words": zipf_stats[k]["uniq"],
            "zipf_a": zipf_stats[k]["zipf_a"],
            "rmse": zipf_stats[k]["rmse"]
        }
        for k in unique_parses
    }).T.to_latex(float_format='%.2f'))

if __name__ == "__main__":
    main()
