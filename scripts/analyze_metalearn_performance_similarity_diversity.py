import argparse
import os
import numpy as np
import torch
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
from sentence_transformers import SentenceTransformer

from analyze_failure_cases import get_metaseq2seq_predictions_from_model
from train_meta_encdec_big_symbol_transformer import determine_padding, determine_state_profile

def mean_std(array):
    return [array.mean(), array.std()]


def normalize(vector):
    return vector / np.linalg.norm(vector, axis=-1)[:, None]


def batch_measure_performance_similarities_diversity(
    transformer_module,
    sentence_transformer_checkpoint,
    dataset,
    use_cuda,
    batch_size,
    only_exacts,
    idx2word,
    pad_word
):
    query_sentences, support_sentences = list(zip(*[
        (
            (
                " ".join([
                    idx2word[w] for w in query_instruction
                    if w != pad_word
                ])
            ),
            (
                [
                    " ".join([
                        idx2word[w] for w in support_instruction
                        if w != pad_word
                    ])
                    for support_instruction in filter(
                        lambda x: x[0] != pad_word,
                        support_instructions
                    )
                ]
            )
        )
        for state, support_states, query_instruction, query_targets, support_instructions, support_actions in dataset
    ]))

    with torch.inference_mode():
        sentence_transformer_checkpoint.cuda()

        # We can have an uneven number of support instructions per support set,
        # so we need to keep track of the assignment indices
        support_sentence_assignment_indices = np.concatenate([
            [i] * len(b)
            for i, b in enumerate(support_sentences)
        ])
        query_sentence_encodings = sentence_transformer_checkpoint.encode(query_sentences, show_progress_bar=True)
        support_sentence_encodings = [
            np.stack(list(map(lambda x: x[1], group)))
            for key, group in itertools.groupby(
                enumerate(sentence_transformer_checkpoint.encode(list(itertools.chain.from_iterable(support_sentences)), show_progress_bar=True)),
                key=lambda x: support_sentence_assignment_indices[x[0]]
            )
        ]
        sentence_transformer_checkpoint.cpu()

    relevances = [
        # (1 x E) * (E x S) => S => 1
        (normalize(q[None]) @ normalize(s).T)
        for q, s in zip(query_sentence_encodings, support_sentence_encodings)
    ]

    mean_relevances = [r.mean() for r in relevances]
    max_relevances = [r.max() for r in relevances]
    top4_mean_relevances = [r.flatten()[r.flatten().argsort()][-4:].mean() for r in relevances]

    diversities = [
        # (S x 1 x E) - (1 x S x E) => (S x S x E) => (S x S) => S
        np.linalg.norm(s[:, None, :] - s[None, :, :], axis=-1).sum() / ((s.shape[0] ** 2 - s.shape[0]) or 1)
        for s in support_sentence_encodings
    ]

    transformer_module.cuda()
    exacts_stacked = get_metaseq2seq_predictions_from_model(
        transformer_module,
        dataset,
        use_cuda=use_cuda,
        batch_size=batch_size,
        only_exacts=only_exacts,
        validate_first=True
    )
    transformer_module.cpu()

    return list(zip(exacts_stacked.numpy(), mean_relevances, max_relevances, top4_mean_relevances, diversities)) 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--data-directory", type=str, required=True)
    parser.add_argument("--output-directory", type=str, required=True)
    parser.add_argument("--transformer-checkpoint", type=str, required=True)
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--limit-load", type=int, default=None)
    parser.add_argument("--pad-instructions-to", type=int, default=8)
    parser.add_argument("--pad-actions-to", type=int, default=128)
    parser.add_argument("--pad-state-to", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--only-splits", type=str, nargs="*")
    parser.add_argument("--determine-padding", action="store_true")
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

    pad_state_to = args.pad_state_to
    pad_actions_to = args.pad_actions_to
    pad_instructions_to = args.pad_instructions_to

    pad_instructions_to, pad_actions_to, pad_state_to = determine_padding(train_demonstrations)

    state_component_max_len, state_feat_len = determine_state_profile(
        train_demonstrations,
        valid_demonstrations_dict
    )
    print(BigSymbolTransformerLearner.load_from_checkpoint(args.transformer_checkpoint))

    # Here we calculate the sentence-transformer cosine similarity of
    # the support instructions to the query instruction.
    #
    # Diversity score is given by the average of the L2 distances of
    # each vector from every other vector
    sentence_transformer_model = SentenceTransformer('all-mpnet-base-v2')

    per_n_per_split_exacts = list(itertools.chain.from_iterable([
        list(map(lambda x: (k, *x), batch_measure_performance_similarities_diversity(
            args.transformer_checkpoint,
            sentence_transformer_model,
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
                    16,
                ),
                (
                    (pad_state_to, state_feat_len),
                    (16, pad_state_to, state_feat_len),
                    pad_instructions_to,
                    pad_actions_to,
                    (16, pad_instructions_to),
                    (16, pad_actions_to),
                ),
                (pad_state, pad_state, pad_word, pad_action, pad_word, pad_action),
            ),
            not args.disable_cuda,
            batch_size=args.batch_size,
            only_exacts=True,
            idx2word=IDX2WORD,
            pad_word=pad_word
        )))
        for k, demonstrations in tqdm(valid_demonstrations_dict.items())
    ]))

    predictor_metrics = ["Mean Relevance", "Max Relevance", "Top 4 Mean Relevance", "Diversity"]
    plot_data = pd.DataFrame(per_n_per_split_exacts, columns=["Split", "Exact Match Fraction"] + predictor_metrics)

    for split in valid_demonstrations_dict:
        for metric in predictor_metrics:
            cats, bins = pd.cut(plot_data.loc[plot_data['Split'] == split][metric], 10, retbins=True)
            plot_data.loc[plot_data['Split'] == split, "Bin " + metric] = bins[cats.cat.codes]

    for metric in predictor_metrics:
        groupby_df = plot_data.groupby(['Bin ' + metric, 'Split']).agg({'Exact Match Fraction': ['mean', 'std'], metric: 'count'})
        groupby_df.columns = ['mean', 'std', 'count']
        groupby_df = groupby_df.sort_values(['Split', 'Bin ' + metric])
        print(groupby_df)

    import pdb
    pdb.set_trace()

    sns.lineplot(data=plot_data, x="Number of Demonstrations", y="Exact Match Fraction", hue="Split")
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(args.output_directory, "demonstrations-efficiency.pdf"))

    for split in valid_demonstrations_dict.keys():
        plt.clf()
        sns.lineplot(data=plot_data.loc[plot_data["Split"] == split],
                     x="Number of Demonstrations",
                     y="Exact Match Fraction",
                     hue="Split")
        plt.ylim(0, 1.1)
        plt.savefig(os.path.join(args.output_directory, f"demonstrations-efficiency-{split}.pdf"))

    plot_data.to_csv(os.path.join(args.output_directory, "results.csv"))

if __name__ == "__main__":
    main()