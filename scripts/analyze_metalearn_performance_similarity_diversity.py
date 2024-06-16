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
from analyze_nearest_neighbour_similarities import situations_to_dense_situations, situation_to_dense

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
    (
        query_sentences,
        support_sentences,
        query_states,
        support_states
    ) = list(zip(*[
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
            ),
            (
                situation_to_dense(state[~(state == 0).all(axis=-1)])
            ),
            (
                np.stack([
                    situation_to_dense(s[~(s == 0).all(axis=-1)])
                    for s in support_states if s.any()
                ])
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
    state_relevances = [
        (query_state[None] == ss).all(axis=-1).astype(np.float32).mean(axis=-1)
        for query_state, ss in zip(query_states, support_states)
    ]

    mean_relevances = [r.mean() for r in relevances]
    max_relevances = [r.max() for r in relevances]
    top4_mean_relevances = [r.flatten()[r.flatten().argsort()][-4:].mean() for r in relevances]

    state_mean_relevances = [r.mean() for r in state_relevances]
    state_max_relevances = [r.max() for r in state_relevances]
    state_top4_mean_relevances = [r.flatten()[r.flatten().argsort()][-4:].mean() for r in state_relevances]

    diversities = [
        # (S x 1 x E) - (1 x S x E) => (S x S x E) => (S x S) => S
        np.linalg.norm(s[:, None, :] - s[None, :, :], axis=-1).sum() / ((s.shape[0] ** 2 - s.shape[0]) or 1)
        for s in support_sentence_encodings
    ]
    state_diversities = [
        # 1 x S x M x T == S x 1 x T => S x S x M x T => S x S
        np.triu(1 - (ss[None] == ss[:, None]).all(axis=-1).mean(axis=-1), 1).sum() / ((ss.shape[0] * (ss.shape[0] - 1)) // 2 or 1)
        for ss in support_states
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

    return list(zip(exacts_stacked.numpy(), mean_relevances, max_relevances, top4_mean_relevances, diversities, state_mean_relevances, state_max_relevances, state_top4_mean_relevances, state_diversities)) 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--data-directory", type=str, required=True)
    parser.add_argument("--transformer-checkpoint", type=str, required=True)
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--limit-load", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--only-splits", type=str, nargs="*")
    parser.add_argument("--determine-padding", action="store_true")
    parser.add_argument("--output-directory", type=str, required=True)
    parser.add_argument("--limit-per-split", type=int, default=None)
    args = parser.parse_args()


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

    pad_instructions_to, pad_actions_to, pad_state_to = determine_padding(
        itertools.chain.from_iterable([
            train_demonstrations,
            *valid_demonstrations_dict.values()
        ])
    )

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
    module = BigSymbolTransformerLearner.load_from_checkpoint(args.transformer_checkpoint)
    module.hparams.predict_only_exacts = True

    np.random.seed(0)
    per_n_per_split_exacts = list(itertools.chain.from_iterable([
        list(map(lambda x: (k, *x), batch_measure_performance_similarities_diversity(
            module,
            sentence_transformer_model,
            PaddingDataset(
                ReorderSupportsByDistanceDataset(
                    MapDataset(
                        MapDataset(
                            Subset(demonstrations, np.random.permutation(len(demonstrations))[:args.limit_per_split])
                            if args.limit_per_split else demonstrations,
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
                    (pad_state_to, None),
                    (16, pad_state_to, None),
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
        if len(demonstrations)
    ]))

    predictor_metrics = ["Mean Relevance", "Max Relevance", "Top 4 Mean Relevance", "Diversity", "State Mean Relevance", "State Max Relevance", "State Top 4 Mean Relevance", "State Diversity"]
    plot_data = pd.DataFrame(per_n_per_split_exacts, columns=["Split", "Exact Match Fraction"] + predictor_metrics)

    for split in valid_demonstrations_dict:
        for metric in predictor_metrics:
            if len(plot_data.loc[plot_data['Split'] == split][metric]):
                cats, bins = pd.cut(plot_data.loc[plot_data['Split'] == split][metric], 10, retbins=True)
                plot_data.loc[plot_data['Split'] == split, "Bin " + metric] = bins[cats.cat.codes]

    os.makedirs(args.output_directory, exist_ok=True)

    for metric in predictor_metrics:
        groupby_df = plot_data.groupby(['Bin ' + metric, 'Split']).agg({'Exact Match Fraction': ['mean', 'std'], metric: 'count'})
        groupby_df.columns = ['mean', 'std', 'count']
        groupby_df = groupby_df.sort_values(['Split', 'Bin ' + metric])

        with open(os.path.join(args.output_directory, metric + ".tex"), "w") as f:
            print(groupby_df.to_latex(float_format='%.2f'), file=f)

        groupby_df.to_csv(os.path.join(args.output_directory, metric + ".csv"))

    for split in valid_demonstrations_dict:
        plot_data[plot_data.Split == split][predictor_metrics + ["Exact Match Fraction"]].corr(method='pearson').to_csv(
            os.path.join(args.output_directory, split + ".corr.csv")
        )

if __name__ == "__main__":
    main()