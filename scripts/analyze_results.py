import argparse
import matplotlib
import seaborn as sns
import rliable
import fnmatch
import operator
import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import tqdm.auto as tqdm
import json
from collections import Counter, defaultdict, deque


def truncate_at_key(df, key, limit):
    return df[df[key] <= limit]


def exclude_worst_performing_by_metric(dfs, metric, n_exclude, rolling=50):
    best_rolling_max = [
        (k, dfs[k][metric].dropna().rolling(rolling).mean().fillna(0).max())
        for k in range(len(dfs))
    ]
    sort_keys = sorted(best_rolling_max, key=lambda k: k[1])
    print(sort_keys)

    return [dfs[k[0]] for k in sort_keys[n_exclude:]]


def get_top_values_for_corresponding_value(dfs, corresponding, values, rolling=1):
    cols = values

    nonrolling_dfs = [df[cols].dropna() for df in dfs]

    rolling_dfs = [df[cols].rolling(rolling).mean().fillna(0) for df in nonrolling_dfs]

    argwheres = [
        np.argwhere(
            rolling_df[corresponding].values == rolling_df[corresponding].max()
        )[-1][0]
        for rolling_df in rolling_dfs
    ]
    print(list(zip(argwheres, map(lambda df: df.shape[0], rolling_dfs))))

    return pd.DataFrame(
        np.stack(
            [
                nonrolling_df[cols].iloc[argwhere].values
                for argwhere, nonrolling_df in zip(argwheres, nonrolling_dfs)
            ]
        ),
        columns=cols,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--limit", type=int, default=30000)
    parser.add_argument("--ablations-limit", type=int, default=30000)
    parser.add_argument("--drop-bad-seeds", type=int, default=3)
    parser.add_argument("--ablations-drop-bad-seeds", type=int, default=3)
    parser.add_argument("--exclude-by-a-smoothing", type=int, default=50)
    parser.add_argument("--result-smoothing", type=int, default=1)
    args = parser.parse_args()

    SCRIPTS_LOGS_DIR = args.logs_dir

    all_transformer_encoder_only_metrics_dfs = [
        truncate_at_key(
            pd.read_csv(
                os.path.join(
                    SCRIPTS_LOGS_DIR,
                    f"gscan_s_{seed}_m_transformer_encoder_only_decode_actions_l_28_h_4_d_128_it_50000_b_4096_d_gscan_t_none_drop_0.0/transformer_encoder_only_decode_actions/gscan/{seed}/lightning_logs/version_100/metrics.csv",
                )
            ),
            "step",
            args.limit,
        )
        for seed in range(10)
    ]
    all_transformer_encoder_only_metrics_dfs = exclude_worst_performing_by_metric(
        all_transformer_encoder_only_metrics_dfs,
        "vexact/dataloader_idx_0",
        args.drop_bad_seeds,
        args.exclude_by_a_smoothing,
    )

    all_meta_gscan_oracle_metrics_dfs = [
        truncate_at_key(
            pd.read_csv(
                os.path.join(
                    SCRIPTS_LOGS_DIR,
                    f"meta_gscan_s_{seed}_m_meta_imagination_transformer_l_8_h_4_d_128_it_50000_b_4096_d_gscan_metalearn_fixed_t_none_drop_0.0/meta_imagination_transformer/gscan_metalearn_fixed/{seed}/lightning_logs/100/metrics.csv",
                )
            ),
            "step",
            args.limit,
        )
        for seed in range(10)
    ]
    meta_gscan_oracle_metrics_dfs = exclude_worst_performing_by_metric(
        all_meta_gscan_oracle_metrics_dfs,
        "vexact/dataloader_idx_0",
        args.drop_bad_seeds,
        args.exclude_by_a_smoothing,
    )

    all_meta_gscan_oracle_metrics_dfs_20k = [
        truncate_at_key(
            pd.read_csv(
                os.path.join(
                    SCRIPTS_LOGS_DIR,
                    f"meta_gscan_s_{seed}_m_meta_imagination_transformer_l_8_h_4_d_128_it_50000_b_4096_d_gscan_metalearn_fixed_t_none_drop_0.0/meta_imagination_transformer/gscan_metalearn_fixed/{seed}/lightning_logs/100/metrics.csv",
                )
            ),
            "step",
            args.ablations_limit,
        )
        for seed in range(10)
    ]
    meta_gscan_oracle_metrics_dfs_20k = exclude_worst_performing_by_metric(
        all_meta_gscan_oracle_metrics_dfs_20k,
        "vexact/dataloader_idx_0",
        args.ablations_drop_bad_seeds,
        args.exclude_by_a_smoothing,
    )

    all_meta_gscan_oracle_noshuffle_metrics_dfs_20k = [
        truncate_at_key(
            pd.read_csv(
                os.path.join(
                    SCRIPTS_LOGS_DIR,
                    f"meta_gscan_s_{seed}_m_meta_imagination_transformer_l_8_h_4_d_128_it_50000_b_4096_d_gscan_metalearn_fixed_t_noshuffle_drop_0.0/meta_imagination_transformer/gscan_metalearn_fixed/{seed}/lightning_logs/100/metrics.csv",
                )
            ),
            "step",
            args.ablations_limit,
        )
        for seed in range(10)
    ]
    meta_gscan_oracle_noshuffle_metrics_dfs_20k = exclude_worst_performing_by_metric(
        all_meta_gscan_oracle_noshuffle_metrics_dfs_20k,
        "vexact/dataloader_idx_0",
        args.ablations_drop_bad_seeds,
        args.exclude_by_a_smoothing,
    )

    all_meta_gscan_imagine_actions_metrics_dfs_20k = [
        truncate_at_key(
            pd.read_csv(
                os.path.join(
                    SCRIPTS_LOGS_DIR,
                    f"meta_gscan_s_{seed}_m_meta_imagination_transformer_l_8_h_4_d_128_it_50000_b_4096_d_gscan_imagine_actions_fixed_t_none_drop_0.0/meta_imagination_transformer/gscan_imagine_actions_fixed/{seed}/lightning_logs/100/metrics.csv",
                )
            ),
            "step",
            args.ablations_limit,
        )
        for seed in range(10)
    ]
    meta_gscan_imagine_actions_metrics_dfs_20k = exclude_worst_performing_by_metric(
        all_meta_gscan_imagine_actions_metrics_dfs_20k,
        "vexact/dataloader_idx_0",
        args.ablations_drop_bad_seeds,
        args.exclude_by_a_smoothing,
    )

    all_meta_gscan_metalearn_distractors_metrics_dfs_20k = [
        truncate_at_key(
            pd.read_csv(
                os.path.join(
                    SCRIPTS_LOGS_DIR,
                    f"meta_gscan_s_{seed}_m_meta_imagination_transformer_l_8_h_4_d_128_it_50000_b_4096_d_gscan_metalearn_distractors_fixed_t_none_drop_0.0/meta_imagination_transformer/gscan_metalearn_distractors_fixed/{seed}/lightning_logs/100/metrics.csv",
                )
            ),
            "step",
            args.ablations_limit,
        )
        for seed in range(10)
    ]
    meta_gscan_metalearn_distractors_metrics_dfs_20k = (
        exclude_worst_performing_by_metric(
            all_meta_gscan_metalearn_distractors_metrics_dfs_20k,
            "vexact/dataloader_idx_0",
            args.ablations_drop_bad_seeds,
            args.exclude_by_a_smoothing,
        )
    )

    all_meta_gscan_metalearn_sample_environments_metrics_dfs_20k = [
        truncate_at_key(
            pd.read_csv(
                os.path.join(
                    SCRIPTS_LOGS_DIR,
                    f"meta_gscan_s_{seed}_m_meta_imagination_transformer_l_8_h_4_d_128_it_50000_b_4096_d_gscan_metalearn_sample_environments_fixed_t_none_drop_0.0/meta_imagination_transformer/gscan_metalearn_sample_environments_fixed/{seed}/lightning_logs/100/metrics.csv",
                )
            ),
            "step",
            args.ablations_limit,
        )
        for seed in range(10)
    ]
    meta_gscan_metalearn_sample_environments_metrics_dfs_20k = (
        exclude_worst_performing_by_metric(
            all_meta_gscan_metalearn_sample_environments_metrics_dfs_20k,
            "vexact/dataloader_idx_0",
            args.ablations_drop_bad_seeds,
            args.exclude_by_a_smoothing,
        )
    )

    all_meta_gscan_metalearn_only_random_metrics_dfs_20k = [
        truncate_at_key(
            pd.read_csv(
                os.path.join(
                    SCRIPTS_LOGS_DIR,
                    f"meta_gscan_s_{seed}_m_meta_imagination_transformer_l_8_h_4_d_128_it_50000_b_4096_d_gscan_metalearn_only_random_t_none_drop_0.0/meta_imagination_transformer/gscan_metalearn_only_random/{seed}/lightning_logs/100/metrics.csv",
                )
            ),
            "step",
            args.ablations_limit,
        )
        for seed in range(10)
    ]
    meta_gscan_metalearn_only_random_metrics_dfs_20k = (
        exclude_worst_performing_by_metric(
            all_meta_gscan_metalearn_only_random_metrics_dfs_20k,
            "vexact/dataloader_idx_0",
            args.ablations_drop_bad_seeds,
            args.exclude_by_a_smoothing,
        )
    )

    meta_gscan_oracle_performance_at_best_0 = get_top_values_for_corresponding_value(
        meta_gscan_oracle_metrics_dfs,
        "vexact/dataloader_idx_0",
        [
            "vexact/dataloader_idx_0",
            "vexact/dataloader_idx_1",
            "vexact/dataloader_idx_2",
            "vexact/dataloader_idx_3",
            "vexact/dataloader_idx_4",
            "vexact/dataloader_idx_5",
            "vexact/dataloader_idx_6",
        ],
        args.result_smoothing,
    ).describe()

    meta_gscan_oracle_performance_at_best_6 = get_top_values_for_corresponding_value(
        meta_gscan_oracle_metrics_dfs,
        "vexact/dataloader_idx_6",
        [
            "vexact/dataloader_idx_0",
            "vexact/dataloader_idx_1",
            "vexact/dataloader_idx_2",
            "vexact/dataloader_idx_3",
            "vexact/dataloader_idx_4",
            "vexact/dataloader_idx_5",
            "vexact/dataloader_idx_6",
        ],
        args.result_smoothing,
    ).describe()
    meta_gscan_oracle_performance_at_best_6

    meta_gscan_oracle_performance_at_best_6_20k = (
        get_top_values_for_corresponding_value(
            meta_gscan_oracle_metrics_dfs_20k,
            "vexact/dataloader_idx_0",
            [
                "vexact/dataloader_idx_0",
                "vexact/dataloader_idx_1",
                "vexact/dataloader_idx_2",
                "vexact/dataloader_idx_3",
                "vexact/dataloader_idx_4",
                "vexact/dataloader_idx_5",
                "vexact/dataloader_idx_6",
            ],
            args.result_smoothing,
        ).describe()
    )
    meta_gscan_oracle_performance_at_best_6_20k

    meta_gscan_oracle_noshuffle_performance_at_best_6_20k = (
        get_top_values_for_corresponding_value(
            meta_gscan_oracle_noshuffle_metrics_dfs_20k,
            "vexact/dataloader_idx_0",
            [
                "vexact/dataloader_idx_0",
                "vexact/dataloader_idx_1",
                "vexact/dataloader_idx_2",
                "vexact/dataloader_idx_3",
                "vexact/dataloader_idx_4",
                "vexact/dataloader_idx_5",
                "vexact/dataloader_idx_6",
            ],
            args.result_smoothing,
        ).describe()
    )
    meta_gscan_oracle_noshuffle_performance_at_best_6_20k

    meta_gscan_imagine_actions_performance_at_best_6_20k = (
        get_top_values_for_corresponding_value(
            meta_gscan_imagine_actions_metrics_dfs_20k,
            "vexact/dataloader_idx_0",
            [
                "vexact/dataloader_idx_0",
                "vexact/dataloader_idx_1",
                "vexact/dataloader_idx_2",
                "vexact/dataloader_idx_3",
                "vexact/dataloader_idx_4",
                "vexact/dataloader_idx_5",
                "vexact/dataloader_idx_6",
            ],
            args.result_smoothing,
        ).describe()
    )
    meta_gscan_imagine_actions_performance_at_best_6_20k

    meta_gscan_metalearn_distractors_performance_at_best_6_20k = (
        get_top_values_for_corresponding_value(
            meta_gscan_metalearn_distractors_metrics_dfs_20k,
            "vexact/dataloader_idx_0",
            [
                "vexact/dataloader_idx_0",
                "vexact/dataloader_idx_1",
                "vexact/dataloader_idx_2",
                "vexact/dataloader_idx_3",
                "vexact/dataloader_idx_4",
                "vexact/dataloader_idx_5",
                "vexact/dataloader_idx_6",
            ],
            args.result_smoothing,
        ).describe()
    )
    meta_gscan_metalearn_distractors_performance_at_best_6_20k

    meta_gscan_metalearn_only_random_performance_at_best_6_20k = (
        get_top_values_for_corresponding_value(
            meta_gscan_metalearn_only_random_metrics_dfs_20k,
            "vexact/dataloader_idx_0",
            [
                "vexact/dataloader_idx_0",
                "vexact/dataloader_idx_1",
                "vexact/dataloader_idx_2",
                "vexact/dataloader_idx_3",
                "vexact/dataloader_idx_4",
                "vexact/dataloader_idx_5",
                "vexact/dataloader_idx_6",
            ],
            args.result_smoothing,
        ).describe()
    )
    meta_gscan_metalearn_only_random_performance_at_best_6_20k

    meta_gscan_metalearn_sample_environments_performance_at_best_6_20k = (
        get_top_values_for_corresponding_value(
            meta_gscan_metalearn_sample_environments_metrics_dfs_20k,
            "vexact/dataloader_idx_0",
            [
                "vexact/dataloader_idx_0",
                "vexact/dataloader_idx_1",
                "vexact/dataloader_idx_2",
                "vexact/dataloader_idx_3",
                "vexact/dataloader_idx_4",
                "vexact/dataloader_idx_5",
                "vexact/dataloader_idx_6",
            ],
            args.result_smoothing,
        ).describe()
    )
    meta_gscan_metalearn_sample_environments_performance_at_best_6_20k

    gscan_transformer_performance_at_best_0 = get_top_values_for_corresponding_value(
        all_transformer_encoder_only_metrics_dfs,
        "vexact/dataloader_idx_0",
        [
            "vexact/dataloader_idx_0",
            "vexact/dataloader_idx_1",
            "vexact/dataloader_idx_2",
            "vexact/dataloader_idx_3",
            "vexact/dataloader_idx_4",
            "vexact/dataloader_idx_5",
            "vexact/dataloader_idx_6",
        ],
        args.result_smoothing,
    ).describe()
    gscan_transformer_performance_at_best_0

    results_table = (
        pd.concat(
            [
                gscan_transformer_performance_at_best_0.T["mean"],
                meta_gscan_oracle_performance_at_best_0.T["mean"],
                meta_gscan_oracle_performance_at_best_6.T["mean"],
            ],
            axis=1,
        )
        .T.round(2)
        .astype(str)
        .reset_index()
        + " ± "
        + (
            pd.concat(
                [
                    gscan_transformer_performance_at_best_0.T["std"],
                    meta_gscan_oracle_performance_at_best_0.T["std"],
                    meta_gscan_oracle_performance_at_best_6.T["std"],
                ],
                axis=1,
            )
            .T.round(2)
            .astype(str)
            .reset_index()
        )
    ).T
    results_table.index = ["index", "A", "B", "C", "D", "E", "F", "H"]
    results_table.columns = ["Transformer", "Ours(o, A)", "Ours(o)"]
    print("Table 1")
    print(results_table.to_latex(float_format="%.2f", escape=False))

    ablation_study_table = (
        pd.concat(
            [
                meta_gscan_oracle_performance_at_best_6_20k.T["mean"],
                meta_gscan_oracle_noshuffle_performance_at_best_6_20k.T["mean"],
                meta_gscan_imagine_actions_performance_at_best_6_20k.T["mean"],
                meta_gscan_metalearn_distractors_performance_at_best_6_20k.T["mean"],
                meta_gscan_metalearn_only_random_performance_at_best_6_20k.T["mean"],
                meta_gscan_metalearn_sample_environments_performance_at_best_6_20k.T[
                    "mean"
                ],
            ],
            axis=1,
        )
        .T.round(2)
        .astype(str)
        .reset_index()
        + " ± "
        + (
            pd.concat(
                [
                    meta_gscan_oracle_performance_at_best_6_20k.T["std"],
                    meta_gscan_oracle_noshuffle_performance_at_best_6_20k.T["std"],
                    meta_gscan_imagine_actions_performance_at_best_6_20k.T["std"],
                    meta_gscan_metalearn_distractors_performance_at_best_6_20k.T["std"],
                    meta_gscan_metalearn_only_random_performance_at_best_6_20k.T["std"],
                    meta_gscan_metalearn_sample_environments_performance_at_best_6_20k.T[
                        "std"
                    ],
                ],
                axis=1,
            )
            .T.round(2)
            .astype(str)
            .reset_index()
        )
    )
    ablation_study_table.columns = ["index", "A", "B", "C", "D", "E", "F", "H"]
    ablation_study_table.index = [
        "Ours(o, A)",
        "No permutations",
        "Transformer Actions",
        "Distractors",
        "Random Instructions",
        "Sample Environments",
    ]
    ablation_study_table = ablation_study_table.drop("index", axis=1)

    print("Table 2")
    print(ablation_study_table.T.to_latex())


if __name__ == "__main__":
    main()
