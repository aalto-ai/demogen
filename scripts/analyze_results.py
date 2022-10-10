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


def format_experiment_name(experiment_config, params):
    name_dict = {**experiment_config, **params}
    return "_".join(
        map(
            str,
            [
                name_dict["headline"],
                "s",
                name_dict["seed"],
                "m",
                name_dict["model"],
                "l",
                name_dict["layers"],
                "h",
                name_dict["heads"],
                "d",
                name_dict["hidden"],
                "it",
                name_dict["iterations"],
                "b",
                name_dict["batch_size"],
                "d",
                name_dict["dataset"],
                "t",
                name_dict["tag"],
                "drop",
                name_dict["dropout"],
            ],
        )
    )


def format_model_name(experiment_config, include_hparams=False):
    return "_".join(
        map(
            str,
            [experiment_config["model"]]
            + (
                [
                    "l",
                    experiment_config["layers"],
                    "h",
                    experiment_config["heads"],
                    "d",
                    experiment_config["hidden"],
                ]
                if include_hparams
                else []
            ),
        )
    )


def format_log_path(experiment_config, params, model_include_hparams=False):
    return os.path.join(
        format_experiment_name(experiment_config, params),
        format_model_name(experiment_config, include_hparams=model_include_hparams),
        experiment_config["dataset"],
        str(params["seed"]),
        "lightning_logs",
        "100",
        "metrics.csv",
    )


BASE_EXPERIMENT_CONFIGS = {
    "transformer": {
        "headline": "gscan",
        "model": "transformer_encoder_only_decode_actions",
        "layers": 28,
        "heads": 4,
        "hidden": 128,
        "iterations": 50000,
        "batch_size": 4096,
        "dataset": "gscan",
        "tag": "none",
        "dropout": 0.0,
    },
    "meta_gscan_oracle": {
        "headline": "meta_gscan",
        "model": "meta_imagination_transformer",
        "layers": 8,
        "heads": 4,
        "hidden": 128,
        "iterations": 50000,
        "batch_size": 4096,
        "dataset": "gscan_metalearn_fixed",
        "tag": "none",
        "dropout": 0.0,
    },
}
ABLATION_EXPERIMENT_CONFIGS = {
    "meta_gscan_oracle_noshuffle": {
        **BASE_EXPERIMENT_CONFIGS["meta_gscan_oracle"],
        "tag": "noshuffle",
    },
    "meta_gscan_imagine_actions": {
        **BASE_EXPERIMENT_CONFIGS["meta_gscan_oracle"],
        "dataset": "gscan_imagine_actions_fixed",
    },
    "meta_gscan_distractors": {
        **BASE_EXPERIMENT_CONFIGS["meta_gscan_oracle"],
        "dataset": "gscan_metalearn_distractors_fixed",
    },
    "meta_gscan_sample_environments": {
        **BASE_EXPERIMENT_CONFIGS["meta_gscan_oracle"],
        "dataset": "gscan_metalearn_sample_environments_fixed",
    },
    "meta_gscan_only_random": {
        **BASE_EXPERIMENT_CONFIGS["meta_gscan_oracle"],
        "dataset": "gscan_metalearn_only_random",
    },
}


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

    all_transformer_encoder_only_metrics_dfs = [
        truncate_at_key(
            pd.read_csv(
                os.path.join(
                    args.logs_dir,
                    format_log_path(
                        BASE_EXPERIMENT_CONFIGS["transformer"], {"seed": seed}
                    ),
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
                    args.logs_dir,
                    format_log_path(
                        BASE_EXPERIMENT_CONFIGS["meta_gscan_oracle"], {"seed": seed}
                    ),
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
                    args.logs_dir,
                    format_log_path(
                        BASE_EXPERIMENT_CONFIGS["meta_gscan_oracle"], {"seed": seed}
                    ),
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
                    args.logs_dir,
                    format_log_path(
                        ABLATION_EXPERIMENT_CONFIGS["meta_gscan_oracle_noshuffle"],
                        {"seed": seed},
                    ),
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
                    args.logs_dir,
                    format_log_path(
                        ABLATION_EXPERIMENT_CONFIGS["meta_gscan_imagine_actions"],
                        {"seed": seed},
                    ),
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
                    args.logs_dir,
                    format_log_path(
                        ABLATION_EXPERIMENT_CONFIGS["meta_gscan_distractors"],
                        {"seed": seed},
                    ),
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
                    args.logs_dir,
                    format_log_path(
                        ABLATION_EXPERIMENT_CONFIGS["meta_gscan_sample_environments"],
                        {"seed": seed},
                    ),
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
                    args.logs_dir,
                    format_log_path(
                        ABLATION_EXPERIMENT_CONFIGS["meta_gscan_only_random"],
                        {"seed": seed},
                    ),
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
        "Different States",
    ]
    ablation_study_table = ablation_study_table.drop("index", axis=1)

    print("Table 2")
    print(ablation_study_table.T.to_latex())


if __name__ == "__main__":
    main()
