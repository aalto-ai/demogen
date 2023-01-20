import argparse
import os
import pandas as pd
import numpy as np


def truncate_at_key(df, key, limit):
    return df[df[key] <= limit]


def exclude_worst_performing_by_metric(
    dfs, metric, n_exclude, rolling=50, descending=False
):
    best_rolling_max = [
        (
            k,
            getattr(
                dfs[k][metric].dropna().rolling(rolling).mean().fillna(0),
                "max" if not descending else "min",
            )(),
        )
        for k in range(len(dfs))
    ]
    sort_keys = sorted(
        best_rolling_max, key=lambda k: k[1] if not descending else -1 * k[1]
    )
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
EXPERIMENT_CONFIGS = {**BASE_EXPERIMENT_CONFIGS, **ABLATION_EXPERIMENT_CONFIGS}


def read_all_csv_files_for_seeds_and_limit(logs_dir, experiment_config, limit):
    return [
        truncate_at_key(
            pd.read_csv(
                os.path.join(
                    logs_dir,
                    format_log_path(experiment_config, {"seed": seed}),
                )
            ),
            "step",
            limit,
        )
        for seed in range(10)
    ]


GSCAN_TEST_SPLIT_DATALOADER_NAMES = [
    "vexact/dataloader_idx_0",
    "vexact/dataloader_idx_1",
    "vexact/dataloader_idx_2",
    "vexact/dataloader_idx_3",
    "vexact/dataloader_idx_4",
    "vexact/dataloader_idx_5",
    "vexact/dataloader_idx_6",
]


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

    table_configs = {
        "transformer_full": {"config_name": "transformer", "limit": args.limit},
        "gscan_oracle_full": {"config_name": "meta_gscan_oracle", "limit": args.limit},
        "gscan_oracle_ablations": {
            "config_name": "meta_gscan_oracle",
            "limit": args.ablations_limit,
        },
        "gscan_oracle_noshuffle": {
            "config_name": "meta_gscan_oracle_noshuffle",
            "limit": args.ablations_limit,
        },
        "gscan_imagine_actions": {
            "config_name": "meta_gscan_imagine_actions",
            "limit": args.ablations_limit,
        },
        "gscan_metalearn_distractors": {
            "config_name": "meta_gscan_distractors",
            "limit": args.ablations_limit,
        },
        "gscan_metalearn_sample_environments": {
            "config_name": "meta_gscan_sample_environments",
            "limit": args.ablations_limit,
        },
        "gscan_metalearn_only_random": {
            "config_name": "meta_gscan_only_random",
            "limit": args.ablations_limit,
        },
    }

    read_metrics_dfs_at_limit = {
        name: read_all_csv_files_for_seeds_and_limit(
            args.logs_dir,
            EXPERIMENT_CONFIGS[table_config["config_name"]],
            table_config["limit"],
        )
        for name, table_config in table_configs.items()
    }
    read_metrics_dfs_excluded = {
        name: exclude_worst_performing_by_metric(
            read_metrics_df_at_limit,
            "vexact/dataloader_idx_0",
            args.drop_bad_seeds,
            args.exclude_by_a_smoothing,
        )
        for name, read_metrics_df_at_limit in read_metrics_dfs_at_limit.items()
    }

    read_metrics_dfs_best_at_0 = {
        name: get_top_values_for_corresponding_value(
            read_metrics_df_excluded,
            "vexact/dataloader_idx_0",
            GSCAN_TEST_SPLIT_DATALOADER_NAMES,
            args.result_smoothing,
        ).describe()
        for name, read_metrics_df_excluded in read_metrics_dfs_excluded.items()
    }
    read_metrics_dfs_best_at_6 = {
        name: get_top_values_for_corresponding_value(
            read_metrics_df_excluded,
            "vexact/dataloader_idx_6",
            GSCAN_TEST_SPLIT_DATALOADER_NAMES,
            args.result_smoothing,
        ).describe()
        for name, read_metrics_df_excluded in read_metrics_dfs_excluded.items()
    }

    results_table = (
        pd.concat(
            [
                read_metrics_dfs_best_at_0["transformer_full"].T["mean"],
                read_metrics_dfs_best_at_0["gscan_oracle_full"].T["mean"],
                read_metrics_dfs_best_at_6["gscan_oracle_full"].T["mean"],
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
                    read_metrics_dfs_best_at_0["transformer_full"].T["std"],
                    read_metrics_dfs_best_at_0["gscan_oracle_full"].T["std"],
                    read_metrics_dfs_best_at_6["gscan_oracle_full"].T["std"],
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
                read_metrics_dfs_best_at_6["gscan_oracle_ablations"].T["mean"],
                read_metrics_dfs_best_at_6["gscan_oracle_noshuffle"].T["mean"],
                read_metrics_dfs_best_at_6["gscan_imagine_actions"].T["mean"],
                read_metrics_dfs_best_at_6["gscan_metalearn_distractors"].T["mean"],
                read_metrics_dfs_best_at_6["gscan_metalearn_only_random"].T["mean"],
                read_metrics_dfs_best_at_6["gscan_metalearn_sample_environments"].T[
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
                    read_metrics_dfs_best_at_6["gscan_oracle_ablations"].T["std"],
                    read_metrics_dfs_best_at_6["gscan_oracle_noshuffle"].T["std"],
                    read_metrics_dfs_best_at_6["gscan_imagine_actions"].T["std"],
                    read_metrics_dfs_best_at_6["gscan_metalearn_distractors"].T["std"],
                    read_metrics_dfs_best_at_6["gscan_metalearn_only_random"].T["std"],
                    read_metrics_dfs_best_at_6["gscan_metalearn_sample_environments"].T[
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
