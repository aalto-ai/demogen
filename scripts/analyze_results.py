import argparse
import os
import itertools
import pandas as pd
import numpy as np
import re


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


def get_top_values_for_corresponding_value(
    dfs, corresponding, values, rolling=1, descending=False
):
    select_cols = values + ([corresponding] if corresponding not in values else [])

    nonrolling_dfs = [df[select_cols].dropna() for df in dfs]

    rolling_dfs = [
        df[select_cols].rolling(rolling).mean().fillna(0) for df in nonrolling_dfs
    ]

    argwheres = [
        np.argwhere(
            rolling_df[corresponding].values
            == getattr(rolling_df[corresponding], "max" if not descending else "min")()
        )[-1][0]
        for rolling_df in rolling_dfs
    ]
    print(list(zip(argwheres, map(lambda df: df.shape[0], rolling_dfs))))

    return pd.DataFrame(
        np.stack(
            [
                nonrolling_df[values].iloc[argwhere].values
                for argwhere, nonrolling_df in zip(argwheres, nonrolling_dfs)
            ]
        ),
        columns=values,
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
            ]
            + (
                ["ml_d_limit", name_dict["ml_d_limit"]]
                if "ml_d_limit" in name_dict and name_dict["ml_d_limit"] is not None
                else []
            ),
        )
    )


def format_model_name(experiment_config, include_hparams=True):
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


def format_log_path(logs_dir, experiment_config, params, model_include_hparams=True):
    # We have to do some testing of paths here, which doesn't scale all
    # that well, but its fine for a small number of paths
    base_path = os.path.join(
        format_experiment_name(experiment_config, params),
        format_model_name(experiment_config, include_hparams=model_include_hparams),
        experiment_config["dataset"],
        str(params["seed"]),
        "lightning_logs",
    )

    if os.path.exists(os.path.join(logs_dir, base_path, "100", "metrics.csv")):
        return os.path.join(base_path, "100", "metrics.csv")

    return os.path.join(base_path, "version_100", "metrics.csv")


def read_csv_and_truncate(path, truncate_key, truncate_value):
    try:
        df = pd.read_csv(path)
        print(f"Read csv {path} - {df.shape} values")
    except Exception as e:
        print(f"Could not read csv {path} - {e}")
        return None
    truncated = truncate_at_key(df, truncate_key, truncate_value)

    return truncated


GSCAN_TEST_SPLIT_DATALOADER_NAMES = [

]

_RE_DIRECTORY_NAME = r"(?P<headline>[a-z_]+)_s_(?P<seed>[0-9])_m_(?P<model>[0-9a-z_]+)_l_(?P<layers>[0-9]+)_h_(?P<heads>[0-9]+)_d_(?P<hidden>[0-9]+)_it_(?P<iterations>[0-9]+)_b_(?P<batch_size>[0-9]+)_d_(?P<dataset>[0-9a-z_]+)_t_(?P<tag>[a-z_0-9]+)_drop_(?P<dropout>[0-9\.]+)(?:_ml_d_limit_(?P<ml_d_limit>[0-9]+))?"


def collate_func_key(x, keys):
    return tuple([x[k] for k in keys if k != "seed"])


def read_and_collate_from_directory(logs_dir, limit, exclude_seeds=[]):
    listing = os.listdir(logs_dir)
    parsed_listing = list(
        map(lambda x: re.match(_RE_DIRECTORY_NAME, x).groupdict(), listing)
    )
    keys = sorted(parsed_listing[0].keys())
    keys_not_seed = [k for k in keys if k != "seed"]
    grouped_listing_indices = [
        (
            {k: v for k, v in zip(keys_not_seed, key_tuple)},
            map(
                lambda index: (index, parsed_listing[index]["seed"]),
                list(zip(*group))[0],
            ),
        )
        for key_tuple, group in itertools.groupby(
            sorted(
                list(enumerate(parsed_listing)),
                key=lambda x: collate_func_key(x[1], keys_not_seed),
            ),
            lambda x: collate_func_key(x[1], keys_not_seed),
        )
    ]

    return [
        (
            config,
            [
                (
                    seed,
                    read_csv_and_truncate(
                        os.path.join(
                            logs_dir, format_log_path(logs_dir, config, {"seed": seed})
                        ),
                        "step",
                        limit,
                    ),
                )
                for index, seed in values
                if seed not in exclude_seeds
                and os.path.exists(
                    os.path.join(
                        logs_dir, format_log_path(logs_dir, config, {"seed": seed})
                    )
                )
            ],
        )
        for config, values in grouped_listing_indices
    ]


MATCH_CONFIGS = {
    "transformer_full": {
        "model": "vilbert_cross_encoder_decode_actions",
        "headline": "gscan",
    },
    "i2g": {"dataset": "i2g", "headline": "meta_gscan"},
    "gandr": {"dataset": "gandr", "headline": "meta_gscan"},
    "gscan_oracle_full": {"dataset": "metalearn_allow_any", "headline": "meta_gscan"},
    "gscan_metalearn_only_random": {
        "dataset": "metalearn_random_instructions_same_layout_allow_any",
        "headline": "meta_gscan",
    },
    "gscan_metalearn_sample_environments": {
        "dataset": "metalearn_find_matching_instruction_demos_allow_any",
        "headline": "meta_gscan",
    },
}


def match_to_configs(configs, configs_and_results_tuples):
    return {
        name: [
            results
            for config, results in configs_and_results_tuples
            if all([config[k] == requested_config[k] for k in requested_config.keys()])
        ][0]
        for name, requested_config in configs.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--limit", type=int, default=30000)
    parser.add_argument("--drop-bad-seeds", type=int, default=3)
    parser.add_argument("--exclude-by-a-smoothing", type=int, default=50)
    parser.add_argument("--result-smoothing", type=int, default=1)
    parser.add_argument("--exclude-seeds", nargs="*")
    args = parser.parse_args()

    read_metrics_dfs_at_limit_by_config = read_and_collate_from_directory(
        args.logs_dir, args.limit, exclude_seeds=args.exclude_seeds or []
    )

    read_metrics_dfs_excluded = [
        (
            config,
            exclude_worst_performing_by_metric(
                # Take the df from the seed/df pair
                list(zip(*read_metrics_df_at_limit_and_seeds))[1],
                "vexact/dataloader_idx_0",
                args.drop_bad_seeds,
                args.exclude_by_a_smoothing,
                descending=False,
            ),
        )
        for config, read_metrics_df_at_limit_and_seeds in read_metrics_dfs_at_limit_by_config
    ]

    read_metrics_dfs_best_at_0 = match_to_configs(
        MATCH_CONFIGS,
        [
            (
                config,
                get_top_values_for_corresponding_value(
                    read_metrics_df_excluded,
                    "vexact/dataloader_idx_0",
                    GSCAN_TEST_SPLIT_DATALOADER_NAMES,
                    args.result_smoothing,
                    descending=False,
                ).describe(),
            )
            for config, read_metrics_df_excluded in read_metrics_dfs_excluded
        ],
    )

    results_table = (
        pd.concat(
            [
                read_metrics_dfs_best_at_0["transformer_full"].T["mean"],
                read_metrics_dfs_best_at_0["i2g"].T["mean"],
                read_metrics_dfs_best_at_0["gandr"].T["mean"],
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
                    read_metrics_dfs_best_at_0["i2g"].T["std"],
                    read_metrics_dfs_best_at_0["gandr"].T["std"],
                ],
                axis=1,
            )
            .T.round(2)
            .astype(str)
            .reset_index()
        )
    ).T
    results_table.index = ["index", "A", "B", "C", "D", "E", "F", "G", "H"]
    results_table.columns = ["ViLBERT", "DemoGen", "GandR"]
    print("Table 1")
    print(results_table.to_latex(float_format="%.2f", escape=False))

    ablation_study_table = (
        pd.concat(
            [
                read_metrics_dfs_best_at_0["gscan_oracle_full"].T["mean"],
                read_metrics_dfs_best_at_0["gscan_metalearn_only_random"].T["mean"],
                read_metrics_dfs_best_at_0["gscan_metalearn_sample_environments"].T[
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
                    read_metrics_dfs_best_at_0["gscan_oracle_full"].T["std"],
                    read_metrics_dfs_best_at_0["gscan_metalearn_only_random"].T["std"],
                    read_metrics_dfs_best_at_0["gscan_metalearn_sample_environments"].T[
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
    ablation_study_table.columns = ["index", "A", "B", "C", "D", "E", "F", "G", "H"]
    ablation_study_table.index = [
        "Apriori Oracle",
        "Random",
        "Retrieval",
    ]
    ablation_study_table = ablation_study_table.drop("index", axis=1)

    print("Table 2")
    print(ablation_study_table.T.to_latex())


if __name__ == "__main__":
    main()
