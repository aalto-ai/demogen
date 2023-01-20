import numpy as np
import argparse

from gscan_metaseq2seq.util.load_data import load_data, load_data_directories
from gscan_metaseq2seq.gscan.world import (
    Situation,
    ObjectVocabulary,
    World,
    INT_TO_DIR,
    Object,
    PositionedObject,
    Position,
)
from gscan_metaseq2seq.gscan.vocabulary import Vocabulary
from gscan_metaseq2seq.util.dataset import MapDataset
from torch.utils.data import Dataset
import torch
import os
from tqdm.auto import tqdm
from scipy.stats import hmean
import itertools
import json
import multiprocessing
import pandas as pd

from typing import Optional

TYPE_GRAMMAR = "adverb"
INTRANSITIVE_VERBS = "walk"
TRANSITIVE_VERBS = "pull,push"
ADVERBS = "cautiously,while spinning,hesitantly,while zigzagging"
NOUNS = "square,cylinder,circle,box"
COLOR_ADJECTIVES = "red,green,yellow,blue"
SIZE_ADJECTIVES = "big,small"
MIN_OTHER_OBJECTS = 0
MAX_OBJECTS = 2
MIN_OBJECT_SIZE = 1
MAX_OBJECT_SIZE = 4
GRID_SIZE = 6


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
            ],
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


def read_all_csv_files_for_seeds_and_limit(logs_dir, experiment_config, limit):
    return [
        truncate_at_key(
            pd.read_csv(
                os.path.join(
                    logs_dir,
                    format_log_path(logs_dir, experiment_config, {"seed": seed}),
                )
            ),
            "step",
            limit,
        )
        for seed in range(10)
    ]


def read_csv_and_truncate(path, truncate_key, truncate_value):
    df = pd.read_csv(path)
    truncated = truncate_at_key(df, truncate_key, truncate_value)

    return truncated


GSCAN_TEST_SPLIT_DATALOADER_NAMES = [
    "vexact/dataloader_idx_0",
    "vexact/dataloader_idx_1",
    "vexact/dataloader_idx_2",
    "vexact/dataloader_idx_3",
    "vexact/dataloader_idx_5",
    "vexact/dataloader_idx_6",
    "vexact/dataloader_idx_7",
    "vexact/dataloader_idx_8",
]

_RE_DIRECTORY_NAME = r"(?P<headline>[a-z_]+)_s_(?P<seed>[0-9])_m_(?P<model>[0-9a-z_]+)_l_(?P<layers>[0-9]+)_h_(?P<heads>[0-9]+)_d_(?P<hidden>[0-9]+)_it_(?P<iterations>[0-9]+)_b_(?P<batch_size>[0-9]+)_d_(?P<dataset>[0-9a-z_]+)_t_(?P<tag>[a-z_]+)_drop_(?P<dropout>[0-9\.]+)"


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


def create_world(
    vocabulary,
    grid_size: Optional[int] = None,
    min_object_size: Optional[int] = None,
    max_object_size: Optional[int] = None,
    type_grammar: Optional[str] = None,
):

    # Object vocabulary.
    object_vocabulary = ObjectVocabulary(
        shapes=vocabulary.get_semantic_shapes(),
        colors=vocabulary.get_semantic_colors(),
        min_size=min_object_size or MIN_OBJECT_SIZE,
        max_size=max_object_size or MAX_OBJECT_SIZE,
    )

    # Initialize the world.
    return World(
        grid_size=grid_size or GRID_SIZE,
        colors=vocabulary.get_semantic_colors(),
        object_vocabulary=object_vocabulary,
        shapes=vocabulary.get_semantic_shapes(),
        save_directory=None,
    )


def reinitialize_world(
    world,
    situation: Situation,
    vocabulary,
    mission="",
    manner=None,
    verb=None,
    end_pos=None,
    required_push=0,
    required_pull=0,
    num_instructions=0,
):
    objects = []
    for positioned_object in situation.placed_objects:
        objects.append((positioned_object.object, positioned_object.position))
    world.initialize(
        objects,
        agent_position=situation.agent_pos,
        agent_direction=situation.agent_direction,
        target_object=situation.target_object,
        carrying=situation.carrying,
    )
    if mission:
        is_transitive = False
        if verb in vocabulary.get_transitive_verbs():
            is_transitive = True
        world.set_mission(
            mission,
            manner=manner,
            verb=verb,
            is_transitive=is_transitive,
            end_pos=end_pos,
            required_push=required_push,
            required_pull=required_pull,
            num_instructions=num_instructions,
        )

    return world


def segment_instruction(query_instruction, word2idx, colors, nouns):
    verb_words = [
        [word2idx[w] for w in v] for v in [["walk", "to"], ["push"], ["pull"]]
    ]
    adverb_words = [
        [word2idx[w] for w in v]
        for v in [
            ["while spinning"],
            ["while zigzagging"],
            ["hesitantly"],
            ["cautiously"],
        ]
    ]
    size_words = [word2idx[w] for w in ["small", "big"]]
    color_words = [word2idx[w] for w in list(colors)]
    noun_words = [word2idx[w] for w in list(nouns) if w in word2idx]

    query_verb_words = [
        v for v in verb_words if all([w in query_instruction for w in v])
    ]
    query_adverb_words = [
        v for v in adverb_words if all([w in query_instruction for w in v])
    ]
    query_size_words = [v for v in size_words if v in query_instruction]
    query_color_words = [v for v in color_words if v in query_instruction]
    query_noun_words = [v for v in noun_words if v in query_instruction]

    return (
        query_verb_words,
        query_adverb_words,
        query_size_words,
        query_color_words,
        query_noun_words,
    )


def find_agent_position(state):
    return [s for s in state if s[3] != 0][0]


def find_target_object(state, size, color, noun, idx2word, idx2color, idx2noun):
    color_word = [idx2word[c] for c in color]
    noun_word = [idx2word[c] for c in noun]
    size_word = [idx2word[c] for c in size]

    # Find any state elements with a matching noun, then
    # filter by matching color
    states_with_matching_noun = [
        s for s in state if s[2] and idx2noun[s[2] - 1] in noun_word
    ]
    states_with_matching_color = [
        s
        for s in states_with_matching_noun
        if s[1] and idx2color[s[1] - 1] in color_word or not color_word
    ]
    sorted_by_size = sorted(states_with_matching_color, key=lambda x: x[0])

    if not sorted_by_size:
        return None

    if size_word and size_word[0] == "small":
        return sorted_by_size[0]

    if size_word and size_word[0] == "big":
        return sorted_by_size[-1]

    return sorted_by_size[0]


def state_to_situation(query_instruction, state, word2idx, colors, nouns):
    idx2word = [w for w in word2idx if w != word2idx["[pad]"]]
    verb, adverb, size, color, noun = segment_instruction(
        query_instruction, word2idx, colors, nouns
    )
    agent = find_agent_position(state)
    target_object = find_target_object(
        state, size, color, noun, idx2word, colors, nouns
    )
    return (
        [idx2word[w] for w in query_instruction],
        Situation(
            grid_size=6,
            agent_position=Position(agent[-1], agent[-2]),
            agent_direction=INT_TO_DIR[agent[-3] - 1],
            target_object=None
            if target_object is None
            else PositionedObject(
                object=Object(
                    shape=nouns[target_object[2] - 1],
                    color=colors[target_object[1] - 1],
                    size=target_object[0],
                ),
                position=Position(target_object[-1], target_object[-2]),
                vector=[],
            ),
            placed_objects=[
                PositionedObject(
                    object=Object(
                        shape=nouns[o[2] - 1], color=colors[o[1] - 1], size=o[0]
                    ),
                    position=Position(o[-1], o[-2]),
                    vector=[],
                )
                for o in state[1:]
                if not (o == 0).all()
            ],
        ),
    )


def demonstrate_command_oracle(
    world,
    vocabulary,
    vocabulary_colors,
    vocabulary_nouns,
    command,
    target_object,
    initial_situation,
):
    """
    Demonstrate a command derivation and situation pair. Done by extracting the events from the logical form
    of the command derivation, extracting the arguments of each event. The argument of the event gets located in the
    situation of the world and the path to that target gets calculated. Based on whether the verb in the command is
    transitive or not, the agent interacts with the object.
    :param derivation:
    :param initial_situation:
    :returns
    """
    action_words = []
    article_words = []
    description_words = []
    adverb_words = []

    if not target_object:
        return []

    for w in command:
        if w in ["walk", "to", "push", "pull"]:
            action_words.append(w)

        if w in ["a"]:
            article_words.append(w)

        if w in vocabulary_colors + vocabulary_nouns + ["big", "small"]:
            description_words.append(w)

        if w in ["while spinning", "while zigzagging", "hesitantly", "cautiously"]:
            adverb_words.append(w)

    # Initialize the world based on the initial situation and the command.
    reinitialize_world(world, initial_situation, vocabulary, mission=command)

    # Our commands are quite simple
    manner = adverb_words[0] if adverb_words else ""
    world.go_to_position(
        position=target_object.position,
        manner=manner,
        primitive_command="walk",
    )

    # Then if the primitive command is push or pull, we have to move the object to the wall
    if action_words == ["pull"] or action_words == ["push"]:
        world.move_object_to_wall(action=action_words[0], manner=manner)

    # Done
    target_commands, _ = world.get_current_observations()
    return target_commands


def instruction_is_correct(
    encoded_instruction,
    encoded_state,
    encoded_targets,
    word2idx,
    action2idx,
    color_dictionary,
    noun_dictionary,
    world,
    vocab,
):
    encoded_instruction = encoded_instruction[encoded_instruction != word2idx["[pad]"]]
    encoded_targets = encoded_targets[encoded_targets != action2idx["[pad]"]]
    encoded_state = [s for s in encoded_state if not (s == 0).all()]

    instr, situation = state_to_situation(
        encoded_instruction, encoded_state, word2idx, color_dictionary, noun_dictionary
    )

    oracle_actions = [
        action2idx[a]
        for a in demonstrate_command_oracle(
            world,
            vocab,
            color_dictionary,
            noun_dictionary,
            instr,
            situation.target_object,
            situation,
        )
        + ["[eos]"]
    ]
    dataset_actions = encoded_targets.tolist()

    print(
        oracle_actions[: len(dataset_actions)],
        dataset_actions[: len(oracle_actions)],
        np.array(oracle_actions[: len(dataset_actions)])
        == np.array(dataset_actions[: len(oracle_actions)]),
        len(oracle_actions),
        len(dataset_actions),
        oracle_actions == dataset_actions,
        situation.target_object is not None,
    )

    return oracle_actions == dataset_actions, situation.target_object is not None


def compute_num_correct_and_valid(
    example, word2idx, action2idx, color_dictionary, noun_dictionary
):
    vocab = Vocabulary.initialize(
        intransitive_verbs=(INTRANSITIVE_VERBS).split(","),
        transitive_verbs=(TRANSITIVE_VERBS).split(","),
        adverbs=(ADVERBS).split(","),
        nouns=(NOUNS).split(","),
        color_adjectives=(COLOR_ADJECTIVES).split(","),
        size_adjectives=(SIZE_ADJECTIVES).split(","),
    )
    world = create_world(vocab)

    (
        query,
        target,
        state,
        support_state,
        support_query,
        support_target,
        ranking,
    ) = example
    assert instruction_is_correct(
        query, state, target, word2idx, action2idx, color_dictionary, noun_dictionary
    )[0]

    support_state = (
        [support_state] * len(support_query)
        if isinstance(support_state[0], np.ndarray)
        else support_state
    )

    num_correct_and_valid = np.array(
        [
            instruction_is_correct(
                sq,
                ss,
                st,
                word2idx,
                action2idx,
                color_dictionary,
                noun_dictionary,
                world,
                vocab,
            )
            for sq, st, ss in zip(support_query, support_target, support_state)
        ]
    )

    return num_correct_and_valid


def compute_num_correct_and_valid_star(args):
    return compute_num_correct_and_valid(*args)


def validate_all_instructions(
    demonstrations, word2idx, action2idx, color_dictionary, noun_dictionary, num_procs=8
):
    with multiprocessing.Pool(num_procs) as pool:
        yield from pool.imap_unordered(
            compute_num_correct_and_valid_star,
            map(
                lambda x: (x, word2idx, action2idx, color_dictionary, noun_dictionary),
                tqdm(demonstrations),
            ),
            chunksize=100,
        )


def count_corrects_for_split(
    demonstrations, word2idx, action2idx, color_dictionary, noun_dictionary
):
    corrects, support_counts = list(
        zip(
            *validate_all_instructions(
                demonstrations, word2idx, action2idx, color_dictionary, noun_dictionary
            )
        )
    )
    return np.array(corrects).sum() / np.array(support_counts).sum()


def compute_stats(corrects_and_valids):
    corrects_and_valids = np.concatenate([c for c in corrects_and_valids if len(c)])
    return {
        "correct": corrects_and_valids[..., 0].mean(),
        "valid": corrects_and_valids[..., 1].mean(),
        "correct_if_valid": corrects_and_valids[..., 0][
            corrects_and_valids[..., 1]
        ].mean(),
        "correct_and_valid": (
            corrects_and_valids[..., 0] & corrects_and_valids[..., 1]
        ).mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--limit-load", default=None, type=int)
    args = parser.parse_args()

    (
        (
            WORD2IDX,
            ACTION2IDX,
            color_dictionary,
            noun_dictionary,
        ),
        (meta_train_demonstrations, meta_valid_demonstrations_dict),
    ) = load_data_directories(
        args.data_directory, args.dictionary, limit_load=args.limit_load
    )

    color_dictionary = sorted(color_dictionary)
    noun_dictionary = sorted(noun_dictionary)

    corrects_by_split = {
        split: list(
            validate_all_instructions(
                examples, WORD2IDX, ACTION2IDX, color_dictionary, noun_dictionary
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
    }

    split_stats = {
        split: compute_stats(corrects_and_valids)
        for split, corrects_and_valids in corrects_by_split.items()
    }

    print(
        pd.DataFrame.from_dict(split_stats)[["a", "b", "c", "d", "e", "f", "g", "h"]]
        .T[["valid", "correct_and_valid"]]
        .to_latex(float_format="%.2f", escape=False)
    )


if __name__ == "__main__":
    main()
