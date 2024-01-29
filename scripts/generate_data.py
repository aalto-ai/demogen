import argparse
from bisect import bisect_left
import ujson as json
import itertools
import numpy as np
from typing import List, Optional, Tuple
import os
from collections import defaultdict
import pickle
import multiprocessing
import random
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler, normalize

from tqdm.auto import tqdm, trange

from gscan_metaseq2seq.gscan.world import Situation
from gscan_metaseq2seq.util.solver import (
    create_vocabulary,
    create_world,
    demonstrate_command_oracle,
)

from gscan_metaseq2seq.util.dataset import PaddingIterableDataset

DIR_TO_INT = {"west": 3, "east": 1, "north": 2, "south": 0}
INT_TO_DIR = {
    direction_int: direction for direction, direction_int in DIR_TO_INT.items()
}


def parse_command_repr(command_repr: str) -> List[str]:
    return command_repr.split(",")

def sort_indices_by_command(examples):
    command_examples = defaultdict(list)

    # Necessarily sorted, because we iterate front to back and append
    for index, example in enumerate(examples):
        command_examples[" ".join(example["command"].split(","))].append(index)

    return dict(command_examples)


def sort_indices_by_offsets(examples):
    commands_by_offsets = defaultdict(lambda: defaultdict(list))

    for index, example in enumerate(examples):
        situation = Situation.from_representation(example["situation"])
        x_offset = situation.target_object.position.column - situation.agent_pos.column
        y_offset = situation.target_object.position.row - situation.agent_pos.row

        commands_by_offsets[x_offset][y_offset].append(index)

    return commands_by_offsets


def sort_indices_by_target_positions(examples):
    commands_by_positions = defaultdict(lambda: defaultdict(list))

    for index, example in enumerate(examples):
        situation = Situation.from_representation(example["situation"])
        x = situation.target_object.position.column
        y = situation.target_object.position.row

        commands_by_positions[x][y].append(index)

    return commands_by_positions


def sort_indices_by_target_diff_and_description(examples):
    # x
    commands_by_offsets_and_descriptions = defaultdict(
        # y
        lambda: defaultdict(
            # size
            lambda: defaultdict(
                # shape
                lambda: defaultdict(
                    # color
                    lambda: defaultdict(list)
                )
            )
        )
    )

    for index, example in enumerate(examples):
        situation = Situation.from_representation(example["situation"])
        x_diff = situation.target_object.position.column - situation.agent_pos.column
        y_diff = situation.target_object.position.row - situation.agent_pos.row
        size = situation.target_object.object.size
        shape = situation.target_object.object.shape
        color = situation.target_object.object.color

        commands_by_offsets_and_descriptions[x_diff][y_diff][size][shape][color].append(
            index
        )

    return commands_by_offsets_and_descriptions


def sort_indices_by_serialized_situation(examples):
    command_examples = defaultdict(list)

    for index, example in enumerate(examples):
        situation = Situation.from_representation(example["situation"])
        command_examples[serialize_situation(situation)].append(index)

    return command_examples


ALL_ACTION_OPTIONS = [["walk", "to"], ["push"], ["pull"]]
ALL_ADVERB_OPTIONS = [["while spinning"], ["while zigzagging"], ["hesitantly"], []]


def is_prohibited_action_adverb_combo(action_words, adverb_words):
    # Split H: push while spinning
    if "pull" in action_words and "while spinning" in adverb_words:
        return "H"

    return None


def is_prohibited_description(
    agent_pos, target_object, description_words, allow_demonstration_splits=None
):
    allow_demonstration_splits = allow_demonstration_splits or []

    # Split B: "yellow square". We cannot have an example of this
    if (
        "yellow" in description_words
        and "square" in description_words
        and "B" not in allow_demonstration_splits
    ):
        return "B"

    # Split C: red square as target. We cannot have an example of this
    if (
        target_object.object.color == "red"
        and target_object.object.shape == "square"
        and "C" not in allow_demonstration_splits
    ):
        return "C"

    # Split D: object is to the southwest of the agent
    if (
        agent_pos.row < target_object.position.row
        and agent_pos.column > target_object.position.column
        and "D" not in allow_demonstration_splits
    ):
        return "D"

    # Split E: circle of size 2 is the target and "small" in the instruction
    if (
        "small" in description_words
        and target_object.object.size == 2
        and target_object.object.shape == "circle"
        and "E" not in allow_demonstration_splits
    ):
        return "E"

    # Split F: pushing a square of size 3
    if (
        "push" in description_words
        and target_object.object.size == 3
        and target_object.object.shape == "square"
        and "F" not in allow_demonstration_splits
    ):
        return "F"

    return None


def generate_description_words_options(situation, description_words):
    """Generate targets and description words from base description words.

    In some cases description_words will contain something that we can't
    generate, because it would be leaking the target object. So we have to
    use another object in its place.
    """
    object_types_max_sizes = defaultdict(lambda: defaultdict(int))
    object_types_min_sizes = defaultdict(lambda: defaultdict(lambda: 10))
    for positioned_object in situation.placed_objects:
        object_types_max_sizes[positioned_object.object.shape][
            positioned_object.object.color
        ] = max(
            object_types_max_sizes[positioned_object.object.shape][
                positioned_object.object.color
            ],
            positioned_object.object.size,
        )
        object_types_min_sizes[positioned_object.object.shape][
            positioned_object.object.color
        ] = min(
            object_types_min_sizes[positioned_object.object.shape][
                positioned_object.object.color
            ],
            positioned_object.object.size,
        )

    options = [(description_words, situation.target_object)] + [
        (
            (
                (["big"])
                if positioned_object.object.size
                == object_types_max_sizes[positioned_object.object.shape][
                    positioned_object.object.color
                ]
                else (
                    (["small"])
                    if positioned_object.object.size
                    == object_types_min_sizes[positioned_object.object.shape][
                        positioned_object.object.color
                    ]
                    else []
                )
            )
            + [positioned_object.object.color, positioned_object.object.shape],
            positioned_object,
        )
        for positioned_object in situation.placed_objects
        if positioned_object.position != situation.target_object.position
    ]

    return options


def sort_description_words_by_query_match(
    description_words_options,
    query_description_words,
    vocabulary_colors,
    vocabulary_nouns,
):
    size_words = ("big", "small")

    description_words_options_colors = [
        [w for w in dw[0] if w in vocabulary_colors] for dw in description_words_options
    ]
    description_words_options_objects = [
        [w for w in dw[0] if w in vocabulary_nouns] for dw in description_words_options
    ]
    description_words_options_sizes = [
        [w for w in dw[0] if w in size_words] for dw in description_words_options
    ]

    option_indices = list(range(len(description_words_options)))
    option_indices_set = set(option_indices)

    matches_color_word = [
        any([c in query_description_words for c in dwc])
        for dwc in description_words_options_colors
    ]
    matches_color_word_indices = sorted(
        option_indices, key=lambda i: matches_color_word[i], reverse=True
    )
    matches_object_word = [
        any([c in query_description_words for c in dwc])
        for dwc in description_words_options_objects
    ]
    matches_object_word_indices = sorted(
        option_indices, key=lambda i: matches_object_word[i], reverse=True
    )
    matches_size_word = [
        any([c in query_description_words for c in dwc])
        for dwc in description_words_options_sizes
    ]
    matches_size_word_indices = sorted(
        option_indices, key=lambda i: matches_size_word[i], reverse=True
    )

    indices_indexes = {"color": 0, "object": 0, "size": 0}
    indices_indexes_array = {
        "color": matches_color_word_indices,
        "object": matches_object_word_indices,
        "size": matches_size_word_indices,
    }
    option_matches = {
        "color": matches_color_word,
        "object": matches_object_word,
        "size": matches_size_word,
    }
    next_mode = {"color": "object", "object": "size", "size": "color"}
    mode = "color"

    while any(
        [
            index_value < len(index_array)
            for index_value, index_array in zip(
                indices_indexes.values(), indices_indexes_array.values()
            )
        ]
    ):
        # If we run out of indices for this mode, we have to go to the next one
        if indices_indexes[mode] >= len(indices_indexes_array[mode]):
            mode = next_mode[mode]
            continue

        # Try to take the next thing in the current mode
        possible_option = indices_indexes_array[mode][indices_indexes[mode]]

        # If we don't have this in the set anymore because it was already taken,
        # or it doesn't match the attribute, then we skip it without removing it
        # from the set
        if (
            possible_option not in option_indices_set
            or not option_matches[mode][possible_option]
        ):
            indices_indexes[mode] += 1
            continue

        # We got a match! We should yield this one, remove it from the
        # set of possible options and then advance the index
        yield description_words_options[possible_option]

        option_indices_set.remove(possible_option)
        indices_indexes[mode] += 1
        mode = next_mode[mode]


def generate_words_options(words, possible_options, limit_to_words):
    if not limit_to_words:
        return possible_options

    return [o for o in possible_options if all([w in words for w in o])]


def generate_limited_adverb_verb_combos(
    possible_verbs,
    possible_adverbs,
    verb_words_in_instruction,
    adverb_words_in_instruction,
):
    for possible_verb, possible_adverb in itertools.product(
        possible_verbs, possible_adverbs
    ):
        verb_is_in_instruction = all(
            [w in verb_words_in_instruction for w in possible_verb]
        )
        adverb_is_in_instruction = (
            possible_adverb
            and all([w in adverb_words_in_instruction for w in possible_adverb])
        ) or (not adverb_words_in_instruction and not possible_adverb)

        # If the adverb is in the instruction, we can generate all possible verbs
        if adverb_is_in_instruction:
            yield (possible_verb, possible_adverb)
            continue

        # If the verb is in the instruction, we can generate all possible adverbs
        if verb_is_in_instruction:
            yield (possible_verb, possible_adverb)
            continue


def generate_relevant_instructions_gscan_oracle(
    query_instruction,
    situation,
    vocabulary_colors,
    vocabulary_nouns,
    n_description_options=None,
    demonstrate_target=True,
    allow_demonstration_splits=None,
    allow_any_example=False,
    num_demos=16,
    pick_random=False,
    limit_verb_adverb=False,
):
    action_words = []
    article_words = []
    description_words = []
    adverb_words = []

    support_instructions = []

    # We generate "cautiously" only if "cautiously" appears in the query instrunction
    real_adverb_options = ALL_ADVERB_OPTIONS
    if "cautiously" in query_instruction:
        real_adverb_options = ALL_ADVERB_OPTIONS + [["cautiously"]]

    vocabulary_descriptors = vocabulary_colors + vocabulary_nouns + ["big", "small"]
    vocabulary_verbs = ["walk", "to", "push", "pull"]
    vocabulary_adverbs = [
        "while spinning",
        "while zigzagging",
        "hesitantly",
        "cautiously",
    ]

    for w in query_instruction:
        if w in vocabulary_verbs:
            action_words.append(w)

        if w in ["a"]:
            article_words.append(w)

        if w in vocabulary_descriptors:
            description_words.append(w)

        if w in vocabulary_adverbs:
            adverb_words.append(w)

    description_words_options = generate_description_words_options(
        situation, description_words
    )
    adverb_verb_combos = list(
        itertools.product(ALL_ACTION_OPTIONS, real_adverb_options)
        if not limit_verb_adverb
        else generate_limited_adverb_verb_combos(
            ALL_ACTION_OPTIONS, real_adverb_options, action_words, adverb_words
        )
    )

    # Split into the actual target and other possible targets
    target_description_words, other_description_words = (
        description_words_options[:1],
        description_words_options[1:],
    )

    # We sort the other possible targets by similarity to the target itself
    # so eg you get one point for every word that overlaps. The order is
    # descending
    #
    # XXX: We need to think about how to sort this, eg, we want to sort it
    # so that we get 3 matching words first, then 2 matching words
    # and so on. Also we want to sort so that we get a different matching
    # thing every time, so first we get a matching color, then a matching
    # object, then a matching size and so on.
    sorted_other_description_words = list(
        sort_description_words_by_query_match(
            other_description_words,
            target_description_words[0][0],
            vocabulary_colors=vocabulary_colors,
            vocabulary_nouns=vocabulary_nouns,
        )
    )

    # Filter out anything that is not allowed to be a target according to the
    # gSCAN rules in the sorted_other_description_words. Its important that
    # we do this before the first check for target_description_words.
    #
    # Note that in the context, we allow demonstrations of "prohibited"
    # descriptions if they appear in splits other than the one that we are
    # currently testing. So in the training set, we don't allow any examples
    # of things in the other splits (since we can just exclude those data points)
    # but when testing split B, we allow an example from split D. Otherwise
    # there's a risk that there would just be no supports and we would
    # have to exclude the entire data point.
    filtered_sorted_other_description_words = (
        sorted_other_description_words
        if allow_any_example
        else list(
            filter(
                lambda description_words: not is_prohibited_description(
                    situation.agent_pos,
                    description_words[1],
                    description_words[0],
                    allow_demonstration_splits=allow_demonstration_splits,
                ),
                sorted_other_description_words,
            )
        )
    )

    # Reassign to emptylist if we cannot take this target. The
    # net effect is that we take the "next best" target that we are
    # allowed to take.
    #
    # Exception: The target is the only "permitted" target in the whole environment
    # - in this case we allow a demonstration (such cases are not very
    # interesting anyway, since there are no distractors, you just have to
    # identify any non-empty cell)
    if filtered_sorted_other_description_words:
        if not demonstrate_target:
            target_description_words = []
        elif not allow_any_example and is_prohibited_description(
            situation.agent_pos,
            target_description_words[0][1],
            target_description_words[0][0],
        ):
            target_description_words = []

    # We have these options. Then we take then in accordance with n_description_options.
    # n_description_options == 1 basically means only show the target or the next
    # best descriptor.
    description_words_options = (
        target_description_words + filtered_sorted_other_description_words
    )
    description_words_options = description_words_options[:n_description_options]

    # Start generating action/adverb/target combos.
    #
    # We start first with the description/target loop,
    # so that the target object gets generated first and all of its
    # adverb/actions get priority
    for description_words, target_object in description_words_options:
        for action_option, adverb_option in adverb_verb_combos:
            # We might be prohibited on the basis of the chosen action/adverb combination
            # so check that again here
            if (
                is_prohibited_action_adverb_combo(action_option, adverb_option)
                and not allow_any_example
            ):
                continue

            proposed_support_instruction = (
                action_option + article_words + description_words + adverb_option,
                target_object,
            )

            # Don't generate any instruction which is exactly the same as the query instruction
            if proposed_support_instruction[0] == query_instruction:
                continue

            support_instructions.append(proposed_support_instruction)

    # We can skip this data point if we cannot make any
    # demonstrations for it because none are allowed
    if not allow_demonstration_splits and not support_instructions:
        return []

    assert len(support_instructions) > 0

    # Now we downsample if necessary
    if pick_random:
        return [
            support_instructions[i]
            for i in np.random.permutation(len(support_instructions))[:num_demos]
        ]

    # Ordered by priority
    return support_instructions[:num_demos]


def labelled_situation_to_demonstration_tuple(
    labelled_situation, input_word2idx, action_word2idx
):
    return (
        np.array([input_word2idx[w] for w in labelled_situation["input"]]),
        np.array([action_word2idx[w] for w in labelled_situation["target"]]),
    )


def parse_sparse_situation(
    situation_representation: dict,
    grid_size: int,
    color2idx,
    noun2idx,
    world_encoding_scheme,
    reascan_boxes,
) -> np.ndarray:
    """
    Each grid cell in a situation is fully specified by a vector:
    [_ _ _ _ _ _ _   _       _      _       _   _ _ _ _]
     1 2 3 4 r g b circle square cylinder agent E S W N
     _______ _____ ______________________ _____ _______
       size  color        shape           agent agent dir.
    :param situation_representation: data from dataset.txt at key "situation".
    :param grid_size: int determining row/column number.
    :return: grid to be parsed by computational models.
    """
    # Place the agent.
    agent_row = int(situation_representation["agent_position"].row)
    agent_column = int(situation_representation["agent_position"].column)
    agent_direction = DIR_TO_INT[situation_representation["agent_direction"].name]

    grid = None

    if world_encoding_scheme == "sequence":
        # attribute bits + agent + agent direction + location
        num_grid_channels = 7 + (3 if reascan_boxes else 0)
        grid = []
        agent_representation = np.zeros([num_grid_channels], dtype=np.int32)
        agent_representation[3] = 1
        agent_representation[4] = agent_direction
        agent_representation[5] = agent_row
        agent_representation[6] = agent_column
        grid.append(agent_representation)

        for placed_object in situation_representation["objects"]:
            object_row = int(placed_object.position.row)
            object_column = int(placed_object.position.column)
            if placed_object.object.shape != "box":
                grid.append(
                    np.array(
                        [
                            int(placed_object.object.size),
                            int(color2idx[placed_object.object.color]),
                            int(noun2idx[placed_object.object.shape]),
                            0,
                            0,
                            object_row,
                            object_column,
                        ]
                        + ([0] * 3 if reascan_boxes else [])
                    )
                )
            elif reascan_boxes:
                grid.append(
                    np.array(
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                            object_row,
                            object_column,
                            int(placed_object.object.size),
                            int(color2idx[placed_object.object.color]),
                            1,
                        ]
                    )
                )

    elif world_encoding_scheme == "all":
        # attribute bits + agent + agent direction
        num_grid_channels = 5 + (3 if reascan_boxes else 0)
        grid = np.zeros([grid_size, grid_size, num_grid_channels], dtype=int)
        agent_representation = np.zeros([num_grid_channels], dtype=np.int32)
        agent_representation[-2] = 1
        agent_representation[-1] = agent_direction

        grid[agent_row, agent_column, :] = agent_representation

        # Loop over the objects in the world and place them.
        for placed_object in situation_representation["objects"]:
            object_row = int(placed_object.position.row)
            object_column = int(placed_object.position.column)

            if placed_object.object.shape != "box":
                grid[object_row, object_column, 0] = int(placed_object.object.size)
                grid[object_row, object_column, 1] = int(
                    color2idx[placed_object.object.color]
                )
                grid[object_row, object_column, 2] = int(
                    noun2idx[placed_object.object.shape]
                )

            if reascan_boxes and placed_object.object.shape == "box":
                grid[object_row, object_column, 5] = int(placed_object.object.size)
                grid[object_row, object_column, 6] = int(
                    color2idx[placed_object.object.color]
                )
                grid[object_row, object_column, 7] = 1

        grid = add_positional_information_to_grid(grid)

    return grid


def add_positional_information_to_grid(grid):
    grid_pos = np.concatenate(
        [
            grid,
            np.ones_like(grid[..., 0]).cumsum(axis=0)[..., None],
            np.ones_like(grid[..., 0]).cumsum(axis=1)[..., None],
        ],
        axis=-1,
    )
    grid_pos = grid_pos.reshape(-1, grid_pos.shape[-1])

    return grid_pos


def add_positional_information_to_observation(observations):
    observations_pos = np.concatenate(
        [
            observations[0],
            np.ones_like(observations[0][..., 0]).cumsum(axis=0)[..., None],
            np.ones_like(observations[0][..., 0]).cumsum(axis=1)[..., None],
        ],
        axis=-1,
    )
    observations_pos = observations_pos.reshape(-1, observations_pos.shape[-1])

    return observations_pos


def add_eos_to_actions(actions, eos_token_idx):
    actions_pos = np.concatenate(
        [actions, np.ones_like(actions[:1]) * eos_token_idx], axis=-1
    )

    return actions_pos


def generate_relevant_supports_oracle(
    index, command, target_commands, situation, world, vocabulary, payload, options
):
    (colors, nouns, allow_demonstration_splits) = payload

    n_description_options = options.get("n_description_options", None)
    demonstrate_target = options.get("demonstrate_target", True)
    allow_any_example = options.get("allow_any_example", False)
    num_demos = options.get("num_demos", 16)
    pick_random = options.get("pick_random", False)
    limit_verb_adverb = options.get("limit_verb_adverb", False)

    support_instructions = []
    support_targets = []
    support_layouts = []

    relevant_instructions = generate_relevant_instructions_gscan_oracle(
        command,
        situation,
        colors,
        nouns,
        n_description_options=n_description_options,
        demonstrate_target=demonstrate_target,
        allow_demonstration_splits=allow_demonstration_splits,
        allow_any_example=allow_any_example,
        num_demos=num_demos,
        pick_random=pick_random,
        limit_verb_adverb=limit_verb_adverb,
    )

    if not relevant_instructions:
        return (
            f"Skipping for {command} {situation.target_object} / {situation.placed_objects} as no demonstrations are possible and it is training or Split A test data.\n",
            (None, None, None),
        )

    for support_instruction_command, target_object in relevant_instructions:
        # Demonstrate the command using the oracle
        support_target_commands = demonstrate_command_oracle(
            world,
            vocabulary,
            colors,
            nouns,
            support_instruction_command,
            target_object,
            situation,
        )
        support_instructions.append(support_instruction_command)
        support_targets.append(support_target_commands)

    return (None, (support_instructions, support_targets, support_layouts))


def generate_instructions_find_support_in_any_layout(
    index, command, target_commands, situation, world, vocabulary, payload, options
):
    (
        sorted_example_indices_by_command,
        train_examples,
        colors,
        nouns,
        allow_demonstration_splits,
    ) = payload
    n_description_options = options.get("n_description_options", None)
    demonstrate_target = options.get("demonstrate_target", True)
    limit_verb_adverb = options.get("limit_verb_adverb", False)

    support_instructions = []
    support_targets = []
    support_layouts = []

    relevant_instructions = generate_relevant_instructions_gscan_oracle(
        command,
        situation,
        colors,
        nouns,
        n_description_options=n_description_options,
        demonstrate_target=demonstrate_target,
        allow_demonstration_splits=allow_demonstration_splits,
        limit_verb_adverb=limit_verb_adverb,
    )

    for support_instruction_command, target_object in relevant_instructions:
        key = " ".join(support_instruction_command)

        # If its not there, we just don't add it. This might result
        # in some of the supports being empty, but that's fine.
        if key not in sorted_example_indices_by_command:
            continue

        relevant_example_idx = np.random.choice(sorted_example_indices_by_command[key])
        relevant_example = train_examples[relevant_example_idx]
        relevant_situation = Situation.from_representation(
            relevant_example["situation"]
        )
        relevant_target_commands = parse_command_repr(
            relevant_example["target_commands"]
        )
        support_layouts.append(relevant_situation.to_dict())
        support_instructions.append(support_instruction_command)
        support_targets.append(relevant_target_commands)

    return (None, (support_instructions, support_targets, support_layouts))


def encode_situation_as_onehot(situation, color2idx, noun2idx):
    return (
        parse_sparse_situation(
            situation.to_dict(),
            6,
            color2idx,
            noun2idx,
            "all",
            False
        )[:, :-2, None] == np.arange(5, dtype=np.int32)[None, None]
    ).reshape(36, -1).astype(np.int32)


def stack_as_passed(numpy_arrays):
    max_len = max([len(a) for a in numpy_arrays])
    return np.stack([
        np.concatenate([
            a, np.zeros_like(a)[:1].repeat(max_len - len(a), axis=0)
        ], axis=0)
        for a in numpy_arrays
    ])


def compute_sorted_bsr(
    query_embeds,
    demonstration_embeds,
    target_limit
):
    indices = []
    coverage_value = float('-inf')

    examples_list = []
    coverage_bits_list = []
    lengths_list = []

    demonstration_embeds_pad = stack_as_passed(demonstration_embeds)
    current_matching = query_embeds[None] @ demonstration_embeds_pad.transpose(0, 2, 1)
    current_matching_max_scores = (current_matching.max(axis=-1) > 0.8)

    # Pick examples that gets us coverage for every query token
    # but also maximizes the total weight.
    #
    # We can do this with an iterative algorithm that keeps track of
    # each token's coverage. We remove the "lightest" example from
    # the set on each iteration as long as it wouldn't break the overall
    # coverage
    current_overall_token_coverages = current_matching_max_scores.sum(axis=0)
    token_coverage_weights = current_matching_max_scores.sum(axis=-1)
    examples_in_set = np.arange(len(demonstration_embeds))
    examples_in_set_sorted_by_weights = examples_in_set[token_coverage_weights.argsort()]
    zero_mask = current_overall_token_coverages == 0

    while examples_in_set_sorted_by_weights.shape[0] > target_limit:
        removed = False

        for i in range(examples_in_set_sorted_by_weights.shape[0]):
            # Check if removing this example would make anything newly-zero in the coverage
            index = examples_in_set_sorted_by_weights[i]
            scores = current_matching_max_scores[index].astype(int)
            if (
                ((current_overall_token_coverages - scores) == 0) ^
                zero_mask
            ).sum() > 0:
                continue

            # Otherwise, we can remove this example
            examples_in_set_sorted_by_weights = examples_in_set_sorted_by_weights[1:]
            current_overall_token_coverages = current_overall_token_coverages - scores
            removed = True
            break

        # We could not remove anything without reducing coverage. Lets return the whole set
        if not removed:
            break

    examples_in_set_sorted_by_weights = list(reversed(examples_in_set_sorted_by_weights))

    # Finally, remove any coverage-dupes (eg, same coverage
    # as the one before)
    exact_dupe_indices = [
        i for i, first, second in zip(
            range(1, len(examples_in_set_sorted_by_weights)),
            examples_in_set_sorted_by_weights[:-1],
            examples_in_set_sorted_by_weights[1:]
        )
        if (current_matching_max_scores[first] == current_matching_max_scores[second]).all()
    ]

    examples_in_set_sorted_by_weights = [
        examples_in_set_sorted_by_weights[i]
        for i in range(len(examples_in_set_sorted_by_weights))
        if i not in exact_dupe_indices
    ]

    return examples_in_set_sorted_by_weights


def compute_sorted_set_bsr(
    query_embeds,
    demonstration_embeds
):
    indices = []
    coverage_value = float('-inf')

    examples_list = []
    coverage_bits_list = []
    lengths_list = []

    for i, demo_embed in enumerate(demonstration_embeds):
        current_matching = query_embeds @ demonstration_embeds[i].T
        current_matching_max_scores = (current_matching.max(axis=-1) > 0.8)
        mask = np.zeros_like(current_matching_max_scores)

        index = 0
        
        for j in range(len(coverage_bits_list)):
            index = j
            # Check if it covers more than what's in the current list index
            disjunction = np.logical_xor(
                current_matching_max_scores,
                coverage_bits_list[j]
            ) & ~mask
            unique_current = np.logical_and(disjunction, current_matching_max_scores)
            unique_in_list = np.logical_and(disjunction, coverage_bits_list[j])

            if unique_current.sum() > coverage_bits_list[j].sum():
                break

            # If this example would give the same coverage, check
            # to see if it is shorter. If so, we should prefer it over
            # the one currently in the set, on the intuition that shorter
            # examples probably contain less distracting information
            if unique_current.sum() == coverage_bits_list[j].sum():
                if demonstration_embeds[i].shape[0] < lengths_list[j]:
                    break

            mask |= unique_in_list

        j = index

        # Would this add anything new? If not, not worth it
        if np.logical_and(~mask, current_matching_max_scores).any():
            # Insert the current matching in the list
            examples_list.insert(j, i)
            lengths_list.insert(j, demonstration_embeds[i].shape[0])
            coverage_bits_list.insert(j, current_matching_max_scores)

            try:
                coverage_list_cumsum = np.concatenate([
                    np.zeros_like(current_matching_max_scores)[None],
                    coverage_bits_list
                ]).cumsum(axis=0).astype(bool)
                coverage_list_stack = np.concatenate([
                    coverage_bits_list,
                    np.zeros_like(current_matching_max_scores)[None],
                ], axis=0)
                coverage_list_land = np.logical_and(
                    ~coverage_list_cumsum,
                    coverage_list_stack
                ).sum(axis=-1)[:-1]
            except:
                import pdb
                pdb.set_trace()
            
            len_examples_list = len(examples_list)
            
            examples_list = [
                examples_list[j]
                for j in range(len_examples_list)
                if coverage_list_land[j]
            ]
            lengths_list = [
                lengths_list[j]
                for j in range(len_examples_list)
                if coverage_list_land[j]
            ]
            coverage_bits_list = [
                coverage_bits_list[j]
                for j in range(len_examples_list)
                if coverage_list_land[j]
            ]
            
            assert len(examples_list) == len(coverage_bits_list)

    return examples_list


def retrieve_layout_instruction_coverage(
    index, command, target_commands, situation, world, vocabulary, payload, options
):
    # Similar to Gupta et al 2023, we walk down the list and try to greedily
    # find things that would maximize coverage of the query according to the
    # "bert-score"
    (
        retrievals,
        train_examples,
        word2idx,
        train_unique_indices,
        train_unique_encodings,
        train_unique_token_encodings,
        split_sentences_to_unique_index,
        split_sentences_unique_all_token_encodings
    ) = payload

    token_embeddings = split_sentences_unique_all_token_encodings[split_sentences_to_unique_index[index]]

    retrievals_from_training_set = retrievals[index]
    retrievals_from_training_set = retrievals_from_training_set[
        np.array([
            command != train_examples[i]["command"].split(",")
            for i in retrievals_from_training_set
        ])
    ]

    retrieved_token_encodings = [
        train_unique_token_encodings[train_unique_indices[i]]
        for i in retrievals_from_training_set
    ]
    sorted_examples_by_set_coverage = np.array(compute_sorted_bsr(token_embeddings, retrieved_token_encodings, 16))
    retrievals_from_training_set[sorted_examples_by_set_coverage]

    selected_examples = retrievals_from_training_set[:16]

    return (
        None,
        tuple(list(zip(*[
            (
                parse_command_repr(train_examples[i]["command"]),
                parse_command_repr(train_examples[i]["target_commands"]),
                Situation.from_representation(
                    train_examples[i]["situation"]
                ).to_dict(),
            )
            for i in selected_examples
        ])))
    )


def generate_random_instructions_find_support_in_any_layout(
    index, command, target_commands, situation, world, vocabulary, payload, options
):
    (
        sorted_example_indices_by_command,
        train_examples,
        nouns,
        colors,
        allow_demonstration_splits,
    ) = payload
    num_demos = options.get("num_demos", 16)

    support_instructions = []
    support_targets = []
    support_layouts = []

    command_key = " ".join(command)
    random_instructions = np.random.choice(
        list(set(sorted_example_indices_by_command.keys()) - set([command_key])),
        size=num_demos,
        replace=False,
    )

    for support_instruction_command in random_instructions:
        key = " ".join(support_instruction_command)

        # If its not there, we just don't add it. This might result
        # in some of the supports being empty, but that's fine.
        if key not in sorted_example_indices_by_command:
            continue

        relevant_example_idx = np.random.choice(sorted_example_indices_by_command[key])
        relevant_example = train_examples[relevant_example_idx]
        relevant_situation = Situation.from_representation(
            relevant_example["situation"]
        )
        relevant_target_commands = parse_command_repr(
            relevant_example["target_commands"]
        )
        support_layouts.append(relevant_situation.to_dict())
        support_instructions.append(support_instruction_command)
        support_targets.append(relevant_target_commands)

    return (None, (support_instructions, support_targets, support_layouts))


def find_in_sorted_list(a, x):
    "Locate the leftmost value exactly equal to x"
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return True
    return False


def find_supports_with_same_agent_target_offset(
    index, command, target_commands, situation, world, vocabulary, payload, options
):
    (
        sorted_example_indices_by_x_y_distance_to_agent,
        sorted_example_indices_by_command,
        train_examples,
    ) = payload
    num_demos = options.get("num_demos", 16)

    support_instructions = []
    support_targets = []
    support_layouts = []

    agent_pos = situation.agent_pos
    target_object = situation.target_object

    x_diff = target_object.position.column - agent_pos.column
    y_diff = target_object.position.row - agent_pos.row

    command_key = " ".join(command)
    possible_demos = sorted_example_indices_by_x_y_distance_to_agent[x_diff][y_diff]
    possible_demos = [
        p
        for p in possible_demos
        if not find_in_sorted_list(sorted_example_indices_by_command[command_key], p)
    ]
    relevant_demos = np.random.choice(
        possible_demos, size=min(len(possible_demos), num_demos), replace=True
    )

    for example_idx in relevant_demos:
        relevant_example = train_examples[example_idx]
        support_instruction_command = parse_command_repr(relevant_example["command"])
        relevant_situation = Situation.from_representation(
            relevant_example["situation"]
        )
        relevant_target_commands = parse_command_repr(
            relevant_example["target_commands"]
        )
        support_layouts.append(relevant_situation.to_dict())
        support_instructions.append(support_instruction_command)
        support_targets.append(relevant_target_commands)

    return (None, (support_instructions, support_targets, support_layouts))


def find_supports_with_any_target_object_in_same_position(
    index, command, target_commands, situation, world, vocabulary, payload, options
):
    (
        sorted_example_indices_by_target_x_y,
        sorted_example_indices_by_command,
        train_examples,
    ) = payload
    num_demos = options.get("num_demos", 16)

    support_instructions = []
    support_targets = []
    support_layouts = []

    target_object = situation.target_object

    x_pos = target_object.position.column
    y_pos = target_object.position.row

    command_key = " ".join(command)
    possible_demos = sorted_example_indices_by_target_x_y[x_pos][y_pos]
    possible_demos = [
        p
        for p in possible_demos
        if not find_in_sorted_list(sorted_example_indices_by_command[command_key], p)
    ]

    relevant_demos = np.random.choice(
        possible_demos, size=min(len(possible_demos), num_demos), replace=True
    )

    for example_idx in relevant_demos:
        relevant_example = train_examples[example_idx]
        support_instruction_command = parse_command_repr(relevant_example["command"])
        relevant_situation = Situation.from_representation(
            relevant_example["situation"]
        )
        relevant_target_commands = parse_command_repr(
            relevant_example["target_commands"]
        )
        support_layouts.append(relevant_situation.to_dict())
        support_instructions.append(support_instruction_command)
        support_targets.append(relevant_target_commands)

    return (None, (support_instructions, support_targets, support_layouts))


def find_supports_by_matching_object_in_same_diff(
    index, command, target_commands, situation, world, vocabulary, payload, options
):
    (
        example_indices_by_target_x_y_diff_object,
        sorted_example_indices_by_command,
        train_examples,
    ) = payload
    num_demos = options.get("num_demos", 16)

    support_instructions = []
    support_targets = []
    support_layouts = []

    agent_pos = situation.agent_pos
    target_object = situation.target_object
    target_object_size = target_object.object.size
    target_object_shape = target_object.object.shape
    target_object_color = target_object.object.color

    x_diff = target_object.position.column - agent_pos.column
    y_diff = target_object.position.row - agent_pos.row

    command_key = " ".join(command)
    possible_demos = example_indices_by_target_x_y_diff_object[x_diff][y_diff][
        target_object_size
    ][target_object_shape][target_object_color]
    possible_demos = [
        p
        for p in possible_demos
        if not find_in_sorted_list(sorted_example_indices_by_command[command_key], p)
    ]
    relevant_demos = np.random.choice(
        possible_demos, size=min(len(possible_demos), num_demos), replace=False
    )

    for example_idx in relevant_demos:
        relevant_example = train_examples[example_idx]
        support_instruction_command = parse_command_repr(relevant_example["command"])
        relevant_situation = Situation.from_representation(
            relevant_example["situation"]
        )
        relevant_target_commands = parse_command_repr(
            relevant_example["target_commands"]
        )
        support_layouts.append(relevant_situation.to_dict())
        support_instructions.append(support_instruction_command)
        support_targets.append(relevant_target_commands)

    return (None, (support_instructions, support_targets, support_layouts))


def serialize_situation(situation):
    return "_".join(
        [
            f"agent_{situation.agent_pos.column}_{situation.agent_pos.row}"
            f"target_{''.join(map(str, situation.target_object.vector))}_{situation.target_object.position.row}_{situation.target_object.position.column}"
        ]
        + [
            f"object_{''.join(map(str, o.vector))}_{o.position.row}_{o.position.column}"
            for o in sorted(
                situation.placed_objects,
                key=lambda x: (x.position.row, x.position.column),
            )
        ]
    )


def find_supports_by_matching_environment_layout(
    index, command, target_commands, situation, world, vocabulary, payload, options
):
    (
        example_indices_by_matching_environment,
        sorted_example_indices_by_command,
        train_examples,
    ) = payload
    num_demos = options.get("num_demos", 16)

    support_instructions = []
    support_targets = []
    support_layouts = []

    environment_string = serialize_situation(situation)

    command_key = " ".join(command)
    possible_demos = example_indices_by_matching_environment[environment_string]
    possible_demos = [
        p
        for p in possible_demos
        if not find_in_sorted_list(sorted_example_indices_by_command[command_key], p)
    ]
    relevant_demos = np.random.choice(
        possible_demos, size=min(len(possible_demos), num_demos), replace=True
    )

    for example_idx in relevant_demos:
        relevant_example = train_examples[example_idx]
        support_instruction_command = parse_command_repr(relevant_example["command"])
        relevant_situation = Situation.from_representation(
            relevant_example["situation"]
        )
        relevant_target_commands = parse_command_repr(
            relevant_example["target_commands"]
        )
        support_layouts.append(relevant_situation.to_dict())
        support_instructions.append(support_instruction_command)
        support_targets.append(relevant_target_commands)

    return (None, (support_instructions, support_targets, support_layouts))


GENERATION_STRATEGIES = {
    "generate_oracle": generate_relevant_supports_oracle,
    "generate_find_matching": generate_instructions_find_support_in_any_layout,
    "retrieve_similar_state": retrieve_layout_instruction_coverage,
    "random_find_matching": generate_random_instructions_find_support_in_any_layout,
    "find_by_environment_layout": find_supports_by_matching_environment_layout,
    "find_by_matching_same_object_in_same_diff": find_supports_by_matching_object_in_same_diff,
    "find_by_matching_any_object_in_same_target_position": find_supports_with_any_target_object_in_same_position,
    "find_by_matching_any_object_in_same_diff": find_supports_with_same_agent_target_offset,
}


def generate_supports_for_data_point(
    data_example,
    index,
    world,
    vocabulary,
    generation_mode,
    generation_payload,
    generation_options,
):
    command = parse_command_repr(data_example["command"])
    target_commands = parse_command_repr(data_example["target_commands"])
    situation = Situation.from_representation(data_example["situation"])

    error, (
        support_instruction_commands,
        support_target_commands,
        support_layouts,
    ) = GENERATION_STRATEGIES[generation_mode](
        index,
        command,
        target_commands,
        situation,
        world,
        vocabulary,
        generation_payload,
        generation_options,
    )

    return (
        error,
        (
            command,
            target_commands,
            data_example["situation"],
            support_instruction_commands,
            support_target_commands,
            support_layouts,
        ),
    )


def generate_supports_for_data_point_star(args):
    return generate_supports_for_data_point(*args)


def yield_metalearning_examples(
    examples_set,
    world,
    vocabulary,
    generation_mode,
    generation_payload,
    generation_options,
    n_procs=8,
):
    if generation_options.get("can_parallel", False):
        # Fast path for generation, we don't need to keep the whole dataset
        # in memory in order to search for matching examples
        with multiprocessing.Pool(processes=n_procs) as pool:
            for error, result in pool.imap_unordered(
                generate_supports_for_data_point_star,
                map(
                    lambda x, i: (
                        x,
                        i,
                        world,
                        vocabulary,
                        generation_mode,
                        generation_payload,
                        generation_options,
                    ),
                    enumerate(examples_set),
                ),
                chunksize=100,
            ):
                if error is not None:
                    tqdm.write(error)
                    continue

                yield result
    else:
        # Slow path, we need to keep the whole dataset
        for i, example in enumerate(examples_set):
            error, result = generate_supports_for_data_point(
                example,
                i,
                world,
                vocabulary,
                generation_mode,
                generation_payload,
                generation_options,
            )

            if error is not None:
                tqdm.write(error)
                continue

            yield result


def encode_metalearning_example(
    world_encoding_scheme,
    reascan_boxes,
    instruction_word2idx,
    action_word2idx,
    color2idx,
    noun2idx,
    example,
):
    (
        command,
        target_commands,
        situation_representation,
        support_instructions,
        support_targets,
        support_situation_representations,
    ) = example
    situation = Situation.from_representation(situation_representation)

    world_layout = parse_sparse_situation(
        situation.to_dict(),
        situation.grid_size,
        color2idx,
        noun2idx,
        world_encoding_scheme,
        reascan_boxes,
    )
    query_instruction, query_target = labelled_situation_to_demonstration_tuple(
        {"input": command, "target": target_commands},
        instruction_word2idx,
        action_word2idx,
    )
    support_layouts = [
        parse_sparse_situation(
            support_situation_representation,
            support_situation_representation["grid_size"],
            color2idx,
            noun2idx,
            world_encoding_scheme,
            reascan_boxes,
        )
        for support_situation_representation in support_situation_representations
    ]
    if support_instructions:
        support_instructions, support_targets = list(
            zip(
                *[
                    labelled_situation_to_demonstration_tuple(
                        {"input": support_instruction, "target": support_target},
                        instruction_word2idx,
                        action_word2idx,
                    )
                    for support_instruction, support_target in zip(
                        support_instructions, support_targets
                    )
                ]
            )
        )
    else:
        support_instructions = np.array([], dtype=int)
        support_targets = np.array([], dtype=int)

    return (
        query_instruction,
        add_eos_to_actions(query_target, action_word2idx["[eos]"]),
        world_layout,
        world_layout if not support_layouts else support_layouts,
        support_instructions,
        [
            add_eos_to_actions(support_target, action_word2idx["[eos]"])
            for support_target in support_targets
        ],
        # Priorities, which in this case, are always ordered.
        np.array(list(reversed(range(len(support_instructions))))),
    )


def encode_metalearning_examples(
    metalearning_examples,
    world_encoding_scheme,
    reascan_boxes,
    instruction_word2idx,
    action_word2idx,
    color2idx,
    noun2idx,
):
    for ml_example in metalearning_examples:
        yield encode_metalearning_example(
            world_encoding_scheme,
            reascan_boxes,
            instruction_word2idx,
            action_word2idx,
            color2idx,
            noun2idx,
            ml_example,
        )


def demonstrate_target_commands(
    command: str, initial_situation: Situation, target_commands: List[str], world
) -> Tuple[List[str], List[Situation], Optional[int], Optional[int]]:
    """Executes a sequence of commands starting from initial_situation."""
    return target_commands, [initial_situation], None, None  # end_column, end_row


def parse_example(data_example: dict):
    """Take an example as written in a file and parse it to its internal representations such that we can interact
    with it."""
    command = parse_command_repr(data_example["command"])
    meaning = parse_command_repr(data_example["meaning"])
    situation = Situation.from_representation(data_example["situation"])
    target_commands = parse_command_repr(data_example["target_commands"])
    manner = data_example["manner"]

    return command, meaning, situation, target_commands, manner


def yield_situations_from_examples_set(examples_set):
    for data_example in tqdm(examples_set):
        command, meaning, situation, target_commands, manner = parse_example(
            data_example
        )
        yield {"input": command, "target": target_commands, "situation": situation}


def yield_situations(d, split):
    yield from yield_situations_from_examples_set(d["examples"][split])


def yield_baseline_examples(
    situations,
    instruction_word2idx,
    action_word2idx,
    color2idx,
    noun2idx,
    world_encoding_scheme,
    reascan_boxes,
):
    for situation in situations:
        instruction, target = labelled_situation_to_demonstration_tuple(
            {
                "input": parse_command_repr(situation["command"]),
                "target": parse_command_repr(situation["target_commands"]),
            },
            instruction_word2idx,
            action_word2idx,
        )
        situation_object = Situation.from_representation(situation["situation"])
        world_layout = parse_sparse_situation(
            situation_object.to_dict(),
            situation_object.grid_size,
            color2idx,
            noun2idx,
            world_encoding_scheme,
            reascan_boxes,
        )

        yield (
            instruction,
            add_eos_to_actions(target, action_word2idx["[eos]"]),
            world_layout,
        )


def baseline_payload(dataset, vocabulary, word2idx, current_split):
    return None


def generate_oracle_payload(dataset, vocabulary, word2idx, current_split, global_payload, params):
    return (
        vocabulary.get_nouns(),
        vocabulary.get_color_adjectives(),
        SPLITS_ALLOW_ORACLE_DEMONSTRATIONS.get(current_split, []),
    )


def generate_oracle_find_any_matching_payload(dataset, vocabulary, word2idx, current_split, global_payload, params):
    sorted_example_indices_by_command = sort_indices_by_command(
        dataset["examples"]["train"]
    )

    return (
        sorted_example_indices_by_command,
        dataset["examples"]["train"],
        vocabulary.get_nouns(),
        vocabulary.get_color_adjectives(),
        SPLITS_ALLOW_ORACLE_DEMONSTRATIONS.get(current_split, []),
    )


def to_count_matrix(word_arrays, word_vocab_size):
    count_matrix = np.zeros(
        (len(word_arrays), word_vocab_size)
    )

    for i, word_array in enumerate(word_arrays):
        for element in word_array:
            count_matrix[i, element] += 1

    return count_matrix


def to_tfidf(tfidf_transformer, count_matrix):
    return tfidf_transformer.transform(count_matrix).todense().astype("float32")


def retrieve_similar_state_payload(dataset, vocabulary, word2idx, current_split, global_payload, params):
    model, index, state_scaler, state_pca, train_unique_encodings, train_unique_token_encodings, train_unique_indices = global_payload
    colors = sorted(vocabulary.get_color_adjectives())
    COLOR2IDX = {c: i + 1 for i, c in enumerate(colors)}

    nouns = sorted(vocabulary.get_nouns())
    NOUN2IDX = {n: i + 1 for i, n in enumerate(nouns)}

    split_state_vectors = vectorize_all_example_situations(
        tqdm(dataset["examples"][current_split]),
        COLOR2IDX,
        NOUN2IDX
    )
    normalized_split_state_vectors = normalize(split_state_vectors)

    pca_split_state_vectors = state_pca.apply(
        state_scaler.transform(split_state_vectors)
    ).astype(np.float32)

    split_sentences_index_dict = defaultdict(list)
    for i, example in enumerate(tqdm(dataset["examples"][current_split])):
        split_sentences_index_dict[" ".join(example["command"].split(","))].append(i)
    split_sentences_unique = sorted(list(split_sentences_index_dict.keys()))
    split_sentences_to_unique_index = {
        t: i for i, t in enumerate(split_sentences_unique)
    }
    split_sentences_unique_list_lookup = np.zeros(len(dataset["examples"][current_split]), dtype=np.int32)
    for t, indices in split_sentences_index_dict.items():
        for i in indices:
            split_sentences_unique_list_lookup[i] = split_sentences_to_unique_index[t]

    split_sentences_unique_all_token_encodings = list(map(lambda x: x.cpu().numpy(), model.encode(
        split_sentences_unique,
        output_value='token_embeddings',
        normalize_embeddings=True
    )))
    split_sentences_unique_all_token_encodings = [
        v / (np.linalg.norm(v, axis=-1)[:, None] + 1e-7)
        for v in split_sentences_unique_all_token_encodings
    ]
    split_sentences_unique_all_sentence_encodings = model.encode(
        split_sentences_unique,
        normalize_embeddings=True
    )

    normalized_split_vectors = normalize(np.concatenate([
        np.array(pca_split_state_vectors),
        split_sentences_unique_all_sentence_encodings[split_sentences_unique_list_lookup]
    ], axis=-1), axis=1).astype(np.float32)

    # Once we're at this point, lets release a bunch of stuff we don't need anymore
    del split_sentences_unique_all_sentence_encodings
    del split_state_vectors

    search_results = np.concatenate([
        index.search(normalized_split_vectors[b * 128:(b + 1) * 128], 128)[1]
        for b in trange(
            (params.limit or normalized_split_vectors.shape[0]) // 128 + 1,
            desc=f"Finding near neighbours for split {current_split}"
        )
    ], axis=0)

    return (
        search_results,
        dataset["examples"]["train"],
        word2idx,
        train_unique_indices,
        train_unique_encodings,
        train_unique_token_encodings,
        split_sentences_unique_list_lookup,
        split_sentences_unique_all_token_encodings
    )


def generate_random_instructions_find_in_any_layout_payload(
    dataset, vocabulary, word2idx, current_split, global_payload
):
    sorted_example_indices_by_command = sort_indices_by_command(
        dataset["examples"]["train"]
    )

    return (sorted_example_indices_by_command, dataset["examples"]["train"])


def find_supports_with_same_agent_target_offset_payload(
    dataset, vocabulary, word2idx, current_split, global_payload
):
    sorted_example_indices_by_offsets = sort_indices_by_offsets(
        dataset["examples"]["train"]
    )
    sorted_example_indices_by_command = sort_indices_by_command(
        dataset["examples"]["train"]
    )

    return (
        sorted_example_indices_by_offsets,
        sorted_example_indices_by_command,
        dataset["examples"]["train"],
    )


def find_supports_with_any_target_object_in_same_position_payload(
    dataset, vocabulary, word2idx, current_split, global_payload
):
    sorted_example_indices_by_target_positions = sort_indices_by_target_positions(
        dataset["examples"]["train"]
    )
    sorted_example_indices_by_command = sort_indices_by_command(
        dataset["examples"]["train"]
    )

    return (
        sorted_example_indices_by_target_positions,
        sorted_example_indices_by_command,
        dataset["examples"]["train"],
    )


def find_supports_by_matching_object_in_same_diff_payload(
    dataset, vocabulary, word2idx, current_split, global_payload
):
    sorted_example_indices_by_diff_and_description = (
        sort_indices_by_target_diff_and_description(dataset["examples"]["train"])
    )
    sorted_example_indices_by_command = sort_indices_by_command(
        dataset["examples"]["train"]
    )

    return (
        sorted_example_indices_by_diff_and_description,
        sorted_example_indices_by_command,
        dataset["examples"]["train"],
    )


def find_supports_by_matching_environment_layout_payload(
    dataset, vocabulary, word2idx, current_split, global_payload
):
    sorted_examples_by_serialized_layouts = sort_indices_by_serialized_situation(
        dataset["examples"]["train"]
    )
    sorted_example_indices_by_command = sort_indices_by_command(
        dataset["examples"]["train"]
    )

    return (
        sorted_examples_by_serialized_layouts,
        sorted_example_indices_by_command,
        dataset["examples"]["train"],
    )


def vectorize_state(situation, grid_size, color2dix, noun2idx, encoding_scheme, reascan_boxes):
    return (parse_sparse_situation(
        Situation.from_representation(situation).to_dict(), 6, color2dix, noun2idx, "all", False
    )[:, :-2, None] == np.arange(5, dtype=np.int32)[None, None]).reshape(-1).astype(np.float32)


def vectorize_state_star(args):
    return vectorize_state(*args)


def vectorize_all_example_situations(examples, color2idx, noun2idx):
    with multiprocessing.Pool() as pool:
        return np.stack(
            list(
                pool.map(
                    vectorize_state_star,
                    map(
                        lambda e: (
                            e["situation"],
                            6,
                            color2idx,
                            noun2idx,
                            "all",
                            False
                        ),
                        examples
                    )
                )
            )
        )


# sqrt((a*2) + (b*2)) = 1
# if sqrt(a*2) == sqrt(b*2)
# sum |a| == sum |b|
# b = b / (|b| / |a|) 
def balance_dims(first, *rest, factors=None):
    first_abs_sum = np.linalg.norm(first, axis=-1)
    rest_abs_sums = [
        np.linalg.norm(r, axis=-1) for r in rest
    ]
    rest_abs_ratios = [
        first_abs_sum / rs for rs in rest_abs_sums
    ]

    return np.concatenate([
        first
    ] + [
        r * ra[..., None] * factor
        for r, ra, factor in zip(rest, rest_abs_ratios, factors or ([1] * len(rest_abs_ratios)))
    ], axis=-1)


def premultiply_balance_dimensions(vectors, dim_other):
    return vectors * (dim_other / vectors.shape[-1])


def vectorize_example_text(word_counts, tfidf, dim_state):
    unscaled_train_text_tfidf = tfidf.fit_transform(word_counts)
    # Multiply by d_pca / d_tfidf to ensure that each component has the same
    # contribution in the vector search
    return premultiply_balance_dimensions(unscaled_train_text_tfidf, dim_state)


def retrieve_similar_state_global_payload(dataset, vocabulary, word2idx):
    colors = sorted(vocabulary.get_color_adjectives())
    COLOR2IDX = {c: i + 1 for i, c in enumerate(colors)}

    nouns = sorted(vocabulary.get_nouns())
    NOUN2IDX = {n: i + 1 for i, n in enumerate(nouns)}

    train_state_vectors = vectorize_all_example_situations(
        tqdm(dataset["examples"]["train"]),
        COLOR2IDX,
        NOUN2IDX
    )

    state_scaler = StandardScaler()
    state_scaler.fit(np.array(train_state_vectors))
    scaled_train_state_vectors = state_scaler.transform(train_state_vectors).astype(np.float32)

    # Used for sanity checks below, but here so that everything is together
    normalized_train_state_vectors = normalize(train_state_vectors, axis=1)
    normalized_scaled_train_state_vectors = normalize(scaled_train_state_vectors, axis=1)

    state_pca = faiss.PCAMatrix(scaled_train_state_vectors.shape[-1], 320)
    state_pca.train(scaled_train_state_vectors)

    pca_train_state_vectors = np.array(state_pca.apply(scaled_train_state_vectors))

    # Sanity check, how well do we reconstruct the original layouts
    state_pca_b = faiss.vector_to_array(state_pca.b)
    state_pca_A = faiss.vector_to_array(state_pca.A).reshape(state_pca.d_out, state_pca.d_in)
    reconstruction_error = ((((pca_train_state_vectors - state_pca_b[None]) @ state_pca_A) - scaled_train_state_vectors) ** 2).mean()

    print(f"PCA reconstruction error {reconstruction_error}")

    # Make a lookup table of train sentences to indices - this will allow
    # us to just encode the unique sentences and save some memory.
    train_sentences_index_dict = defaultdict(list)
    for i, example in enumerate(tqdm(dataset["examples"]["train"])):
        train_sentences_index_dict[" ".join(example["command"].split(","))].append(i)
    train_sentences_unique = sorted(list(train_sentences_index_dict.keys()))
    train_sentences_to_unique_index = {
        t: i for i, t in enumerate(train_sentences_unique)
    }
    train_sentences_unique_list_lookup = np.zeros(len(dataset["examples"]["train"]), dtype=np.int32)
    for t, indices in train_sentences_index_dict.items():
        for i in indices:
            train_sentences_unique_list_lookup[i] = train_sentences_to_unique_index[t]


    model = SentenceTransformer('all-mpnet-base-v2')
    train_sentences_unique_all_token_encodings = list(map(lambda x: x.cpu().numpy(), model.encode(
        train_sentences_unique,
        output_value='token_embeddings',
        normalize_embeddings=True
    )))
    train_sentences_unique_all_token_encodings = [
        v / (np.linalg.norm(v, axis=-1)[:, None] + 1e-7)
        for v in train_sentences_unique_all_token_encodings
    ]
    train_sentences_unique_all_sentence_encodings = model.encode(
        train_sentences_unique,
        normalize_embeddings=True
    )

    if False:
        train_word_counts = to_count_matrix([
            np.array([word2idx[w] for w in e["command"].split(",")])
            for e in dataset["examples"]["train"]
        ], len(word2idx))
        train_text_tfidf = TfidfTransformer()
        train_text_tfidf.fit(train_word_counts)
        unscaled_train_text_tfidf = train_text_tfidf.transform(train_word_counts)

        unscaled_train_text_tfidf = vectorize_example_text(
            train_word_counts,
            train_text_tfidf,
            pca_train_state_vectors.shape[-1]
        )

    scaled_train_vectors = normalize(
        balance_dims(
            np.array(pca_train_state_vectors),
            train_sentences_unique_all_sentence_encodings[train_sentences_unique_list_lookup],
            factors=[(1 / 2)]
        ),
        axis=1
    ).astype(np.float32)

    # Voronoi index to improve retrieval speed
    base_index = faiss.IndexFlatIP(scaled_train_vectors.shape[-1])
    index = faiss.IndexIVFFlat(base_index, scaled_train_vectors.shape[-1], 512)
    index.train(scaled_train_vectors[np.random.permutation(scaled_train_vectors.shape[0])[:512 * 40]])
    index.nprobe = 10
    index.add(scaled_train_vectors)

    # Sanity check, run NN search on some random subset of the vectors and check
    # similarity of the states
    search_sample_indices = np.random.permutation(scaled_train_vectors.shape[0])[:512]
    search_sample_train_vectors = scaled_train_vectors[search_sample_indices]
    normalized_search_sample_scaled_train_state_vectors = normalized_train_state_vectors[search_sample_indices]
    sample_retrieved_indices = index.search(
        search_sample_train_vectors,
        128
    )[1][:, 1:]

    sample_mean_similarities = np.stack([
        (normalized_train_state_vectors[indices] @ vector[:, None]).mean(axis=0)
        for vector, indices in zip(normalized_search_sample_scaled_train_state_vectors, sample_retrieved_indices)
    ]).mean()
    print(f"Mean similarity of retrieved states {sample_mean_similarities}")
    sample_mean_sentence_similarities = np.stack([
        train_sentences_unique_all_sentence_encodings[train_sentences_unique_list_lookup[indices]] @ train_sentences_unique_all_sentence_encodings[train_sentences_unique_list_lookup[query_index]].T
        for indices, query_index in zip(sample_retrieved_indices, search_sample_indices)
    ]).mean()
    print(f"Mean similarity of retrieved sentences {sample_mean_sentence_similarities}")

    # Compare with baseline where we create an index just from the state vectors
    baseline_index = faiss.IndexFlatIP(scaled_train_state_vectors.shape[-1])
    baseline_index.add(normalized_train_state_vectors)
    baseline_search_sample_train_vectors = normalized_train_state_vectors[search_sample_indices]
    baseline_sample_retrieved_indices = baseline_index.search(baseline_search_sample_train_vectors, 2)[1][:, 1:]

    baseline_sample_mean_similarities = np.stack([
        (normalized_train_state_vectors[indices] @ vector[:, None]).mean(axis=0)
        for vector, indices in zip(baseline_search_sample_train_vectors, baseline_sample_retrieved_indices)
    ]).mean()
    print(f"Baseline mean similarity of retrieved states {baseline_sample_mean_similarities}")

    return (
        model,
        index,
        state_scaler,
        state_pca,
        train_sentences_unique_all_sentence_encodings,
        train_sentences_unique_all_token_encodings,
        train_sentences_unique_list_lookup
    )

def null_global_payload(dataset, vocabulary, word2idx):
    return None


GENERATION_CONFIGS = {
    "baseline": {"yield_func": "baseline", "generate_mode": "baseline"},
    "metalearn": {
        "yield_func": "metalearning",
        "generate_mode": "generate_oracle",
        "kwargs": {
            "n_description_options": 1,
            "can_parallel": True,
            "limit_verb_adverb": True,
        },
    },
    "metalearn_allow_any": {
        "yield_func": "metalearning",
        "generate_mode": "generate_oracle",
        "kwargs": {
            "n_description_options": 1,
            "can_parallel": True,
            "allow_any_example": True,
            "limit_verb_adverb": True,
        },
    },
    "metalearn_distractors": {
        "yield_func": "metalearning",
        "generate_mode": "generate_oracle",
        "kwargs": {"n_description_options": 3, "can_parallel": True},
    },
    "metalearn_all": {
        "yield_func": "metalearning",
        "generate_mode": "generate_oracle",
        "kwargs": {"n_description_options": None, "can_parallel": True},
    },
    "metalearn_all_allow_any": {
        "yield_func": "metalearning",
        "generate_mode": "generate_oracle",
        "kwargs": {
            "n_description_options": None,
            "can_parallel": True,
            "allow_any_example": True,
        },
    },
    "metalearn_random_instructions_same_layout": {
        "yield_func": "metalearning",
        "generate_mode": "generate_oracle",
        "kwargs": {
            "n_description_options": None,
            "demonstrate_target": False,
            "can_parallel": True,
            "num_demos": 16,
            "pick_random": True,
        },
    },
    "metalearn_random_instructions_same_layout_allow_any": {
        "yield_func": "metalearning",
        "generate_mode": "generate_oracle",
        "kwargs": {
            "n_description_options": None,
            "demonstrate_target": False,
            "can_parallel": True,
            "num_demos": 16,
            "pick_random": True,
            "allow_any_example": True,
        },
    },
    "metalearn_random_layouts": {
        "yield_func": "metalearning",
        "generate_mode": "random_find_matching",
        "kwargs": {"num_demos": 16, "can_parallel": False},
    },
    "metalearn_find_matching_instruction_demos": {
        "yield_func": "metalearning",
        "generate_mode": "generate_find_matching",
        "kwargs": {"can_parallel": False, "limit_verb_adverb": True},
    },
    "metalearn_find_matching_instruction_demos_allow_any": {
        "yield_func": "metalearning",
        "generate_mode": "generate_find_matching",
        "kwargs": {
            "can_parallel": False,
            "allow_any_example": True,
            "limit_verb_adverb": True,
        },
    },
    "metalearn_find_matching_environment_layout": {
        "yield_func": "metalearning",
        "generate_mode": "find_by_environment_layout",
        "kwargs": {"can_parallel": False, "num_demos": 16},
    },
    "metalearn_find_matching_target_location_demos": {
        "yield_func": "metalearning",
        "generate_mode": "find_by_matching_any_object_in_same_target_position",
        "kwargs": {"can_parallel": False, "num_demos": 16},
    },
    "metalearn_find_matching_diff_demos": {
        "yield_func": "metalearning",
        "generate_mode": "find_by_matching_any_object_in_same_diff",
        "kwargs": {"can_parallel": False, "num_demos": 16},
    },
    "metalearn_find_matching_object_same_diff_demos": {
        "yield_func": "metalearning",
        "generate_mode": "find_by_matching_same_object_in_same_diff",
        "kwargs": {"can_parallel": False, "num_demos": 16},
    },
    "metalearn_retrieve_state_coverage": {
        "yield_func": "metalearning",
        "generate_mode": "retrieve_similar_state",
        "kwargs": {"can_parallel": False, "num_demos": 16}
    }
}


SPLITS_NAMES_MAP = {
    "train": "train",
    "test": "a",
    "visual_easier": "b",
    "visual": "c",
    "situational_1": "d",
    "situational_2": "e",
    "contextual": "f",
    "adverb_2": "h",
    "adverb_1": "g",
}

SPLITS_ALLOW_ORACLE_DEMONSTRATIONS = {
    "train": [],
    "test": ["A", "B", "C", "D", "E", "F", "G", "H"],
    "visual_easier": ["A", "C", "D", "E", "F", "G", "H"],
    "visual": ["A", "B", "D", "E", "F", "G", "H"],
    "situational_1": ["A", "B", "C", "E", "F", "G", "H"],
    "situational_2": ["A", "B", "C", "D", "F", "G", "H"],
    "contextual": ["A", "B", "C", "D", "E", "G", "H"],
    "adverb_1": ["A", "B", "C", "D", "E", "F", "G", "H"],
    "adverb_2": ["A", "B", "C", "D", "E", "F", "G"],
}

PREPROCESSING_GLOBAL_PAYLOAD_GENERATOR = {
    "baseline": null_global_payload,
    "generate_oracle": null_global_payload,
    "generate_find_matching": null_global_payload,
    "retrieve_similar_state": retrieve_similar_state_global_payload,
    "random_find_matching": null_global_payload,
    "find_by_environment_layout": null_global_payload,
    "find_by_matching_same_object_in_same_diff": null_global_payload,
    "find_by_matching_any_object_in_same_target_position": null_global_payload,
    "find_by_matching_any_object_in_same_diff": null_global_payload,
}

PREPROCESSING_PAYLOAD_GENERATOR = {
    "baseline": baseline_payload,
    "generate_oracle": generate_oracle_payload,
    "generate_find_matching": generate_oracle_find_any_matching_payload,
    "retrieve_similar_state": retrieve_similar_state_payload,
    "random_find_matching": generate_oracle_find_any_matching_payload,
    "find_by_environment_layout": find_supports_by_matching_environment_layout_payload,
    "find_by_matching_same_object_in_same_diff": find_supports_by_matching_object_in_same_diff_payload,
    "find_by_matching_any_object_in_same_target_position": find_supports_with_any_target_object_in_same_position_payload,
    "find_by_matching_any_object_in_same_diff": find_supports_with_same_agent_target_offset_payload,
}


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))

        if not batch:
            break

        yield batch


def main():
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gscan-dataset", type=str, required=True)
    parser.add_argument("--output-directory", type=str, required=True)
    parser.add_argument(
        "--generate-mode", choices=tuple(GENERATION_CONFIGS.keys()), required=True
    )
    parser.add_argument("--limit", type=int, help="Data generation limit", default=None)
    parser.add_argument("--only-splits", nargs="*", type=str)
    parser.add_argument(
        "--world-encoding-scheme", choices=("sequence", "all"), default="sequence"
    )
    parser.add_argument("--reascan-boxes", action="store_true")
    args = parser.parse_args()

    with open(args.gscan_dataset, "r") as f:
        d = json.load(f)

    print("Number of examples per split :" + "\n".join([
        f"- {key}: {len(values)}"
        for key, values in d["examples"].items()
    ]))

    vocabulary = create_vocabulary()
    world = create_world(vocabulary)

    colors = sorted(vocabulary.get_color_adjectives())
    COLOR2IDX = {c: i + 1 for i, c in enumerate(colors)}

    nouns = sorted(vocabulary.get_nouns())
    NOUN2IDX = {n: i + 1 for i, n in enumerate(nouns)}

    INPUT_WORD2IDX = {
        w: i
        for i, w in enumerate(
            sorted(
                list(
                    set(
                        itertools.chain.from_iterable(
                            itertools.chain.from_iterable(
                                map(
                                    lambda s: s["command"].split(","),
                                    d["examples"][split],
                                )
                            )
                            for split in d["examples"].keys()
                        )
                    )
                )
            )
        )
    }
    ACTION_WORD2IDX = {
        w: i
        for i, w in enumerate(
            sorted(
                list(
                    set(
                        itertools.chain.from_iterable(
                            map(
                                lambda s: s["target_commands"].split(","),
                                d["examples"]["train"],
                            )
                        )
                    )
                )
            )
        )
    }
    INPUT_WORD2IDX["[pad]"] = len(INPUT_WORD2IDX.values())
    INPUT_WORD2IDX["[sos]"] = len(INPUT_WORD2IDX.values())
    ACTION_WORD2IDX["[pad]"] = len(ACTION_WORD2IDX.values())
    ACTION_WORD2IDX["[sos]"] = len(ACTION_WORD2IDX.values())
    ACTION_WORD2IDX["[eos]"] = len(ACTION_WORD2IDX.values())

    bound_funcs = {
        "baseline": lambda examples, payload, kwargs: yield_baseline_examples(
            examples,
            INPUT_WORD2IDX,
            ACTION_WORD2IDX,
            COLOR2IDX,
            NOUN2IDX,
            args.world_encoding_scheme,
            args.reascan_boxes,
        ),
        "metalearning": lambda examples, payload, kwargs: encode_metalearning_examples(
            yield_metalearning_examples(
                examples,
                world,
                vocabulary,
                GENERATION_CONFIGS[args.generate_mode]["generate_mode"],
                payload,
                kwargs,
            ),
            args.world_encoding_scheme,
            args.reascan_boxes,
            INPUT_WORD2IDX,
            ACTION_WORD2IDX,
            COLOR2IDX,
            NOUN2IDX,
        ),
    }

    splits = {s: SPLITS_NAMES_MAP.get(s, s) for s in list(d["examples"].keys())}

    if args.only_splits:
        splits = {k: v for k, v in splits.items() if v in args.only_splits}

    os.makedirs(f"{args.output_directory}/valid", exist_ok=True)

    global_payload = PREPROCESSING_GLOBAL_PAYLOAD_GENERATOR[
        GENERATION_CONFIGS[args.generate_mode]["generate_mode"]
    ](d, vocabulary, INPUT_WORD2IDX)

    for split, split_name in tqdm(splits.items()):
        payload = PREPROCESSING_PAYLOAD_GENERATOR[
            GENERATION_CONFIGS[args.generate_mode]["generate_mode"]
        ](d, vocabulary, INPUT_WORD2IDX, split, global_payload, args)

        os.makedirs(f"{args.output_directory}/{split_name}", exist_ok=True)
        iterable = bound_funcs[GENERATION_CONFIGS[args.generate_mode]["yield_func"]](
            tqdm(d["examples"][split][: args.limit]),
            payload,
            GENERATION_CONFIGS[args.generate_mode].get("kwargs", {}),
        )

        for i, batch in enumerate(batched(iterable, 10000)):
            with open(f"{args.output_directory}/{split_name}/{i}.pb", "wb") as f:
                pickle.dump(batch, f)

    with open(f"{args.output_directory}/dictionary.pb", "wb") as f:
        pickle.dump(
            (
                INPUT_WORD2IDX,
                ACTION_WORD2IDX,
                vocabulary.get_color_adjectives(),
                vocabulary.get_nouns(),
            ),
            f,
        )


if __name__ == "__main__":
    main()
