import argparse
import json
import itertools
import numpy as np
from typing import List, Optional, Tuple
import os
from collections import defaultdict
import pickle
import multiprocessing
import random

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset

from tqdm.auto import tqdm

from gscan_metaseq2seq.gscan.world import Situation, ObjectVocabulary, World
from gscan_metaseq2seq.gscan.vocabulary import Vocabulary

from gscan_metaseq2seq.util.dataset import PaddingIterableDataset
from train_transformer import TransformerLearner

GRID_SIZE = 6
MIN_OTHER_OBJECTS = 0
MAX_OBJECTS = 2
MIN_OBJECT_SIZE = 1
MAX_OBJECT_SIZE = 4
OTHER_OBJECTS_SAMPLE_PERCENTAGE = 0.5

TYPE_GRAMMAR = "adverb"
INTRANSITIVE_VERBS = "walk"
TRANSITIVE_VERBS = "pull,push"
ADVERBS = "cautiously,while spinning,hesitantly,while zigzagging"
NOUNS = "square,cylinder,circle,box"
COLOR_ADJECTIVES = "red,green,yellow,blue"
SIZE_ADJECTIVES = "big,small"

DIR_TO_INT = {"west": 3, "east": 1, "north": 2, "south": 0}
INT_TO_DIR = {
    direction_int: direction for direction, direction_int in DIR_TO_INT.items()
}


def parse_command_repr(command_repr: str) -> List[str]:
    return command_repr.split(",")


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


def sort_indices_by_command(examples):
    command_examples = defaultdict(list)

    # Necessarily sorted, because we iterate front to back and append
    for index, example in enumerate(examples):
        command_examples[" ".join(example["command"].split(","))].append(index)

    return dict(command_examples)


action_options = [["walk", "to"], ["push"], ["pull"]]
adverb_options = [["while spinning"], ["while zigzagging"], ["hesitantly"], []]


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
):
    action_words = []
    article_words = []
    description_words = []
    adverb_words = []

    support_instructions = []

    real_adverb_options = adverb_options

    # We generate "cautiously" only if "cautiously" appears in the query instrunction
    if "cautiously" in query_instruction:
        real_adverb_options = real_adverb_options + [["cautiously"]]

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
        for action_option in action_options:
            for adverb_option in real_adverb_options:
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
        num_grid_channels = 7
        grid = []
        agent_representation = np.zeros([num_grid_channels], dtype=np.int)
        agent_representation[-4] = 1
        agent_representation[-3] = agent_direction
        agent_representation[-2] = agent_row
        agent_representation[-1] = agent_column
        grid.append(agent_representation)

        for placed_object in situation_representation["objects"]:
            object_row = int(placed_object.position.row)
            object_column = int(placed_object.position.column)
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
                )
            )
    elif world_encoding_scheme == "all":
        # attribute bits + agent + agent direction
        num_grid_channels = 5
        grid = np.zeros([grid_size, grid_size, num_grid_channels], dtype=int)
        agent_representation = np.zeros([num_grid_channels], dtype=np.int)
        agent_representation[-2] = 1
        agent_representation[-1] = agent_direction

        grid[agent_row, agent_column, :] = agent_representation

        # Loop over the objects in the world and place them.
        for placed_object in situation_representation["objects"]:
            object_row = int(placed_object.position.row)
            object_column = int(placed_object.position.column)
            grid[object_row, object_column, 0] = int(placed_object.object.size)
            grid[object_row, object_column, 1] = int(
                color2idx[placed_object.object.color]
            )
            grid[object_row, object_column, 2] = int(
                noun2idx[placed_object.object.shape]
            )

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


def generate_supports_for_data_point(
    data_example,
    world,
    vocabulary,
    sorted_example_indices_by_command,
    train_examples,
    instruction_word2idx,
    action_word2idx,
    color2idx,
    noun2idx,
    world_encoding_scheme,
    num_irrelevant=0,
    generate_relevant=True,
    search_relevant=False,
    allow_demonstration_splits=None,
):
    command = parse_command_repr(data_example["command"])
    target_commands = parse_command_repr(data_example["target_commands"])
    situation = Situation.from_representation(data_example["situation"])

    world_layout = parse_sparse_situation(
        situation.to_dict(),
        situation.grid_size,
        color2idx,
        noun2idx,
        world_encoding_scheme,
    )
    query_instruction, query_target = labelled_situation_to_demonstration_tuple(
        {"input": command, "target": target_commands},
        instruction_word2idx,
        action_word2idx,
    )
    support_instructions = []
    support_targets = []
    support_layouts = []

    relevant_instructions = generate_relevant_instructions_gscan_oracle(
        command,
        situation,
        list(color2idx.keys()),
        list(noun2idx.keys()),
        n_description_options=num_irrelevant + 1
        if num_irrelevant is not None
        else None,
        demonstrate_target=generate_relevant,
        allow_demonstration_splits=allow_demonstration_splits,
    )

    if not relevant_instructions:
        return (
            f"Skipping for {command} {situation.target_object} / {situation.placed_objects} as no demonstrations are possible and it is training or Split A test data.\n",
            None,
        )

    for support_instruction_command, target_object in relevant_instructions:
        if search_relevant:
            key = " ".join(support_instruction_command)

            if search_relevant and key in sorted_example_indices_by_command:
                relevant_example_idx = np.random.choice(
                    sorted_example_indices_by_command[key]
                )
                relevant_example = train_examples[relevant_example_idx]
                relevant_situation = Situation.from_representation(
                    relevant_example["situation"]
                )
                relevant_layout = parse_sparse_situation(
                    relevant_situation.to_dict(),
                    relevant_situation.grid_size,
                    color2idx,
                    noun2idx,
                    world_encoding_scheme,
                )
                relevant_target_commands = parse_command_repr(
                    relevant_example["target_commands"]
                )
                (
                    support_instruction,
                    support_target,
                ) = labelled_situation_to_demonstration_tuple(
                    {
                        "input": support_instruction_command,
                        "target": relevant_target_commands,
                    },
                    instruction_word2idx,
                    action_word2idx,
                )
                support_layouts.append(relevant_layout)
                support_instructions.append(support_instruction)
                support_targets.append(support_target)
                continue

        # Otherwise we have to generate
        support_target_commands = demonstrate_command_oracle(
            world,
            vocabulary,
            list(color2idx.keys()),
            list(noun2idx.keys()),
            support_instruction_command,
            target_object,
            situation,
        )
        (
            support_instruction,
            support_target,
        ) = labelled_situation_to_demonstration_tuple(
            {
                "input": support_instruction_command,
                "target": support_target_commands,
            },
            instruction_word2idx,
            action_word2idx,
        )
        support_instructions.append(support_instruction)
        support_targets.append(support_target)

        # We append here for consistency
        if search_relevant:
            support_layouts.append(world_layout)

    return (
        None,
        (
            world_layout,
            world_layout if not support_layouts else support_layouts,
            query_instruction,
            add_eos_to_actions(query_target, action_word2idx["[eos]"]),
            support_instructions,
            [
                add_eos_to_actions(support_target, action_word2idx["[eos]"])
                for support_target in support_targets
            ],
        ),
    )


def generate_supports_for_data_point_star(args):
    return generate_supports_for_data_point(*args)


def yield_metalearning_examples(
    examples_set,
    world,
    vocabulary,
    sorted_example_indices_by_command,
    train_examples,
    instruction_word2idx,
    action_word2idx,
    color2idx,
    noun2idx,
    world_encoding_scheme,
    num_irrelevant=0,
    generate_relevant=True,
    search_relevant=False,
    allow_demonstration_splits=None,
    n_procs=8,
):
    if not search_relevant:
        # Fast path for generation, we don't need to keep the whole dataset
        # in memory in order to search for matching examples
        with multiprocessing.Pool(processes=n_procs) as pool:
            for error, result in pool.imap_unordered(
                generate_supports_for_data_point_star,
                map(
                    lambda x: (
                        x,
                        world,
                        vocabulary,
                        None,  # sorted_example_indices_by_command,
                        None,  # train_examples,
                        instruction_word2idx,
                        action_word2idx,
                        color2idx,
                        noun2idx,
                        world_encoding_scheme,
                        num_irrelevant,
                        generate_relevant,
                        search_relevant,
                        allow_demonstration_splits,
                    ),
                    examples_set,
                ),
                chunksize=100,
            ):
                if error is not None:
                    tqdm.write(error)
                    continue

                yield result
    else:
        # Slow path, we need to keep the whole dataset
        for example in examples_set:
            error, result = generate_supports_for_data_point(
                example,
                world,
                vocabulary,
                sorted_example_indices_by_command,
                train_examples,
                instruction_word2idx,
                action_word2idx,
                color2idx,
                noun2idx,
                num_irrelevant,
                generate_relevant,
                search_relevant,
                allow_demonstration_splits,
            )

            if error is not None:
                tqdm.write(error)
                continue

            yield result


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
        )

        yield (
            instruction,
            add_eos_to_actions(target, action_word2idx["[eos]"]),
            world_layout,
        )


class MetalearnSetToTransformerInputsDataset(IterableDataset):
    def __init__(self, metalearn_examples):
        super().__init__()
        self.count = 0
        self.support_count = 0
        self.metalearn_examples = metalearn_examples

    def __iter__(self):
        self.count = 0
        self.support_count = 0
        return self

    def __next__(self):
        (
            query_state,
            support_state,
            query_instruction,
            target,
            support_instructions,
            support_targets,
        ) = self.metalearn_examples[self.count]

        if self.support_count == len(support_instructions):
            self.support_count = 0
            self.count += 1

            if self.count == len(self.metalearn_examples):
                raise StopIteration()

            (
                query_state,
                support_state,
                query_instruction,
                target,
                support_instructions,
                support_targets,
            ) = self.metalearn_examples[self.count]

        result = (
            support_instructions[self.support_count],
            support_targets[self.support_count],
            query_state,
            self.count,
        )

        self.support_count += 1
        return result


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))

        if not batch:
            break

        yield batch


def yield_transformer_model_examples(
    examples_set,
    transformer_model_path,
    world,
    vocabulary,
    instruction_word2idx,
    action_word2idx,
    color2idx,
    noun2idx,
    world_encoding_scheme,
    transformer_batch_size=64,
    allow_demonstration_splits=None,
):
    module = TransformerLearner.load_from_checkpoint(transformer_model_path)
    trainer = pl.Trainer(precision=16, accelerator="gpu")

    # We need to buffer up the metalearning_examples as we
    # need the ability to do random access later on
    metalearning_examples = list(
        yield_metalearning_examples(
            examples_set,
            world,
            vocabulary,
            {},
            [],
            instruction_word2idx,
            action_word2idx,
            color2idx,
            noun2idx,
            world_encoding_scheme,
            allow_demonstration_splits=allow_demonstration_splits,
        )
    )
    train_preds = trainer.predict(
        module,
        DataLoader(
            PaddingIterableDataset(
                MetalearnSetToTransformerInputsDataset(metalearning_examples),
                (8, 128, 36, None),
                (instruction_word2idx["[pad]"], action_word2idx["[pad]"], 0, None),
            ),
            batch_size=transformer_batch_size,
        ),
    )

    num_datapoints = (max([b[-1].max() for b in train_preds]) + 1).item()

    data_array = [[None, None, None, None, [], []] for _ in range(num_datapoints)]

    for (
        state_batch,
        support_instruction,
        decoded,
        logits,
        exacts,
        target,
        index,
    ) in tqdm(train_preds):
        for i, index in enumerate(index.numpy()):
            data_array[index][0] = state_batch[i]
            data_array[index][1] = state_batch[i]
            data_array[index][2] = metalearning_examples[index][2]
            data_array[index][3] = metalearning_examples[index][3]
            data_array[index][4].append(support_instruction[i])
            data_array[index][5].append(decoded[i])

    yield from data_array


GENERATION_CONFIGS = {
    "baseline": {"yield_func": "baseline"},
    "metalearn": {"yield_func": "metalearning"},
    "metalearn_distractors": {
        "yield_func": "metalearning",
        "kwargs": {"num_irrelevant": 3},
    },
    "metalearn_all": {
        "yield_func": "metalearning",
        "kwargs": {"num_irrelevant": None},
    },
    "metalearn_random_only": {
        "yield_func": "metalearning",
        "kwargs": {"num_irrelevant": 4, "generate_relevant": False},
    },
    "metalearn_sample_environments": {
        "yield_func": "metalearning",
        "kwargs": {"search_relevant": True},
    },
    "metalearn_transformer_actions": {
        "yield_func": "metalearning_transformer",
        "kwargs": {},
    },
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

SPLITS_ALLOW_ORACLE_DEMONSTRAITONS = {
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


def main():
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gscan-dataset", type=str, required=True)
    parser.add_argument("--output-directory", type=str, required=True)
    parser.add_argument(
        "--generate-mode",
        choices=tuple(GENERATION_CONFIGS.keys()),
    )
    parser.add_argument("--limit", type=int, help="Data generation limit", default=None)
    parser.add_argument("--transformer-model", type=str, help="Transformer model")
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--only-splits", nargs="*", type=str)
    parser.add_argument(
        "--world-encoding-scheme", choices=("sequence", "all"), default="sequence"
    )
    args = parser.parse_args()

    with open(args.gscan_dataset, "r") as f:
        d = json.load(f)

    vocabulary = Vocabulary.initialize(
        intransitive_verbs=(INTRANSITIVE_VERBS).split(","),
        transitive_verbs=(TRANSITIVE_VERBS).split(","),
        adverbs=(ADVERBS).split(","),
        nouns=(NOUNS).split(","),
        color_adjectives=(COLOR_ADJECTIVES).split(","),
        size_adjectives=(SIZE_ADJECTIVES).split(","),
    )
    world = create_world(vocabulary)

    colors = sorted(vocabulary.get_color_adjectives())
    COLOR2IDX = {c: i + 1 for i, c in enumerate(colors)}

    nouns = sorted(vocabulary.get_nouns())
    NOUN2IDX = {n: i + 1 for i, n in enumerate(nouns)}

    seen_instructions = list(
        map(lambda x: x.split(","), set([e["command"] for e in d["examples"]["train"]]))
    )

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

    sorted_example_indices_by_command = sort_indices_by_command(d["examples"]["train"])

    bound_funcs = {
        "baseline": lambda examples, **kwargs: yield_baseline_examples(
            examples,
            INPUT_WORD2IDX,
            ACTION_WORD2IDX,
            COLOR2IDX,
            NOUN2IDX,
            args.world_encoding_scheme,
        ),
        "metalearning": lambda examples, **kwargs: yield_metalearning_examples(
            examples,
            world,
            vocabulary,
            sorted_example_indices_by_command,
            d["examples"]["train"],
            INPUT_WORD2IDX,
            ACTION_WORD2IDX,
            COLOR2IDX,
            NOUN2IDX,
            args.world_encoding_scheme,
            **kwargs,
        ),
        "metalearning_transformer": lambda examples, **kwargs: yield_transformer_model_examples(
            examples,
            args.transformer_model,
            world,
            vocabulary,
            INPUT_WORD2IDX,
            ACTION_WORD2IDX,
            COLOR2IDX,
            NOUN2IDX,
            args.world_encoding_scheme,
            transformer_batch_size=args.inference_batch_size,
            **kwargs,
        ),
    }

    splits = {s: SPLITS_NAMES_MAP.get(s, s) for s in list(d["examples"].keys())}

    if args.only_splits:
        splits = {k: v for k, v in splits.items() if k in args.only_splits}

    os.makedirs(f"{args.output_directory}/valid", exist_ok=True)

    for split, split_name in tqdm(splits.items()):
        os.makedirs(f"{args.output_directory}/{split_name}", exist_ok=True)
        iterable = bound_funcs[GENERATION_CONFIGS[args.generate_mode]["yield_func"]](
            tqdm(d["examples"][split][: args.limit]),
            allow_demonstration_splits=SPLITS_ALLOW_ORACLE_DEMONSTRAITONS.get(
                split, []
            ),
            **GENERATION_CONFIGS[args.generate_mode].get("kwargs", {}),
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
