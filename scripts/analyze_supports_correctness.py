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
    example, word2idx, action2idx, color_dictionary, noun_dictionary, limit_demos
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
        query, state, target, word2idx, action2idx, color_dictionary, noun_dictionary, world, vocab
    )[0]

    support_state = (
        [support_state] * len(support_query)
        if isinstance(support_state[0], np.ndarray)
        else support_state
    )

    support_state = support_state[:limit_demos]
    support_query = support_query[:limit_demos]
    support_target = support_target[:limit_demos]

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
    demonstrations, word2idx, action2idx, color_dictionary, noun_dictionary, limit_demos=None, num_procs=8
):
    with multiprocessing.Pool(num_procs) as pool:
        yield from pool.imap_unordered(
            compute_num_correct_and_valid_star,
            map(
                lambda x: (x, word2idx, action2idx, color_dictionary, noun_dictionary, limit_demos),
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
    parser.add_argument("--only-splits", default=None, nargs="*")
    parser.add_argument("--limit-demos", default=None, type=int)
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
                examples, WORD2IDX, ACTION2IDX, color_dictionary, noun_dictionary, limit_demos=args.limit_demos
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
        if not args.only_splits or split in args.only_splits
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
