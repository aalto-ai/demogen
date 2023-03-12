import numpy as np
import argparse

from gscan_metaseq2seq.util.load_data import load_data, load_data_directories
from tqdm.auto import tqdm
import itertools
import multiprocessing
import pandas as pd

from gscan_metaseq2seq.util.solver import (
    create_vocabulary,
    create_world,
    state_to_situation,
    reinitialize_world,
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
    vocab = create_vocabulary()
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
    parser.add_argument("--show-columns", nargs="+", default=("valid", "correct_and_valid"), choices=("valid", "correct", "correct_if_valid", "correct_and_valid"))
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
        pd.DataFrame.from_dict(split_stats)[["train", "a", "b", "c", "d", "e", "f", "g", "h"] if not args.only_splits else args.only_splits]
        .T[args.show_columns]
        .to_latex(float_format="%.2f", escape=False)
    )


if __name__ == "__main__":
    main()
