import argparse
import json
import itertools
import numpy as np
from typing import List, Optional, Tuple
import os
from collections import defaultdict
import pickle

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
NOUNS = "square,cylinder,circle"
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

    for index, example in enumerate(examples):
        command_examples[" ".join(example["command"].split(","))].append(index)

    return dict(command_examples)


action_options = [["walk", "to"], ["push"], ["pull"]]
adverb_options = [["while spinning"], ["while zigzagging"], ["hesitantly"], []]


def generate_relevant_instructions_gscan_oracle(
    query_instruction,
    seen_instructions,
    vocabulary_colors,
    vocabulary_nouns,
    num_irrelevant=0,
    generate_relevant=True,
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

    if generate_relevant:
        for w in query_instruction:
            if w in ["walk", "to", "push", "pull"]:
                action_words.append(w)

            if w in ["a"]:
                article_words.append(w)

            if w in vocabulary_colors + vocabulary_nouns + ["big", "small"]:
                description_words.append(w)

            if w in ["while spinning", "while zigzagging", "hesitantly", "cautiously"]:
                adverb_words.append(w)

        for action_option in action_options:
            # print(action_option, " - ", article_words + description_words + adverb_words)
            if action_option != action_words:
                if not (
                    action_option == ["pull"] and adverb_words == ["while spinning"]
                ):
                    support_instructions.append(
                        action_option + article_words + description_words + adverb_words
                    )

        for adverb_option in real_adverb_options:
            # print(action_words + article_words + description_words, " - ", adverb_words)
            if adverb_option != adverb_words:
                if not (
                    adverb_option == ["while spinning"] and action_words == ["pull"]
                ):
                    support_instructions.append(
                        action_words + article_words + description_words + adverb_option
                    )

    support_instructions += np.random.choice(
        seen_instructions, num_irrelevant, replace=True
    ).tolist()

    return support_instructions


# Perhaps this can be modified to demonstrate from the command itself and not
# the derivation


def demonstrate_command_oracle(
    world, vocabulary, vocabulary_colors, vocabulary_nouns, command, initial_situation
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

    support_instructions = []

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
        position=initial_situation.target_object.position,
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
    situation_representation: dict, grid_size: int, color2idx, noun2idx
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
    # attribute bits + agent + agent direction
    num_grid_channels = 5

    # Initialize the grid.
    grid = np.zeros([grid_size, grid_size, num_grid_channels], dtype=int)

    # Place the agent.
    agent_row = int(situation_representation["agent_position"].row)
    agent_column = int(situation_representation["agent_position"].column)
    agent_direction = DIR_TO_INT[situation_representation["agent_direction"].name]
    agent_representation = np.zeros([num_grid_channels], dtype=np.int)
    agent_representation[-2] = 1
    agent_representation[-1] = agent_direction
    grid[agent_row, agent_column, :] = agent_representation

    # Loop over the objects in the world and place them.
    for placed_object in situation_representation["objects"]:
        object_row = int(placed_object.position.row)
        object_column = int(placed_object.position.column)
        grid[object_row, object_column, 0] = int(placed_object.object.size)
        grid[object_row, object_column, 1] = int(color2idx[placed_object.object.color])
        grid[object_row, object_column, 2] = int(noun2idx[placed_object.object.shape])
    return grid


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


def yield_metalearning_examples(
    examples_set,
    world,
    vocabulary,
    seen_instructions,
    sorted_example_indices_by_command,
    train_examples,
    instruction_word2idx,
    action_word2idx,
    color2idx,
    noun2idx,
    num_irrelevant=0,
    generate_relevant=True,
    search_relevant=False,
):
    for data_example in examples_set:
        command = parse_command_repr(data_example["command"])
        target_commands = parse_command_repr(data_example["target_commands"])
        situation = Situation.from_representation(data_example["situation"])

        world_layout = add_positional_information_to_observation(
            np.stack(
                [
                    parse_sparse_situation(
                        t.to_dict(), t.grid_size, color2idx, noun2idx
                    )
                    for t in [situation]
                ]
            )
        )
        query_instruction, query_target = labelled_situation_to_demonstration_tuple(
            {"input": command, "target": target_commands},
            instruction_word2idx,
            action_word2idx,
        )
        support_instructions = []
        support_targets = []
        support_layouts = []

        for support_instruction_command in generate_relevant_instructions_gscan_oracle(
            command,
            seen_instructions,
            list(color2idx.keys()),
            list(noun2idx.keys()),
            num_irrelevant=num_irrelevant,
            generate_relevant=generate_relevant,
        ):
            key = " ".join(support_instruction_command)

            if search_relevant and key in sorted_example_indices_by_command:
                relevant_example_idx = np.random.choice(
                    sorted_example_indices_by_command[key]
                )
                relevant_example = train_examples[relevant_example_idx]
                relevant_layout = add_positional_information_to_observation(
                    np.stack(
                        [
                            parse_sparse_situation(
                                t.to_dict(), t.grid_size, color2idx, noun2idx
                            )
                            for t in [
                                Situation.from_representation(
                                    relevant_example["situation"]
                                )
                            ]
                        ]
                    )
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

        yield (
            world_layout,
            world_layout if not support_layouts else support_layouts,
            query_instruction,
            add_eos_to_actions(query_target, action_word2idx["[eos]"]),
            support_instructions,
            [
                add_eos_to_actions(support_target, action_word2idx["[eos]"])
                for support_target in support_targets
            ],
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
    situations, instruction_word2idx, action_word2idx, color2idx, noun2idx
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
        world_layout = add_positional_information_to_observation(
            np.stack(
                [
                    parse_sparse_situation(
                        t.to_dict(), t.grid_size, color2idx, noun2idx
                    )
                    for t in [Situation.from_representation(situation["situation"])]
                ]
            )
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


def yield_transformer_model_examples(
    examples_set,
    transformer_model_path,
    world,
    vocabulary,
    instruction_word2idx,
    action_word2idx,
    color2idx,
    noun2idx,
    transformer_batch_size=64,
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
            [],
            {},
            [],
            instruction_word2idx,
            action_word2idx,
            color2idx,
            noun2idx,
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
    "metalearn_random_only": {
        "yield_func": "metalearning",
        "kwargs": {"num_irrelevant": 8, "generate_relevant": False},
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


def main():
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
                            map(
                                lambda s: s["command"].split(","),
                                d["examples"]["train"],
                            )
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
            examples, INPUT_WORD2IDX, ACTION_WORD2IDX, COLOR2IDX, NOUN2IDX
        ),
        "metalearning": lambda examples, **kwargs: yield_metalearning_examples(
            examples,
            world,
            vocabulary,
            seen_instructions,
            sorted_example_indices_by_command,
            d["examples"]["train"],
            INPUT_WORD2IDX,
            ACTION_WORD2IDX,
            COLOR2IDX,
            NOUN2IDX,
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
            transformer_batch_size=args.inference_batch_size,
            **kwargs,
        ),
    }

    splits = {
        "train": "train",
        "test": "a",
        "visual_easier": "b",
        "visual": "c",
        "situational_1": "d",
        "situational_2": "e",
        "contextual": "f",
        "adverb_2": "h",
    }

    split_examples = {
        split_name: list(
            bound_funcs[GENERATION_CONFIGS[args.generate_mode]["yield_func"]](
                tqdm(d["examples"][split][: args.limit]),
                **GENERATION_CONFIGS[args.generate_mode].get("kwargs", {}),
            )
        )
        for split, split_name in splits.items()
    }

    os.makedirs(f"{args.output_directory}/valid", exist_ok=True)

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

    for split_name, examples in tqdm(split_examples.items()):
        if split_name == "train":
            with open(f"{args.output_directory}/train.pb", "wb") as f:
                pickle.dump(examples, f)
        else:
            with open(f"{args.output_directory}/valid/{split_name}.pb", "wb") as f:
                pickle.dump(examples, f)


if __name__ == "__main__":
    main()
