import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gscan_metaseq2seq.util.load_data import load_data_directories

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
from gscan_metaseq2seq.gscan.grammar import Derivation
from gscan_metaseq2seq.gscan.vocabulary import Vocabulary

from typing import Optional, Union, Dict, List, Tuple

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
                if not (o[:3] == 0).all()
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
    vocabulary,
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
            vocabulary,
            color_dictionary,
            noun_dictionary,
            instr,
            situation.target_object,
            situation,
        )
        + ["[eos]"]
    ]
    dataset_actions = encoded_targets.tolist()

    return oracle_actions == dataset_actions, situation.target_object is not None


REMAP_ACTIONS = {
    "pull": "PULL",
    "push": "PUSH",
    "walk": "WALK",
    "turn left": "LTURN",
    "turn right": "RTURN",
    "stay": "STAY",
    "[eos]": "",
}


def compress_tokens(tokens):
    count = 0
    current_token = None
    for token in tokens:
        if current_token is not None and current_token != token:
            yield f"{current_token}" + ("" if count == 1 else f"({count})")
            count = 0

        current_token = token
        count += 1

    if current_token is not None and current_token != token:
        yield f"{current_token}" + ("" if count == 1 else f"({count})")


def which_color_box(correct, valid, relevant):
    if not valid:
        return "black"

    if relevant and correct:
        return "green"

    if not relevant and correct:
        return "yellow"

    if relevant and not correct:
        return "orange"

    return "red"


def compare_objects(left, right):
    print(left, right)
    return (left.color == right.color) and (left.shape == right.shape)


def plot_at_index(
    examples,
    dataset_name,
    index,
    output_dir,
    word2idx,
    action2idx,
    color_dictionary,
    noun_dictionary,
):
    idx2word = [w for w in word2idx]
    idx2action = [w for w in action2idx]
    vocabulary = Vocabulary.initialize(
        intransitive_verbs=(INTRANSITIVE_VERBS).split(","),
        transitive_verbs=(TRANSITIVE_VERBS).split(","),
        adverbs=(ADVERBS).split(","),
        nouns=(NOUNS).split(","),
        color_adjectives=(COLOR_ADJECTIVES).split(","),
        size_adjectives=(SIZE_ADJECTIVES).split(","),
    )
    world = create_world(vocabulary)

    instr, situation = state_to_situation(
        examples[index][0],
        examples[index][2],
        word2idx,
        color_dictionary,
        noun_dictionary,
    )
    query_target_object = situation.target_object

    world = reinitialize_world(world, situation, vocabulary)

    plt.imshow(world.render(mode="rgb_array"))
    plt.axis("off")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_{index}.pdf"))
    plt.clf()

    print(index)
    print(" ".join([w for w in instr if w != "[pad]"]))

    support_states = meta_valid_demonstrations_dict["h"][index][3]
    support_states = (
        [support_states] * len(examples[index][-3])
        if isinstance(support_states[0], np.ndarray)
        else support_states
    )

    corrects_and_valids = [
        instruction_is_correct(
            demo_instr,
            demo_state,
            demo_acts,
            word2idx,
            action2idx,
            color_dictionary,
            noun_dictionary,
            world,
            vocabulary,
        )
        for demo_instr, demo_acts, demo_state in zip(
            examples[index][-3], examples[index][-2], support_states
        )
    ]
    relevants = [
        compare_objects(
            query_target_object.object,
            state_to_situation(
                demo_instr, demo_state, word2idx, color_dictionary, noun_dictionary
            )[1].target_object.object,
        )
        for demo_instr, demo_state in zip(examples[index][-3], support_states)
    ]

    sorted_instrs = [
        ((demo_instr, demo_acts, demo_state), (correct, valid), relevant)
        for (demo_instr, demo_acts, demo_state), (correct, valid), relevant in sorted(
            list(
                zip(
                    zip(examples[index][-3], examples[index][-2], support_states),
                    corrects_and_valids,
                    relevants,
                )
            ),
            key=lambda x: (~x[-1], ~x[-2][1], ~x[-2][0]),
        )
    ]

    for (
        (demo_instr, demo_acts, demo_state),
        (correct, valid),
        relevant,
    ) in sorted_instrs:

        print(
            " ".join([idx2word[w] for w in demo_instr if w != word2idx["[pad]"]])
            + " "
            + " ".join(map(str, [relevant, valid, correct]))
        )
        print(
            " -- "
            + " ".join(
                compress_tokens(
                    [
                        REMAP_ACTIONS[idx2action[w]]
                        for w in demo_acts
                        if w != action2idx["[pad]"]
                    ]
                )
            )
        )

    print(
        r"""\begin{figure*}[ht]
    \resizebox{\textwidth}{!}{
    \begin{tikzpicture}[
        title/.style={font=\fontsize{6}{6}\color{black!50}\ttfamily},
        node distance = 10mm, 
    ]
    \node [fill=black!10,rounded corners, inner sep=3pt] (query) {
        \begin{tikzpicture}[node distance=0mm]
            \node [anchor=west] (title) {\footnotesize{Query}};
            \node [anchor=north, below=of title.south] (state)
        {\includegraphics[width=.35\textwidth,keepaspectratio=1,trim={4cm 1cm, 3.5cm 1cm},clip]{img/figures/"""
        + f"{dataset_name}_{index}.pdf"
        + r"""}};
            \node[inner sep=0pt, below=of state.south] (query)
        {\footnotesize{$I^Q$ = ``"""
        + " ".join(instr)
        + r"""\"}};
        \end{tikzpicture}
    };
    \node [fill=black!10, rounded corners, inner sep = 5pt, right=of query.east, minimum width=50mm, anchor=south, rotate=90] (ig) {Instruction Generator};
    """
        + (
            r"\node [fill="
            + which_color_box(*sorted_instrs[0][1], sorted_instrs[0][2])
            + r"!10, rounded corners, inner sep = 5pt, right= 7mm of ig.east, anchor=west, yshift=7mm] (i1) {\footnotesize{$I_1$ = ``"
            + " ".join(
                [idx2word[w] for w in sorted_instrs[0][0][0] if w != word2idx["[pad]"]]
            )
            + '"}};\n'
        )
        + (
            "\n".join(
                [
                    r"\node [fill="
                    + which_color_box(correct, valid, relevant)
                    + r"!10, rounded corners, inner sep = 5pt, below=7mm of i"
                    + str(i + 1)
                    + r".west, anchor=west] (i"
                    + str(i + 2)
                    + r") {\footnotesize{$I_"
                    + str(i + 2)
                    + "$ = ``"
                    + (
                        " ".join(
                            [idx2word[w] for w in demo_instr if w != word2idx["[pad]"]]
                        )
                        + "}};"
                    )
                    for i, (
                        (demo_instr, demo_acts, demo_state),
                        (correct, valid),
                        relevant,
                    ) in enumerate(sorted_instrs[1:])
                ]
            )
        )
        + r"""
    \node [fill=black!10, rounded corners, inner sep = 5pt, right=9.5cm of query.east, minimum width=50mm, anchor=south, rotate=90] (at) {Transformer};
    """
        + (
            r"\node [fill="
            + which_color_box(*sorted_instrs[0][1], sorted_instrs[0][2])
            + r"!10, rounded corners, inner sep = 5pt, right= 7mm of at.east, anchor=west, yshift=7mm] (a1) {\footnotesize{$A_1$ = ``"
            + " ".join(
                compress_tokens(
                    [
                        REMAP_ACTIONS[idx2action[w]]
                        for w in sorted_instrs[0][0][1]
                        if w != action2idx["[pad]"]
                    ]
                )
            )
            + '"}};\n'
        )
        + (
            "\n".join(
                [
                    r"\node [fill="
                    + which_color_box(correct, valid, relevant)
                    + r"!10, rounded corners, inner sep=5pt, below=7mm of a"
                    + str(i + 1)
                    + r".west, anchor=west] (a"
                    + str(i + 2)
                    + r") {\footnotesize{$A_"
                    + str(i + 2)
                    + "$ = ``"
                    + (
                        " ".join(
                            compress_tokens(
                                [
                                    REMAP_ACTIONS[idx2action[w]]
                                    for w in demo_acts
                                    if w != action2idx["[pad]"]
                                ]
                            )
                        )
                        + "}};"
                    )
                    for i, (
                        (demo_instr, demo_acts, demo_state),
                        (correct, valid),
                        relevant,
                    ) in enumerate(sorted_instrs[1:])
                ]
            )
        )
        + r"""
    \end{tikzpicture}
    }
    \end{figure*}"""
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-directory", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--img-output-directory", required=True)
    parser.add_argument("--limit-load", type=int)
    parser.add_argument("--split", required=True)
    parser.add_argument("--index", type=int, required=True)
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

    plot_at_index(
        meta_valid_demonstrations_dict[args.split],
        args.dataset_name,
        args.index,
        args.img_output_directory,
        WORD2IDX,
        ACTION2IDX,
        color_dictionary,
        noun_dictionary,
    )

if __name__ == "__main__":
    main()
