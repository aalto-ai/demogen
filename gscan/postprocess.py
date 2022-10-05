import numpy as np
from typing import Optional, List, Tuple

from tqdm.auto import tqdm

from .world import Situation, ObjectVocabulary, World
from .vocabulary import Vocabulary


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


def parse_command_repr(command_repr: str) -> List[str]:
    return command_repr.split(",")


def parse_example(data_example: dict):
    """Take an example as written in a file and parse it to its internal representations such that we can interact
    with it."""
    command = parse_command_repr(data_example["command"])
    meaning = parse_command_repr(data_example["meaning"])
    situation = Situation.from_representation(data_example["situation"])
    target_commands = parse_command_repr(data_example["target_commands"])
    manner = data_example.get("manner", None)

    return command, meaning, situation, target_commands, manner


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


def initialize_world(
    situation: Situation,
    mission="",
    manner=None,
    verb=None,
    end_pos=None,
    required_push=0,
    required_pull=0,
    num_instructions=0,
):
    """
    Initializes the world with the passed situation.
    :param situation: class describing the current situation in the world, fully determined by a grid size,
    agent position, agent direction, list of placed objects, an optional target object and optional carrying object.
    :param mission: a string defining a command (e.g. "Walk to a green circle."
    :param manner: a string defining the manner of the mission (e.g., "while spinning")
    :param verb: a string defining the transitive verb of the mission (e.g., "push")
    :param end_pos: position tuple of where the agent and target object should end up (e.g., (3, 4))
    :param required_push: amount of push-actions required in a correct demonstration of this mission (e.g., 2)
    :param required_pull: amount of pull-actions required in a correct demonstration of this mission (e.g., 0)
    :param num_instructions: amount of actions in expert target demonstration
    """
    vocabulary = Vocabulary.initialize(
        intransitive_verbs=(INTRANSITIVE_VERBS).split(","),
        transitive_verbs=(TRANSITIVE_VERBS).split(","),
        adverbs=(ADVERBS).split(","),
        nouns=(NOUNS).split(","),
        color_adjectives=(COLOR_ADJECTIVES).split(","),
        size_adjectives=(SIZE_ADJECTIVES).split(","),
    )
    world = create_world(vocabulary)

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


def demonstrate_target_commands(
    command: str, initial_situation: Situation, target_commands: List[str]
) -> Tuple[List[str], List[Situation], int, int]:
    """Executes a sequence of commands starting from initial_situation."""
    # Initialize the world based on the initial situation and the command.
    world = initialize_world(initial_situation, mission=command)

    for target_command in target_commands:
        world.execute_command(target_command)

    target_commands, target_demonstration = world.get_current_observations()
    end_column, end_row = world.agent_pos

    return target_commands, target_demonstration, end_column, end_row


def parse_sparse_situation(
    situation_representation: dict, grid_size: int
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
    num_object_attributes = len(
        [int(bit) for bit in situation_representation["target_object"]["vector"]]
    )
    # Object representation + agent bit + agent direction bits (see docstring).
    num_grid_channels = num_object_attributes + 1 + 4

    # attribute bits + agent + agent direction
    num_grid_channels = 5

    # Initialize the grid.
    grid = np.zeros([grid_size, grid_size, num_grid_channels], dtype=int)

    # Place the agent.
    agent_row = int(situation_representation["agent_position"]["row"])
    agent_column = int(situation_representation["agent_position"]["column"])
    agent_direction = int(situation_representation["agent_direction"])
    agent_representation = np.zeros([num_grid_channels], dtype=np.int)
    agent_representation[-2] = 1
    agent_representation[-1] = agent_direction
    grid[agent_row, agent_column, :] = agent_representation

    # Loop over the objects in the world and place them.
    for placed_object in situation_representation["placed_objects"].values():
        object_vector = np.array(
            [int(bit) for bit in placed_object["vector"]], dtype=np.int
        )
        object_row = int(placed_object["position"]["row"])
        object_column = int(placed_object["position"]["column"])
        grid[object_row, object_column, 0] = (
            0 if (object_vector[:4] == 0).all() else (np.argmax(object_vector[:4]) + 1)
        )
        grid[object_row, object_column, 1] = (
            0
            if (object_vector[4:7] == 0).all()
            else (np.argmax(object_vector[4:7]) + 1)
        )
        grid[object_row, object_column, 2] = (
            0
            if (object_vector[7:10] == 0).all()
            else (np.argmax(object_vector[7:10]) + 1)
        )
    return grid


def labelled_situation_to_demonstration_tuple(
    labelled_situation, input_word2idx, action_word2idx
):
    (
        target_commands,
        target_demonstration,
        end_column,
        end_row,
    ) = demonstrate_target_commands(
        labelled_situation["input"],
        labelled_situation["situation"],
        labelled_situation["target"],
    )

    return (
        np.array([input_word2idx[w] for w in labelled_situation["input"]]),
        np.array([action_word2idx[w] for w in target_commands]),
        np.stack(
            [
                parse_sparse_situation(t.to_representation(), t.grid_size)
                for t in target_demonstration
            ]
        ),
    )


def yield_situations(d, split):
    for data_example in tqdm(d["examples"][split]):
        command, meaning, situation, target_commands, manner = parse_example(
            data_example
        )
        yield {"input": command, "target": target_commands, "situation": situation}


def yield_demonstrations(labelled_situations):
    for labelled_situation in labelled_situations:
        yield labelled_situation_to_demonstration_tuple(labelled_situation)
