import argparse
import json
import itertools
import functools
import tqdm
import re
import numpy as np
import pandas as pd
import pprint
import os
from gscan_metaseq2seq.util.load_data import load_data_directories

def gscan_encode_individual_state(s, idx2color, idx2object):
    size, color, obj, agent, direction, y, x = s

    if agent == 1:
        return f"agent d: {direction} x: {x} y: {y}"

    return f"{idx2color[color - 1]} {idx2object[obj - 1]} s: {size} x: {x} y: {y}"


def reascan_gscan_encode_individual_state(s, idx2color, idx2object):
    size, color, obj, agent, direction, y, x, box_size, box_color, is_box = s

    if is_box == 1:
        return f"{idx2color[box_color - 1]} box s: {box_size} x: {x} y {y}"

    if agent == 1:
        return f"agent d: {direction} x: {x} y: {y}"

    return f"{idx2color[color - 1]} {idx2object[obj - 1]} s: {size} x: {x} y: {y}"


STATE_PROFILES = {
    "gscan": gscan_encode_individual_state,
    "reascan": reascan_gscan_encode_individual_state
}

STATE_PROFILE_SORT = {
    "gscan": lambda x: sorted(x, key=lambda y: (y[-2], y[-1])),
    "reascan": lambda x: sorted(x, key=lambda y: (y[-5], y[-4])),
}


def encode_state(state, idx2color, idx2object, state_profile):
    return ", ".join(map(lambda s: STATE_PROFILES[state_profile](s, idx2color, idx2object),
                         STATE_PROFILE_SORT[state_profile](list(filter(lambda s: (s != 0).any(), state)))))


def encode_instr(instr, idx2word, word2idx):
    return " ".join([idx2word[w] for w in instr if w != word2idx['[pad]']])


def encode_targets(targets, idx2action, action2idx):
    return " ".join([idx2action[w] for w in targets if w != action2idx['[pad]']])


def handle_is_metalearn(example):
    if len(example) == 3:
        return (*example, [], [], [], [])

    return example

def convert_to_text_representation(example, idx2word, idx2action, word2idx, action2idx, idx2color, idx2object, state_profile, max_examples=16, reverse=False):
    query_instr, query_targets, query_state, support_states, support_instrs, support_targets, scores = handle_is_metalearn(example)

    scores = np.array(scores)
    rev_stride = -1 if reverse else 1

    # highest scores go last
    support_instrs = [support_instrs[i] for i in scores.argsort()[::rev_stride]]
    support_targets = [support_targets[i] for i in scores.argsort()[::rev_stride]]

    if len(support_states) > 0 and isinstance(support_states[0], np.ndarray):
        state_line = f"State: {encode_state(query_state, idx2color, idx2object, state_profile)}\n"
        return ("Complete based on the following. Base the answer on Inputs Output pairs that are relevant to the Query Input:\n" + state_line + "\n".join([
            f"Input: {encode_instr(instr, idx2word, word2idx)}\nOutput: {encode_targets(targets, idx2action, action2idx)}"
            for instr, targets in zip(
                support_instrs[-max_examples:],
                support_targets[-max_examples:]
            )
        ]) + f"\nQuery Input: {encode_instr(query_instr, idx2word, word2idx)}\nOutput:", encode_targets(query_targets, idx2action, action2idx))

    support_states = [support_states[i] for i in scores.argsort()[::rev_stride]]

    state_str = f"State: {encode_state(query_state, idx2color, idx2object, state_profile)}\n"
    query_input_str = f"Query Input: {encode_instr(query_instr, idx2word, word2idx)}"
    supports_str = "\n".join([
        f"State: {encode_state(state, idx2color, idx2object, state_profile)}\nInput: {encode_instr(instr, idx2word, word2idx)}\nOutput: {encode_targets(targets, idx2action, action2idx)}"
        for state, instr, targets in zip(
            support_states[-max_examples:],
            support_instrs[-max_examples:],
            support_targets[-max_examples:]
        )
    ])

    return "\n".join([
        "Complete based on the following. Base the answer on Input Output pairs that are relevant to the Query Input:",
        supports_str,
        state_str,
        query_input_str,
        "Output:"
    ]), encode_targets(query_targets, idx2action, action2idx)


def print_through(val):
    tqdm.tqdm.write(pprint.pformat(val))
    return val


def stream_json_list_to_file(iterable, f, append=False):
    if not append:
        f.write("[\n  ")
    for i, element in enumerate(iterable):
        if i != 0 or append:
            f.write(",\n  ")
        json.dump(element, f)
    f.write("]\n")


def try_json_parse(line):
    try:
        return json.loads(line.strip().rstrip(","))
    except json.decoder.JSONDecodeError as e:
        import pdb
        pdb.set_trace()
        return None


def check_already_done(path):
    try:
        with open(path, "r") as f:
            file_lines = f.readlines()
    except FileNotFoundError:
        return []

    import pdb
    pdb.set_trace()

    return map(lambda obj: obj["orig_instruction"],
               filter(lambda x: x,
                      map(lambda line: try_json_parse(line),
                          file_lines)))


action_remap = {
    'turn left': 'lturn',
    'turn right': 'rturn'
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="List of input sentences")
    parser.add_argument("--output-responses", type=str, required=True, help="Where to write outputs")
    parser.add_argument("--only-splits", nargs="*")
    parser.add_argument("--limit-load", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-examples", type=int, default=8)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--state-profile", type=str, choices=STATE_PROFILES.keys(), required=True)
    args = parser.parse_args()

    (
        dictionaries,
        (train_demonstrations, valid_demonstrations_dict),
    ) = load_data_directories(
        args.dataset,
        os.path.join(args.dataset, "dictionary.pb"),
        limit_load=args.limit_load,
        only_splits=args.only_splits
    )

    word2idx, action2idx, idx2color, idx2object = dictionaries
    idx2word = list(word2idx.keys())
    idx2action = list(map(lambda x: action_remap.get(x, x), action2idx.keys()))
    #idx2action = list(map(str, range(len(idx2action))))
    #idx2action[action2idx['[pad]']] = '[pad]'
    #action2idx = {a: i for i, a in enumerate(idx2action)} # 

    idx2color = sorted(idx2color)
    idx2object = sorted(idx2object)

    os.makedirs(args.output_responses, exist_ok=True)

    for key, examples in {
        "train": train_demonstrations,
        **valid_demonstrations_dict
    }.items():
        if len(examples) == 0:
            df = pd.DataFrame(columns=["text", "label"])
        else:
            df = pd.DataFrame(
                list(map(
                    lambda e: convert_to_text_representation(
                        e,
                        idx2word,
                        idx2action,
                        word2idx,
                        action2idx,
                        idx2color,
                        idx2object,
                        args.state_profile,
                        args.limit_examples,
                        args.reverse
                    ),
                    tqdm.tqdm(examples[:args.limit])
                ))
            )
            df.columns = ["text", "label"]
        df.to_csv(os.path.join(args.output_responses, f"{key}.tsv"), sep='\t')


if __name__ == "__main__":
    main()
