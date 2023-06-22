import argparse
import json
import os
import re
import pprint
from typing import Any, List, Tuple
import random
import tqdm


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_chatgpt_response_into_list(chatgpt_response) -> Tuple[str, List[str]]:
    return (
        chatgpt_response["orig_instruction"],
        [
            re.split(r"\d+\.", l, maxsplit=1)[1].lower().strip()
            for l in chatgpt_response["choices"][0]["message"]["content"].splitlines()
            if l and not l.endswith(":")
        ]
    )


def parse_command(command):
    return command.replace(",", " ")


def to_command(command):
    return command.replace(" ", ",")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="The original gscan compositional splits dataset")
    parser.add_argument("--paraphrases-outputs", required=True, help="JSON outputs containing list of paraphrases from ChatGPT")
    parser.add_argument("--dataset-output", required=True, help="Where to save the resulting dataset")
    args = parser.parse_args()

    sentences, paraphrases_outputs = list(zip(*list(map(parse_chatgpt_response_into_list, read_json(args.paraphrases_outputs)))))

    sentence_paraphrases_map = {
        sentence: [sentence] + paraphrases
        for sentence, paraphrases in zip(sentences, paraphrases_outputs)
    }

    print(json.dumps(sentence_paraphrases_map, indent=2))

    dataset_json = read_json(args.dataset)
    dataset_examples = dataset_json["examples"]

    del dataset_json["examples"]

    with open(args.dataset_output, "w") as f:
        json.dump({
            **dataset_json,
            "examples": {
                split: [
                    {
                        **example,
                        "command": to_command(random.choice(sentence_paraphrases_map[parse_command(example["command"])]))
                    }
                    for example in tqdm.tqdm(examples, desc=split)
                ]
                for split, examples in dataset_examples.items()
            }
        }, f, indent=2)


if __name__ == "__main__":
    main()