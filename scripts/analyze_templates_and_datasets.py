import argparse
import itertools
import json
import numpy as np
import re

from generate_gscan_dataset_natural_language import parse_chatgpt_response_into_list

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


REPLACE_WITH_TEMPLATE = ("big", "small", "red", "green", "blue", "yellow", "circle", "square", "cylinder")
REPLACE_WITH_TEMPLATE_REGEX = re.compile(f"((?:(?:{'|'.join(REPLACE_WITH_TEMPLATE)})\\s?)+)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-examples", type=int, default=3)
    args = parser.parse_args()

    responses = load_json(args.templates)
    dataset = load_json(args.dataset)

    print(f"Examples: {len(dataset['examples']['train'])}")

    uniq_in_train = set(list(map(lambda x: x['command'].replace(",", " "), dataset['examples']['train'])))
    print(f"Unique Instructions: {len(uniq_in_train)}")

    uniq_gen_instrs = set(list(itertools.chain.from_iterable([
        parse_chatgpt_response_into_list(r)[-1]
        for r in responses if r["orig_instruction"] in uniq_in_train
    ])))
    print(f"Unique Generated Instructions: {len(uniq_gen_instrs)}")

    uniq_seed_templates = set(list(map(lambda s: REPLACE_WITH_TEMPLATE_REGEX.sub("OBJ ", s).strip(), uniq_in_train)))
    uniq_gen_templates = set(list(map(lambda s: REPLACE_WITH_TEMPLATE_REGEX.sub("OBJ ", s).strip(), uniq_gen_instrs)))

    print(f"Seed Templates: {len(uniq_seed_templates)}")
    print(f"Gen Templates: {len(uniq_gen_templates)}")

    example_indices = np.random.permutation(len(responses))[:args.num_examples]

    print("Sample Responses:")

    for index in example_indices:
        print("ORIG", responses[index]["orig_instruction"])
        print("PP", responses[index]["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()