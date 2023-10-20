import argparse
import json
import os
import fnmatch
import itertools
from tqdm import tqdm

def read_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toplevel-data-directory", required=True)
    parser.add_argument("--dataset-filename-pattern", required=True)
    parser.add_argument("--write-sentences", required=True)
    args = parser.parse_args()

    filenames_to_extract_from = fnmatch.filter(itertools.chain.from_iterable([
        [os.path.join(root, filename) for filename in filenames]
        for root, dirnames, filenames in os.walk(args.toplevel_data_directory)
    ]), args.dataset_filename_pattern)

    print(json.dumps(filenames_to_extract_from, indent=2))

    sentences = sorted(list(set(list(itertools.chain.from_iterable([
        list(set(itertools.chain.from_iterable([
            list(set([
                example["command"].replace(",", " ")
                for example in tqdm(examples, desc=split)
            ]))
            for split, examples in tqdm(read_json(path)["examples"].items(), desc=path)
        ])))
        for path in tqdm(filenames_to_extract_from, desc="Total")
    ])))))

    import pdb
    pdb.set_trace()

    with open(args.write_sentences, "w") as f:
        json.dump(sentences, f, indent=2)


if __name__ == "__main__":
    main()
