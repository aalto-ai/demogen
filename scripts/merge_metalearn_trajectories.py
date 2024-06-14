import argparse
import os
import itertools
import re
import pickle
import pprint
import shutil
import functools

from tqdm.auto import tqdm


def assemble_files_and_splits(directories):
    directory_split_subdirectories = {
        directory: list(
            filter(
                lambda x: os.path.isdir(os.path.join(directory, x)),
                os.listdir(directory),
            )
        )
        for directory in directories
    }

    all_splits = sorted(
        list(
            set(
                list(
                    itertools.chain.from_iterable(
                        directory_split_subdirectories.values()
                    )
                )
            )
        )
    )

    all_split_paths = {
        s: sorted(
            list(
                itertools.chain.from_iterable(
                    map(
                        lambda toplevel: map(
                            lambda x: os.path.join(toplevel, s, x),
                            os.listdir(os.path.join(toplevel, s)),
                        ),
                        directory_split_subdirectories.keys(),
                    )
                )
            ),
            key=lambda x: list(map(int, re.match(r".*_(\d+).*?(\d+).pb", x).groups())),
        )
        for s in all_splits
    }

    return all_split_paths


def read_pb_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def postprocess(item, postprocess_limit):
    query, targets, query_state, support_state, support_x, support_y, score = item
    sort_idx = sorted(list(range(len(support_x))), key=lambda i: -score[i])[
        :postprocess_limit
    ]

    if isinstance(support_state, list):
        support_state = [support_state[i] for i in sort_idx]

    support_x = [support_x[i] for i in sort_idx]
    support_y = [support_y[i] for i in sort_idx]

    return (query, targets, query_state, support_state, support_x, support_y, score)


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))

        if not batch:
            break

        yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directories", nargs="+")
    parser.add_argument("--limit", default=32, type=int)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--batch-size", default=10000, type=int)
    args = parser.parse_args()

    all_split_paths = assemble_files_and_splits(args.directories)

    pprint.pprint(all_split_paths)

    os.makedirs(args.output_directory, exist_ok=True)
    shutil.copyfile(args.dictionary, os.path.join(args.output_directory, "dictionary.pb"))

    for split, paths in tqdm(all_split_paths.items()):
        os.makedirs(os.path.join(args.output_directory, split), exist_ok=True)

        for i, batch in enumerate(
            batched(
                itertools.chain.from_iterable(map(read_pb_file, tqdm(paths))),
                args.batch_size,
            )
        ):
            with open(os.path.join(args.output_directory, split, f"{i}.pb"), "wb") as f:
                pickle.dump(batch, f)


if __name__ == "__main__":
    main()
