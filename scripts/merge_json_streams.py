import argparse
import json_stream
import os
from json_stream import streamable_dict, streamable_list
from json_stream.dump import JSONStreamEncoder
import json

SPLITS = ["a", "b", "c", "d", "train"]

def try_get(stream, key, default):
    try:
        return stream[key]
    except json_stream.base.TransientAccessException:
        return default

@streamable_list
def generate_examples_from_stream_splits(split, streams):
    for tuple_of_examples in zip(*[try_get(stream, split, []) for stream in streams]):
        for example in tuple_of_examples:
            yield example.persistent()

@streamable_dict
def generate_splits_from_streams(streams, splits):
    for split in splits:
        print(f"Generating for split {split}")
        yield split, generate_examples_from_stream_splits(split, streams)


@streamable_dict
def generate_from_input_streams(streams, splits):
    yield "examples", generate_splits_from_streams(streams, splits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", nargs="+", type=str)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    args = parser.parse_args()

    load_str = f"Loading: " + '\n'.join(args.input_files)
    print(load_str)

    streams = [
        json_stream.load(open(input_file, "r"))["examples"]
        for input_file in args.input_files
    ]

    with open(args.output_file, "w") as f:
        json.dump(
            generate_from_input_streams(streams, args.splits),
            f,
            cls=JSONStreamEncoder
        )


if __name__ == "__main__":
    main()