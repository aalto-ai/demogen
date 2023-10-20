import argparse
import openai
import json
import itertools
import functools
import tqdm
import re
import pprint
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)

PROMPT_WITH_ADVERB = """
Here are 10 similar statements to “push a red square cautiously”

Cautiously push a red square
Push a red square with caution
Heave a red square with caution
Push a red square while looking
Go to the red square and push it cautiously
Walk to the red square and push it cautiously
Find a red square and push it cautiously
Shove a red square cautiously
Go to a red square and push it with caution
Go to a red square and push it and be cautious
"""

PROMPT_WITHOUT_ADVERB = """
Here are 10 similar statements to “push a red square"

Push the red square
Move a red square
Shove the red square
Go to the red square and shove it
Go to the red square and push it
Walk to the red square and push it
Find a red square and push it
Locate a red square and push it
Get to the red square and move it along
Walk up to the red square and then really push it
"""

PROMPT_SPATIAL_RELATIONS = """
Here are 10 similar statements to "push a red circle that is south east of a blue circle"

Find the red circle south east of the blue circle and push it
First find the blue circle then push the red circle south west of it
Shove the red circle on the south east of the blue circle
There is a red circle south east of the blue circle, push it
Locate the blue circle, now look south east and find the red circle, then push the red circle
Move a red circle south east of a blue circle forward
Push a red circle that is south of a blue circle and east of a blue circle
Push a red circle that has a blue circle north west of it
There is a red circle with a blue circle north west of it, push it
Locate the red circle which has a blue circle to the north west, then push it
"""

PROMPT_REASCAN = """
Here are 10 similar statements to "pull the yellow square that is inside of a big red box and in the same row as a small red circle and in the same column as a small cylinder while spinning"

Take hold of the large red box's yellow square that shares a row with a tiny red circle and a column with a little cylinder while spinning
Grasp the yellow square from the big red box that's on the same row as a small red circle and the same column as a small cylinder and spin
While twirling, retrieve the yellow square situated on the same row as a small red circle and the same column as a small cylinder from the big red box
Seize the yellow square on the same row as a small red circle and in the same column as a small cylinder from the big red box and spin around
Get a grip on the yellow square that's in the same row as a small red circle and in the same column as a small cylinder from the large red box but keep turning as you approach
Pick up the yellow square that's on the same row as a small red circle and in the same column as a small cylinder from the big red box and spin on each step
Take the yellow square from the big red box (while spinning around) that's located on the same row as a small red circle and in the same column as a small cylinder
Extract the yellow square (while twirling) from the big red box that's situated on the same row as a small red circle and in the same column as a small cylinder
Remove the yellow square from the big red box, which is located on the same row as a small red circle and in the same column as a small cylinder
Spin around and pull out the yellow square that is on the same row as a small red circle and in the same column as a small cylinder from the big red box
Retrieve the yellow square from the big red box that shares a row with a small red circle and a column with a small cylinder while spinning
"""

TEMPLATE = "Can you generate 25 similar statements for {tgt} in English?"

PROMPT_CHOICES = {
    "simple": PROMPT_WITHOUT_ADVERB,
    "with_adverb": PROMPT_WITH_ADVERB,
    "relational": PROMPT_SPATIAL_RELATIONS,
    "reascan": PROMPT_REASCAN
}

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def make_prompt(prompt, sentence):
    return prompt + "\n" + TEMPLATE.format(tgt=sentence)


PLACEHOLDER_WORDS = ("small", "big", "object", "red", "blue", "green", "yellow", "box", "cylinder", "square", "circle")

def sentence_to_placeholder_pair(sentence):
    placeholder_phrases = [[]]
    sentence_words = []
    for w in sentence.split():
        if w in PLACEHOLDER_WORDS:
            placeholder_phrases[-1].append(w)
        else:
            if placeholder_phrases[-1]:
                sentence_words.append("TGT" + str(len(placeholder_phrases)))
                placeholder_phrases.append([])

            sentence_words.append(w)

    if placeholder_phrases[-1]:
        sentence_words.append("TGT" + str(len(placeholder_phrases)))
    else:
        placeholder_phrases = placeholder_phrases[:-1]

    return " ".join(sentence_words), list(map(lambda x: " ".join(x), placeholder_phrases))


def replace_placeholders_in(template_text, placeholders):
    for i, placeholder in enumerate(placeholders):
        template_text = template_text.replace(
            f"TGT{i + 1}", placeholder
        )

    return template_text


def print_through(val):
    tqdm.tqdm.write(pprint.pformat(val))
    return val



def expand_templates(placeholder_instruction_to_placeholders, original_instructions, pair):
    instruction, response = pair
    expanded_sentences, responses_contents = list(zip(*
        map(
            lambda placeholders: (
                replace_placeholders_in(instruction, placeholders),
                replace_placeholders_in(
                    response.to_dict()["choices"][0]["message"]["content"],
                    placeholders
                )
            ),
            placeholder_instruction_to_placeholders[instruction]
        ),
    ))
    responses = [
        {
            "orig_instruction": expanded_instruction,
            **response,
            "choices": [
                {
                    **response["choices"][0],
                    "message": {
                        **response["choices"][0]["message"],
                        "content": response_content
                    }
                }
            ]
        }
        for expanded_instruction, response_content in zip(expanded_sentences, responses_contents)
    ]
    assert all([s in original_instructions for s in expanded_sentences])

    return responses


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--sentence-list-input", type=str, required=True, help="List of input sentences")
    parser.add_argument("--responses-list-output", type=str, required=True, help="Where to write chatgpt responses")
    parser.add_argument("--mode", choices=("paraphrase_all", "paraphrase_placeholder"), default="paraphrase_all")
    parser.add_argument("--prompt", choices=("simple", "with_adverb", "relational", "reascan"), default="simple")
    args = parser.parse_args()

    openai.api_key = args.api_key

    with open(args.sentence_list_input) as f:
        original_instructions = json.load(f)

    already_done_set = set(list(check_already_done(args.responses_list_output)))
    original_instructions = list(filter(lambda x: x not in already_done_set, original_instructions))

    if args.mode == "paraphrase_placeholder":
        instructions, grouped_paraphrase_placeholders = list(zip(*map(
            lambda x: (x[0], list(map(lambda y: y[1], x[1]))),
            itertools.groupby(
                sorted(
                    list(map(sentence_to_placeholder_pair, original_instructions)),
                    key=lambda x: x[0]
                ),
                key=lambda x: x[0]
            )
        )))
        placeholder_instruction_to_placeholders = dict(zip(instructions, grouped_paraphrase_placeholders))
    else:
        instructions = original_instructions

    prompt = PROMPT_CHOICES[args.prompt]

    print(prompt)
    print(f"Generating paraphrases for {len(instructions)} instructions")

    with open(args.responses_list_output, "a" if already_done_set else "w") as f:
        expand_func = (
            functools.partial(expand_templates, placeholder_instruction_to_placeholders, original_instructions)
            if args.mode == "paraphrase_placeholder"
            else lambda x: [{**x[1].to_dict(), "orig_instruction": x[0]}]
        )
        stream_json_list_to_file(
            itertools.chain.from_iterable(
                map(expand_func,
                    map(
                        lambda instruction: (instruction, print_through(completion_with_backoff(model="gpt-3.5-turbo", messages=[{"role": "user", "content": make_prompt(prompt, print_through(instruction))}], temperature=0))),
                        tqdm.tqdm(instructions)
                    )
                )
            ),
            f,
            append=True if already_done_set else False
        )


if __name__ == "__main__":
    main()