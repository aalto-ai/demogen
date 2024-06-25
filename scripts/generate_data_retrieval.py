import argparse
from bisect import bisect_left
import json
import itertools
import numpy as np
from typing import List, Optional, Tuple
import os
from collections import defaultdict
import pickle
import multiprocessing
import random
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler, normalize
import functools

from tqdm.auto import tqdm, trange
from gscan_metaseq2seq.util.load_data import load_data_directories


def stack_as_passed(numpy_arrays):
    max_len = max([len(a) for a in numpy_arrays])
    return np.stack([
        np.concatenate([
            a, np.zeros_like(a)[:1].repeat(max_len - len(a), axis=0)
        ], axis=0)
        for a in numpy_arrays
    ])


def compute_sorted_bsr(
    query_embeds,
    demonstration_embeds,
    target_limit
):
    indices = []
    coverage_value = float('-inf')

    examples_list = []
    coverage_bits_list = []
    lengths_list = []

    demonstration_embeds_pad = stack_as_passed(demonstration_embeds)
    current_matching = query_embeds[None] @ demonstration_embeds_pad.transpose(0, 2, 1)
    current_matching_max_scores = (current_matching.max(axis=-1) > 0.8)

    # Pick examples that gets us coverage for every query token
    # but also maximizes the total weight.
    #
    # We can do this with an iterative algorithm that keeps track of
    # each token's coverage. We remove the "lightest" example from
    # the set on each iteration as long as it wouldn't break the overall
    # coverage
    current_overall_token_coverages = current_matching_max_scores.sum(axis=0)
    token_coverage_weights = current_matching_max_scores.sum(axis=-1)
    examples_in_set = np.arange(len(demonstration_embeds))
    examples_in_set_sorted_by_weights = examples_in_set[token_coverage_weights.argsort()]
    zero_mask = current_overall_token_coverages == 0

    while examples_in_set_sorted_by_weights.shape[0] > target_limit:
        removed = False

        for i in range(examples_in_set_sorted_by_weights.shape[0]):
            # Check if removing this example would make anything newly-zero in the coverage
            index = examples_in_set_sorted_by_weights[i]
            scores = current_matching_max_scores[index].astype(int)
            if (
                ((current_overall_token_coverages - scores) == 0) ^
                zero_mask
            ).sum() > 0:
                continue

            # Otherwise, we can remove this example
            examples_in_set_sorted_by_weights = np.concatenate([
                examples_in_set_sorted_by_weights[:i],
                examples_in_set_sorted_by_weights[i + 1:]
            ], axis=0)
            current_overall_token_coverages = current_overall_token_coverages - scores
            removed = True
            break

        # We could not remove anything without reducing coverage. Lets return the whole set
        if not removed:
            break

    examples_in_set_sorted_by_weights = list(reversed(examples_in_set_sorted_by_weights))

    # Finally, remove any coverage-dupes (eg, same coverage
    # as the one before)
    exact_dupe_indices = [
        i for i, first, second in zip(
            range(1, len(examples_in_set_sorted_by_weights)),
            examples_in_set_sorted_by_weights[:-1],
            examples_in_set_sorted_by_weights[1:]
        )
        if (current_matching_max_scores[first] == current_matching_max_scores[second]).all()
    ]

    examples_in_set_sorted_by_weights = [
        examples_in_set_sorted_by_weights[i]
        for i in range(len(examples_in_set_sorted_by_weights))
        if i not in exact_dupe_indices
    ]

    return examples_in_set_sorted_by_weights


def compute_sorted_set_bsr(
    query_embeds,
    demonstration_embeds
):
    indices = []
    coverage_value = float('-inf')

    examples_list = []
    coverage_bits_list = []
    lengths_list = []

    for i, demo_embed in enumerate(demonstration_embeds):
        current_matching = query_embeds @ demonstration_embeds[i].T
        current_matching_max_scores = (current_matching.max(axis=-1) > 0.8)
        mask = np.zeros_like(current_matching_max_scores)

        index = 0
        
        for j in range(len(coverage_bits_list)):
            index = j
            # Check if it covers more than what's in the current list index
            disjunction = np.logical_xor(
                current_matching_max_scores,
                coverage_bits_list[j]
            ) & ~mask
            unique_current = np.logical_and(disjunction, current_matching_max_scores)
            unique_in_list = np.logical_and(disjunction, coverage_bits_list[j])

            if unique_current.sum() > coverage_bits_list[j].sum():
                break

            # If this example would give the same coverage, check
            # to see if it is shorter. If so, we should prefer it over
            # the one currently in the set, on the intuition that shorter
            # examples probably contain less distracting information
            if unique_current.sum() == coverage_bits_list[j].sum():
                if demonstration_embeds[i].shape[0] < lengths_list[j]:
                    break

            mask |= unique_in_list

        j = index

        # Would this add anything new? If not, not worth it
        if np.logical_and(~mask, current_matching_max_scores).any():
            # Insert the current matching in the list
            examples_list.insert(j, i)
            lengths_list.insert(j, demonstration_embeds[i].shape[0])
            coverage_bits_list.insert(j, current_matching_max_scores)

            try:
                coverage_list_cumsum = np.concatenate([
                    np.zeros_like(current_matching_max_scores)[None],
                    coverage_bits_list
                ]).cumsum(axis=0).astype(bool)
                coverage_list_stack = np.concatenate([
                    coverage_bits_list,
                    np.zeros_like(current_matching_max_scores)[None],
                ], axis=0)
                coverage_list_land = np.logical_and(
                    ~coverage_list_cumsum,
                    coverage_list_stack
                ).sum(axis=-1)[:-1]
            except:
                import pdb
                pdb.set_trace()
            
            len_examples_list = len(examples_list)
            
            examples_list = [
                examples_list[j]
                for j in range(len_examples_list)
                if coverage_list_land[j]
            ]
            lengths_list = [
                lengths_list[j]
                for j in range(len_examples_list)
                if coverage_list_land[j]
            ]
            coverage_bits_list = [
                coverage_bits_list[j]
                for j in range(len_examples_list)
                if coverage_list_land[j]
            ]
            
            assert len(examples_list) == len(coverage_bits_list)

    return examples_list


def retrieve_layout_instruction_coverage(
    index, command, target_commands, state, payload, options
):
    # Similar to Gupta et al 2023, we walk down the list and try to greedily
    # find things that would maximize coverage of the query according to the
    # "bert-score"
    (
        retrievals,
        train_examples,
        word2idx,
        train_unique_indices,
        train_unique_encodings,
        train_unique_token_encodings,
        split_sentences_to_unique_index,
        split_sentences_unique_all_token_encodings
    ) = payload

    token_embeddings = split_sentences_unique_all_token_encodings[split_sentences_to_unique_index[index]]

    retrievals_from_training_set = retrievals[index]
    filtered_retrievals_from_training_set = retrievals_from_training_set[
        np.array([
            command.shape[0] != train_examples[i][0].shape[0] or (command != train_examples[i][0]).any()
            for i in retrievals_from_training_set
        ])
    ]

    # If we just don't get any retrievals, then we have to use the originals
    if filtered_retrievals_from_training_set.shape[0] == 0:
        filtered_retrievals_from_training_set = retrievals_from_training_set

    retrievals_from_training_set = filtered_retrievals_from_training_set

    retrieved_token_encodings = [
        train_unique_token_encodings[train_unique_indices[i]]
        for i in retrievals_from_training_set
    ]
    sorted_examples_by_set_coverage = np.array(compute_sorted_bsr(token_embeddings, retrieved_token_encodings, 16))
    retrievals_from_training_set[sorted_examples_by_set_coverage]

    selected_examples = retrievals_from_training_set[:16]

    return (
        None,
        tuple(list(zip(*[
            train_examples[i]
            for i in selected_examples
        ])))
    )


def retrieve_layout_by_score(
    index, command, target_commands, state, payload, options
):
    (
        retrievals,
        train_examples,
        word2idx,
        train_unique_indices,
        train_unique_encodings,
        train_unique_token_encodings,
        split_sentences_to_unique_index,
        split_sentences_unique_all_token_encodings
    ) = payload
    idx2word = [w for w in word2idx]

    retrievals_from_training_set = retrievals[index]
    filtered_retrievals_from_training_set = retrievals_from_training_set[
        np.array([
            command != train_examples[i][0]
            for i in retrievals_from_training_set
        ])
    ]

    # If we just don't get any retrievals, then we have to use the originals
    if filtered_retrievals_from_training_set.shape[0] == 0:
        filtered_retrievals_from_training_set = retrievals_from_training_set

    retrievals_from_training_set = filtered_retrievals_from_training_set
    selected_examples = retrievals_from_training_set[:16]

    return (
        None,
        tuple(list(zip(*[
            train_examples[i]
            for i in selected_examples
        ])))
    )


GENERATION_STRATEGIES = {
    "retrieve_similar_state": retrieve_layout_instruction_coverage
}


def generate_supports_for_data_point(
    data_example,
    index,
    generation_mode,
    generation_payload,
    generation_options,
):
    instruction, targets, state = data_example

    error, (
        support_instruction_commands,
        support_target_commands,
        support_layouts,
    ) = GENERATION_STRATEGIES[generation_mode](
        index,
        instruction,
        targets,
        state,
        generation_payload,
        generation_options,
    )

    return (
        error,
        (
            instruction,
            targets,
            state,
            support_instruction_commands,
            support_target_commands,
            support_layouts,
        ),
    )


def generate_supports_for_data_point_star(args):
    return generate_supports_for_data_point(*args)


def yield_metalearning_examples(
    examples_set,
    generation_mode,
    generation_payload,
    generation_options,
    n_procs=8,
):
    if generation_options.get("can_parallel", False):
        # Fast path for generation, we don't need to keep the whole dataset
        # in memory in order to search for matching examples
        with multiprocessing.Pool(processes=n_procs) as pool:
            for error, result in pool.imap_unordered(
                generate_supports_for_data_point_star,
                map(
                    lambda x, i: (
                        x,
                        i,
                        generation_mode,
                        generation_payload,
                        generation_options,
                    ),
                    enumerate(examples_set),
                ),
                chunksize=100,
            ):
                if error is not None:
                    tqdm.write(error)
                    continue

                yield result
    else:
        # Slow path, we need to keep the whole dataset
        for i, example in enumerate(examples_set):
            error, result = generate_supports_for_data_point(
                example,
                i,
                generation_mode,
                generation_payload,
                generation_options,
            )

            if error is not None:
                tqdm.write(error)
                continue

            yield result


def encode_metalearning_example(
    example
):
    (
        command,
        target_commands,
        state,
        support_instructions,
        support_targets,
        support_states,
    ) = example

    if not support_instructions:
        support_instructions = np.array([], dtype=int)
        support_targets = np.array([], dtype=int)

    return (
        command,
        target_commands,
        state,
        state if not support_states else support_states,
        support_instructions,
        support_targets,
        # Priorities, which in this case, are always ordered.
        np.array(list(reversed(range(len(support_instructions))))),
    )


def encode_metalearning_examples(
    metalearning_examples
):
    for ml_example in metalearning_examples:
        yield encode_metalearning_example(
            ml_example
        )


def to_count_matrix(word_arrays, word_vocab_size):
    count_matrix = np.zeros(
        (len(word_arrays), word_vocab_size)
    )

    for i, word_array in enumerate(word_arrays):
        for element in word_array:
            count_matrix[i, element] += 1

    return count_matrix


def to_tfidf(tfidf_transformer, count_matrix):
    return tfidf_transformer.transform(count_matrix).todense().astype("float32")


def retrieve_similar_state_payload(examples, dictionaries, current_split, global_payload, params):
    model, index, max_state_component_lengths, state_scaler, state_pca, train_unique_encodings, train_unique_token_encodings, train_unique_indices = global_payload
    WORD2IDX = dictionaries[0]
    IDX2WORD = [w for w in WORD2IDX]
    ACTION2IDX = dictionaries[1]

    split_state_vectors = vectorize_all_example_situations(
        tqdm(examples[current_split], desc=f"Vectorizing examples from split {current_split}"),
        max_state_component_lengths
    )

    pca_split_state_vectors = state_pca.transform(
        state_scaler.transform(split_state_vectors)
    ).astype(np.float32)

    del split_state_vectors

    split_sentences_index_dict = defaultdict(list)
    for i, example in enumerate(tqdm(examples[current_split])):
        split_sentences_index_dict[" ".join([
            IDX2WORD[w] for w in example[0]
        ])].append(i)
    split_sentences_unique = sorted(list(split_sentences_index_dict.keys()))
    split_sentences_to_unique_index = {
        t: i for i, t in enumerate(split_sentences_unique)
    }
    split_sentences_unique_list_lookup = np.zeros(len(examples[current_split]), dtype=np.int32)
    for t, indices in split_sentences_index_dict.items():
        for i in indices:
            split_sentences_unique_list_lookup[i] = split_sentences_to_unique_index[t]

    split_sentences_unique_all_token_encodings = list(itertools.chain.from_iterable([
        list(map(lambda x: x.cpu().numpy(), model.encode(
            split_sentences_unique[batch_index * 128:(batch_index + 1) * 128],
            output_value='token_embeddings',
            normalize_embeddings=True,
            convert_to_numpy=True
        )))
        for batch_index in trange(len(split_sentences_unique) // 128 + 1)
    ]))
    split_sentences_unique_all_token_encodings = [
        v / (np.linalg.norm(v, axis=-1)[:, None] + 1e-7)
        for v in split_sentences_unique_all_token_encodings
    ]
    split_sentences_unique_all_sentence_encodings = model.encode(
        split_sentences_unique,
        normalize_embeddings=True,
        batch_size=params.sbert_batch_size
    )

    normalized_split_vectors = normalize(
        balance_dims(
            np.array(pca_split_state_vectors),
            split_sentences_unique_all_sentence_encodings[split_sentences_unique_list_lookup],
            factors=[params.retrieval_sentence_state_tradeoff]
        ),
        axis=1
    ).astype(np.float32)

    # Once we're at this point, lets release a bunch of stuff we don't need anymore
    del split_sentences_unique_all_sentence_encodings

    search_results = np.concatenate([
        index.search(normalized_split_vectors[b * 128:(b + 1) * 128], 128)[1]
        for b in trange(
            (params.limit or normalized_split_vectors.shape[0]) // 128 + 1,
            desc=f"Finding near neighbours for split {current_split}"
        )
    ], axis=0)

    return (
        search_results,
        examples["train"],
        WORD2IDX,
        train_unique_indices,
        train_unique_encodings,
        train_unique_token_encodings,
        split_sentences_unique_list_lookup,
        split_sentences_unique_all_token_encodings
    )


def vectorize_state_star(args):
    return vectorize_state(*args)

def vectorize_state(state, component_max_sizes):
    max_size_of_component = max(component_max_sizes)

    # Assume that the last two components of state are the position information
    positions_1d = state[:, -2] * component_max_sizes[-1] + state[:, -1]
    data = state[..., :-2]

    data_cube_flat = np.zeros((component_max_sizes[-2] * component_max_sizes[-1], len(component_max_sizes[:-2])))
    data_cube_flat[positions_1d] = data

    # H x W x E x C
    state_cube = (np.arange(max_size_of_component)[None, None, None].repeat(
        component_max_sizes[-2],
        axis=0
    ).repeat(
        component_max_sizes[-1],
        axis=1
    ).repeat(
        len(component_max_sizes[:-2]),
        axis=2
    ))

    flat_state_cube = state_cube.reshape(-1, *state_cube.shape[-2:])
    one_hot_state_cube = (flat_state_cube == data_cube_flat[..., None].repeat(
        max_size_of_component,
        axis=-1
    )).reshape(flat_state_cube.shape[0], -1)

    # E x C => (E x C)
    select_mask = (np.arange(max_size_of_component)[None].repeat(len(component_max_sizes[:-2]), axis=0) < (
        np.array(component_max_sizes[:-2])[:, None].repeat(max_size_of_component, axis=-1)
    )).flatten()

    # Should be (H x W) x K
    reselected_flat_state_cube = one_hot_state_cube.T[select_mask].reshape(-1, flat_state_cube.shape[0]).T

    return reselected_flat_state_cube.reshape(-1)


def vectorize_all_example_situations(examples, component_max_sizes):
    return np.stack(
        list(
            map(lambda x: vectorize_state(np.stack(x[-1]), component_max_sizes), examples)
        )
    )


# sqrt((a*2) + (b*2)) = 1
# if sqrt(a*2) == sqrt(b*2)
# sum |a| == sum |b|
# b = b / (|b| / |a|) 
def balance_dims(first, *rest, factors=None):
    first_abs_sum = np.linalg.norm(first, axis=-1)
    rest_abs_sums = [
        np.linalg.norm(r, axis=-1) for r in rest
    ]
    rest_abs_ratios = [
        first_abs_sum / rs for rs in rest_abs_sums
    ]

    return np.concatenate([
        first
    ] + [
        r * ra[..., None] * factor
        for r, ra, factor in zip(rest, rest_abs_ratios, factors or ([1] * len(rest_abs_ratios)))
    ], axis=-1)


def premultiply_balance_dimensions(vectors, dim_other):
    return vectors * (dim_other / vectors.shape[-1])


def vectorize_example_text(word_counts, tfidf, dim_state):
    unscaled_train_text_tfidf = tfidf.fit_transform(word_counts)
    # Multiply by d_pca / d_tfidf to ensure that each component has the same
    # contribution in the vector search
    return premultiply_balance_dimensions(unscaled_train_text_tfidf, dim_state)


def shift_bit_length(x):
    return 1 << ((x - 1).bit_length() - 1)


def lower_pow2(num):
    return shift_bit_length(num)


def retrieve_similar_state_global_payload(examples, dictionaries, args):
    WORD2IDX = dictionaries[0]
    IDX2WORD = [w for w in WORD2IDX]
    ACTION2IDX = dictionaries[1]

    # Determine the max length of each state component
    max_state_component_lengths = functools.reduce(
        lambda x, o: np.stack([x, o]).max(axis=0),
        map(lambda x: np.stack(x[-1]).max(axis=0), itertools.chain.from_iterable([
            examples[split] for split in examples
        ]))
    ) + 1

    train_state_vectors = vectorize_all_example_situations(
        tqdm(examples["train"], desc="Vectorizing examples"),
        max_state_component_lengths
    )

    train_state_vectors_fit_perm = np.random.permutation(train_state_vectors.shape[0])[:8192]
    state_scaler = StandardScaler()
    state_scaler.fit(train_state_vectors[train_state_vectors_fit_perm].astype(np.float32))
    sample_scaled_train_state_vectors = state_scaler.transform(train_state_vectors[train_state_vectors_fit_perm]).astype(np.float32)

    n_components = min(
        lower_pow2(sample_scaled_train_state_vectors.shape[-1]),
        args.retrieval_state_pca_dim
    )
    print(f"Fitting PCA {(sample_scaled_train_state_vectors.shape[-1], n_components)}")
    state_pca = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components)
    )
    state_pca.fit(sample_scaled_train_state_vectors)

    print(f"Applying PCA to train state vectors")
    pca_train_state_vectors = np.concatenate([
        state_pca.transform(state_scaler.transform(train_state_vectors[i * 1024:(i + 1) * 1024]).astype(np.float32))
        for i in trange(train_state_vectors.shape[0] // 1024 + 1, desc="Applying PCA")
    ], axis=0)

    # Sanity check, how well do we reconstruct the original layouts
    print(f"Sanity check: computing reconstruction error")
    reconstruction_error_sample = np.random.permutation(pca_train_state_vectors.shape[0])[:8192]
    reconstruction_error = ((
        state_pca.inverse_transform(pca_train_state_vectors[reconstruction_error_sample]) -
        state_scaler.transform(train_state_vectors[reconstruction_error_sample].astype(np.float32))
    ) ** 2).mean()

    print(f"PCA reconstruction error {reconstruction_error}")

    # Make a lookup table of train sentences to indices - this will allow
    # us to just encode the unique sentences and save some memory.
    train_sentences_index_dict = defaultdict(list)
    for i, example in enumerate(tqdm(examples["train"])):
        train_sentences_index_dict[" ".join([
            IDX2WORD[w] for w in example[0]
        ])].append(i)
    train_sentences_unique = sorted(list(train_sentences_index_dict.keys()))
    train_sentences_to_unique_index = {
        t: i for i, t in enumerate(train_sentences_unique)
    }
    train_sentences_unique_list_lookup = np.zeros(len(examples["train"]), dtype=np.int32)
    for t, indices in train_sentences_index_dict.items():
        for i in indices:
            train_sentences_unique_list_lookup[i] = train_sentences_to_unique_index[t]


    model = SentenceTransformer('all-mpnet-base-v2')
    # We have to do batching ourselves because otherwise sbert holds everything in the GPU memory
    train_sentences_unique_all_token_encodings = list(itertools.chain.from_iterable([
        list(map(lambda x: x.cpu().numpy(), model.encode(
            train_sentences_unique[batch_index * 128:(batch_index + 1) * 128],
            output_value='token_embeddings',
            normalize_embeddings=True,
            convert_to_numpy=True
        )))
        for batch_index in trange(len(train_sentences_unique) // 128 + 1)
    ]))
    train_sentences_unique_all_token_encodings = [
        v / (np.linalg.norm(v, axis=-1)[:, None] + 1e-7)
        for v in train_sentences_unique_all_token_encodings
    ]
    print("Encoding unique sentences")
    train_sentences_unique_all_sentence_encodings = model.encode(
        train_sentences_unique,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=args.sbert_batch_size
    )

    if False:
        train_word_counts = to_count_matrix([
            np.array([word2idx[w] for w in e["command"].split(",")])
            for e in examples["train"]
        ], len(word2idx))
        train_text_tfidf = TfidfTransformer()
        train_text_tfidf.fit(train_word_counts)
        unscaled_train_text_tfidf = train_text_tfidf.transform(train_word_counts)

        unscaled_train_text_tfidf = vectorize_example_text(
            train_word_counts,
            train_text_tfidf,
            pca_train_state_vectors.shape[-1]
        )

    scaled_train_vectors = normalize(
        balance_dims(
            np.array(pca_train_state_vectors),
            train_sentences_unique_all_sentence_encodings[train_sentences_unique_list_lookup],
            factors=[args.retrieval_sentence_state_tradeoff]
        ),
        axis=1
    ).astype(np.float32)

    # Voronoi index to improve retrieval speed
    base_index = faiss.IndexFlatIP(scaled_train_vectors.shape[-1])
    index = faiss.IndexIVFFlat(base_index, scaled_train_vectors.shape[-1], 512)
    index.train(scaled_train_vectors[np.random.permutation(scaled_train_vectors.shape[0])[:512 * 40]])
    index.nprobe = 10
    index.add(scaled_train_vectors)

    # Sanity check, run NN search on some random subset of the vectors and check
    # similarity of the states
    sanity_index_sample_indices = np.random.permutation(scaled_train_vectors.shape[0])[:8192]
    print("Performing sanity checks")
    sanity_base_index = faiss.IndexFlatIP(scaled_train_vectors.shape[-1])
    sanity_index = faiss.IndexIVFFlat(sanity_base_index, scaled_train_vectors.shape[-1], 512)
    sanity_index.train(scaled_train_vectors[np.random.permutation(scaled_train_vectors.shape[0])[:512 * 40]])
    sanity_index.nprobe = 10
    sanity_index.add(scaled_train_vectors[sanity_index_sample_indices])

    search_sample_indices = np.random.permutation(scaled_train_vectors.shape[0])[:512]
    search_sample_train_vectors = scaled_train_vectors[search_sample_indices]
    normalized_search_sample_scaled_train_state_vectors = normalize(train_state_vectors[search_sample_indices].astype(np.float32))
    sample_retrieved_indices = index.search(
        search_sample_train_vectors,
        128
    )[1][:, 1:]

    sample_mean_similarities = np.stack([
        (normalize(train_state_vectors[indices].astype(np.float32)) @ vector[:, None]).mean(axis=0)
        for vector, indices in zip(normalized_search_sample_scaled_train_state_vectors, sample_retrieved_indices)
    ]).mean()
    print(f"Mean similarity of retrieved states {sample_mean_similarities}")
    sample_mean_sentence_similarities = np.stack([
        train_sentences_unique_all_sentence_encodings[train_sentences_unique_list_lookup[indices]] @ train_sentences_unique_all_sentence_encodings[train_sentences_unique_list_lookup[query_index]].T
        for indices, query_index in zip(sample_retrieved_indices, search_sample_indices)
    ]).mean()
    print(f"Mean similarity of retrieved sentences {sample_mean_sentence_similarities}")

    # Compare with baseline where we create an index just from a random subsample of the state vectors
    baseline_index = faiss.IndexFlatIP(train_state_vectors.shape[-1])
    baseline_index.add(normalize(train_state_vectors[sanity_index_sample_indices].astype(np.float32)))
    baseline_search_sample_train_vectors = normalize(train_state_vectors[search_sample_indices].astype(np.float32))
    baseline_sample_retrieved_indices = baseline_index.search(baseline_search_sample_train_vectors, 2)[1][:, 1:]

    baseline_sample_mean_similarities = np.stack([
        (normalize(train_state_vectors[sanity_index_sample_indices[indices]]) @ vector[:, None]).mean(axis=0)
        for vector, indices in zip(baseline_search_sample_train_vectors, baseline_sample_retrieved_indices)
    ]).mean()
    print(f"Baseline mean similarity of retrieved states {baseline_sample_mean_similarities}")

    # Compare also baseline index of just sentences
    baseline_sentence_index = faiss.IndexFlatIP(train_sentences_unique_all_sentence_encodings.shape[-1])
    baseline_sentence_index.add(train_sentences_unique_all_sentence_encodings[train_sentences_unique_list_lookup[sanity_index_sample_indices]])
    baseline_sentence_search_sample_train_vectors = train_sentences_unique_all_sentence_encodings[train_sentences_unique_list_lookup[search_sample_indices]]
    baseline_sentence_sample_retrieved_indices = baseline_sentence_index.search(baseline_sentence_search_sample_train_vectors, 2)[1][:, 1:]

    baseline_sentence_sample_mean_similarities = np.stack([
        (train_sentences_unique_all_sentence_encodings[train_sentences_unique_list_lookup[sanity_index_sample_indices[indices]]] @ vector[:, None]).mean(axis=0)
        for vector, indices in zip(baseline_sentence_search_sample_train_vectors, baseline_sentence_sample_retrieved_indices)
    ]).mean()
    print(f"Baseline mean similarity of retrieved sentences {baseline_sentence_sample_mean_similarities}")

    return (
        model,
        index,
        max_state_component_lengths,
        state_scaler,
        state_pca,
        train_sentences_unique_all_sentence_encodings,
        train_sentences_unique_all_token_encodings,
        train_sentences_unique_list_lookup
    )


GENERATION_CONFIGS = {
    "metalearn_retrieve_state_coverage": {
        "yield_func": "metalearning",
        "generate_mode": "retrieve_similar_state",
        "kwargs": {"can_parallel": False, "num_demos": 16}
    }
}

PREPROCESSING_GLOBAL_PAYLOAD_GENERATOR = {
    "retrieve_similar_state": retrieve_similar_state_global_payload
}

PREPROCESSING_PAYLOAD_GENERATOR = {
    "retrieve_similar_state": retrieve_similar_state_payload
}


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
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--data-output-directory", type=str, required=True)
    parser.add_argument("--limit-load", type=int, help="Data loading limit", default=None)
    parser.add_argument("--limit", type=int, help="Data generation limit", default=None)
    parser.add_argument("--only-splits", nargs="*", type=str)
    parser.add_argument("--retrieval-sentence-state-tradeoff", type=float, default=(4 / 3))
    parser.add_argument("--retrieval-state-pca-dim", type=int, default=1024)
    parser.add_argument("--sbert-batch-size", type=int, default=16)
    args = parser.parse_args()

    dictionaries, (train_dataset, examples) = load_data_directories(
        args.data_directory,
        args.dictionary,
        limit_load=args.limit_load,
        only_splits=args.only_splits
    )

    examples["train"] = train_dataset

    print("Number of examples per split :" + "\n".join([
        f"- {key}: {len(values)}"
        for key, values in examples.items()
    ]))

    bound_funcs = {
        "metalearning": lambda examples, payload, kwargs: encode_metalearning_examples(
            yield_metalearning_examples(
                examples,
                GENERATION_CONFIGS["metalearn_retrieve_state_coverage"]["generate_mode"],
                payload,
                kwargs,
            )
        ),
    }

    splits = list(examples.keys())

    if args.only_splits:
        splits = [k for k in splits if k in args.only_splits]

    os.makedirs(f"{args.data_output_directory}", exist_ok=True)

    global_payload = PREPROCESSING_GLOBAL_PAYLOAD_GENERATOR[
        GENERATION_CONFIGS["metalearn_retrieve_state_coverage"]["generate_mode"]
    ](examples, dictionaries, args)

    for split in tqdm(splits):
        # Note we still make always make the directory,
        # this is to ensure that the dataloader indices align correctly
        os.makedirs(f"{args.data_output_directory}/{split}", exist_ok=True)

        if not examples[split]:
            print(f"Skip {split} as it is empty")
            continue

        payload = PREPROCESSING_PAYLOAD_GENERATOR[
            GENERATION_CONFIGS["metalearn_retrieve_state_coverage"]["generate_mode"]
        ](examples, dictionaries, split, global_payload, args)

        iterable = bound_funcs[GENERATION_CONFIGS["metalearn_retrieve_state_coverage"]["yield_func"]](
            tqdm(examples[split][: args.limit], desc=f"Generating for split {split}"),
            payload,
            GENERATION_CONFIGS["metalearn_retrieve_state_coverage"].get("kwargs", {}),
        )

        for i, batch in enumerate(batched(iterable, 10000)):
            with open(f"{args.data_output_directory}/{split}/{i}.pb", "wb") as f:
                pickle.dump(batch, f)

    with open(f"{args.data_output_directory}/dictionary.pb", "wb") as f:
        pickle.dump(
            dictionaries,
            f,
        )


if __name__ == "__main__":
    main()
