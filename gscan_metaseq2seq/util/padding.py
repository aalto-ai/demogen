import collections.abc
import numpy as np

def recursive_mod(sequence, depth, func):
    if depth == 0:
        return func(sequence)

    return [recursive_mod(subseq, depth - 1, func) for subseq in sequence]


def pad_subsequence_to(subsequence, length, pad):
    if len(subsequence) == length:
        return subsequence

    subsequence = np.asarray(subsequence)

    pad_width = tuple(
        [(0, length - min(len(subsequence), length))]
        + [(0, 0)] * (subsequence.ndim - 1)
    )
    padded = np.pad(
        subsequence[:length], pad_width, mode="constant", constant_values=pad
    )
    return padded


def fast_array_pad(np_array, expected_shape, pad_value):
    expected_shape = (
        [expected_shape] if isinstance(expected_shape, int) else expected_shape
    )
    expected_shape = tuple(
        [(s or np_a_s) for s, np_a_s in zip(expected_shape, np_array.shape)]
    )
    truncated_np_array = np_array[tuple([slice(0, s) for s in expected_shape])]
    pad_width = tuple(
        [(0, s - np_a_s) for s, np_a_s in zip(expected_shape, truncated_np_array.shape)]
    )
    return np.pad(
        truncated_np_array, pad_width, mode="constant", constant_values=pad_value
    )


def fast_2d_pad(list_of_arrays, expected_shape, pad_value):
    list_of_arrays = list_of_arrays[:expected_shape[0]]
    lens = np.array([len(item) for item in list_of_arrays])
    expected_shape = (expected_shape[0], expected_shape[-1] if expected_shape[-1] else lens.max())

    mask = lens[:, None] > np.arange(expected_shape[-1])
    out = np.ones(expected_shape, dtype=int) * pad_value
    out[mask] = np.concatenate(list_of_arrays)
    return out


def pad_to(sequence, length, pad=-1):
    if length is None:
        return sequence

    if len(sequence) == 0:
        return np.ones(length, dtype=np.int32) * pad

    length = (length,) if isinstance(length, int) else length
    # pad_width = [(0, l - sequence.shape[i]) for i, l in enumerate(length)]

    # First pad the sequences on the last dimension
    dim_idx = len(length)

    # Some fast paths, first if we already have an array
    # we can just pad it directly
    if isinstance(sequence, np.ndarray):
        return fast_array_pad(sequence, length, pad_value=pad)

    # If we have a list of arrays and they're all the same size, just stack
    # and pad
    if (
        isinstance(sequence[0], np.ndarray)
        and np.array([len(s) for s in sequence]).max()
        == np.array([len(s) for s in sequence]).min()
    ):
        return fast_array_pad(np.stack(sequence), length, pad_value=pad)

    # Two dimensional ragged arrays have a fast padding method
    if dim_idx == 2:
        return fast_2d_pad(sequence, length, pad_value=pad)

    # Slow recursive fallback
    while dim_idx != 0:
        dim_idx -= 1

        if length[dim_idx] != None:
            if dim_idx == 0:
                sequence = pad_subsequence_to(sequence, length[dim_idx], pad)
            else:
                sequence = recursive_mod(
                    sequence,
                    dim_idx,
                    lambda x: pad_subsequence_to(np.stack(x), length[dim_idx], pad),
                )

    return np.stack(sequence)


def recursive_pad_array(item, max_lengths, pad_value):
    if max_lengths == None:
        return item

    if isinstance(item, np.ndarray):
        assert isinstance(max_lengths, int)
        return pad_to(item, max_lengths, pad=pad_value)
    elif isinstance(item, collections.abc.Mapping):
        return type(item)(
            {
                k: recursive_pad_array(
                    item[k],
                    max_lengths,
                    pad_value[k]
                    if isinstance(pad_value, collections.abc.Mapping)
                    else pad_value,
                )
                for k in item
            }
        )
    elif isinstance(item, collections.abc.Sequence):
        assert isinstance(max_lengths, collections.abc.Sequence)
        return type(item)(
            [
                recursive_pad_array(
                    e,
                    l,
                    pad_value[i]
                    if isinstance(pad_value, collections.abc.Sequence)
                    else pad_value,
                )
                for i, (e, l) in enumerate(zip(item, max_lengths))
            ]
        )
    else:
        return item
