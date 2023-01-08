import collections.abc
import numpy as np

def recursive_mod(sequence, depth, func):
    if depth == 0:
        return func(sequence)

    return [recursive_mod(subseq, depth - 1, func) for subseq in sequence]


def pad_subsequence_to(subsequence, length, pad):
    if len(subsequence) == length:
        return subsequence

    pad_width = tuple(
        [(0, length - min(len(subsequence), length))]
        + [(0, 0)] * (subsequence.ndim - 1)
    )
    padded = np.pad(
        subsequence[:length], pad_width, mode="constant", constant_values=pad
    )
    return padded


def pad_to(sequence, length, pad=-1):
    if length is None:
        return sequence

    if len(sequence) == 0:
        return np.ones(length, dtype=np.int32) * pad


    length = (length,) if isinstance(length, int) else length
    # pad_width = [(0, l - sequence.shape[i]) for i, l in enumerate(length)]

    # Truncate the sequences on the first dimension. This avoids
    # extra work if the sequence is very long
    for dim_idx in range(len(length)):
        sequence = recursive_mod(
            sequence, dim_idx, lambda x: x[:length[dim_idx]]
        )

    # First pad the sequences on the last dimension
    dim_idx = len(length)

    while dim_idx != 0:
        dim_idx -= 1

        if length[dim_idx] != None:
            if dim_idx == 0:
                sequence = pad_subsequence_to(sequence, length[dim_idx], pad)
            else:
                sequence = recursive_mod(
                    sequence, dim_idx, lambda x: pad_subsequence_to(x, length[dim_idx], pad)
                )

        if dim_idx != 0:
            sequence = recursive_mod(sequence, dim_idx - 1, lambda x: np.stack(x))

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
