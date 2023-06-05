import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from .padding import pad_to, recursive_pad_array


class ReorderSupportsByDistanceDataset(Dataset):
    def __init__(self, dataset, limit):
        super().__init__()
        self.dataset = dataset
        self.limit = limit

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (
            query_state,
            support_state,
            queries,
            targets,
            x_supports,
            y_supports,
            similarity_logit,
        ) = self.dataset[idx]

        order = (-np.array(similarity_logit)).argsort()[: self.limit]

        return (
            query_state,
            [support_state[i] for i in order]
            if isinstance(support_state, list)
            else support_state,
            queries,
            targets,
            [x_supports[i] for i in order],
            [y_supports[i] for i in order],
        )


class MapDataset(Dataset):
    def __init__(self, dataset, map_func):
        super().__init__()
        self.dataset = dataset
        self.map_func = map_func

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.map_func(self.dataset[i])


class PaddingDataset(Dataset):
    def __init__(self, dataset, paddings, pad_values):
        super().__init__()
        self.dataset = dataset
        self.paddings = paddings
        self.pad_values = pad_values

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]

        if isinstance(i, np.ndarray):
            return tuple(
                [
                    pad_to(a, p, v)
                    for a, p, v in zip(
                        item, [self.paddings] * item.shape[0], self.pad_values
                    )
                ]
            )
        else:
            return tuple(
                [
                    pad_to(a, p, v)
                    for a, p, v in zip(item, self.paddings, self.pad_values)
                ]
            )


class PaddingIterableDataset(IterableDataset):
    def __init__(self, dataset, paddings, pad_values):
        super().__init__()
        self.dataset = dataset
        self.paddings = paddings
        self.pad_values = pad_values
        self.iterable = None

    def __iter__(self):
        self.iterable = iter(self.dataset)
        return self

    def __next__(self):
        item = next(self.iterable)
        return recursive_pad_array(item, self.paddings, pad_value=self.pad_values)


class ReshuffleOnIndexZeroDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.indices = torch.randperm(len(dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        if i == 0:
            self.indices = torch.randperm(len(self.dataset))

        return self.dataset[self.indices[i]]
