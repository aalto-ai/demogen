import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
import random
from .padding import pad_to, recursive_pad_array


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
            items = self.dataset[i]
            return tuple(
                [
                    pad_to(a, p, v)
                    for a, p, v in zip(
                        items, [self.paddings] * items.shape[0], self.pad_values
                    )
                ]
            )
        else:
            item = self.dataset[i]
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


class AddRandomNoiseDataset(Dataset):
    def __init__(self, dataset, ACTION2IDX, prob=0.05):
        super().__init__()
        self.dataset = dataset
        self.ACTION2IDX = ACTION2IDX
        self.prob = prob

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]

        rand = random.uniform(0, 1)

        def add_random_noise(instruction):
            if len(np.where(instruction == self.ACTION2IDX["walk"])[0]) > 0:
                walk_index = random.choice(np.where(instruction == self.ACTION2IDX["walk"])[0])
                instruction[walk_index] = self.ACTION2IDX["turn left"]

                return instruction

            return instruction

        if rand <= self.prob:
            return item[0], add_random_noise(item[1]), item[2]

        return item

