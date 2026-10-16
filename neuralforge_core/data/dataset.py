"""Dataset and DataLoader abstractions for batching and shuffling."""

from typing import Tuple, List, Optional, Iterator
import math
import numpy as np
from neuralforge_core.tensor import Tensor, to_tensor


class Dataset:
    """Base class for all datasets."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class ArrayDataset(Dataset):
    """Dataset backed by numpy arrays or tensors."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        assert len(features) == len(targets), "Features and targets must have matching length"
        self.features = np.asarray(features, dtype=np.float64)
        self.targets = np.asarray(targets)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.features[idx], self.targets[idx]


class DataLoader:
    """Mini-batch generator with optional shuffling and deterministic seed."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self) -> Iterator[Tuple[Tensor, np.ndarray]]:
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            self.rng.shuffle(indices)

        for i in range(0, n, self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                break

            # Collect items
            batch_x_list = []
            batch_y_list = []
            for idx in batch_idx:
                x_item, y_item = self.dataset[idx]
                batch_x_list.append(x_item)
                batch_y_list.append(y_item)

            batch_x = np.stack(batch_x_list, axis=0)
            batch_y = np.stack(batch_y_list, axis=0)

            yield Tensor(batch_x, requires_grad=False), batch_y
