from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChronologicalSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def chronological_split(
    n: int, *, train_fraction: float = 0.6, validation_fraction: float = 0.2
) -> ChronologicalSplit:
    if n < 3:
        raise ValueError("need at least three observations")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("fractions must lie strictly between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test interval")
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + validation_fraction))
    idx = np.arange(n)
    return ChronologicalSplit(idx[:train_end], idx[train_end:val_end], idx[val_end:])
