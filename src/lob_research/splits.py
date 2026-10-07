from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChronologicalSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def chronological_split(
    n: int,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    purge: int = 0,
) -> ChronologicalSplit:
    """Chronological split with optional gaps to purge overlapping forward labels."""
    if n < 3:
        raise ValueError("need at least three observations")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("fractions must lie strictly between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test interval")
    if purge < 0:
        raise ValueError("purge must be non-negative")

    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + validation_fraction))
    val_start = train_end + purge
    test_start = val_end + purge
    if val_start >= val_end or test_start >= n:
        raise ValueError("purge is too large for the requested split")

    idx = np.arange(n)
    return ChronologicalSplit(idx[:train_end], idx[val_start:val_end], idx[test_start:])
