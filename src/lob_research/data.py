from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LOBSTERData:
    messages: np.ndarray
    book: np.ndarray
    levels: int


def load_lobster_pair(message_path: str | Path, book_path: str | Path) -> LOBSTERData:
    """Load aligned headerless LOBSTER message and order-book CSV files."""
    messages = np.loadtxt(message_path, delimiter=",", dtype=np.float64, ndmin=2)
    book = np.loadtxt(book_path, delimiter=",", dtype=np.float64, ndmin=2)
    if messages.shape[1] != 6:
        raise ValueError("message file must have exactly 6 columns")
    if book.shape[1] == 0 or book.shape[1] % 4 != 0:
        raise ValueError("order-book file must have 4 columns per level")
    if len(messages) != len(book):
        raise ValueError("message and order-book rows must be aligned")
    if not np.isfinite(messages).all() or not np.isfinite(book).all():
        raise ValueError("LOBSTER inputs must contain finite values")
    if np.any(np.diff(messages[:, 0]) < 0):
        raise ValueError("message timestamps must be non-decreasing")
    if np.any(book[:, 1::2] < 0):
        raise ValueError("order-book quantities must be non-negative")
    return LOBSTERData(messages=messages, book=book, levels=book.shape[1] // 4)
