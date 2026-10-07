"""Classification diagnostics with explicit class order and temporal dependence."""
from __future__ import annotations

import numpy as np


def confusion_counts(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    y, pred = np.asarray(y), np.asarray(pred)
    if y.ndim != 1 or y.shape != pred.shape or len(y) == 0:
        raise ValueError("expected non-empty, aligned one-dimensional labels")
    if not np.isin(y, [-1, 0, 1]).all() or not np.isin(pred, [-1, 0, 1]).all():
        raise ValueError("labels must be down=-1, stationary=0 or up=1")
    return np.bincount((y.astype(int) + 1) * 3 + pred.astype(int) + 1,
                       minlength=9).reshape(3, 3)


def macro_f1_from_counts(counts: np.ndarray) -> float:
    denominator = counts.sum(0) + counts.sum(1)
    scores = np.divide(2 * counts.diagonal(), denominator,
                       out=np.zeros(3, dtype=float), where=denominator > 0)
    return float(scores.mean())


def classification_diagnostics(y: np.ndarray, pred: np.ndarray) -> dict:
    counts = confusion_counts(y, pred)
    support, predicted = counts.sum(1), counts.sum(0)
    precision = np.divide(counts.diagonal(), predicted, out=np.zeros(3), where=predicted > 0)
    recall = np.divide(counts.diagonal(), support, out=np.zeros(3), where=support > 0)
    return {"class_order": [-1, 0, 1], "confusion_true_rows_predicted_columns": counts.tolist(),
            "precision": precision.tolist(), "recall": recall.tolist(),
            "support": support.tolist(), "predicted_counts": predicted.tolist(),
            "macro_f1": macro_f1_from_counts(counts)}


def paired_block_bootstrap_f1(
    y: np.ndarray, first: np.ndarray, second: np.ndarray, *, block_size: int,
    resamples: int = 2000, seed: int = 2026,
) -> dict:
    """Resample aligned, non-overlapping event blocks; return first-minus-second F1.

    A block preserves adjacent labels/predictions. This does not establish that
    blocks are independent or replace resampling independent trading days.
    """
    if block_size <= 0 or resamples <= 0:
        raise ValueError("block_size and resamples must be positive")
    first_counts, second_counts = [], []
    # Validate the full arrays, including equal lengths, before slicing blocks.
    actual = macro_f1_from_counts(confusion_counts(y, first))
    actual -= macro_f1_from_counts(confusion_counts(y, second))
    for start in range(0, len(y), block_size):
        interval = slice(start, start + block_size)
        first_counts.append(confusion_counts(y[interval], first[interval]))
        second_counts.append(confusion_counts(y[interval], second[interval]))
    if len(first_counts) < 2:
        raise ValueError("need at least two temporal blocks")
    a, b = np.array(first_counts), np.array(second_counts)
    generator = np.random.default_rng(seed)
    deltas = []
    for _ in range(resamples):
        selected = generator.integers(0, len(a), size=len(a))
        deltas.append(macro_f1_from_counts(a[selected].sum(0))
                      - macro_f1_from_counts(b[selected].sum(0)))
    return {"difference": actual, "ci95": np.quantile(deltas, [.025, .975]).tolist(),
            "block_size_events": block_size, "n_blocks": len(a), "resamples": resamples,
            "seed": seed, "method": "paired non-overlapping temporal block bootstrap"}
