from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean_ = x.mean(axis=0)
        scale = x.std(axis=0)
        self.scale_ = np.where(scale == 0, 1.0, scale)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("standardizer must be fit before transform")
        return (x - self.mean_) / self.scale_


def majority_class(y_train: np.ndarray) -> int:
    if len(y_train) == 0:
        raise ValueError("training labels cannot be empty")
    values, counts = np.unique(y_train, return_counts=True)
    return int(values[np.argmax(counts)])


def majority_predict(y_train: np.ndarray, n: int) -> np.ndarray:
    return np.full(n, majority_class(y_train), dtype=np.int8)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: tuple[int, ...] = (-1, 0, 1)) -> float:
    scores = []
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else float(2 * tp / denom))
    return float(np.mean(scores))


def balanced_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, classes: tuple[int, ...] = (-1, 0, 1)
) -> float:
    recalls = []
    for cls in classes:
        mask = y_true == cls
        if mask.any():
            recalls.append(float(np.mean(y_pred[mask] == cls)))
    return float(np.mean(recalls)) if recalls else 0.0
