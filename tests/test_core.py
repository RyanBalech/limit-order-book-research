import numpy as np
import pytest

from lob_research.data import LOBSTERData
from lob_research.features import build_features
from lob_research.labels import forward_midprice_labels
from lob_research.splits import chronological_split


def _toy_data() -> LOBSTERData:
    messages = np.array([
        [1, 1, 1, 10, 10000, 1],
        [2, 1, 2, 20, 10100, -1],
        [3, 4, 1, 5, 10000, 1],
        [4, 2, 2, 5, 10100, -1],
    ], dtype=float)
    book = np.array([
        [10100, 10, 9900, 10],
        [10100, 8, 9900, 12],
        [10200, 10, 10000, 10],
        [10300, 10, 10100, 10],
    ], dtype=float)
    return LOBSTERData(messages, book, 1)


def test_features_are_finite_and_causal_shape():
    x = build_features(_toy_data(), depth=1)
    assert x.shape == (4, 7)
    assert np.isfinite(x).all()


def test_forward_labels_drop_rows_without_future_target():
    idx, y = forward_midprice_labels(_toy_data().book, horizon=1)
    assert idx.tolist() == [0, 1, 2]
    assert y.tolist() == [0, 1, 1]


def test_chronological_split_has_no_overlap_or_reordering():
    split = chronological_split(10)
    assert split.train.tolist() == list(range(6))
    assert split.validation.tolist() == [6, 7]
    assert split.test.tolist() == [8, 9]


def test_invalid_depth_rejected():
    with pytest.raises(ValueError):
        build_features(_toy_data(), depth=2)
