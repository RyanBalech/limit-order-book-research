import numpy as np
import pytest

from lob_research.data import LOBSTERData, load_lobster_pair
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


def test_feature_prefix_is_unchanged_when_future_book_and_events_change():
    data = _toy_data()
    original = build_features(data, depth=1)
    messages, book = data.messages.copy(), data.book.copy()
    messages[2:, 3] = 9999
    book[2:, 1::2] = 9999
    changed = build_features(LOBSTERData(messages, book, 1), depth=1)
    np.testing.assert_array_equal(original[:2], changed[:2])
    # Unchanged quote prices: bid size grows by 2, ask size falls by 2.
    assert original[1, 6] == 4.


def test_purge_separates_forward_label_endpoints():
    horizon = 100
    split = chronological_split(1000, purge=horizon)
    assert split.train[-1] + horizon < split.validation[0]
    assert split.validation[-1] + horizon < split.test[0]


@pytest.mark.parametrize("failure", ["time", "nan", "size"])
def test_parser_rejects_inputs_that_break_time_or_quantity_contract(tmp_path, failure):
    data = _toy_data()
    messages, book = data.messages.copy(), data.book.copy()
    if failure == "time":
        messages[2, 0] = 0
    elif failure == "nan":
        book[0, 0] = np.nan
    else:
        book[0, 1] = -1
    message_file, book_file = tmp_path / "message.csv", tmp_path / "book.csv"
    np.savetxt(message_file, messages, delimiter=",")
    np.savetxt(book_file, book, delimiter=",")
    with pytest.raises(ValueError):
        load_lobster_pair(message_file, book_file)
