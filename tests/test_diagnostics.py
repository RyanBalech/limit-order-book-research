import numpy as np
import pytest

from lob_research.diagnostics import classification_diagnostics, paired_block_bootstrap_f1


def test_absent_stationary_predictions_have_zero_precision():
    result = classification_diagnostics(np.array([-1, 0, 1]), np.array([-1, -1, 1]))
    assert result["precision"] == [.5, 0., 1.]
    assert result["recall"] == [1., 0., 1.]
    assert result["confusion_true_rows_predicted_columns"] == [[1, 0, 0], [1, 0, 0], [0, 0, 1]]


def test_block_comparison_preserves_pairing_and_is_reproducible():
    y = np.tile([-1, 0, 1], 10)
    pred = np.roll(y, 1)
    result = paired_block_bootstrap_f1(y, y, pred, block_size=6, resamples=100)
    assert result["difference"] == 1.
    assert result["ci95"] == [1., 1.]
    assert result == paired_block_bootstrap_f1(y, y, pred, block_size=6, resamples=100)
    identical = paired_block_bootstrap_f1(y, pred, pred, block_size=6, resamples=100)
    assert identical["ci95"] == [0., 0.]


def test_block_comparison_rejects_misaligned_labels():
    with pytest.raises(ValueError, match="aligned"):
        paired_block_bootstrap_f1(np.array([-1, 0, 1]), np.array([-1, 0]),
                                  np.array([-1, 0, 1]), block_size=1)
