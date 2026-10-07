import numpy as np

from lob_research.baselines import Standardizer, balanced_accuracy, macro_f1, majority_predict
from lob_research.splits import chronological_split


def test_standardizer_uses_fit_data_only():
    train = np.array([[0.0, 2.0], [2.0, 4.0]])
    future = np.array([[100.0, 200.0]])
    scaler = Standardizer().fit(train)
    transformed = scaler.transform(future)
    assert np.allclose(scaler.mean_, [1.0, 3.0])
    assert transformed[0, 0] > 1.0


def test_majority_and_metrics():
    y_train = np.array([0, 0, 0, 1, -1])
    assert majority_predict(y_train, 3).tolist() == [0, 0, 0]
    y = np.array([-1, 0, 1])
    assert macro_f1(y, y) == 1.0
    assert balanced_accuracy(y, y) == 1.0


def test_purge_creates_gaps():
    split = chronological_split(100, purge=5)
    assert split.train[-1] == 59
    assert split.validation[0] == 65
    assert split.validation[-1] == 79
    assert split.test[0] == 85
