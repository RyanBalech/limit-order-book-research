import pytest
import torch

pytest.importorskip("torch")

from lob_research.deep_models import DeepLOBStyleClassifier, make_sequence_windows


def test_sequence_windows_are_backward_looking_and_aligned():
    x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    y = torch.arange(10)
    windows, labels = make_sequence_windows(x, y, sequence_length=4)
    assert windows.shape == (7, 4, 2)
    assert torch.equal(windows[0], x[:4])
    assert torch.equal(windows[-1], x[-4:])
    assert labels.tolist() == list(range(3, 10))


def test_deeplob_style_forward_shape():
    model = DeepLOBStyleClassifier(n_features=6)
    logits = model(torch.randn(8, 20, 6))
    assert logits.shape == (8, 3)


def test_sequence_validation():
    with pytest.raises(ValueError):
        make_sequence_windows(torch.randn(2, 3), torch.zeros(2), sequence_length=3)
