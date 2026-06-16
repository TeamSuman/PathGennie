import numpy as np
import pytest

from pathgennie.core.selection import selection_probs, softmax_select


def test_probs_sum_to_one():
    probs = selection_probs(np.array([0.1, 0.5, 0.9, 0.3]), sigma=0.2)
    assert probs.shape == (4,)
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs >= 0)


def test_degenerate_batch_is_uniform():
    probs = selection_probs(np.array([2.0, 2.0, 2.0]), sigma=0.1)
    assert np.allclose(probs, 1.0 / 3.0)


def test_small_sigma_approaches_argmax():
    metrics = np.array([0.0, 0.2, 1.0, 0.5])
    probs = selection_probs(metrics, sigma=1e-3)
    assert probs.argmax() == metrics.argmax()
    assert probs[metrics.argmax()] > 0.99


def test_large_sigma_approaches_uniform():
    metrics = np.array([0.0, 0.5, 1.0])
    probs = selection_probs(metrics, sigma=1e6)
    assert np.allclose(probs, 1.0 / 3.0, atol=1e-3)


def test_softmax_select_is_reproducible():
    metrics = np.array([0.1, 0.9, 0.4, 0.7])
    a = softmax_select(metrics, 0.3, np.random.default_rng(123))
    b = softmax_select(metrics, 0.3, np.random.default_rng(123))
    assert a == b
    assert 0 <= a < metrics.size


@pytest.mark.parametrize("bad_sigma", [0.0, -1.0])
def test_invalid_sigma_raises(bad_sigma):
    with pytest.raises(ValueError):
        selection_probs(np.array([1.0, 2.0]), sigma=bad_sigma)


def test_nonfinite_metrics_raise():
    with pytest.raises(ValueError):
        selection_probs(np.array([1.0, np.nan]), sigma=0.1)
