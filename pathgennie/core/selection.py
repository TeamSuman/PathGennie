"""Selection logic for PathGennie swarms.

The same Boltzmann/softmax selection over per-trial progress metrics was
previously duplicated across the OpenMM, AMBER and GROMACS backends
(``pg_omm.py``, ``pg_amber.py``, ``pg_gmx.py``).  It now lives here as the single
source of truth so the behaviour is identical regardless of backend and can be
unit tested in isolation.

Given a batch of trial metrics (higher == better progress), the metrics are
min-max scaled onto ``[0, 1]`` and converted to weights ``exp((scaled - 1)/sigma)``.
Shifting by ``-1`` keeps the largest argument at ``0`` so ``exp`` never overflows.
``sigma`` controls exploration: small ``sigma`` approaches ``argmax`` (greedy),
large ``sigma`` approaches a uniform draw.
"""

from __future__ import annotations

import numpy as np

__all__ = ["selection_probs", "boltzmann_probs", "softmax_select"]


def selection_probs(metrics: np.ndarray, sigma: float) -> np.ndarray:
    """Return normalized selection probabilities for a batch of trial metrics.

    Parameters
    ----------
    metrics:
        1-D array of progress metrics (higher is better).
    sigma:
        Selection temperature; must be > 0.

    Notes
    -----
    If every metric is (numerically) identical the result is a uniform
    distribution, matching the degenerate-batch behaviour of the original
    backends.
    """

    metrics = np.asarray(metrics, dtype=float).ravel()
    if metrics.size == 0:
        raise ValueError("metrics must be non-empty")
    if not np.all(np.isfinite(metrics)):
        raise ValueError("metrics contains non-finite values")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")

    m_min = float(metrics.min())
    m_max = float(metrics.max())
    if (m_max - m_min) < 1e-9:
        return np.full(metrics.shape, 1.0 / metrics.size)

    scaled = (metrics - m_min) / (m_max - m_min)
    weights = np.exp((scaled - 1.0) / sigma)
    return weights / weights.sum()


def softmax_select(metrics: np.ndarray, sigma: float, rng: np.random.Generator) -> int:
    """Draw a single trial index using :func:`selection_probs`.

    ``rng`` is an explicit :class:`numpy.random.Generator` so runs are
    reproducible from a single master seed (the original code used NumPy's
    global RNG, which made selection non-reproducible).
    """

    probs = selection_probs(metrics, sigma)
    return int(rng.choice(probs.size, p=probs))


def boltzmann_probs(metrics: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Selection probabilities ``p_i ~ exp(metric_i / sigma)`` with no rescaling.

    :func:`selection_probs` min-max scales the batch first, so the weight a trial
    receives depends on which other trials happen to be in the batch, and the
    same metric difference is amplified or damped from cycle to cycle. Here the
    weight depends on the trial's own metric only.

    That matters when the metric is the logarithm of a committor estimate: with
    ``metric = log q`` and ``sigma = 1`` the swarm is resampled with
    ``p_i ~ q(y_i)``, which is importance resampling of the transition kernel
    conditioned on reaching the product state (the Doob transform
    ``P(x -> y) q(y) / q(x)``), in the limit of many trials. ``sigma < 1``
    sharpens the selection, ``sigma > 1`` flattens it toward unbiased dynamics.
    """

    metrics = np.asarray(metrics, dtype=float).ravel()
    if metrics.size == 0:
        raise ValueError("metrics must be non-empty")
    if not np.all(np.isfinite(metrics)):
        raise ValueError("metrics contains non-finite values")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    weights = np.exp((metrics - metrics.max()) / sigma)
    return weights / weights.sum()
