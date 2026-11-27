"""A learned committor as PathGennie's progress variable.

The metric is ``log q`` when steering toward B and ``log(1 - q)`` when steering
toward A. Used with the driver's ``selection="boltzmann"`` and ``sigma=1`` the
swarm is resampled with probabilities proportional to ``q`` (or ``1 - q``) at
the trial endpoints -- see :func:`pathgennie.core.selection.boltzmann_probs`.

Only *values* of the model at trial endpoints are needed, never a gradient, and
the MD engine is not involved: one forward pass per trial.

``floor`` regularises the committor to ``q_floor = floor + (1 - 2 floor) q`` so
that a region the current model believes to be hopeless (``q = 0``) keeps a
non-zero selection weight and a wrong early model can still be corrected.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

__all__ = ["CommittorProgress"]


class CommittorProgress:
    """:class:`~pathgennie.core.progress.ProgressVariable` backed by a committor model.

    ``model`` is anything with ``logit(features) -> float`` -- a
    :class:`~pathgennie.committor.train.CommittorModel` or an ensemble of them.
    """

    def __init__(self, model, featurize: Callable[[np.ndarray], np.ndarray], *,
                 target: str = "B", floor: float = 1e-3):
        if target not in ("A", "B"):
            raise ValueError("target must be 'A' or 'B'")
        if not 0.0 <= floor < 0.5:
            raise ValueError("floor must be in [0, 0.5)")
        self.model = model
        self.featurize = featurize
        self.target = target
        self.floor = float(floor)

    def committor(self, coords: np.ndarray) -> float:
        z = float(self.model.logit(np.asarray(self.featurize(coords), dtype=float)))
        return float(np.exp(-np.logaddexp(0.0, -z)))      # overflow-safe sigmoid

    def project(self, coords: np.ndarray, cycle: Optional[int] = None) -> np.ndarray:
        """The 1-component "CV" is the regularised log-probability of reaching the target."""
        q = self.floor + (1.0 - 2.0 * self.floor) * self.committor(coords)
        return np.array([np.log(q if self.target == "B" else 1.0 - q)])

    def metric(self, cv: np.ndarray) -> float:
        return float(np.asarray(cv, dtype=float).ravel()[0])
