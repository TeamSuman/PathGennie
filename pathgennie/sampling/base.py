"""Shared data + protocol for PathGennie enhanced-sampling stages.

A :class:`PathEnsemble` is the hand-off from a PathGennie run to a downstream
stage.  It carries the reactive path frames plus the extra information that makes
the path a good *informed seed*: the CV trajectory, optional per-frame metastable
state labels (e.g. from SPIB, for binning / defining WE bins or OPES states), and
optional engine handles so a stage can restart MD from any frame.

A :class:`SamplingStage` consumes a :class:`PathEnsemble` (and an engine) and
returns a :class:`SamplingResult`.  Concrete stages — weighted ensemble (the
paper's "path-informed WESS") and OPES for free-energy surfaces — implement this
single ``run`` method, so they are interchangeable and configured the same way
(``pathgennie.downstream: weighted_ensemble | opes``).

This module is intentionally implementation-free: it fixes the integration
contract so the WE and OPES stages can be added consistently later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, runtime_checkable

import numpy as np

__all__ = ["PathEnsemble", "SamplingResult", "SamplingStage"]


@dataclass
class PathEnsemble:
    """Output of a PathGennie run, usable as seeds for enhanced sampling."""

    frames: np.ndarray                                   # (n_frames, n_atoms, 3) Angstrom
    metrics: np.ndarray                                  # per-cycle progress metric
    cv_trajectory: Optional[np.ndarray] = None           # (n_frames, n_cv)
    state_labels: Optional[np.ndarray] = None            # (n_frames,) metastable-state ids
    handles: List[Any] = field(default_factory=list)     # engine handles for restartable seeds
    metadata: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0]) if self.frames.ndim == 3 else 0


@dataclass
class SamplingResult:
    """Result of an enhanced-sampling stage (free energies / rates / weights)."""

    free_energy: Optional[np.ndarray] = None
    rate_constants: Optional[dict] = None
    weights: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class SamplingStage(Protocol):
    """Consume a PathGennie :class:`PathEnsemble` and produce a result.

    Implementations (planned): ``WeightedEnsembleStage`` (path-informed WE for
    rate constants) and ``OPESStage`` (free-energy surfaces).  ``engine`` is a
    :class:`pathgennie.core.engine.Engine` so the stage can launch further MD
    from the ensemble's seed frames.
    """

    def run(self, ensemble: PathEnsemble, engine: Any, **kwargs: Any) -> SamplingResult:
        ...
