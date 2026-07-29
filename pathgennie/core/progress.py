"""Progress variables and metrics for PathGennie.

A :class:`ProgressVariable` couples a CV projection ``project(coords) -> cv`` with
a scalar ``metric(cv) -> float`` where *higher is better*.  The two built-in
metrics reproduce the original backends:

* :class:`EscapeMetric` — maximise distance from the start CV (escape a minimum /
  blind exploration).  ``escape_metric="cv0"`` keeps the legacy behaviour of just
  maximising the first CV component.
* :class:`TargetMetric` — minimise distance to a target CV (directed exploration);
  the metric is the negated Euclidean distance.

These wrap, verbatim, the logic in ``pg_omm.py:78-83`` and
``pg_amber.py:155-160`` so behaviour is preserved while removing duplication.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence

import numpy as np

__all__ = [
    "ProgressVariable",
    "EscapeMetric",
    "TargetMetric",
    "CallableProjection",
    "periodic_delta",
    "DEFAULT_ESCAPE_METRIC",
]

#: Escape-mode objective used when a case does not set ``escape_metric``.
#:
#: ``"distance_from_start"`` maximises the Euclidean distance from the starting CV,
#: which is the objective the method is published with. The backends previously
#: disagreed -- OpenMM hardcoded this value while AMBER/GROMACS defaulted to the
#: legacy ``"cv0"`` -- so the same input.yaml optimised a different quantity
#: depending on the engine. They now share this default.
DEFAULT_ESCAPE_METRIC = "distance_from_start"


class ProgressVariable(Protocol):
    """Projection + scalar progress metric used to score swarm trials."""

    def project(self, coords: np.ndarray, cycle: Optional[int] = None) -> np.ndarray:
        """Map ``(n_atoms, 3)`` Angstrom coordinates to a CV vector."""
        ...

    def metric(self, cv: np.ndarray) -> float:
        """Score a CV vector; larger means more progress."""
        ...


class CallableProjection:
    """Adapt a plain ``projection_fn(coords, **kwargs)`` to ``project``."""

    def __init__(self, projection_fn: Callable[..., np.ndarray], projection_args: Optional[dict] = None):
        self._fn = projection_fn
        self._args = projection_args or {}

    def project(self, coords: np.ndarray, cycle: Optional[int] = None) -> np.ndarray:
        kwargs = dict(self._args)
        if cycle is not None:
            import inspect
            sig = inspect.signature(self._fn)
            if 'cycle' in sig.parameters:
                kwargs['cycle'] = cycle
        return np.asarray(self._fn(coords, **kwargs), dtype=float)


def periodic_delta(
    delta: np.ndarray, periodic: Optional[Sequence[Optional[float]]]
) -> np.ndarray:
    """Wrap CV differences into the minimum image for periodic components.

    ``periodic`` is ``None`` (every component treated as non-periodic — the historical
    behaviour) or one entry per CV component: a period (e.g. ``360.0`` for degrees,
    ``2*np.pi`` for radians) or ``None``/``0`` for a non-periodic component such as a
    distance or a PCA projection.

    Without this, a dihedral pair straddling the +-180 deg branch cut is scored as ~360 deg
    apart when it is in fact adjacent, which inflates the progress metric and rewards the
    sampler for crossing the cut instead of for real progress.
    """

    delta = np.asarray(delta, dtype=float)
    if periodic is None:
        return delta
    periods = list(periodic)
    if len(periods) != delta.size:
        raise ValueError(
            f"periodic has {len(periods)} entries but the CV has {delta.size} components"
        )
    out = delta.copy()
    for i, period in enumerate(periods):
        if period:
            p = float(period)
            out[i] = (out[i] + 0.5 * p) % p - 0.5 * p
    return out


class EscapeMetric(CallableProjection):
    """Maximise progress away from the starting CV."""

    def __init__(
        self,
        projection_fn: Callable[..., np.ndarray],
        start_cv: np.ndarray,
        *,
        projection_args: Optional[dict] = None,
        escape_metric: str = "distance_from_start",
        periodic: Optional[Sequence[Optional[float]]] = None,
    ):
        super().__init__(projection_fn, projection_args)
        self.start_cv = np.asarray(start_cv, dtype=float)
        self.periodic = periodic
        if escape_metric not in ("distance_from_start", "cv0"):
            raise ValueError("escape_metric must be 'distance_from_start' or 'cv0'")
        self.escape_metric = escape_metric

    def metric(self, cv: np.ndarray) -> float:
        cv = np.asarray(cv, dtype=float)
        if self.escape_metric == "cv0":
            return float(cv.ravel()[0])
        
        s_cv = self.start_cv
        if np.isnan(cv).any():
            valid = ~np.isnan(cv)
            cv = cv[valid]
            s_cv = s_cv[valid]

        if len(cv) < len(s_cv):
            s_cv = s_cv[-len(cv):]

        return float(np.linalg.norm(periodic_delta(cv - s_cv, self.periodic)))


class TargetMetric(CallableProjection):
    """Minimise distance to a target CV (returned negated so higher is better)."""

    def __init__(
        self,
        projection_fn: Callable[..., np.ndarray],
        target_cv: np.ndarray,
        *,
        projection_args: Optional[dict] = None,
        periodic: Optional[Sequence[Optional[float]]] = None,
    ):
        super().__init__(projection_fn, projection_args)
        self.target_cv = np.asarray(target_cv, dtype=float)
        self.periodic = periodic

    def metric(self, cv: np.ndarray) -> float:
        cv = np.asarray(cv, dtype=float)
        t_cv = self.target_cv
        if np.isnan(cv).any():
            valid = ~np.isnan(cv)
            cv = cv[valid]
            t_cv = t_cv[valid]

        if len(cv) < len(t_cv):
            t_cv = t_cv[-len(cv):]

        return float(-np.linalg.norm(periodic_delta(cv - t_cv, self.periodic)))
