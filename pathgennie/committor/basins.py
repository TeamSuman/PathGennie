"""Reactant (A) and product (B) state definitions.

A basin is a predicate on a CV vector. Basins are deliberately defined in CV
space, not on raw coordinates: the same projection function that drives a
PathGennie run can be reused, and the definition stays engine independent.

``BasinPair.label`` returns ``LABEL_A`` (0), ``LABEL_B`` (1) or ``LABEL_NONE``
(-1). The integer values are chosen so that a label ``>= 0`` *is* the committor
boundary value of that state.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from ..core.progress import periodic_delta

__all__ = ["Basin", "SphereBasin", "BoxBasin", "CallableBasin", "BasinPair",
           "LABEL_NONE", "LABEL_A", "LABEL_B"]

LABEL_NONE = -1
LABEL_A = 0
LABEL_B = 1


class Basin(Protocol):
    def contains(self, cv: np.ndarray) -> bool:
        ...


class SphereBasin:
    """All CVs within ``radius`` of ``center`` (minimum image if ``periodic``)."""

    def __init__(self, center: Sequence[float], radius: float,
                 periodic: Optional[Sequence[Optional[float]]] = None):
        self.center = np.asarray(center, dtype=float).ravel()
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError("radius must be > 0")
        self.periodic = periodic

    def contains(self, cv: np.ndarray) -> bool:
        delta = np.asarray(cv, dtype=float).ravel() - self.center
        if self.periodic is not None:
            delta = periodic_delta(delta, self.periodic)
        return bool(np.linalg.norm(delta) < self.radius)


class BoxBasin:
    """All CVs with ``lower <= cv <= upper`` component-wise.

    Use ``-inf``/``+inf`` for an open side, e.g. a one-sided threshold on a
    distance: ``BoxBasin([-np.inf], [3.5])``.
    """

    def __init__(self, lower: Sequence[float], upper: Sequence[float]):
        self.lower = np.asarray(lower, dtype=float).ravel()
        self.upper = np.asarray(upper, dtype=float).ravel()
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape")
        if np.any(self.lower > self.upper):
            raise ValueError("lower must be <= upper")

    def contains(self, cv: np.ndarray) -> bool:
        cv = np.asarray(cv, dtype=float).ravel()
        return bool(np.all(cv >= self.lower) and np.all(cv <= self.upper))


class CallableBasin:
    """Wrap any ``fn(cv) -> bool``."""

    def __init__(self, fn: Callable[[np.ndarray], bool]):
        self._fn = fn

    def contains(self, cv: np.ndarray) -> bool:
        return bool(self._fn(np.asarray(cv, dtype=float)))


class BasinPair:
    """The reactant state A and the product state B of one transition."""

    def __init__(self, A: Basin, B: Basin):
        self.A = A
        self.B = B

    def label(self, cv: np.ndarray) -> int:
        in_a = self.A.contains(cv)
        in_b = self.B.contains(cv)
        if in_a and in_b:
            raise ValueError(f"basins A and B overlap at cv={np.asarray(cv)}")
        if in_a:
            return LABEL_A
        if in_b:
            return LABEL_B
        return LABEL_NONE

    def labels(self, cvs: np.ndarray) -> np.ndarray:
        cvs = np.asarray(cvs, dtype=float)
        if cvs.ndim == 1:
            cvs = cvs[:, None]
        return np.array([self.label(cv) for cv in cvs], dtype=np.int8)
