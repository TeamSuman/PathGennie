"""Coordinate featurization for data-driven CVs (pure NumPy).

SPIB (and any learned CV) needs a fixed-length, roughly translation/rotation
invariant description of a configuration.  This module provides the common
choices used in the PathGennie examples — pairwise distances, soft contacts and
dihedral sin/cos — plus a :class:`Featurizer` that composes them and applies
online standardization, so the learned model sees well-scaled inputs.

Everything here is NumPy-only so it has no heavy dependency and can run inside
the MD worker; the learned model (PyTorch) consumes these features.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

__all__ = ["pairwise_distances", "contact_features", "dihedral_features", "Featurizer"]


def pairwise_distances(coords: np.ndarray, pairs: Sequence[Sequence[int]]) -> np.ndarray:
    """Euclidean distances for a list of atom-index pairs."""
    coords = np.asarray(coords, dtype=float)
    pairs = np.asarray(pairs, dtype=int)
    diff = coords[pairs[:, 0]] - coords[pairs[:, 1]]
    return np.linalg.norm(diff, axis=1)


def contact_features(coords: np.ndarray, pairs: Sequence[Sequence[int]], r0: float = 8.0, n: int = 6, m: int = 12) -> np.ndarray:
    """Smooth rational switching-function contacts (1 when close, 0 when far)."""
    d = pairwise_distances(coords, pairs)
    x = d / float(r0)
    # Standard PLUMED-style switching function (1 - x^n) / (1 - x^m).
    num = 1.0 - np.power(x, n)
    den = 1.0 - np.power(x, m)
    # Avoid 0/0 at x == 1 (limit is n/m).
    out = np.where(np.abs(den) < 1e-9, n / m, num / den)
    return out


def dihedral_features(coords: np.ndarray, quads: Sequence[Sequence[int]]) -> np.ndarray:
    """sin/cos of each dihedral (continuous, periodicity-safe)."""
    coords = np.asarray(coords, dtype=float)
    feats: List[float] = []
    for a, b, c, d in quads:
        p0, p1, p2, p3 = coords[a], coords[b], coords[c], coords[d]
        b0 = -(p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        b1n = b1 / (np.linalg.norm(b1) + 1e-12)
        v = b0 - np.dot(b0, b1n) * b1n
        w = b2 - np.dot(b2, b1n) * b1n
        x = np.dot(v, w)
        y = np.dot(np.cross(b1n, v), w)
        angle = np.arctan2(y, x)
        feats.extend([np.sin(angle), np.cos(angle)])
    return np.asarray(feats, dtype=float)


class Featurizer:
    """Compose feature functions and apply (optional) online standardization.

    Parameters
    ----------
    funcs:
        List of callables ``f(coords) -> 1-D np.ndarray``.  If empty, the raw
        flattened coordinates are used (useful for low-dimensional toy systems).
    standardize:
        If True, subtract mean / divide by std using statistics accumulated via
        :meth:`fit` (Welford-style running moments).
    """

    def __init__(self, funcs: Optional[Sequence[Callable[[np.ndarray], np.ndarray]]] = None, *, standardize: bool = True):
        self.funcs = list(funcs or [])
        self.standardize = standardize
        self._mean: Optional[np.ndarray] = None
        self._m2: Optional[np.ndarray] = None
        self._count = 0

    def raw(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        if not self.funcs:
            return coords.ravel()
        return np.concatenate([np.atleast_1d(np.asarray(f(coords), dtype=float).ravel()) for f in self.funcs])

    def fit(self, coords_batch: Sequence[np.ndarray]) -> "Featurizer":
        """Accumulate standardization statistics from a batch of configurations."""
        for coords in coords_batch:
            x = self.raw(coords)
            self._count += 1
            if self._mean is None:
                self._mean = np.zeros_like(x)
                self._m2 = np.zeros_like(x)
            delta = x - self._mean
            self._mean += delta / self._count
            self._m2 += delta * (x - self._mean)
        return self

    @property
    def std(self) -> Optional[np.ndarray]:
        if self._m2 is None or self._count < 2:
            return None
        return np.sqrt(self._m2 / (self._count - 1)) + 1e-8

    def transform(self, coords: np.ndarray) -> np.ndarray:
        x = self.raw(coords)
        if self.standardize and self._mean is not None and self.std is not None:
            return (x - self._mean) / self.std
        return x

    def transform_batch(self, coords_batch: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack([self.transform(c) for c in coords_batch])

    @property
    def n_features(self) -> Optional[int]:
        if self._mean is not None:
            return int(self._mean.size)
        return None
