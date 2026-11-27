"""Numerically exact committor on a 2-D grid (overdamped Langevin).

For ``dX = -grad V / gamma dt + sqrt(2 kT / gamma) dW`` the committor solves the
backward Kolmogorov equation ``-grad V . grad q + kT lap q = 0`` outside A and B
with ``q = 0`` on A and ``q = 1`` on B. On a grid this is discretised as a
reversible nearest-neighbour jump process with rates

    k(i -> j) = exp(-(V_j - V_i) / (2 kT)) / h^2

which satisfies detailed balance with respect to ``exp(-V / kT)`` exactly and
converges to the generator as ``h -> 0``. (Friction only rescales time, so the
committor does not depend on it.) The domain edge is reflecting, so choose
limits where the potential is many kT above the transition region.

This is a validation tool for low-dimensional test potentials, not something
that scales to molecules -- that is what shooting and the learned model are for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

__all__ = ["GridCommittor", "grid_committor"]


@dataclass
class GridCommittor:
    x: np.ndarray        # (nx,)
    y: np.ndarray        # (ny,)
    q: np.ndarray        # (nx, ny), q[i, j] = committor at (x[i], y[j])
    in_a: np.ndarray     # (nx, ny) bool
    in_b: np.ndarray     # (nx, ny) bool

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of the committor at ``(n, 2)`` points."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        fx = np.clip((pts[:, 0] - self.x[0]) / (self.x[1] - self.x[0]), 0, self.x.size - 1 - 1e-9)
        fy = np.clip((pts[:, 1] - self.y[0]) / (self.y[1] - self.y[0]), 0, self.y.size - 1 - 1e-9)
        ix, iy = fx.astype(int), fy.astype(int)
        tx, ty = fx - ix, fy - iy
        q = self.q
        return ((1 - tx) * (1 - ty) * q[ix, iy] + tx * (1 - ty) * q[ix + 1, iy]
                + (1 - tx) * ty * q[ix, iy + 1] + tx * ty * q[ix + 1, iy + 1])


def grid_committor(
    potential: Callable[[np.ndarray, np.ndarray], np.ndarray],
    in_a: Callable[[np.ndarray, np.ndarray], np.ndarray],
    in_b: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    n: int = 201,
    kT: float = 1.0,
) -> GridCommittor:
    """Solve for the committor on an ``n x n`` grid.

    ``potential``, ``in_a`` and ``in_b`` take broadcastable ``x`` and ``y``
    arrays; the basin predicates return boolean arrays.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    x = np.linspace(xlim[0], xlim[1], n)
    y = np.linspace(ylim[0], ylim[1], n)
    hx, hy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    V = np.asarray(potential(X, Y), dtype=float)
    A = np.asarray(in_a(X, Y), dtype=bool)
    B = np.asarray(in_b(X, Y), dtype=bool)
    if not A.any() or not B.any():
        raise ValueError("a basin contains no grid point; refine the grid or enlarge the basin")
    if (A & B).any():
        raise ValueError("basins A and B overlap")

    index = np.arange(n * n).reshape(n, n)
    fixed = A | B
    rows, cols, vals = [], [], []
    diag = np.zeros((n, n))

    def couple(src, dst, h):
        # rate from every cell in the `src` slice to its neighbour in `dst`
        rate = np.exp(-(V[dst] - V[src]) / (2.0 * kT)) / h**2
        rows.append(index[src].ravel())
        cols.append(index[dst].ravel())
        vals.append(rate.ravel())
        diag[src] -= rate

    lo, hi = slice(None, -1), slice(1, None)
    every = slice(None)
    couple((lo, every), (hi, every), hx)
    couple((hi, every), (lo, every), hx)
    couple((every, lo), (every, hi), hy)
    couple((every, hi), (every, lo), hy)

    rows = np.concatenate(rows + [index.ravel()])
    cols = np.concatenate(cols + [index.ravel()])
    vals = np.concatenate(vals + [diag.ravel()])

    # Replace the equations of basin cells with q = boundary value.
    keep = ~fixed.ravel()[rows]
    rows, cols, vals = rows[keep], cols[keep], vals[keep]
    fixed_idx = index[fixed]
    rows = np.concatenate([rows, fixed_idx])
    cols = np.concatenate([cols, fixed_idx])
    vals = np.concatenate([vals, np.ones(fixed_idx.size)])
    rhs = np.zeros(n * n)
    rhs[index[B]] = 1.0

    L = coo_matrix((vals, (rows, cols)), shape=(n * n, n * n)).tocsr()
    q = spsolve(L, rhs).reshape(n, n)
    return GridCommittor(x=x, y=y, q=np.clip(q, 0.0, 1.0), in_a=A, in_b=B)
