"""Toy 2-D Langevin engine on the Wolfe-Quapp potential.

This is a pure-NumPy :class:`~pathgennie.core.engine.Engine` implementation used
to exercise the *entire* driver (swarm -> selection -> runner -> convergence, and
later beam/RRT policies) without any MD binary or GPU.  It also reproduces the
paper's §2.4.1 Wolfe-Quapp benchmark: two minima connected by two saddle points,
the canonical test for whether a sampler can find competing pathways.

The potential (standard Wolfe-Quapp form)::

    V(x, y) = x^4 + y^4 - 2 x^2 - 4 y^2 + x y + 0.3 x + 0.1 y

is integrated with over-damped Langevin (Brownian) dynamics, so swarm diversity
comes purely from the per-segment random seed — exactly the "selection bias on
unbiased trajectories" PathGennie relies on.

A *handle* is an integer id into an in-process state cache, mirroring the design
of the OpenMM worker pool (state stays put; only ids move), which makes this a
faithful proxy for testing ``clone_anchor`` / ``release`` semantics.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

__all__ = ["wolfe_quapp_potential", "wolfe_quapp_gradient", "ToyLangevinEngine"]


def wolfe_quapp_potential(x: float, y: float) -> float:
    return x**4 + y**4 - 2.0 * x**2 - 4.0 * y**2 + x * y + 0.3 * x + 0.1 * y


def wolfe_quapp_gradient(pos: np.ndarray) -> np.ndarray:
    x, y = float(pos[0]), float(pos[1])
    gx = 4.0 * x**3 - 4.0 * x + y + 0.3
    gy = 4.0 * y**3 - 8.0 * y + x + 0.1
    return np.array([gx, gy], dtype=float)


class ToyLangevinEngine:
    """Over-damped Langevin dynamics on the Wolfe-Quapp surface."""

    def __init__(self, *, dt: float = 0.002, kT: float = 1.0, gamma: float = 1.0):
        self.dt = float(dt)
        self.kT = float(kT)
        self.gamma = float(gamma)
        self._cache: Dict[int, np.ndarray] = {}
        self._next_id = 0

    # -- handle/state management -------------------------------------------------
    def _store(self, pos: np.ndarray) -> int:
        handle = self._next_id
        self._next_id += 1
        self._cache[handle] = np.asarray(pos, dtype=float).copy()
        return handle

    def create_state(self, position) -> int:
        """Seed the cache with an initial 2-D position; returns its handle."""
        return self._store(np.asarray(position, dtype=float)[:2])

    def clone_anchor(self, handle: int) -> int:
        return self._store(self._cache[handle])

    def run_segment(
        self,
        handle: int,
        n_steps: int,
        *,
        randomize_velocities: bool = True,
        seed: int = 0,
        device: Optional[int] = None,
    ) -> int:
        rng = np.random.default_rng(seed)
        pos = self._cache[handle].copy()
        diffusion = self.kT / self.gamma
        noise_scale = np.sqrt(2.0 * diffusion * self.dt)
        for _ in range(int(n_steps)):
            force = -wolfe_quapp_gradient(pos)
            pos = pos + (force / self.gamma) * self.dt + noise_scale * rng.standard_normal(2)
        return self._store(pos)

    def get_coords(self, handle: int) -> np.ndarray:
        pos = self._cache[handle]
        return np.array([[pos[0], pos[1], 0.0]], dtype=float)

    def release(self, handle: int) -> None:
        self._cache.pop(handle, None)
