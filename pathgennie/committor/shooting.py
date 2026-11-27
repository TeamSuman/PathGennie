"""Brute-force shooting committor: the ground truth for validating a model.

From one configuration, launch ``n_shots`` unbiased trajectories with fresh
Maxwell-Boltzmann velocities and count which basin each reaches first. The
estimate ``n_B / (n_A + n_B)`` is the position committor ``q(x)`` (the
phase-space committor averaged over thermal velocities) -- the same quantity a
swarm-trained model estimates.

It works with any :class:`~pathgennie.core.engine.Engine`, so the same call
validates a toy potential, a classical force field or a QM/MM Hamiltonian.

Basin entry is checked every ``check_every`` steps. A trajectory that enters a
basin and leaves it again inside one check interval is missed, so keep the
interval short compared with the residence time in the smaller basin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..core.box import frame_coords
from ..core.parallel import SerialExecutor
from .basins import LABEL_A, LABEL_B, LABEL_NONE, BasinPair

__all__ = ["ShootingResult", "shooting_committor"]


@dataclass
class ShootingResult:
    n_a: int
    n_b: int
    n_open: int          # shots that reached neither basin within max_steps
    total_steps: int     # integrator steps spent (the cost of this estimate)

    @property
    def n_committed(self) -> int:
        return self.n_a + self.n_b

    @property
    def committor(self) -> float:
        """Fraction of committed shots that reached B (NaN if none committed)."""
        return self.n_b / self.n_committed if self.n_committed else float("nan")

    @property
    def stderr(self) -> float:
        """Binomial standard error, from the Agresti-Coull adjusted proportion so
        that an estimate of exactly 0 or 1 does not report zero uncertainty."""
        n = self.n_committed
        if n == 0:
            return float("nan")
        p = (self.n_b + 2.0) / (n + 4.0)
        return float(np.sqrt(p * (1.0 - p) / (n + 4.0)))


def shooting_committor(
    engine,
    handle,
    basins: BasinPair,
    cv_fn: Callable[[np.ndarray], np.ndarray],
    *,
    n_shots: int = 100,
    max_steps: int = 100_000,
    check_every: int = 10,
    seed: Optional[int] = None,
    executor=None,
    refresh_velocities: bool = False,
) -> ShootingResult:
    """Estimate the committor of the configuration behind ``handle``.

    ``refresh_velocities=True`` redraws Maxwell-Boltzmann velocities at *every*
    check interval instead of only at the start. That is not the physical
    committor: it is the committor of the velocity-refreshed chain, which is the
    fixed point a position-only model converges to when trained on swarms of
    length ``check_every``. Comparing the two measures, without any model, how
    much swarms that short distort the committor.

    ``handle`` itself is left untouched. If the configuration is already inside
    a basin the result is that basin's value with zero simulation cost.
    """
    if n_shots < 1 or check_every < 1 or max_steps < 1:
        raise ValueError("n_shots, check_every and max_steps must be >= 1")
    executor = executor or SerialExecutor()

    start_label = basins.label(cv_fn(frame_coords(engine, handle)))
    if start_label == LABEL_A:
        return ShootingResult(n_a=n_shots, n_b=0, n_open=0, total_steps=0)
    if start_label == LABEL_B:
        return ShootingResult(n_a=0, n_b=n_shots, n_open=0, total_steps=0)

    rng = np.random.default_rng(seed)
    # (first-chunk seed, continuation seed base) per shot, drawn up front so the
    # result does not depend on how the executor schedules the shots.
    seeds = rng.integers(1, 2_147_483_647, size=(n_shots, 2))

    def shoot(index: int, device):
        current = engine.clone_anchor(handle)
        steps = 0
        label = LABEL_NONE
        chunk = 0
        try:
            while steps < max_steps:
                n = min(check_every, max_steps - steps)
                seg_seed = int(seeds[index, 0]) if chunk == 0 else int(
                    np.random.SeedSequence([int(seeds[index, 1]), chunk]).generate_state(1)[0]
                    % 2_147_483_646) + 1
                nxt = engine.run_segment(current, n, randomize_velocities=(chunk == 0 or refresh_velocities),
                                         seed=seg_seed, device=device)
                if nxt != current:
                    engine.release(current)
                current = nxt
                steps += n
                chunk += 1
                label = basins.label(cv_fn(frame_coords(engine, current)))
                if label != LABEL_NONE:
                    break
        finally:
            engine.release(current)
        return label, steps

    results = executor.map(shoot, list(range(n_shots)))
    labels = np.array([r[0] for r in results])
    return ShootingResult(
        n_a=int((labels == LABEL_A).sum()),
        n_b=int((labels == LABEL_B).sum()),
        n_open=int((labels == LABEL_NONE).sum()),
        total_steps=int(sum(r[1] for r in results)),
    )
