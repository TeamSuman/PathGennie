"""OpenMM PathGennie runner.

Thin wrapper that builds an :class:`OpenMMEngine` and the shared
:class:`~pathgennie.core.driver.PathGennieDriver`.  The adaptive cycle, selection
and convergence logic now live in :mod:`pathgennie.core`; this class only adapts
the OpenMM-specific construction and preserves the original public API used by
``pg_openmm.py``.

PathGennie reference:
'PathGennie: Rapid Generation of Rare Event Pathways via Direction-Guided
Adaptive Sampling Using Ultrashort Monitored Trajectories', J. Chem. Theory
Comput. 2025.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
from openmm.app import Simulation

from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import SerialExecutor
from pathgennie.core.progress import EscapeMetric, TargetMetric

from .engine import OpenMMEngine


class PathGennieMD:
    NM_TO_ANG = 10.0

    def __init__(
        self,
        simulation: Simulation,
        projection_fn: Callable[..., np.ndarray],
        projection_args: Optional[Dict] = None,
        mode: str = "escape",
        target_projection: Optional[np.ndarray] = None,
        convergence_fn: Optional[Callable] = None,
        convergence_args: Optional[Dict] = None,
        escape_direction: str = "auto",
        temperature: float = 300.0,
        sigma: float = 0.5,
        seed: Optional[int] = None,
    ):
        if mode not in ("escape", "target"):
            raise ValueError("mode must be 'escape' or 'target'")
        if mode == "target" and target_projection is None:
            raise ValueError("target_projection required for target mode")
        if convergence_fn is None:
            raise ValueError("convergence_fn is required")

        self.sim = simulation
        self.mode = mode
        self.proj_fn = projection_fn
        self.proj_args = projection_args or {}
        self.target = np.asarray(target_projection) if target_projection is not None else None
        self.converge_fn = convergence_fn
        self.converge_args = convergence_args or {}
        self.temperature = temperature
        self.sigma = sigma
        self.seed = seed

    def run(
        self,
        initial_pos,
        tau1: int = 200,
        tau2: int = 200,
        max_trial: int = 20,
        max_cycle: int = 5000,
        save_freq: int = 10,
        verbosity: int = 1,
    ):
        engine = OpenMMEngine(self.sim, self.temperature)
        initial_handle = engine.create_state(initial_pos)
        start_cv = np.asarray(self.proj_fn(engine.get_coords(initial_handle), **self.proj_args))

        if self.mode == "escape":
            progress = EscapeMetric(
                self.proj_fn, start_cv, projection_args=self.proj_args,
                escape_metric="distance_from_start",
            )
        else:
            progress = TargetMetric(self.proj_fn, self.target, projection_args=self.proj_args)

        converge_fn = self.converge_fn
        converge_args = self.converge_args

        def convergence(coords: np.ndarray) -> bool:
            return bool(converge_fn(coords, **converge_args))

        driver = PathGennieDriver(
            engine, progress, convergence,
            executor=SerialExecutor(),
            sigma=self.sigma, seed=self.seed, verbosity=verbosity,
        )
        return driver.run(
            initial_handle,
            tau1=tau1, tau2=tau2, max_trial=max_trial,
            max_cycle=max_cycle, save_freq=save_freq,
        )
