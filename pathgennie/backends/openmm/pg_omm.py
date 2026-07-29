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
from pathgennie.core.parallel import SerialExecutor, ThreadDevicePool
from pathgennie.core.progress import DEFAULT_ESCAPE_METRIC, EscapeMetric, TargetMetric

from .engine import OpenMMEngine, resolve_worker_count


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
        escape_metric: str = DEFAULT_ESCAPE_METRIC,
        temperature: float = 300.0,
        sigma: float = 0.5,
        seed: Optional[int] = None,
        workers_per_device=1,
        device: Optional[int] = None,
        save_subframes: bool = False,
        subframe_stride: int = 1,
        checkpoint_freq: int = 0,
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
        self.escape_metric = escape_metric
        self.temperature = temperature
        self.sigma = sigma
        self.seed = seed
        self.workers_per_device = workers_per_device
        self.device = device
        self.save_subframes = bool(save_subframes)
        self.subframe_stride = max(1, int(subframe_stride))
        self.checkpoint_freq = max(0, int(checkpoint_freq))

    def run(
        self,
        initial_pos,
        tau1: int = 200,
        tau2: int = 200,
        max_trial: int = 20,
        max_cycle: int = 5000,
        save_freq: int = 10,
        verbosity: int = 1,
        collect_seeds: bool = False,
        checkpoint_path: Optional[str] = None,
        checkpoint_freq: Optional[int] = None,
    ):
        # Single-GPU saturation: build a pool of concurrent Contexts sized from
        # workers_per_device (an int, or "auto" -> cores capped by free GPU memory).
        n_workers = resolve_worker_count(self.workers_per_device, self.device)
        engine = OpenMMEngine(
            self.sim, self.temperature,
            n_workers=n_workers, device=self.device, verbose=verbosity >= 1,
        )
        self.engine = engine  # exposed so a downstream stage can reuse it
        initial_handle = engine.create_state(initial_pos)
        start_cv = np.asarray(self.proj_fn(engine.get_coords(initial_handle), **self.proj_args))

        if self.mode == "escape":
            progress = EscapeMetric(
                self.proj_fn, start_cv, projection_args=self.proj_args,
                escape_metric=self.escape_metric,
            )
        else:
            progress = TargetMetric(self.proj_fn, self.target, projection_args=self.proj_args)

        converge_fn = self.converge_fn
        converge_args = self.converge_args

        def convergence(coords: np.ndarray) -> bool:
            return bool(converge_fn(coords, **converge_args))

        # One GPU, several concurrent walkers: a thread pool of engine.n_workers
        # runs segments on the single card. Falls back to a plain serial path when
        # only one Context could be built (no regression on tiny/CPU runs).
        dev_list = [self.device] if self.device is not None else None
        if engine.n_workers > 1:
            executor = ThreadDevicePool(devices=dev_list, workers_per_device=engine.n_workers)
        else:
            executor = SerialExecutor(device=self.device)
        self.executor = executor  # exposed so a downstream stage can reuse the pool
        driver = PathGennieDriver(
            engine, progress, convergence,
            executor=executor,
            sigma=self.sigma, seed=self.seed, verbosity=verbosity,
            save_subframes=self.save_subframes,
            subframe_stride=self.subframe_stride,
            checkpoint_freq=self.checkpoint_freq if checkpoint_freq is None else checkpoint_freq,
        )
        return driver.run(
            initial_handle,
            tau1=tau1, tau2=tau2, max_trial=max_trial,
            max_cycle=max_cycle, save_freq=save_freq,
            collect_seeds=collect_seeds,
            checkpoint_path=checkpoint_path,
            checkpoint_freq=checkpoint_freq,
        )
