"""Engine-agnostic samplers for the iterative path-refinement loop.

:class:`~pathrefinement.refiner.PathRefiner` accepts any callable
``sampler(path_cv, start_pt, seed) -> ndarray | None`` for its exploration step.
:class:`EngineSampler` implements that contract on top of the core
:class:`~pathgennie.core.engine.Engine` protocol, so refinement works with **any**
backend -- AMBER, GROMACS, OpenMM or the toy engine -- rather than only OpenMM.

That matters for QM/MM: AMBER is the only backend that can run a QM Hamiltonian,
and before this the refinement loop could not drive it at all.

Usage::

    from pathrefinement.samplers import EngineSampler
    from pathrefinement.refiner import PathRefiner, PathRefinementConfig

    sampler = EngineSampler(
        engine=my_engine,             # any Engine: CoreAmberEngine, OpenMMEngine, ...
        initial_handle=handle,
        feature_fn=lambda xyz: xyz[[0, 1, 5]].ravel(),
        tau1=10, tau2=10, max_trial=30, max_cycle=300,
    )
    refiner = PathRefiner(potential=None, config=cfg, sampler=sampler)

The sampler drives the engine *toward the far end of the current PathCV* (target
mode on the progress coordinate ``s``), which is exactly what the iterative
protocol asks of the exploration step.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

__all__ = ["EngineSampler"]


class EngineSampler:
    """Run one refinement walker on any :class:`Engine`, driven along a PathCV.

    Parameters
    ----------
    engine:
        Anything implementing the core ``Engine`` protocol (``clone_anchor``,
        ``run_segment``, ``get_coords``, ``release``).
    initial_handle:
        Engine handle the walker starts from.
    feature_fn:
        ``coords -> features``. The refinement operates in this feature space, and
        the returned trajectory is expressed in it. Defaults to the flattened
        coordinates.
    tau1, tau2, max_trial, max_cycle, sigma, save_freq:
        Standard PathGennie swarm settings, forwarded to the driver.
    target_s:
        Progress coordinate to drive toward (1.0 = the far end of the path).
    tol:
        Convergence tolerance on ``s``.
    executor:
        Optional ``ParallelExecutor``; the engine's own device pool if it has one.
    """

    def __init__(
        self,
        engine: Any,
        initial_handle: Any,
        *,
        feature_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        tau1: int = 10,
        tau2: int = 10,
        max_trial: int = 20,
        max_cycle: int = 300,
        sigma: float = 0.1,
        save_freq: int = 1,
        target_s: float = 1.0,
        tol: float = 0.05,
        executor: Any = None,
        reject_worse_anchor: bool = True,
        verbosity: int = 0,
    ):
        self.engine = engine
        self.initial_handle = initial_handle
        self.feature_fn = feature_fn or (lambda xyz: np.asarray(xyz, dtype=float).ravel())
        self.tau1 = int(tau1)
        self.tau2 = int(tau2)
        self.max_trial = int(max_trial)
        self.max_cycle = int(max_cycle)
        self.sigma = float(sigma)
        self.save_freq = int(save_freq)
        self.target_s = float(target_s)
        self.tol = float(tol)
        self.executor = executor
        self.reject_worse_anchor = bool(reject_worse_anchor)
        self.verbosity = int(verbosity)

    # -- PathCV helpers ------------------------------------------------------
    def _s_of(self, coords: np.ndarray, path_cv) -> float:
        feats = np.atleast_2d(self.feature_fn(coords))
        s, _z = path_cv.compute(feats)
        return float(s)

    # -- the sampler contract ------------------------------------------------
    def __call__(self, path_cv, start_pt, seed):
        """Return one trajectory in feature space, or ``None`` if it produced nothing."""
        from pathgennie.core.driver import PathGennieDriver
        from pathgennie.core.progress import TargetMetric

        # `path_cv=None` is used by the contract tests: fall back to driving on the
        # first feature component so the sampler is still exercisable without a CV.
        if path_cv is None:
            def project(coords, **_kw):
                return np.atleast_1d(self.feature_fn(coords))[:1]
            target = np.atleast_1d(np.asarray(start_pt, dtype=float)).ravel()[:1]
        else:
            def project(coords, **_kw):
                return np.array([self._s_of(coords, path_cv)])
            target = np.array([self.target_s])

        progress = TargetMetric(project, target)

        def converged(coords: np.ndarray) -> bool:
            return bool(abs(float(project(coords)[0]) - float(target[0])) <= self.tol)

        driver = PathGennieDriver(
            self.engine, progress, converged,
            executor=self.executor,
            sigma=self.sigma,
            seed=int(seed),
            reject_worse_anchor=self.reject_worse_anchor,
            verbosity=self.verbosity,
        )
        frames, _metrics = driver.run(
            self.initial_handle,
            tau1=self.tau1, tau2=self.tau2,
            max_trial=self.max_trial, max_cycle=self.max_cycle,
            save_freq=self.save_freq,
        )
        frames = np.asarray(frames)
        if frames.ndim != 3 or len(frames) == 0:
            return None
        return np.array([np.atleast_1d(self.feature_fn(f)) for f in frames])
