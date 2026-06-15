"""Path-informed Weighted Ensemble (WE) sampling stage.

Weighted Ensemble keeps a population of trajectories ("walkers"), each carrying a
statistical *weight*, propagates them with **unbiased** dynamics, and periodically
*resamples* (splits over-weight walkers, merges under-weight ones) so that walkers
stay spread across bins of a progress coordinate.  No bias force is ever applied —
the weights make the ensemble unbiased — so WE reuses PathGennie's exact
:class:`~pathgennie.core.engine.Engine` and
:class:`~pathgennie.core.parallel.ParallelExecutor` (multi-GPU for free).

This stage is *path-informed*: it seeds walkers from a discovered
:class:`~pathgennie.sampling.base.PathEnsemble` and bins along that path's CV
range, so WE does not have to first find the transition.  With recycling it
yields a steady-state flux (rate constant); the time-averaged bin weights give a
free-energy profile along the CV.

Resampling follows the standard Huber & Kim (1996) split/merge scheme, which
conserves total weight exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from pathgennie.core.engine import Engine, Handle
from pathgennie.core.parallel import ParallelExecutor, SerialExecutor

from .base import PathEnsemble, SamplingResult

__all__ = ["Walker", "GridBinner", "resample", "WeightedEnsembleStage"]


@dataclass
class Walker:
    handle: Handle
    weight: float
    bin: int = -1
    cv: float = float("nan")


class GridBinner:
    """Uniform 1-D bins over a progress coordinate."""

    def __init__(self, edges: Sequence[float]):
        self.edges = np.asarray(edges, dtype=float)
        if self.edges.ndim != 1 or self.edges.size < 2 or not np.all(np.diff(self.edges) > 0):
            raise ValueError("edges must be a strictly increasing 1-D sequence of length >= 2")
        self.n_bins = self.edges.size - 1

    @classmethod
    def from_values(cls, values: Sequence[float], n_bins: int = 12, pad: float = 0.05) -> "GridBinner":
        values = np.asarray(values, dtype=float).ravel()
        lo, hi = float(values.min()), float(values.max())
        if hi <= lo:
            hi = lo + 1.0
        span = hi - lo
        edges = np.linspace(lo - pad * span, hi + pad * span, int(n_bins) + 1)
        return cls(edges)

    def bin_index(self, cv: float) -> int:
        # np.digitize returns 1..len(edges)-1 for interior; clamp to [0, n_bins-1].
        idx = int(np.digitize(float(cv), self.edges)) - 1
        return max(0, min(self.n_bins - 1, idx))

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])


def resample(
    walkers: List[Walker],
    target_count: int,
    rng: np.random.Generator,
    clone_fn: Callable[[Handle], Handle],
    release_fn: Callable[[Handle], None],
) -> List[Walker]:
    """Split/merge a single bin's walkers to exactly ``target_count``, conserving weight.

    Splitting clones the largest-weight walker (halving its weight); merging
    combines the two smallest-weight walkers into one (summed weight), keeping a
    survivor chosen in proportion to weight and releasing the other.
    """

    walkers = list(walkers)
    if not walkers or target_count < 1:
        return walkers

    # Split: grow to target_count by halving the heaviest walker each step.
    while len(walkers) < target_count:
        walkers.sort(key=lambda w: w.weight, reverse=True)
        heavy = walkers[0]
        half = heavy.weight / 2.0
        heavy.weight = half
        walkers.append(Walker(handle=clone_fn(heavy.handle), weight=half, bin=heavy.bin, cv=heavy.cv))

    # Merge: shrink to target_count by combining the two lightest walkers.
    while len(walkers) > target_count:
        walkers.sort(key=lambda w: w.weight)
        a, b = walkers[0], walkers[1]
        total = a.weight + b.weight
        # Survivor chosen proportional to weight; the other is dropped.
        if total <= 0 or rng.random() < (a.weight / total if total > 0 else 0.5):
            survivor, dropped = a, b
        else:
            survivor, dropped = b, a
        survivor.weight = total
        release_fn(dropped.handle)
        walkers = [survivor] + walkers[2:]

    return walkers


class WeightedEnsembleStage:
    """Run path-informed Weighted Ensemble over a :class:`PathEnsemble`.

    ``cv_fn(coords) -> float`` maps an ``(n_atoms, 3)`` configuration to the scalar
    progress coordinate WE bins along (reuse the discovery projection; reduce a
    vector CV to one component or a norm).
    """

    def __init__(
        self,
        cv_fn: Callable[[np.ndarray], float],
        *,
        tau_steps: int,
        n_iterations: int,
        n_bins: int = 12,
        bin_edges: Optional[Sequence[float]] = None,
        target_count: int = 4,
        continue_velocities: bool = True,
        recycle: bool = False,
        source_cv: Optional[float] = None,
        target_cv: Optional[float] = None,
        executor: Optional[ParallelExecutor] = None,
        seed: int = 0,
        timestep_ps: Optional[float] = None,
        kT: float = 1.0,
    ):
        self.cv_fn = cv_fn
        self.tau_steps = int(tau_steps)
        self.n_iterations = int(n_iterations)
        self.n_bins = int(n_bins)
        self.bin_edges = bin_edges
        self.target_count = int(target_count)
        self.continue_velocities = bool(continue_velocities)
        self.recycle = bool(recycle)
        self.source_cv = source_cv
        self.target_cv = target_cv
        self.executor = executor or SerialExecutor()
        self.seed = int(seed)
        self.timestep_ps = timestep_ps
        self.kT = float(kT)
        if self.recycle and (self.source_cv is None or self.target_cv is None):
            raise ValueError("recycle=True requires both source_cv and target_cv")

    # -- seeding -------------------------------------------------------------
    def _seed_handles(self, ensemble: PathEnsemble, engine: Engine) -> List[Handle]:
        if ensemble.handles:
            return list(ensemble.handles)
        create_state = getattr(engine, "create_state", None)
        if create_state is None:
            raise ValueError(
                "PathEnsemble has no restartable handles and the engine has no "
                "create_state(coords); run the driver with collect_seeds=True or "
                "use an engine that supports create_state."
            )
        return [create_state(frame) for frame in ensemble.frames]

    def _cv(self, engine: Engine, handle: Handle) -> float:
        return float(self.cv_fn(engine.get_coords(handle)))

    # -- main loop -----------------------------------------------------------
    def run(self, ensemble: PathEnsemble, engine: Engine, **_: object) -> SamplingResult:
        rng = np.random.default_rng(self.seed)

        # Binner: explicit edges, else from the path CV trajectory / frames.
        if self.bin_edges is not None:
            binner = GridBinner(self.bin_edges)
        elif ensemble.cv_trajectory is not None:
            binner = GridBinner.from_values(ensemble.cv_trajectory[:, 0], self.n_bins)
        else:
            values = [self.cv_fn(f) for f in ensemble.frames]
            binner = GridBinner.from_values(values, self.n_bins)

        seeds = self._seed_handles(ensemble, engine)
        if not seeds:
            raise ValueError("PathEnsemble produced no seed configurations")

        def clone(handle: Handle) -> Handle:
            return engine.clone_anchor(handle)

        def release(handle: Handle) -> None:
            engine.release(handle)

        # Persistent source handle for recycling: an independent clone of the seed
        # nearest source_cv that is never used as a walker (so it is never freed).
        source_handle: Optional[Handle] = None
        if self.recycle:
            cvs = [self._cv(engine, h) for h in seeds]
            nearest = int(np.argmin([abs(c - self.source_cv) for c in cvs]))
            source_handle = engine.clone_anchor(seeds[nearest])

        # Initialise one walker per seed, uniform weight, then balance per bin.
        walkers: List[Walker] = []
        w0 = 1.0 / len(seeds)
        for handle in seeds:
            cv = self._cv(engine, handle)
            walkers.append(Walker(handle=handle, weight=w0, bin=binner.bin_index(cv), cv=cv))
        walkers = self._resample_all(walkers, binner, rng, clone, release)

        bin_weight = np.zeros(binner.n_bins, dtype=float)
        weight_trace: List[float] = []
        flux_total = 0.0

        for _it in range(self.n_iterations):
            seg_seeds = [int(rng.integers(1, 2_147_483_647)) for _ in walkers]

            def worker(item, device):
                walker, seg_seed = item
                new_handle = engine.run_segment(
                    walker.handle, self.tau_steps,
                    randomize_velocities=not self.continue_velocities,
                    seed=seg_seed, device=device,
                )
                return new_handle

            new_handles = self.executor.map(worker, list(zip(walkers, seg_seeds)))
            for walker, new_handle in zip(walkers, new_handles):
                if new_handle is not walker.handle:
                    engine.release(walker.handle)
                walker.handle = new_handle
                walker.cv = self._cv(engine, new_handle)
                walker.bin = binner.bin_index(walker.cv)

            # Recycle walkers that crossed the target back to the source.
            if self.recycle:
                forward = self.target_cv >= self.source_cv
                for walker in walkers:
                    crossed = walker.cv >= self.target_cv if forward else walker.cv <= self.target_cv
                    if crossed:
                        flux_total += walker.weight
                        engine.release(walker.handle)
                        walker.handle = clone(source_handle)
                        walker.cv = self._cv(engine, walker.handle)
                        walker.bin = binner.bin_index(walker.cv)

            for walker in walkers:
                bin_weight[walker.bin] += walker.weight
            weight_trace.append(float(sum(w.weight for w in walkers)))

            walkers = self._resample_all(walkers, binner, rng, clone, release)

        # Free energy from time-averaged occupancy.
        total = bin_weight.sum()
        prob = bin_weight / total if total > 0 else bin_weight
        with np.errstate(divide="ignore"):
            free_energy = -self.kT * np.log(prob)
        finite = free_energy[np.isfinite(free_energy)]
        if finite.size:
            free_energy = free_energy - finite.min()

        rate_constants = None
        if self.recycle:
            flux_per_iter = flux_total / max(1, self.n_iterations)
            if self.timestep_ps is not None:
                tau_time = self.tau_steps * self.timestep_ps
                rate = flux_per_iter / tau_time if tau_time > 0 else float("nan")
            else:
                rate = flux_per_iter  # per-iteration units
            rate_constants = {"flux_per_iter": flux_per_iter, "rate": rate}

        if source_handle is not None:
            engine.release(source_handle)

        return SamplingResult(
            free_energy=free_energy,
            rate_constants=rate_constants,
            weights=np.array([w.weight for w in walkers], dtype=float),
            metadata={
                "bin_centers": binner.centers,
                "bin_probability": prob,
                "weight_trace": np.asarray(weight_trace),
                "n_walkers": len(walkers),
            },
        )

    def _resample_all(self, walkers, binner, rng, clone, release):
        by_bin: dict = {}
        for walker in walkers:
            by_bin.setdefault(walker.bin, []).append(walker)
        out: List[Walker] = []
        for members in by_bin.values():
            out.extend(resample(members, self.target_count, rng, clone, release))
        return out
