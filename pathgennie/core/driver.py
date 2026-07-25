"""Backend-independent PathGennie adaptive-sampling driver.

This is the single implementation of the cycle that the OpenMM, AMBER and
GROMACS backends previously each carried their own copy of:

    for cycle in range(max_cycle):
        run N samplers (tau1) from the anchor with fresh velocities   # swarm
        score each by progress in CV space and softmax-select one     # selection
        extend the chosen sampler for tau2 (runner)                   # commit
        update the anchor, optionally rejecting a worse candidate
        check convergence

The swarm is evaluated through a :class:`ParallelExecutor`, so spreading the
``N`` trials across multiple GPUs is a matter of passing a device pool — the
selection/anchor logic here is untouched by the degree of parallelism.

Correctness fixes relative to the original backends:

* a single master RNG (``seed``) drives both per-segment seeds and the selection
  draw, so a run is reproducible;
* the final, converged frame is always saved (the OpenMM loop could skip it when
  ``cycle % save_freq != 0``);
* trial handles are released each cycle so scratch usage stays bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .engine import Engine, Handle
from .parallel import ParallelExecutor, SerialExecutor
from .progress import ProgressVariable
from .selection import softmax_select

__all__ = ["PathGennieDriver", "TrialResult"]


@dataclass
class TrialResult:
    handle: Handle
    cv: np.ndarray
    metric: float
    coords: np.ndarray
    device: Optional[int] = None


class PathGennieDriver:
    def __init__(
        self,
        engine: Engine,
        progress: ProgressVariable,
        convergence_fn: Callable[[np.ndarray], bool],
        *,
        executor: Optional[ParallelExecutor] = None,
        sigma: float = 0.1,
        seed: Optional[int] = None,
        reject_worse_tau2: bool = False,
        reject_worse_anchor: bool = False,
        verbosity: int = 1,
        save_subframes: bool = False,
        subframe_stride: int = 1,
    ):
        self.engine = engine
        self.progress = progress
        self.convergence_fn = convergence_fn
        self.executor = executor or SerialExecutor()
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)
        self.reject_worse_tau2 = bool(reject_worse_tau2)
        self.reject_worse_anchor = bool(reject_worse_anchor)
        self.verbosity = int(verbosity)
        self.save_subframes = bool(save_subframes)
        self.subframe_stride = max(1, int(subframe_stride))

    def _seed(self) -> int:
        return int(self.rng.integers(1, 2_147_483_647))

    def _evaluate(self, coords: np.ndarray, cycle: Optional[int] = None):
        cv = np.asarray(self.progress.project(coords, cycle=cycle), dtype=float)
        return cv, float(self.progress.metric(cv))

    def run(
        self,
        initial_handle: Handle,
        *,
        tau1: int,
        tau2: int,
        max_trial: int,
        max_cycle: int,
        save_freq: int = 10,
        collect_seeds: bool = False,
        checkpoint_path: Optional[str] = None,
    ):
        """Run the adaptive cycle and return ``(trajectory, metrics)`` arrays.

        ``trajectory`` has shape ``(n_saved, n_atoms, 3)`` in Angstrom;
        ``metrics`` has shape ``(n_cycles,)``.

        When ``collect_seeds`` is True, an independent clone of the committed
        anchor is retained for every saved frame and the call returns
        ``(trajectory, metrics, seed_handles)`` — restartable seeds aligned with
        the trajectory frames, for handing to a downstream sampling stage
        (e.g. Weighted Ensemble). Default behaviour and the 2-tuple return are
        unchanged.
        """

        anchor = initial_handle
        anchor_coords = self.engine.get_coords(anchor)
        anchor_cv, anchor_metric = self._evaluate(anchor_coords, cycle=0)

        trajectory: List[np.ndarray] = []
        metric_history: List[float] = []
        seed_handles: List[Handle] = []
        last_saved_cycle = -1

        if checkpoint_path is not None:
            from .storage import HDF5Storage
            storage = HDF5Storage(checkpoint_path)
        else:
            storage = None

        def save_frame(coords: np.ndarray, handle: Handle, metric_val: float) -> None:
            trajectory.append(coords.copy())
            if collect_seeds:
                seed_handles.append(self.engine.clone_anchor(handle))
            if storage is not None:
                storage.append("trajectory", coords)
                storage.append("metric", np.array([metric_val]))

        cycle_seeds: List[int] = []

        def worker(trial_index: int, device: Optional[int]) -> TrialResult:
            handle = self.engine.clone_anchor(anchor)
            seg = self.engine.run_segment(
                handle, tau1, randomize_velocities=True, seed=cycle_seeds[trial_index], device=device
            )
            coords = self.engine.get_coords(seg)
            cv, metric = self._evaluate(coords, cycle=cycle)
            # The cloned anchor was only the *input* to this segment; run_segment
            # returned a fresh handle (seg), so release the clone now. Otherwise
            # one clone (a scratch restart file for AMBER/GROMACS, a cache entry
            # for OpenMM) leaks per trial per cycle and fills scratch on long runs.
            if seg != handle:
                self.engine.release(handle)
            return TrialResult(handle=seg, cv=cv, metric=metric, coords=coords, device=device)

        converged_at: Optional[int] = None
        for cycle in range(max_cycle):
            previous_anchor = anchor

            # Draw all per-trial seeds up front on the main thread so the
            # seed -> trial mapping is deterministic regardless of executor
            # scheduling (numpy's Generator is not thread-safe). This makes a
            # seeded run reproducible under ThreadDevicePool, matching Serial.
            cycle_seeds = [self._seed() for _ in range(max_trial)]
            trials = self.executor.map(worker, list(range(max_trial)))
            metrics = np.array([t.metric for t in trials], dtype=float)
            chosen_idx = softmax_select(metrics, self.sigma, self.rng)
            chosen = trials[chosen_idx]

            # ---- runner (tau2) from the chosen sampler ----
            tau2_device = getattr(chosen, "device", None)
            if tau2_device is None:
                tau2_device = self.executor.devices[0]
            tau2_seed = self._seed()
            tau2_handle = self.engine.run_segment(
                chosen.handle, tau2, randomize_velocities=False,
                seed=tau2_seed, device=tau2_device,
            )
            tau2_coords = self.engine.get_coords(tau2_handle)
            tau2_cv, tau2_metric = self._evaluate(tau2_coords, cycle=cycle)

            # ---- candidate selection (optional rejection of a worse runner) ----
            if self.reject_worse_tau2 and tau2_metric < chosen.metric:
                cand_handle, cand_coords, cand_cv, cand_metric = (
                    chosen.handle, chosen.coords, chosen.cv, chosen.metric,
                )
                self.engine.release(tau2_handle)
            else:
                cand_handle, cand_coords, cand_cv, cand_metric = (
                    tau2_handle, tau2_coords, tau2_cv, tau2_metric,
                )

            # ---- optional rejection vs the existing anchor ----
            if self.reject_worse_anchor and cand_metric < anchor_metric:
                self.engine.release(cand_handle)
                new_anchor, coords, cv, metric = (
                    previous_anchor, anchor_coords, anchor_cv, anchor_metric,
                )
            else:
                new_anchor, coords, cv, metric = cand_handle, cand_coords, cand_cv, cand_metric

            # ---- release everything we are not keeping ----
            # Compare handles by value (==), not identity (is): handles may be
            # plain ints (CPython caches only -5..256) or round-trip through a
            # distributed executor, so `is` would leak or double-release.
            for index, trial in enumerate(trials):
                if trial.handle != new_anchor and not (index == chosen_idx and cand_handle == chosen.handle):
                    self.engine.release(trial.handle)

            # When subframes are requested for a save-cycle, we need
            # previous_anchor alive for the replay, so defer its release.
            need_replay = (self.save_subframes and cycle % save_freq == 0)
            if not need_replay:
                if new_anchor != previous_anchor and previous_anchor != initial_handle:
                    self.engine.release(previous_anchor)

            anchor, anchor_coords, anchor_cv, anchor_metric = new_anchor, coords, cv, metric
            metric_history.append(metric)

            # Let adaptive progress variables (e.g. SPIB) buffer the path and
            # retrain on the fly. Done once per cycle on the committed anchor so
            # the CV is stable within a cycle's project/metric calls.
            observe = getattr(self.progress, "observe", None)
            if observe is not None:
                observe(coords, cycle)

            if cycle % save_freq == 0:
                # ---- intra-segment subframe capture ----
                if self.save_subframes:
                    # Re-run the winning trial's tau1 with subframe capture.
                    replay_handle = self.engine.clone_anchor(previous_anchor)

                    # Now safe to release previous_anchor.
                    if new_anchor != previous_anchor and previous_anchor != initial_handle:
                        self.engine.release(previous_anchor)

                    tau1_result = self.engine.run_segment(
                        replay_handle, tau1, randomize_velocities=True,
                        seed=cycle_seeds[chosen_idx], device=tau2_device,
                        save_subframes=True, subframe_stride=self.subframe_stride,
                    )
                    tau1_replay_handle, tau1_subframes = tau1_result

                    # Re-run tau2 from the replayed tau1 endpoint.
                    tau2_result = self.engine.run_segment(
                        tau1_replay_handle, tau2, randomize_velocities=False,
                        seed=tau2_seed, device=tau2_device,
                        save_subframes=True, subframe_stride=self.subframe_stride,
                    )
                    tau2_replay_handle, tau2_subframes = tau2_result

                    # Concatenate tau1 + tau2 subframes into trajectory.
                    all_subframes = np.concatenate([tau1_subframes, tau2_subframes], axis=0)
                    for frame in all_subframes:
                        trajectory.append(frame.copy())

                    # Clean up replay handles.
                    self.engine.release(replay_handle)
                    self.engine.release(tau1_replay_handle)
                    self.engine.release(tau2_replay_handle)
                else:
                    save_frame(coords, anchor, metric)

                last_saved_cycle = cycle
                if self.verbosity:
                    print(f"Cycle {cycle}: metric={metric:.4f}, CV={cv}")

            if self.convergence_fn(coords):
                converged_at = cycle
                if last_saved_cycle != cycle:  # always keep the converged frame
                    save_frame(coords, anchor, metric)
                if self.verbosity:
                    print(f"Converged at cycle {cycle}")
                break

        if self.verbosity:
            tail = f" (converged at {converged_at})" if converged_at is not None else ""
            print(f"Final metric: {anchor_metric:.4f}{tail}")

        if storage is not None:
            storage.close()

        if collect_seeds:
            return np.asarray(trajectory), np.asarray(metric_history), seed_handles
        return np.asarray(trajectory), np.asarray(metric_history)
