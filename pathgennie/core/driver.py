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

Two trial schemes (``scheme``):

``parallel`` (default, the scheme above)
    ``N`` samplers per cycle, one drawn by softmax over their metrics.
``serial`` (probe until accepted; the original PathGennie 0.0.0 scheme)
    samplers are run one after another from the anchor, each with fresh
    velocities, until one beats the acceptance threshold; that one is committed.
    The threshold is the *record* -- the highest committed metric so far
    (``accept="record"``, as in 0.0.0) -- or the current anchor metric
    (``accept="anchor"``). There is no softmax and no ``sigma``; the number of
    probes a cycle needs is geometric with the per-probe success probability,
    so it is itself an observable. After ``max_try`` probes without success the
    cycle is *censored* and either the best probe is committed
    (``on_max_try="best"``) or the anchor is kept (``"stay"``).

    Probes from one anchor are independent, so ``batch_size`` of them may run
    concurrently: the accepted probe is the first success *in index order*,
    which has exactly the statistics of a one-at-a-time loop. Probes of the
    batch that come after it are wasted compute but are still logged (flagged
    ``after_accept``). ``batch_size=1`` is strictly sequential.

A :class:`~pathgennie.core.cyclelog.CycleLogger` (``cycle_logger``) streams one
row per cycle and one per evaluated trial while the run proceeds.

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

from .box import FrameCoords, call_with_box, create_handle, frame_coords
from .engine import Engine, Handle
from .parallel import ParallelExecutor, SerialExecutor
from .progress import ProgressVariable
from .selection import selection_probs

__all__ = ["PathGennieDriver", "TrialResult", "scheme_options"]


def scheme_options(pg_cfg: dict) -> dict:
    """Driver keyword arguments for the trial scheme, from the ``pathgennie:`` block.

    ::

        scheme: serial            # parallel (default) | serial
        serial:
          accept: record          # record (beat the best committed metric so far) | anchor
          max_try: 1000           # probes per cycle before the cycle is censored
          batch_size: auto        # probes run concurrently (auto = executor workers; 1 = one at a time)
          on_max_try: best        # best (commit the best probe) | stay (keep the anchor)
          min_gain: 0.0           # accept only if metric > threshold + min_gain
    """

    scheme = pg_cfg.get("scheme") or "parallel"
    opts = {"scheme": scheme}
    if scheme == "serial":
        ser = dict(pg_cfg.get("serial") or {})
        unknown = set(ser) - {"accept", "max_try", "batch_size", "on_max_try", "min_gain"}
        if unknown:
            raise ValueError(f"unknown pathgennie.serial option(s): {sorted(unknown)}")
        opts.update(
            accept=ser.get("accept", "record"),
            max_try=int(ser.get("max_try", 1000)),
            batch_size=ser.get("batch_size", "auto"),
            on_max_try=ser.get("on_max_try", "best"),
            min_gain=float(ser.get("min_gain", 0.0)),
        )
    return opts


@dataclass
class TrialResult:
    handle: Handle
    cv: np.ndarray
    metric: float
    coords: np.ndarray
    device: Optional[int] = None
    index: int = -1   # position in this cycle's seed list (for subframe replay)


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
        checkpoint_freq: int = 0,
        scheme: str = "parallel",
        accept: str = "record",
        max_try: int = 1000,
        batch_size: Optional[int] = None,
        on_max_try: str = "best",
        min_gain: float = 0.0,
        cycle_logger=None,
    ):
        if scheme not in ("parallel", "serial"):
            raise ValueError(f"scheme must be 'parallel' or 'serial', not {scheme!r}")
        if accept not in ("record", "anchor"):
            raise ValueError(f"accept must be 'record' or 'anchor', not {accept!r}")
        if on_max_try not in ("best", "stay"):
            raise ValueError(f"on_max_try must be 'best' or 'stay', not {on_max_try!r}")
        if int(max_try) < 1:
            raise ValueError("max_try must be >= 1")
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
        self.checkpoint_freq = max(0, int(checkpoint_freq))
        self.scheme = scheme
        self.accept = accept
        self.max_try = int(max_try)
        if batch_size in (None, "auto"):
            batch_size = getattr(self.executor, "_max_workers", 1)
        self.batch_size = max(1, int(batch_size))
        self.on_max_try = on_max_try
        self.min_gain = float(min_gain)
        self.cycle_logger = cycle_logger

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
        checkpoint_freq: Optional[int] = None,
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
        ckpt_freq = self.checkpoint_freq if checkpoint_freq is None else max(0, int(checkpoint_freq))
        start_cycle = 0

        trajectory: List[np.ndarray] = []
        metric_history: List[float] = []
        seed_handles: List[Handle] = []
        last_saved_cycle = -1

        if checkpoint_path is not None:
            from .storage import HDF5Storage
            ckpt = HDF5Storage.load_checkpoint(checkpoint_path)
            if ckpt is not None:
                start_cycle = ckpt["cycle"] + 1
                self.rng.bit_generator.state = ckpt["rng_state"]
                # The anchor's own box (None for checkpoints written before boxes
                # were stored): under a barostat the starting box would be wrong.
                anchor_coords = FrameCoords(ckpt["anchor_coords"], ckpt.get("anchor_box"))
                anchor_cv = ckpt["anchor_cv"]
                anchor_metric = float(ckpt["anchor_metric"])
                anchor = create_handle(self.engine, anchor_coords)
                if len(ckpt["trajectory"]) > 0:
                    trajectory = list(ckpt["trajectory"])
                if len(ckpt["metric_history"]) > 0:
                    metric_history = list(ckpt["metric_history"])
                if self.verbosity:
                    print(f"Resuming from checkpoint at cycle {start_cycle} (loaded {len(trajectory)} frames)")
            else:
                anchor = initial_handle
                anchor_coords = frame_coords(self.engine, anchor)
                anchor_cv, anchor_metric = self._evaluate(anchor_coords, cycle=0)
            storage = HDF5Storage(checkpoint_path)
        else:
            anchor = initial_handle
            anchor_coords = frame_coords(self.engine, anchor)
            anchor_cv, anchor_metric = self._evaluate(anchor_coords, cycle=0)
            storage = None

        # A per-frame "box" dataset, aligned with "trajectory", when the engine
        # reports a box. Subframes have no box of their own; they get NaN rows so
        # the two datasets stay aligned.
        store_box = getattr(anchor_coords, "box", None) is not None

        def save_frame(coords: np.ndarray, handle: Handle, metric_val: float) -> None:
            trajectory.append(coords.copy())
            if collect_seeds:
                seed_handles.append(self.engine.clone_anchor(handle))
            if storage is not None:
                storage.append("trajectory", np.asarray(coords))
                storage.append("metric", np.array([metric_val]))
                if store_box:
                    box = getattr(coords, "box", None)
                    storage.append("box", np.full((3, 3), np.nan) if box is None else box)

        def save_subframe_block(frames: np.ndarray, metric_val: float) -> None:
            for frame in np.asarray(frames):
                trajectory.append(frame.copy())
                if collect_seeds:
                    seed_handles.append(self.engine.create_handle(frame))
                if storage is not None:
                    storage.append("trajectory", frame)
                    storage.append("metric", np.array([metric_val]))
                    if store_box:
                        storage.append("box", np.full((3, 3), np.nan))

        cycle_seeds: List[int] = []

        def worker(trial_index: int, device: Optional[int]) -> Optional[TrialResult]:
            # A swarm exists because short segments are unreliable: integrators go
            # unstable, MD subprocesses get killed, scratch writes fail. Letting one
            # casualty out of `max_trial` propagate would discard every completed
            # cycle of a multi-hour run, so a failed trial is quarantined and the
            # cycle proceeds on the survivors. A cycle in which *nothing* survives
            # is a real failure and is raised below.
            handle = None
            seg = None
            try:
                handle = self.engine.clone_anchor(anchor)
                seg = self.engine.run_segment(
                    handle, tau1, randomize_velocities=True,
                    seed=cycle_seeds[trial_index], device=device,
                )
                coords = frame_coords(self.engine, seg)
                cv, metric = self._evaluate(coords, cycle=cycle)
            except Exception as exc:  # noqa: BLE001 - one trial, not the run
                for stray in (seg, handle):
                    if stray is not None:
                        try:
                            self.engine.release(stray)
                        except Exception:  # noqa: BLE001 - already failing
                            pass
                if self.verbosity:
                    print(f"Cycle {cycle}: trial {trial_index} quarantined "
                          f"({type(exc).__name__}: {exc})")
                return None
            # The cloned anchor was only the *input* to this segment; run_segment
            # returned a fresh handle (seg), so release the clone now. Otherwise
            # one clone (a scratch restart file for AMBER/GROMACS, a cache entry
            # for OpenMM) leaks per trial per cycle and fills scratch on long runs.
            if seg != handle:
                self.engine.release(handle)
            return TrialResult(handle=seg, cv=cv, metric=metric, coords=coords, device=device,
                               index=trial_index)

        def all_failed(n: int) -> RuntimeError:
            return RuntimeError(
                f"all {n} trials failed at cycle {cycle}; the run cannot "
                "continue. Quarantining individual trials is deliberate, but a "
                "wholly failed cycle indicates a systematic problem (bad "
                "topology, missing executable, exhausted scratch) rather than an "
                "unlucky segment."
            )

        def trial_row(t_idx, t, batch=0, prob=np.nan, chosen=False, after=False):
            if t is None:
                return dict(cycle=cycle, trial=t_idx, batch=batch, after_accept=after)
            return dict(cycle=cycle, trial=t_idx, batch=batch, metric=t.metric,
                        above_anchor=t.metric > anchor_metric, above_record=t.metric > record,
                        prob=prob, chosen=chosen, after_accept=after)

        def parallel_swarm():
            """N samplers, softmax selection. Returns (trials, chosen_idx, rows, info)."""
            # Draw all per-trial seeds up front on the main thread so the
            # seed -> trial mapping is deterministic regardless of executor
            # scheduling (numpy's Generator is not thread-safe). This makes a
            # seeded run reproducible under ThreadDevicePool, matching Serial.
            cycle_seeds[:] = [self._seed() for _ in range(max_trial)]
            all_trials = self.executor.map(worker, list(range(max_trial)))
            trials = [t for t in all_trials if t is not None]
            if not trials:
                raise all_failed(max_trial)
            metrics = np.array([t.metric for t in trials], dtype=float)
            # Same draw as softmax_select (one rng.choice), so seeded runs are unchanged.
            probs = selection_probs(metrics, self.sigma)
            chosen_idx = int(self.rng.choice(probs.size, p=probs))
            prob_of = {t.index: p for t, p in zip(trials, probs)}
            rows = [trial_row(i, t, prob=prob_of.get(i, np.nan),
                              chosen=(t is not None and t.index == trials[chosen_idx].index))
                    for i, t in enumerate(all_trials)]
            info = dict(n_trials=max_trial, n_failed=max_trial - len(trials), threshold=np.nan,
                        chosen_prob=float(probs[chosen_idx]), accepted=True, censored=False)
            return trials, chosen_idx, rows, info

        def serial_swarm(threshold: float):
            """Probe until one sampler beats ``threshold`` (first success in index order).

            Only the accepted probe and the best probe so far are kept alive, so
            memory/scratch stays bounded however many probes a cycle needs.
            """
            # One master draw per cycle; probe i's seed is derived from (base, i).
            # The master RNG therefore advances identically whatever batch_size is,
            # so batch_size changes only the speed of a seeded run, not its result.
            cycle_seeds.clear()
            base = self._seed()
            accepted = best = None
            n_eval = n_failed = 0
            rows = []
            batch = 0
            while n_eval < self.max_try and accepted is None:
                m = min(self.batch_size, self.max_try - n_eval)
                start = len(cycle_seeds)
                cycle_seeds.extend(
                    int(np.random.SeedSequence([base, i]).generate_state(1)[0] % 2_147_483_646) + 1
                    for i in range(start, start + m))
                results = self.executor.map(worker, list(range(start, start + m)))
                prev_best = best
                for t_idx, t in zip(range(start, start + m), results):
                    after = accepted is not None
                    if t is None:
                        n_failed += 1
                        rows.append(trial_row(t_idx, None, batch, after=after))
                        continue
                    hit = (not after) and t.metric > threshold + self.min_gain
                    rows.append(trial_row(t_idx, t, batch, chosen=hit, after=after))
                    if hit:
                        accepted = t
                    if best is None or t.metric > best.metric:
                        best = t
                for t in results:
                    if t is not None and t is not accepted and t is not best:
                        self.engine.release(t.handle)
                if prev_best is not None and prev_best is not best and prev_best is not accepted:
                    self.engine.release(prev_best.handle)
                n_eval += m
                batch += 1
            if best is None:
                raise all_failed(n_eval)
            censored = accepted is None
            if censored and self.on_max_try == "best":
                chosen = best
                for r in rows:
                    if r.get("trial") == best.index:
                        r["chosen"] = True
            else:
                chosen = accepted
            trials = [accepted] if accepted is not None else []
            if best is not accepted:
                trials.append(best)
            chosen_idx = None if chosen is None else next(i for i, t in enumerate(trials) if t is chosen)
            info = dict(n_trials=n_eval, n_failed=n_failed, threshold=threshold, chosen_prob=np.nan,
                        accepted=not censored, censored=censored)
            return trials, chosen_idx, rows, info

        # The record is the highest committed metric so far (the serial scheme's
        # default acceptance threshold; also logged for the parallel scheme).
        record = max([anchor_metric] + [float(m) for m in metric_history])
        converged_at: Optional[int] = None
        n_quarantined = 0
        n_censored = 0
        for cycle in range(start_cycle, max_cycle):
            previous_anchor = anchor

            if self.scheme == "parallel":
                trials, chosen_idx, trial_rows, info = parallel_swarm()
            else:
                threshold = record if self.accept == "record" else anchor_metric
                trials, chosen_idx, trial_rows, info = serial_swarm(threshold)
                n_censored += int(info["censored"])
            n_quarantined += info["n_failed"]
            tau2_metric = np.nan

            if chosen_idx is None:
                # Serial scheme, max_try reached, on_max_try="stay": keep the anchor.
                for trial in trials:
                    self.engine.release(trial.handle)
                chosen = None
                new_anchor, coords, cv, metric = previous_anchor, anchor_coords, anchor_cv, anchor_metric
                need_replay = False
                tau2_device = None
            else:
                chosen = trials[chosen_idx]

                # ---- runner (tau2) from the chosen sampler ----
                tau2_device = getattr(chosen, "device", None)
                if tau2_device is None:
                    tau2_device = self.executor.devices[0]
                tau2_seed = self._seed()
                # The runner can fail for the same reasons a sampler can. Unlike a
                # sampler it is not one of N, so there is nothing to select among --
                # but the chosen sampler is itself a valid, already-scored state, so
                # fall back to it. That is exactly the state reject_worse_tau2 keeps
                # when the runner merely comes out worse.
                tau2_handle = None
                try:
                    tau2_handle = self.engine.run_segment(
                        chosen.handle, tau2, randomize_velocities=False,
                        seed=tau2_seed, device=tau2_device,
                    )
                    tau2_coords = frame_coords(self.engine, tau2_handle)
                    tau2_cv, tau2_metric = self._evaluate(tau2_coords, cycle=cycle)
                    tau2_failed = False
                except Exception as exc:  # noqa: BLE001 - one runner, not the run
                    if tau2_handle is not None:
                        try:
                            self.engine.release(tau2_handle)
                        except Exception:  # noqa: BLE001 - already failing
                            pass
                    tau2_handle = None
                    tau2_failed = True
                    n_quarantined += 1
                    if self.verbosity:
                        print(f"Cycle {cycle}: runner quarantined "
                              f"({type(exc).__name__}: {exc}); keeping the chosen sampler")

                # ---- candidate selection (optional rejection of a worse runner) ----
                if tau2_failed:
                    cand_handle, cand_coords, cand_cv, cand_metric = (
                        chosen.handle, chosen.coords, chosen.cv, chosen.metric,
                    )
                elif self.reject_worse_tau2 and tau2_metric < chosen.metric:
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

                # When subframes are requested, replay the committed segment before
                # releasing previous_anchor.  Capturing only every save_freq-th cycle
                # creates a discontinuous trajectory because the intervening committed
                # segments are skipped.
                need_replay = self.save_subframes and new_anchor != previous_anchor
                if not need_replay:
                    if new_anchor != previous_anchor and previous_anchor != initial_handle:
                        self.engine.release(previous_anchor)

            if self.cycle_logger is not None:
                self.cycle_logger.log_trials(cycle, trial_rows)
                self.cycle_logger.log_cycle(dict(
                    cycle=cycle, scheme=self.scheme, n_trials=info["n_trials"], n_failed=info["n_failed"],
                    threshold=info["threshold"], anchor_metric=anchor_metric, record=record,
                    n_above_anchor=sum(1 for r in trial_rows if r.get("above_anchor")),
                    n_above_record=sum(1 for r in trial_rows if r.get("above_record")),
                    chosen_trial=None if chosen is None else chosen.index,
                    chosen_metric=None if chosen is None else chosen.metric,
                    chosen_prob=info["chosen_prob"], tau2_metric=tau2_metric, committed_metric=metric,
                    accepted=info["accepted"], censored=info["censored"],
                ), coords=coords)

            anchor, anchor_coords, anchor_cv, anchor_metric = new_anchor, coords, cv, metric
            metric_history.append(metric)
            record = max(record, metric)

            # Let adaptive progress variables (e.g. SPIB) buffer the path and
            # retrain on the fly. Done once per cycle on the committed anchor so
            # the CV is stable within a cycle's project/metric calls.
            observe = getattr(self.progress, "observe", None)
            if observe is not None:
                observe(coords, cycle)

            if need_replay:
                replay_handle = self.engine.clone_anchor(previous_anchor)
                tau1_replay_handle = None
                tau2_replay_handle = None
                try:
                    tau1_result = self.engine.run_segment(
                        replay_handle, tau1, randomize_velocities=True,
                        seed=cycle_seeds[chosen.index], device=tau2_device,
                        save_subframes=True, subframe_stride=self.subframe_stride,
                    )
                    tau1_replay_handle, tau1_subframes = tau1_result

                    subframe_blocks = [tau1_subframes]
                    if new_anchor == tau2_handle:
                        tau2_result = self.engine.run_segment(
                            tau1_replay_handle, tau2, randomize_velocities=False,
                            seed=tau2_seed, device=tau2_device,
                            save_subframes=True, subframe_stride=self.subframe_stride,
                        )
                        tau2_replay_handle, tau2_subframes = tau2_result
                        subframe_blocks.append(tau2_subframes)

                    nonempty_blocks = [block for block in subframe_blocks if len(block) > 0]
                    if nonempty_blocks:
                        save_subframe_block(np.concatenate(nonempty_blocks, axis=0), metric)

                finally:
                    if tau2_replay_handle is not None:
                        self.engine.release(tau2_replay_handle)
                    if tau1_replay_handle is not None:
                        self.engine.release(tau1_replay_handle)
                    self.engine.release(replay_handle)
                    if new_anchor != previous_anchor and previous_anchor != initial_handle:
                        self.engine.release(previous_anchor)
                last_saved_cycle = cycle

            if cycle % save_freq == 0:
                if not self.save_subframes:
                    save_frame(coords, anchor, metric)
                    last_saved_cycle = cycle
                if self.verbosity:
                    extra = ""
                    if self.scheme == "serial":
                        extra = (f", probes={info['n_trials']}" + (" (censored)" if info["censored"] else "")
                                 + f", record={record:.4f}")
                    print(f"Cycle {cycle}: metric={metric:.4f}{extra}, CV={cv}")

            if ckpt_freq > 0 and cycle % ckpt_freq == 0 and storage is not None:
                storage.save_checkpoint(
                    cycle=cycle,
                    rng_state=self.rng.bit_generator.state,
                    anchor_coords=np.asarray(anchor_coords),
                    anchor_box=getattr(anchor_coords, "box", None),
                    anchor_cv=anchor_cv,
                    anchor_metric=anchor_metric,
                    metric_history=metric_history,
                )

            if call_with_box(self.convergence_fn, coords):
                converged_at = cycle
                if last_saved_cycle != cycle:  # always keep the converged frame
                    save_frame(coords, anchor, metric)
                if self.verbosity:
                    print(f"Converged at cycle {cycle}")
                break

        if n_quarantined:
            # Always reported, not gated on verbosity: silently dropping trials
            # would change the effective swarm size without the user knowing.
            print(f"Note: {n_quarantined} trial(s) were quarantined after failing; "
                  "the affected cycles selected from fewer samplers.")
        if n_censored:
            print(f"Note: {n_censored} serial cycle(s) reached max_try={self.max_try} without an "
                  f"accepted probe (on_max_try={self.on_max_try!r}).")
        if self.verbosity:
            tail = f" (converged at {converged_at})" if converged_at is not None else ""
            print(f"Final metric: {anchor_metric:.4f}{tail}")

        if storage is not None:
            storage.close()
        if self.cycle_logger is not None:
            self.cycle_logger.close()

        if collect_seeds:
            return np.asarray(trajectory), np.asarray(metric_history), seed_handles
        return np.asarray(trajectory), np.asarray(metric_history)
