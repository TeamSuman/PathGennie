"""Swarm records: every trial of every cycle, not just the selected one.

Why all trials
--------------
The trials of one PathGennie cycle are independent draws of the *unbiased*
dynamics from the same anchor (fresh Maxwell-Boltzmann velocities, physical
forces). Their endpoints therefore give an unbiased Monte-Carlo estimate of
``E[f(X_tau) | X_0 = anchor]`` for any function ``f`` -- which is what the
committor self-consistency relation ``q(x) = E[q(X_tau) | X_0 = x]`` needs.

The *selected* trial alone does not: it was chosen because of where it ended,
so conditioning on it biases that expectation. Likewise the tau2 runner starts
from a selected sampler with its selected velocities and is a single trajectory;
it is stored (``runner_*``) for path analysis but is not a training sample.

Where the anchors sit is irrelevant to correctness -- the relation holds
pointwise for any distribution of starting points -- so PathGennie's selection
bias on the anchors only decides *where* the committor is learned, not whether
it is learned correctly.

Caveat for the serial scheme: probing stops at the first success, so the number
of probes depends on their outcomes and the plain endpoint average is slightly
biased. Records carry ``scheme`` so that data can be excluded or reweighted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .basins import LABEL_NONE, BasinPair

__all__ = ["SwarmRecorder", "SwarmDataset", "run_swarms"]


def _flatten(coords: np.ndarray) -> np.ndarray:
    return np.asarray(coords, dtype=float).ravel()


@dataclass
class SwarmDataset:
    """Padded arrays of swarm records.

    ``n`` swarms, up to ``K`` trials each, ``d`` features.

    starts       (n, d)     features of the swarm's starting point (the anchor)
    ends         (n, K, d)  features of every trial endpoint (NaN where padded)
    mask         (n, K)     True for a real trial
    start_label  (n,)       basin label of the start (-1 none, 0 A, 1 B)
    end_label    (n, K)     basin label of each endpoint
    prob         (n, K)     selection probability the driver assigned (NaN if none)
    chosen       (n, K)     True for the trial that was selected
    cycle        (n,)       driver cycle (or a running index for standalone swarms)
    lag_steps    (n,)       integrator steps of the trial segments
    """

    starts: np.ndarray
    ends: np.ndarray
    mask: np.ndarray
    start_label: np.ndarray
    end_label: np.ndarray
    prob: np.ndarray
    chosen: np.ndarray
    cycle: np.ndarray
    lag_steps: np.ndarray
    attrs: Dict[str, object] = field(default_factory=dict)

    _FIELDS = ("starts", "ends", "mask", "start_label", "end_label", "prob",
               "chosen", "cycle", "lag_steps")

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.starts.shape[1])

    @property
    def n_trials(self) -> int:
        """Total number of recorded trial segments."""
        return int(self.mask.sum())

    @property
    def total_steps(self) -> int:
        """Integrator steps spent on the recorded trials (the simulation cost)."""
        return int((self.mask.sum(axis=1) * self.lag_steps).sum())

    def subset(self, index) -> "SwarmDataset":
        return SwarmDataset(**{name: getattr(self, name)[index] for name in self._FIELDS},
                            attrs=dict(self.attrs))

    @classmethod
    def concatenate(cls, datasets: Sequence["SwarmDataset"]) -> "SwarmDataset":
        datasets = [d for d in datasets if len(d) > 0]
        if not datasets:
            raise ValueError("no non-empty datasets to concatenate")
        k_max = max(d.ends.shape[1] for d in datasets)

        def pad(arr: np.ndarray, fill) -> np.ndarray:
            if arr.ndim < 2 or arr.shape[1] == k_max:
                return arr
            width = [(0, 0), (0, k_max - arr.shape[1])] + [(0, 0)] * (arr.ndim - 2)
            return np.pad(arr, width, constant_values=fill)

        fills = dict(ends=np.nan, mask=False, end_label=LABEL_NONE, prob=np.nan, chosen=False)
        out = {}
        for name in cls._FIELDS:
            parts = [pad(getattr(d, name), fills[name]) if name in fills else getattr(d, name)
                     for d in datasets]
            out[name] = np.concatenate(parts, axis=0)
        return cls(**out, attrs=dict(datasets[0].attrs))

    def save(self, path: "str | Path") -> None:
        import h5py

        with h5py.File(path, "w") as f:
            for name in self._FIELDS:
                f.create_dataset(name, data=getattr(self, name), compression="gzip")
            for key, value in self.attrs.items():
                f.attrs[key] = value

    @classmethod
    def load(cls, path: "str | Path") -> "SwarmDataset":
        import h5py

        with h5py.File(path, "r") as f:
            data = {name: f[name][()] for name in cls._FIELDS}
            attrs = {k: (v.item() if hasattr(v, "item") and np.ndim(v) == 0 else v)
                     for k, v in f.attrs.items()}
        return cls(**data, attrs=attrs)


class SwarmRecorder:
    """Collects swarm records; pass as ``swarm_recorder=`` to the driver.

    Parameters
    ----------
    featurize:
        ``f(coords) -> 1-D array``: the committor network's input. Defaults to
        the flattened coordinates (fine for toy systems; use internal
        coordinates -- distances, dihedral sin/cos -- for molecules, so the
        input is invariant to translation and rotation).
    basins:
        Optional :class:`BasinPair`. Basin membership is evaluated on the CV the
        driver already computed for each frame, unless ``basin_cv`` is given.
    basin_cv:
        Optional ``f(coords) -> cv`` used *only* for basin membership, for when
        the basins are defined in a different CV than the one steering the run
        (e.g. once a learned progress variable has replaced the geometric one).
    keep_coords:
        Also keep raw coordinates of starts (``coords_starts``). Off by default:
        for a solvated system this is the bulk of the memory.
    """

    def __init__(
        self,
        featurize: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        basins: Optional[BasinPair] = None,
        basin_cv: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        keep_coords: bool = False,
    ):
        self.featurize = featurize or _flatten
        self.basins = basins
        self.basin_cv = basin_cv
        self.keep_coords = bool(keep_coords)
        self._rows: List[dict] = []
        self.coords_starts: List[np.ndarray] = []
        self.runner_starts: List[np.ndarray] = []
        self.runner_ends: List[np.ndarray] = []
        self.runner_cycle: List[int] = []
        self.attrs: Dict[str, object] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def _label(self, coords: np.ndarray, cv: Optional[np.ndarray]) -> int:
        if self.basins is None:
            return LABEL_NONE
        if self.basin_cv is not None:
            cv = self.basin_cv(coords)
        if cv is None:
            raise ValueError("basins were given but no CV is available to evaluate them")
        return self.basins.label(cv)

    def record_swarm(
        self,
        *,
        cycle: int,
        lag_steps: int,
        start_coords: np.ndarray,
        start_cv: Optional[np.ndarray],
        end_coords: Sequence[np.ndarray],
        end_cvs: Optional[Sequence[np.ndarray]] = None,
        probs: Optional[Sequence[float]] = None,
        chosen: Optional[Sequence[bool]] = None,
    ) -> None:
        """Record one swarm: a start and the endpoints of all its trials."""
        k = len(end_coords)
        if k == 0:
            return
        if end_cvs is None:
            end_cvs = [None] * k
        self._rows.append(dict(
            cycle=int(cycle),
            lag_steps=int(lag_steps),
            start=np.asarray(self.featurize(start_coords), dtype=float).ravel(),
            start_label=self._label(start_coords, start_cv),
            ends=np.stack([np.asarray(self.featurize(c), dtype=float).ravel() for c in end_coords]),
            end_label=np.array([self._label(c, cv) for c, cv in zip(end_coords, end_cvs)], dtype=np.int8),
            prob=np.full(k, np.nan) if probs is None else np.asarray(probs, dtype=float),
            chosen=np.zeros(k, dtype=bool) if chosen is None else np.asarray(chosen, dtype=bool),
        ))
        if self.keep_coords:
            self.coords_starts.append(np.array(start_coords, dtype=float))

    def record_runner(self, *, cycle: int, start_coords: np.ndarray, end_coords: np.ndarray) -> None:
        """Record the committed tau2 runner (path analysis only; not a training sample)."""
        self.runner_cycle.append(int(cycle))
        self.runner_starts.append(np.asarray(self.featurize(start_coords), dtype=float).ravel())
        self.runner_ends.append(np.asarray(self.featurize(end_coords), dtype=float).ravel())

    def dataset(self) -> SwarmDataset:
        """Padded-array view of everything recorded so far."""
        if not self._rows:
            raise ValueError("no swarms recorded")
        n = len(self._rows)
        k_max = max(r["ends"].shape[0] for r in self._rows)
        d = self._rows[0]["start"].size
        starts = np.stack([r["start"] for r in self._rows])
        ends = np.full((n, k_max, d), np.nan)
        mask = np.zeros((n, k_max), dtype=bool)
        end_label = np.full((n, k_max), LABEL_NONE, dtype=np.int8)
        prob = np.full((n, k_max), np.nan)
        chosen = np.zeros((n, k_max), dtype=bool)
        for i, r in enumerate(self._rows):
            k = r["ends"].shape[0]
            ends[i, :k] = r["ends"]
            mask[i, :k] = True
            end_label[i, :k] = r["end_label"]
            prob[i, :k] = r["prob"]
            chosen[i, :k] = r["chosen"]
        return SwarmDataset(
            starts=starts, ends=ends, mask=mask,
            start_label=np.array([r["start_label"] for r in self._rows], dtype=np.int8),
            end_label=end_label, prob=prob, chosen=chosen,
            cycle=np.array([r["cycle"] for r in self._rows], dtype=np.int64),
            lag_steps=np.array([r["lag_steps"] for r in self._rows], dtype=np.int64),
            attrs=dict(self.attrs),
        )

    def save(self, path: "str | Path") -> None:
        self.dataset().save(path)


def run_swarms(
    engine,
    handles: Sequence,
    recorder: SwarmRecorder,
    *,
    n_trials: int,
    n_steps: int,
    cv_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    seed: Optional[int] = None,
    executor=None,
) -> SwarmRecorder:
    """Launch a swarm from each of ``handles`` and record it (no selection).

    This is the selection-free counterpart of a driver cycle: use it to add
    swarms at chosen configurations -- along a discovered path, at frames an
    adaptive-sampling run flagged, or at points stratified in the current
    committor estimate. ``handles`` are left untouched; trial handles are
    released.
    """
    from ..core.box import frame_coords
    from ..core.parallel import SerialExecutor

    executor = executor or SerialExecutor()
    rng = np.random.default_rng(seed)
    cv_of = (lambda c: np.asarray(cv_fn(c), dtype=float)) if cv_fn is not None else (lambda c: None)

    for index, start in enumerate(handles):
        start_coords = frame_coords(engine, start)
        seeds = [int(rng.integers(1, 2_147_483_647)) for _ in range(n_trials)]

        def worker(trial: int, device):
            clone = engine.clone_anchor(start)
            seg = engine.run_segment(clone, n_steps, randomize_velocities=True,
                                     seed=seeds[trial], device=device)
            coords = frame_coords(engine, seg)
            if seg != clone:
                engine.release(clone)
            engine.release(seg)
            return coords

        ends = executor.map(worker, list(range(n_trials)))
        recorder.record_swarm(
            cycle=index, lag_steps=n_steps,
            start_coords=start_coords, start_cv=cv_of(start_coords),
            end_coords=ends, end_cvs=[cv_of(c) for c in ends],
        )
    return recorder
