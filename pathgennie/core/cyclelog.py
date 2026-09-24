"""Streaming per-cycle / per-trial log for PathGennie runs.

The metrics file written at the end of a run keeps one number per cycle. For
analysing *how* a path was generated -- how many trials a cycle needed, how often
trials advanced, how sharp the selection was -- the individual trials are needed
too, and a long HPC run should be inspectable while it is still going.

:class:`CycleLogger` therefore writes two CSV files that grow as the run proceeds
(flushed every ``flush_every`` cycles):

``cycles.csv``
    one row per cycle: scheme, number of trials evaluated, the acceptance
    threshold, how many trials beat the anchor / the record, the chosen trial and
    its selection probability, the tau2 and committed metrics, whether the cycle
    was censored (serial scheme: ``max_try`` reached), wall time -- plus any
    user observables (see below).
``trials.csv``
    one row per evaluated trial. The number of rows per cycle is whatever the
    cycle used: ``max_trial`` in the parallel scheme, the (variable) number of
    probes in the serial scheme. Quarantined trials are logged with an empty
    metric.

The log is *dynamic* in two ways: the per-cycle trial count is not fixed, and
user observables add columns. ``log.observables`` names a function
``f(coords_angstrom, **args) -> dict`` evaluated on every committed frame (e.g.
minimum ligand--protein distance, contact count); each key becomes a
``cycles.csv`` column, named from the first call. Keys that appear later are
added to a side file ``cycles_extra.csv`` (cycle, key, value) rather than
breaking the CSV header.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

__all__ = ["CycleLogger", "CYCLE_FIELDS", "TRIAL_FIELDS", "build_cycle_logger"]

CYCLE_FIELDS = [
    "cycle", "scheme", "n_trials", "n_failed", "threshold", "anchor_metric", "record",
    "n_above_anchor", "n_above_record", "chosen_trial", "chosen_metric", "chosen_prob",
    "tau2_metric", "committed_metric", "accepted", "censored", "wall_s",
]
TRIAL_FIELDS = ["cycle", "trial", "batch", "metric", "above_anchor", "above_record", "prob", "chosen", "after_accept"]


def _fmt(x: Any) -> Any:
    if x is None:
        return ""
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return "" if not math.isfinite(float(x)) else f"{float(x):.6g}"
    if isinstance(x, np.integer):
        return int(x)
    return x


class CycleLogger:
    def __init__(
        self,
        directory: Path | str,
        *,
        cycles_file: str = "cycles.csv",
        trials_file: Optional[str] = "trials.csv",
        flush_every: int = 1,
        observables_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        observables_args: Optional[Dict[str, Any]] = None,
        append: bool = False,
    ):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = max(1, int(flush_every))
        self.obs_fn = observables_fn
        self.obs_args = observables_args or {}
        self.append = bool(append)
        self._cycles_path = self.dir / cycles_file
        self._trials_path = self.dir / trials_file if trials_file else None
        self._extra_path = self._cycles_path.with_name(self._cycles_path.stem + "_extra.csv")
        self._cf = self._tf = self._ef = None
        self._cw = self._tw = self._ew = None
        self._obs_keys: Optional[List[str]] = None
        self._n = 0
        self._t0 = time.perf_counter()
        self._last = self._t0
        mode = "a" if self.append else "w"
        if self._trials_path is not None:
            new = not (self.append and self._trials_path.exists() and self._trials_path.stat().st_size)
            self._tf = open(self._trials_path, mode, newline="")
            self._tw = csv.writer(self._tf)
            if new:
                self._tw.writerow(TRIAL_FIELDS)

    # -- writers ---------------------------------------------------------------
    def _open_cycles(self, obs_keys: List[str]) -> None:
        self._obs_keys = obs_keys
        new = not (self.append and self._cycles_path.exists() and self._cycles_path.stat().st_size)
        self._cf = open(self._cycles_path, "a" if self.append else "w", newline="")
        self._cw = csv.writer(self._cf)
        if new:
            self._cw.writerow(CYCLE_FIELDS + obs_keys)
        else:  # resuming: keep the existing header's observable columns
            with open(self._cycles_path) as fh:
                header = next(csv.reader(fh))
            self._obs_keys = [k for k in header if k not in CYCLE_FIELDS]

    def _extra(self, cycle: int, key: str, value: Any) -> None:
        if self._ef is None:
            new = not self._extra_path.exists()
            self._ef = open(self._extra_path, "a", newline="")
            self._ew = csv.writer(self._ef)
            if new:
                self._ew.writerow(["cycle", "key", "value"])
        self._ew.writerow([cycle, key, _fmt(value)])

    def log_trials(self, cycle: int, rows: List[Dict[str, Any]]) -> None:
        if self._tw is None:
            return
        for r in rows:
            self._tw.writerow([_fmt(r.get(k)) for k in TRIAL_FIELDS])

    def log_cycle(self, row: Dict[str, Any], coords: Optional[np.ndarray] = None) -> None:
        now = time.perf_counter()
        row = dict(row)
        row.setdefault("wall_s", now - self._last)
        self._last = now
        obs: Dict[str, Any] = {}
        if self.obs_fn is not None and coords is not None:
            try:
                obs = dict(self.obs_fn(coords, **self.obs_args))
            except Exception as exc:  # noqa: BLE001 - a log observable must not kill the run
                obs = {"observable_error": f"{type(exc).__name__}: {exc}"}
        if self._cw is None:
            self._open_cycles(list(obs))
        for k, v in obs.items():
            if k not in self._obs_keys:
                self._extra(int(row["cycle"]), k, v)
        self._cw.writerow([_fmt(row.get(k)) for k in CYCLE_FIELDS] + [_fmt(obs.get(k)) for k in self._obs_keys])
        self._n += 1
        if self._n % self.flush_every == 0:
            self.flush()

    def flush(self) -> None:
        for fh in (self._cf, self._tf, self._ef):
            if fh is not None:
                fh.flush()

    def close(self) -> None:
        self.flush()
        for fh in (self._cf, self._tf, self._ef):
            if fh is not None:
                fh.close()
        self._cf = self._tf = self._ef = None


def build_cycle_logger(
    cfg: Dict[str, Any],
    output_dir: Path,
    load_function: Callable[[str, str], Callable],
    *,
    append: bool = False,
    enrich: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Optional[CycleLogger]:
    """Build a :class:`CycleLogger` from the top-level ``log:`` block, or ``None``.

    ``load_function(module, function)`` resolves the optional observables
    callable relative to the case directory (each backend has its own loader).
    """

    log_cfg = cfg.get("log")
    if not log_cfg or not log_cfg.get("enabled", True):
        return None
    obs_fn, obs_args = None, {}
    obs_cfg = log_cfg.get("observables")
    if obs_cfg:
        obs_fn = load_function(obs_cfg["module"], obs_cfg["function"])
        obs_args = {k: v for k, v in obs_cfg.items() if k not in {"module", "function"}}
        if enrich is not None:  # same topology-derived arguments the projection gets
            obs_args = enrich(obs_args)
    return CycleLogger(
        output_dir / log_cfg.get("directory", "."),
        cycles_file=log_cfg.get("cycles", "cycles.csv"),
        trials_file=log_cfg.get("trials", "trials.csv") or None,
        flush_every=log_cfg.get("flush_every", 1),
        observables_fn=obs_fn,
        observables_args=obs_args,
        append=append,
    )
