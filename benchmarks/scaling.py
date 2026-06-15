#!/usr/bin/env python
"""Measure how the PathGennie swarm scales across devices.

The multi-GPU win comes from the :class:`~pathgennie.core.parallel.ThreadDevicePool`
spreading the ``N`` per-cycle sampler segments across GPUs.  Real MD segments
spend their wall-time inside ``pmemd.cuda`` / ``gmx mdrun`` / an OpenMM kernel,
i.e. outside the Python GIL, so threads genuinely overlap.

This script reproduces that with a ``SleepEngine`` whose ``run_segment`` sleeps
for a fixed time (GIL released), modelling a GPU segment of known cost.  It then
runs the *real* core driver for a fixed number of cycles with 1, 2, 4, ...
simulated devices and reports cycles/second and speedup, so the dispatch
overhead of the executor itself can be verified to be small.

To benchmark a real backend instead, point ``--case`` at an ``input.yaml`` and
set ``pathgennie.devices`` in it; this harness then just times that run.
"""

from __future__ import annotations

import argparse
import itertools
import threading
import time

import numpy as np

from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import ThreadDevicePool
from pathgennie.core.progress import EscapeMetric


class SleepEngine:
    """Engine whose segments cost a fixed wall-time (models a GPU MD segment)."""

    def __init__(self, segment_seconds: float = 0.01):
        self.segment_seconds = float(segment_seconds)
        self._counter = itertools.count()
        self._lock = threading.Lock()
        self._cache: dict[int, np.ndarray] = {}

    def _store(self, pos):
        with self._lock:
            h = next(self._counter)
        self._cache[h] = np.asarray(pos, dtype=float).copy()
        return h

    def create_state(self, pos):
        return self._store(pos)

    def clone_anchor(self, handle):
        return self._store(self._cache[handle])

    def run_segment(self, handle, n_steps, *, randomize_velocities, seed, device=None):
        time.sleep(self.segment_seconds)  # GIL released, like a real MD call
        rng = np.random.default_rng(seed)
        pos = self._cache[handle] + 0.01 * rng.standard_normal(3)
        return self._store(pos)

    def get_coords(self, handle):
        return self._cache[handle].reshape(1, 3)

    def release(self, handle):
        self._cache.pop(handle, None)


def benchmark(n_devices_list, *, max_trial, max_cycle, segment_seconds):
    print(f"max_trial={max_trial}  max_cycle={max_cycle}  segment={segment_seconds*1e3:.1f} ms\n")
    print(f"{'devices':>8} {'wall (s)':>10} {'cycles/s':>10} {'speedup':>9}")
    baseline = None
    for g in n_devices_list:
        engine = SleepEngine(segment_seconds)
        initial = engine.create_state(np.zeros(3))
        progress = EscapeMetric(lambda c: np.array([c[0, 0]]), start_cv=np.array([0.0]), escape_metric="cv0")
        driver = PathGennieDriver(
            engine, progress, convergence_fn=lambda c: False,
            executor=ThreadDevicePool(devices=list(range(g))),
            sigma=0.1, seed=0, verbosity=0,
        )
        t0 = time.perf_counter()
        driver.run(initial, tau1=1, tau2=1, max_trial=max_trial, max_cycle=max_cycle, save_freq=max_cycle)
        wall = time.perf_counter() - t0
        baseline = baseline or wall
        print(f"{g:>8} {wall:>10.3f} {max_cycle / wall:>10.2f} {baseline / wall:>8.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--devices", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--max-trial", type=int, default=16)
    ap.add_argument("--max-cycle", type=int, default=20)
    ap.add_argument("--segment-ms", type=float, default=10.0)
    args = ap.parse_args()
    benchmark(args.devices, max_trial=args.max_trial, max_cycle=args.max_cycle,
              segment_seconds=args.segment_ms / 1000.0)


if __name__ == "__main__":
    main()
