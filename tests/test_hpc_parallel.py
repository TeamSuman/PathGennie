"""HPC parallel-safety regressions for the shared driver and device pool.

Covers three fixes that matter at scale on shared clusters:
  * scheduler-aware CUDA_VISIBLE_DEVICES resolution (never escape the Slurm/PBS
    GPU allocation);
  * deterministic seed -> trial mapping under a threaded device pool
    (numpy Generator is not thread-safe, so seeds are pre-drawn on the main
    thread);
  * the per-trial cloned-anchor handle is released (no scratch/cache leak).
"""

from __future__ import annotations

import numpy as np

from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import (
    SerialExecutor,
    ThreadDevicePool,
    resolve_cuda_visible_device,
)
from pathgennie.core.progress import EscapeMetric
from pathgennie.core.toy import ToyLangevinEngine


# --- scheduler-aware device masking ----------------------------------------

def test_resolve_device_none_returns_none():
    assert resolve_cuda_visible_device(None, {}) is None


def test_resolve_device_absolute_without_mask():
    # No scheduler mask -> logical index used as an absolute id.
    assert resolve_cuda_visible_device(0, {}) == "0"
    assert resolve_cuda_visible_device(3, {}) == "3"


def test_resolve_device_indexes_into_scheduler_mask():
    env = {"CUDA_VISIBLE_DEVICES": "2,3"}
    # Logical 0/1 map to the allocated physical GPUs 2/3, not 0/1.
    assert resolve_cuda_visible_device(0, env) == "2"
    assert resolve_cuda_visible_device(1, env) == "3"
    # Requesting more logical devices than allocated wraps within the allocation
    # (never targets an unallocated GPU).
    assert resolve_cuda_visible_device(2, env) == "2"


def test_resolve_device_handles_uuid_mask():
    env = {"CUDA_VISIBLE_DEVICES": "GPU-aaaa, GPU-bbbb"}
    assert resolve_cuda_visible_device(1, env) == "GPU-bbbb"


# --- driver reproducibility + leak safety ----------------------------------

def _make_driver(executor, seed=7):
    engine = ToyLangevinEngine(dt=0.002, kT=1.0)
    progress = EscapeMetric(lambda c: np.array([c[0, 0]]), start_cv=np.array([0.0]), escape_metric="cv0")
    driver = PathGennieDriver(
        engine, progress, convergence_fn=lambda c: False,
        executor=executor, sigma=0.2, seed=seed, verbosity=0,
    )
    return engine, driver


def test_threaded_run_is_reproducible_from_seed():
    """Same master seed -> identical metric history, even threaded (pre-drawn seeds)."""
    _, d1 = _make_driver(ThreadDevicePool(devices=[0, 1, 2, 3], workers_per_device=1))
    init1 = d1.engine.create_state([-1.0, -1.0, 0.0])
    _, m1 = d1.run(init1, tau1=3, tau2=3, max_trial=8, max_cycle=5, save_freq=1)

    _, d2 = _make_driver(ThreadDevicePool(devices=[0, 1, 2, 3], workers_per_device=1))
    init2 = d2.engine.create_state([-1.0, -1.0, 0.0])
    _, m2 = d2.run(init2, tau1=3, tau2=3, max_trial=8, max_cycle=5, save_freq=1)

    np.testing.assert_allclose(m1, m2)


def test_threaded_matches_serial_for_same_seed():
    """The threaded pool must give the same result as the serial reference."""
    _, ds = _make_driver(SerialExecutor())
    inits = ds.engine.create_state([-1.0, -1.0, 0.0])
    _, ms = ds.run(inits, tau1=3, tau2=3, max_trial=8, max_cycle=5, save_freq=1)

    _, dt = _make_driver(ThreadDevicePool(devices=[0, 1], workers_per_device=2))
    initt = dt.engine.create_state([-1.0, -1.0, 0.0])
    _, mt = dt.run(initt, tau1=3, tau2=3, max_trial=8, max_cycle=5, save_freq=1)

    np.testing.assert_allclose(ms, mt)


def test_cloned_anchor_handles_do_not_leak():
    """After N cycles the engine cache must stay bounded, not grow ~N*max_trial."""
    engine, driver = _make_driver(SerialExecutor())
    init = engine.create_state([-1.0, -1.0, 0.0])
    max_trial, max_cycle = 8, 20
    driver.run(init, tau1=2, tau2=2, max_trial=max_trial, max_cycle=max_cycle, save_freq=5)
    # Without the release fix this would be on the order of max_cycle*max_trial
    # (~160) live entries. It should stay tiny (anchor + a few bookkeeping states).
    assert len(engine._cache) < max_trial + 5
