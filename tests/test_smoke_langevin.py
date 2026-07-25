"""Full-driver smoke + reference tests on the toy Wolfe-Quapp engine.

These exercise the entire adaptive cycle (swarm -> selection -> runner ->
convergence) with no MD binary or GPU, so they run in CI in well under a second
while still reproducing the paper's two-minima Wolfe-Quapp behaviour.
"""

import numpy as np

from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import SerialExecutor, ThreadDevicePool
from pathgennie.core.progress import TargetMetric
from pathgennie.core.toy import ToyLangevinEngine, wolfe_quapp_gradient

# Wolfe-Quapp minima (numerical stationary points of the potential below).
SOURCE = (-1.174, 1.477)
TARGET = np.array([1.124, -1.485])


def _xy(coords):
    return np.array([coords[0, 0], coords[0, 1]])


def test_minima_are_stationary_points():
    # Sanity check the potential: gradient should be near zero at the minima.
    assert np.linalg.norm(wolfe_quapp_gradient(np.array(SOURCE))) < 1.0
    assert np.linalg.norm(wolfe_quapp_gradient(TARGET)) < 1.0


def _make_driver(executor, seed):
    engine = ToyLangevinEngine(dt=0.005, kT=1.0, gamma=1.0)
    initial = engine.create_state(SOURCE)
    progress = TargetMetric(_xy, target_cv=TARGET)

    def converged(coords):
        return bool(np.linalg.norm(_xy(coords) - TARGET) < 0.5)

    driver = PathGennieDriver(
        engine, progress, converged,
        executor=executor, sigma=0.1, seed=seed, verbosity=0,
    )
    return driver, initial


def test_driver_reaches_target():
    driver, initial = _make_driver(SerialExecutor(), seed=7)
    traj, metrics = driver.run(
        initial, tau1=20, tau2=40, max_trial=16, max_cycle=400, save_freq=5
    )
    assert traj.shape[0] >= 1
    assert traj.shape[1:] == (1, 3)
    # The last saved frame should be at (or past) the target basin.
    assert np.linalg.norm(_xy(traj[-1]) - TARGET) < 0.8
    # Metric (negated distance to target) should improve overall.
    assert metrics[-1] > metrics[0]


def test_serial_equals_threadpool_single_device():
    """A device pool with one slot must match serial output for a fixed seed."""
    d1, i1 = _make_driver(SerialExecutor(), seed=42)
    t1, m1 = d1.run(i1, tau1=15, tau2=30, max_trial=8, max_cycle=120, save_freq=5)

    d2, i2 = _make_driver(ThreadDevicePool(devices=[0]), seed=42)
    t2, m2 = d2.run(i2, tau1=15, tau2=30, max_trial=8, max_cycle=120, save_freq=5)

    np.testing.assert_allclose(m1, m2)
    assert t1.shape == t2.shape
    np.testing.assert_allclose(t1, t2)


def test_round_robin_device_assignment():
    """ThreadDevicePool spreads work items across devices in order."""
    pool = ThreadDevicePool(devices=[0, 1, 2, 3])
    seen = pool.map(lambda item, device: (item, device), list(range(8)))
    # Items preserve order and devices cycle 0,1,2,3,0,1,2,3.
    assert [s[0] for s in seen] == list(range(8))
    assert [s[1] for s in seen] == [0, 1, 2, 3, 0, 1, 2, 3]


def _make_driver_with_subframes(executor, seed, *, save_subframes=False, subframe_stride=1):
    engine = ToyLangevinEngine(dt=0.005, kT=1.0, gamma=1.0)
    initial = engine.create_state(SOURCE)
    progress = TargetMetric(_xy, target_cv=TARGET)

    def converged(coords):
        return bool(np.linalg.norm(_xy(coords) - TARGET) < 0.5)

    driver = PathGennieDriver(
        engine, progress, converged,
        executor=executor, sigma=0.1, seed=seed, verbosity=0,
        save_subframes=save_subframes,
        subframe_stride=subframe_stride,
    )
    return driver, initial


def test_subframes_capture_intermediate_positions():
    """With subframes enabled, the trajectory should have many more frames than
    the endpoint-only baseline (one frame per cycle → many frames per cycle)."""
    tau1, tau2, stride = 20, 40, 5

    # Baseline without subframes.
    d_base, i_base = _make_driver(SerialExecutor(), seed=7)
    traj_base, _ = d_base.run(
        i_base, tau1=tau1, tau2=tau2, max_trial=8, max_cycle=60, save_freq=5,
    )
    n_base = traj_base.shape[0]

    # With subframes: each saved cycle produces (tau1+tau2)/stride subframes
    # instead of 1 endpoint frame.
    d_sub, i_sub = _make_driver_with_subframes(
        SerialExecutor(), seed=7, save_subframes=True, subframe_stride=stride,
    )
    traj_sub, _ = d_sub.run(
        i_sub, tau1=tau1, tau2=tau2, max_trial=8, max_cycle=60, save_freq=5,
    )
    n_sub = traj_sub.shape[0]

    # Subframes should produce strictly more frames than baseline.
    assert n_sub > n_base, f"Expected more subframes ({n_sub}) than baseline ({n_base})"
    # Each saved cycle should produce (tau1+tau2)/stride = 12 frames vs 1.
    expected_ratio = (tau1 + tau2) // stride
    assert n_sub >= n_base * expected_ratio * 0.8  # allow some margin for convergence


def test_subframes_off_matches_baseline():
    """save_subframes=False (default) must produce identical output to a driver
    constructed without the subframes parameters at all."""
    d1, i1 = _make_driver(SerialExecutor(), seed=42)
    t1, m1 = d1.run(i1, tau1=15, tau2=30, max_trial=8, max_cycle=120, save_freq=5)

    d2, i2 = _make_driver_with_subframes(
        SerialExecutor(), seed=42, save_subframes=False, subframe_stride=5,
    )
    t2, m2 = d2.run(i2, tau1=15, tau2=30, max_trial=8, max_cycle=120, save_freq=5)

    np.testing.assert_allclose(m1, m2)
    assert t1.shape == t2.shape
    np.testing.assert_allclose(t1, t2)


def test_driver_checkpoint_resume(tmp_path):
    """Running with checkpoint_freq saves periodic checkpoints, and a subsequent
    run with the same checkpoint_path resumes from the saved cycle."""
    ckpt_path = tmp_path / "checkpoint.h5"

    engine = ToyLangevinEngine(dt=0.005, kT=1.0, gamma=1.0)
    initial = engine.create_state(SOURCE)
    progress = TargetMetric(_xy, target_cv=TARGET)

    # Never converge early so loop runs full cycles
    never_converge = lambda coords: False

    # Part 1: Run 25 cycles with checkpoint_freq=10 (saves at cycles 0, 10, 20)
    d1 = PathGennieDriver(
        engine, progress, never_converge,
        executor=SerialExecutor(), sigma=0.1, seed=42, verbosity=0,
    )
    t1, m1 = d1.run(
        initial, tau1=10, tau2=10, max_trial=4, max_cycle=25, save_freq=5,
        checkpoint_path=str(ckpt_path), checkpoint_freq=10,
    )
    assert ckpt_path.exists()
    assert len(m1) == 25

    # Part 2: Resume from checkpoint (starts at cycle 20+1 = 21) and run to 40 cycles
    d2 = PathGennieDriver(
        engine, progress, never_converge,
        executor=SerialExecutor(), sigma=0.1, seed=42, verbosity=0,
    )
    t2, m2 = d2.run(
        initial, tau1=10, tau2=10, max_trial=4, max_cycle=40, save_freq=5,
        checkpoint_path=str(ckpt_path), checkpoint_freq=10,
    )

    # Resumed run should complete cycles 0..39 (total 40 metrics)
    assert len(m2) == 40
    # Metrics from cycles 0..20 should match between Part 1 and Part 2
    np.testing.assert_allclose(m2[:21], m1[:21])


