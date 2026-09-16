"""Tests for Task A4's per-cycle `schedule` hook on `PathGennieDriver.run`."""

from __future__ import annotations

import numpy as np

from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import SerialExecutor
from pathgennie.core.progress import EscapeMetric


class RecordingEngine:
    """A minimal fake engine that records the segment lengths it was asked to
    run, split by sampler (randomize_velocities=True) vs runner (=False), so a
    test can verify the driver actually used the per-cycle (tau1, tau2,
    max_trial) a schedule hook returned instead of the values passed to run()."""

    def __init__(self):
        self.state = {0: np.zeros((1, 3))}
        self.next_id = 1
        self.sampler_calls: list[int] = []
        self.runner_calls: list[int] = []

    def _new(self, pos):
        h = self.next_id
        self.next_id += 1
        self.state[h] = pos
        return h

    def create_state(self, coords):
        return self._new(np.asarray(coords, dtype=float).reshape(1, 3))

    create_handle = create_state

    def clone_anchor(self, handle):
        return self._new(self.state[handle].copy())

    def run_segment(self, handle, n_steps, *, randomize_velocities, seed, device=None,
                     save_subframes=False, subframe_stride=1):
        pos = self.state[handle].copy()
        if randomize_velocities:
            self.sampler_calls.append(int(n_steps))
        else:
            self.runner_calls.append(int(n_steps))
        pos[0, 0] += 0.1 + 0.001 * (seed % 7)
        return self._new(pos)

    def get_coords(self, handle):
        return self.state[handle]

    def release(self, handle):
        self.state.pop(handle, None)


def _make_driver(engine):
    progress = EscapeMetric(lambda c: np.array([float(c[0, 0])]),
                             start_cv=np.array([0.0]), escape_metric="cv0")
    return PathGennieDriver(engine, progress, lambda c: False,
                             executor=SerialExecutor(), sigma=0.3, seed=0, verbosity=0)


def test_schedule_called_once_per_cycle_with_cycle_and_anchor_cv():
    engine = RecordingEngine()
    handle = engine.create_state([0.0, 0.0, 0.0])
    driver = _make_driver(engine)

    schedule_calls = []

    def schedule(cycle, anchor_cv):
        schedule_calls.append((cycle, np.asarray(anchor_cv, dtype=float).copy()))
        return (2, 3, 3) if cycle % 2 == 0 else (4, 3, 5)

    max_cycle = 6
    driver.run(handle, tau1=999, tau2=999, max_trial=999, max_cycle=max_cycle,
               save_freq=1, schedule=schedule)

    assert [c for c, _ in schedule_calls] == list(range(max_cycle))
    # The very first call sees the anchor CV of the initial handle (x == 0).
    assert schedule_calls[0][1] == np.array([0.0])


def test_schedule_history_matches_cycles_and_drives_the_worker():
    engine = RecordingEngine()
    handle = engine.create_state([0.0, 0.0, 0.0])
    driver = _make_driver(engine)

    def schedule(cycle, anchor_cv):
        return (2, 3, 3) if cycle % 2 == 0 else (4, 3, 5)

    max_cycle = 6
    driver.run(handle, tau1=999, tau2=999, max_trial=999, max_cycle=max_cycle,
               save_freq=1, schedule=schedule)

    expected = [(2, 3, 3), (4, 3, 5)] * (max_cycle // 2)
    assert driver.schedule_history == expected
    assert len(driver.schedule_history) == max_cycle

    # Replay the recorded engine calls cycle by cycle: each cycle's sampler
    # trials must equal that cycle's scheduled tau1/max_trial, and its runner
    # segment must equal that cycle's scheduled tau2 -- never the tau1=999,
    # tau2=999, max_trial=999 passed to run().
    idx_s = idx_r = 0
    for tau1, tau2, max_trial in driver.schedule_history:
        batch = engine.sampler_calls[idx_s: idx_s + max_trial]
        assert len(batch) == max_trial
        assert all(n == tau1 for n in batch)
        idx_s += max_trial
        assert engine.runner_calls[idx_r] == tau2
        idx_r += 1
    assert idx_s == len(engine.sampler_calls)
    assert idx_r == len(engine.runner_calls)


def test_without_schedule_history_is_empty_and_behaviour_unchanged():
    engine = RecordingEngine()
    handle = engine.create_state([0.0, 0.0, 0.0])
    driver = _make_driver(engine)
    traj, metrics = driver.run(handle, tau1=2, tau2=3, max_trial=4, max_cycle=5, save_freq=1)
    assert driver.schedule_history == []
    assert len(engine.sampler_calls) == 4 * 5
    assert all(n == 2 for n in engine.sampler_calls)
    assert all(n == 3 for n in engine.runner_calls)
