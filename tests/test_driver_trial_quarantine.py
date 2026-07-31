"""One bad trial must not kill the whole run.

The swarm exists precisely because individual short segments are unreliable:
an integrator can blow up, an MD subprocess can be OOM-killed, a scratch write can
fail. The engines already detect the worst of it -- all three raise ``ValueError``
from ``get_coords`` on non-finite coordinates.

But the driver's trial loop had no guard, so a single failing trial out of ``N``
propagated straight out of ``executor.map`` and ended the run. On a multi-hour
production job that discards every completed cycle because one walker of thirty
went unstable.

The swarm should quarantine the casualty and select from the survivors, and only
give up when a whole cycle fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import SerialExecutor
from pathgennie.core.progress import EscapeMetric


class FlakyEngine:
    """1-D drift engine where selected trials misbehave.

    ``blow_up_every`` makes every n-th segment produce NaN coordinates (what an
    unstable integrator does); ``raise_every`` makes it raise outright (what a
    failed subprocess does).
    """

    def __init__(self, blow_up_every=0, raise_every=0, fail_all=False):
        self.blow_up_every = blow_up_every
        self.raise_every = raise_every
        self.fail_all = fail_all
        self.state = {0: np.zeros((1, 3))}
        self.next_id = 1
        self.n_segments = 0
        self.released = []

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
        self.n_segments += 1
        n = self.n_segments
        if self.fail_all or (self.raise_every and n % self.raise_every == 0):
            raise RuntimeError("simulated MD subprocess failure")
        pos = self.state[handle].copy()
        if self.blow_up_every and n % self.blow_up_every == 0:
            pos[:] = np.nan
        else:
            pos[0, 0] += 0.1
        return self._new(pos)

    def get_coords(self, handle):
        coords = self.state[handle]
        if not np.all(np.isfinite(coords)):
            raise ValueError("engine produced non-finite coordinates")
        return coords

    def release(self, handle):
        self.released.append(handle)
        self.state.pop(handle, None)


def _drive(engine, max_trial=6, max_cycle=8):
    handle = engine.create_state([0.0, 0.0, 0.0])
    progress = EscapeMetric(lambda c: np.array([float(c[0, 0])]),
                            start_cv=np.array([0.0]), escape_metric="cv0")
    driver = PathGennieDriver(engine, progress, lambda c: False,
                              executor=SerialExecutor(), sigma=0.3, seed=0, verbosity=0)
    return driver.run(handle, tau1=2, tau2=2, max_trial=max_trial,
                      max_cycle=max_cycle, save_freq=1)


def test_a_blown_up_trial_does_not_kill_the_run():
    """NaN coordinates in some trials: the run must complete on the survivors."""
    engine = FlakyEngine(blow_up_every=4)
    traj, metrics = _drive(engine)
    assert len(traj) > 0
    assert np.all(np.isfinite(np.asarray(traj)))


def test_a_raising_trial_does_not_kill_the_run():
    """A failed MD subprocess is just as ordinary as an unstable integrator."""
    engine = FlakyEngine(raise_every=3)
    traj, metrics = _drive(engine)
    assert len(traj) > 0
    assert np.all(np.isfinite(np.asarray(traj)))


def test_quarantined_trials_are_released():
    """A discarded trial still holds a scratch file / cache entry."""
    engine = FlakyEngine(blow_up_every=3)
    _drive(engine)
    assert engine.released, "failed trials leaked their handles"


def test_a_wholly_failed_cycle_still_raises():
    """Quarantine must not become silent failure: if nothing survives, say so."""
    engine = FlakyEngine(fail_all=True)
    with pytest.raises(RuntimeError):
        _drive(engine)


def test_a_clean_run_is_unaffected():
    """No failures, no behaviour change."""
    engine = FlakyEngine()
    traj, metrics = _drive(engine)
    assert len(traj) > 0
    assert engine.n_segments > 0
