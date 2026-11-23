"""selection="first_accept": the selection rule the PathGennie papers describe.

Samplers are drawn in order; the first one whose metric beats the anchor's is accepted and extended by tau2. If
none of max_trial advances, one unbiased tau1+tau2 fallback segment from the anchor supplies the candidate.
Samplers run in parallel batches of ``accept_batch``; within a batch the lowest-index advancing sampler wins, so
the outcome is that of the sequential rule, and only the batches actually run are spent.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import SerialExecutor
from pathgennie.core.progress import EscapeMetric


class ScriptedEngine:
    """1-D engine: each segment moves x by a step chosen from the segment's seed (``step_of(seed)``), and every
    call is recorded as (n_steps, seed, randomize_velocities)."""

    def __init__(self, step_of):
        self.step_of = step_of
        self.state = {}
        self.next_id = 1
        self.calls = []

    def _new(self, pos):
        h = self.next_id
        self.next_id += 1
        self.state[h] = np.asarray(pos, dtype=float).reshape(1, 3).copy()
        return h

    def create_handle(self, coords):
        return self._new(coords)

    create_state = create_handle

    def clone_anchor(self, handle):
        return self._new(self.state[handle])

    def run_segment(self, handle, n_steps, *, randomize_velocities, seed, device=None,
                    save_subframes=False, subframe_stride=1):
        self.calls.append((int(n_steps), int(seed), bool(randomize_velocities)))
        pos = self.state[handle].copy()
        pos[0, 0] += self.step_of(seed, n_steps)
        h = self._new(pos)
        if save_subframes:
            return h, np.repeat(pos[None], 2, axis=0)
        return h

    def get_coords(self, handle):
        return self.state[handle]

    def release(self, handle):
        self.state.pop(handle, None)


def _driver(engine, **kw):
    progress = EscapeMetric(lambda c: np.array([float(c[0, 0])]), start_cv=np.array([0.0]), escape_metric="cv0")
    return PathGennieDriver(engine, progress, lambda c: False, executor=SerialExecutor(), seed=3, verbosity=0, **kw)


def test_the_first_advancing_sampler_is_taken_and_later_batches_are_not_run():
    # every sampler advances: the first batch's first sampler wins, so a cycle costs one batch + one runner
    engine = ScriptedEngine(lambda seed, n: 0.1)
    drv = _driver(engine, selection="first_accept", accept_batch=4)
    drv.run(engine.create_handle([0, 0, 0]), tau1=2, tau2=3, max_trial=12, max_cycle=1, save_freq=1)
    samplers = [c for c in engine.calls if c[2] and c[0] == 2]
    runners = [c for c in engine.calls if not c[2] and c[0] == 3]
    assert len(samplers) == 4          # one batch of 4, not all 12
    assert len(runners) == 1


def test_the_lowest_index_advancing_sampler_wins_within_a_batch():
    # samplers alternate backward/forward by seed parity; the accepted one must be the first forward one in index
    # order, i.e. the runner continues from that sampler's position
    engine = ScriptedEngine(lambda seed, n: (0.1 * (1 + seed % 7)) if seed % 2 else -0.5)
    drv = _driver(engine, selection="first_accept", accept_batch=8)
    traj, metrics = drv.run(engine.create_handle([0, 0, 0]), tau1=2, tau2=2, max_trial=8, max_cycle=1, save_freq=1)
    sampler_seeds = [c[1] for c in engine.calls if c[2] and c[0] == 2][:8]
    first_fwd = next(s for s in sampler_seeds if s % 2)
    expected_after_sampler = 0.1 * (1 + first_fwd % 7)
    runner_seed = [c[1] for c in engine.calls if not c[2]][0]
    expected = expected_after_sampler + engine.step_of(runner_seed, 2)
    assert np.isclose(metrics[-1], expected)


def test_no_advancing_sampler_runs_one_tau1_plus_tau2_fallback():
    # every sampler goes backward: all max_trial are spent, then one fallback segment of tau1+tau2 steps with fresh
    # velocities, and no tau2 runner
    engine = ScriptedEngine(lambda seed, n: -0.2 if n == 2 else 0.3)
    drv = _driver(engine, selection="first_accept", accept_batch=3)
    traj, metrics = drv.run(engine.create_handle([0, 0, 0]), tau1=2, tau2=3, max_trial=6, max_cycle=1, save_freq=1)
    samplers = [c for c in engine.calls if c[0] == 2]
    fallback = [c for c in engine.calls if c[0] == 5]
    runners = [c for c in engine.calls if c[0] == 3]
    assert len(samplers) == 6 and len(fallback) == 1 and fallback[0][2] and not runners
    assert np.isclose(metrics[-1], 0.3)    # the fallback endpoint became the anchor


def test_fallback_is_rejected_when_worse_and_reject_worse_anchor_is_on():
    engine = ScriptedEngine(lambda seed, n: -0.2)
    drv = _driver(engine, selection="first_accept", accept_batch=2, reject_worse_anchor=True)
    traj, metrics = drv.run(engine.create_handle([0, 0, 0]), tau1=2, tau2=3, max_trial=4, max_cycle=2, save_freq=1)
    assert np.allclose(metrics, 0.0)


def test_subframe_replay_handles_the_fallback():
    engine = ScriptedEngine(lambda seed, n: -0.2 if n == 2 else 0.3)
    drv = _driver(engine, selection="first_accept", accept_batch=2, save_subframes=True)
    traj, metrics = drv.run(engine.create_handle([0, 0, 0]), tau1=2, tau2=3, max_trial=2, max_cycle=1, save_freq=1)
    replays = [c for c in engine.calls if c[0] == 5]
    assert len(replays) == 2               # the fallback and its replay, both tau1 + tau2 steps
    assert len(traj) > 0


def test_softmax_remains_the_default_and_bad_names_are_refused():
    engine = ScriptedEngine(lambda seed, n: 0.1)
    drv = _driver(engine)
    assert drv.selection == "softmax"
    drv.run(engine.create_handle([0, 0, 0]), tau1=2, tau2=2, max_trial=5, max_cycle=1, save_freq=1)
    assert len([c for c in engine.calls if c[2]]) == 5      # all samplers run under softmax
    with pytest.raises(ValueError):
        _driver(engine, selection="greedy")
