"""trial_log: one JSON line per cycle with what every sampler did, so acceptance, gain and the part of the gain
kept after tau2 can be read per cycle (the input of on-the-fly tuning, phaseX/TUNING_STRATEGY.md §5)."""

from __future__ import annotations

import json

import numpy as np

from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import SerialExecutor
from pathgennie.core.progress import EscapeMetric
from tests.test_driver_first_accept import ScriptedEngine


def _run(tmp_path, engine, **kw):
    progress = EscapeMetric(lambda c: np.array([float(c[0, 0])]), start_cv=np.array([0.0]), escape_metric="cv0")
    log = tmp_path / "trials.jsonl"
    drv = PathGennieDriver(engine, progress, lambda c: False, executor=SerialExecutor(), seed=3, verbosity=0,
                           trial_log=str(log), **kw)
    drv.run(engine.create_handle([0, 0, 0]), tau1=2, tau2=3, max_trial=4, max_cycle=2, save_freq=1)
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_each_cycle_logs_every_sampler_run_and_the_runner(tmp_path):
    # samplers (2 steps, fresh velocities) gain 0.1; the runner (3 steps) loses 0.04
    engine = ScriptedEngine(lambda seed, n: 0.1 if n == 2 else -0.04)
    rows = _run(tmp_path, engine, selection="first_accept", accept_batch=2)
    assert [r["cycle"] for r in rows] == [0, 1]
    r = rows[0]
    assert (r["tau1"], r["tau2"], r["n_tried"], r["fallback"]) == (2, 3, 2, False)
    np.testing.assert_allclose(r["anchor_metric"], 0.0)
    np.testing.assert_allclose(r["trial_metrics"], [0.1, 0.1])
    assert r["chosen"] == 0
    np.testing.assert_allclose([r["sampler_metric"], r["runner_metric"], r["metric"]], [0.1, 0.06, 0.06])
    np.testing.assert_allclose(rows[1]["anchor_metric"], 0.06)


def test_a_fallback_cycle_is_logged_as_such(tmp_path):
    engine = ScriptedEngine(lambda seed, n: -0.1)
    r = _run(tmp_path, engine, selection="first_accept", accept_batch=2)[0]
    assert r["fallback"] is True and r["chosen"] is None and r["sampler_metric"] is None and r["n_tried"] == 4
    assert len(r["trial_metrics"]) == 4


def test_softmax_cycles_are_logged_too(tmp_path):
    engine = ScriptedEngine(lambda seed, n: 0.1)
    r = _run(tmp_path, engine, selection="softmax", sigma=0.05)[0]
    assert len(r["trial_metrics"]) == 4 and r["chosen"] in range(4) and r["fallback"] is False


def test_no_log_is_written_unless_asked(tmp_path):
    engine = ScriptedEngine(lambda seed, n: 0.1)
    progress = EscapeMetric(lambda c: np.array([float(c[0, 0])]), start_cv=np.array([0.0]), escape_metric="cv0")
    PathGennieDriver(engine, progress, lambda c: False, executor=SerialExecutor(), seed=3, verbosity=0).run(
        engine.create_handle([0, 0, 0]), tau1=2, tau2=3, max_trial=4, max_cycle=1)
    assert list(tmp_path.iterdir()) == []
