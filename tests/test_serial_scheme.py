"""Serial (probe-until-accepted) trial scheme and the streaming cycle log.

The serial scheme runs samplers one after another from the anchor until one
beats the record (the best committed metric so far) and commits that one, as in
PathGennie 0.0.0. These tests pin down its contract on a deterministic 1-D engine:

* the committed probe is the *first* success in index order, and it beats the threshold;
* ``batch_size`` changes only how many probes run at once, never the result;
* ``max_try`` censors a cycle, with ``on_max_try`` choosing "best" or "stay";
* probe handles are released, so a cycle with many probes does not leak state;
* the parallel scheme is unchanged by the refactor and by attaching a logger;
* the log files grow per cycle, with a variable number of trial rows per cycle and
  observable columns added dynamically.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from pathgennie.core.cyclelog import CycleLogger
from pathgennie.core.driver import PathGennieDriver, scheme_options
from pathgennie.core.parallel import SerialExecutor, ThreadDevicePool
from pathgennie.core.progress import EscapeMetric


class Walk1D:
    """x += drift + noise(seed); the escape metric is |x - x0|. Handles are ints."""

    def __init__(self, drift=0.0, noise=1.0):
        self.drift, self.noise = drift, noise
        self.state = {}
        self.next_id = 0
        self.n_segments = 0

    def _new(self, pos):
        h = self.next_id
        self.next_id += 1
        self.state[h] = np.asarray(pos, dtype=float).reshape(1, 3).copy()
        return h

    def create_state(self, coords):
        return self._new(coords)

    create_handle = create_state

    def clone_anchor(self, handle):
        return self._new(self.state[handle])

    def run_segment(self, handle, n_steps, *, randomize_velocities, seed, device=None,
                    save_subframes=False, subframe_stride=1):
        self.n_segments += 1
        pos = self.state[handle].copy()
        pos[0, 0] += self.drift + self.noise * np.random.default_rng(seed).standard_normal()
        return self._new(pos)

    def get_coords(self, handle):
        return self.state[handle]

    def release(self, handle):
        self.state.pop(handle, None)


def make_driver(engine, *, executor=None, logger=None, **kw):
    x0 = np.zeros((1, 3))
    progress = EscapeMetric(lambda c: c[:, 0], np.array([0.0]), escape_metric="distance_from_start")
    driver = PathGennieDriver(engine, progress, lambda c: abs(c[0, 0]) > 6.0, executor=executor,
                              seed=7, verbosity=0, cycle_logger=logger, **kw)
    return driver, engine.create_state(x0)


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def run_serial(tmp_path, name, **kw):
    eng = Walk1D()
    logger = CycleLogger(tmp_path / name)
    driver, h0 = make_driver(eng, logger=logger, scheme="serial", **kw)
    _, metrics = driver.run(h0, tau1=1, tau2=1, max_trial=5, max_cycle=60, save_freq=1)
    return eng, metrics, read(tmp_path / name / "cycles.csv"), read(tmp_path / name / "trials.csv")


def test_serial_commits_first_probe_beating_the_record(tmp_path):
    _, metrics, cycles, trials = run_serial(tmp_path, "a", batch_size=1)
    assert len(cycles) == len(metrics)
    for c in cycles:
        rows = [t for t in trials if t["cycle"] == c["cycle"]]
        assert len(rows) == int(c["n_trials"])            # one trial row per probe
        assert c["censored"] == "0"
        thr = float(c["threshold"])
        assert thr == pytest.approx(float(c["record"]))   # accept="record" by default
        chosen = [t for t in rows if t["chosen"] == "1"]
        assert len(chosen) == 1
        # first success: every earlier probe failed, the chosen one beats the threshold
        k = int(chosen[0]["trial"])
        assert all(float(t["metric"]) <= thr for t in rows[:k])
        assert float(chosen[0]["metric"]) > thr
        assert int(c["chosen_trial"]) == k
        assert rows[-1]["trial"] == str(k)                 # batch_size=1: nothing after the success


def test_record_is_the_running_max_of_committed_metrics(tmp_path):
    _, metrics, cycles, _ = run_serial(tmp_path, "b", batch_size=1)
    rec = np.maximum.accumulate(np.r_[0.0, metrics[:-1]])
    assert np.allclose([float(c["record"]) for c in cycles], rec, atol=1e-5)


def test_batch_size_changes_speed_not_result(tmp_path):
    _, m1, c1, _ = run_serial(tmp_path, "b1", batch_size=1)
    _, m4, c4, t4 = run_serial(tmp_path, "b4", batch_size=4)
    assert np.array_equal(m1, m4)
    assert [c["chosen_trial"] for c in c1] == [c["chosen_trial"] for c in c4]
    # batched cycles evaluate a whole number of batches; extra probes are flagged
    assert all(int(c["n_trials"]) % 4 == 0 for c in c4)
    assert any(t["after_accept"] == "1" for t in t4)


def test_thread_pool_matches_serial_executor(tmp_path):
    eng_a, eng_b = Walk1D(), Walk1D()
    da, ha = make_driver(eng_a, scheme="serial", batch_size=4)
    db, hb = make_driver(eng_b, executor=ThreadDevicePool(workers_per_device=4), scheme="serial")
    assert db.batch_size == 4                                # "auto" = executor workers
    _, ma = da.run(ha, tau1=1, tau2=1, max_trial=5, max_cycle=40)
    _, mb = db.run(hb, tau1=1, tau2=1, max_trial=5, max_cycle=40)
    assert np.array_equal(ma, mb)


@pytest.mark.parametrize("policy", ["best", "stay"])
def test_max_try_censors_the_cycle(tmp_path, policy):
    # Anchor at x=-10 (metric |x| = 10); every probe drifts +2.5 towards the start, so
    # none beats the anchor and each cycle hits max_try.
    eng = Walk1D(drift=2.5, noise=0.01)
    logger = CycleLogger(tmp_path / policy)
    driver, _ = make_driver(eng, logger=logger, scheme="serial", accept="anchor",
                            max_try=6, batch_size=4, on_max_try=policy)
    start = eng.create_state(np.array([[-10.0, 0, 0]]))
    n_live = len(eng.state)
    _, m = driver.run(start, tau1=1, tau2=1, max_trial=5, max_cycle=2)
    cyc = read(tmp_path / policy / "cycles.csv")
    assert all(c["censored"] == "1" and c["accepted"] == "0" and c["n_trials"] == "6" for c in cyc)
    if policy == "stay":
        assert np.allclose(m, 10.0)               # anchor kept, no tau2 run
        assert all(c["chosen_trial"] == "" for c in cyc)
    else:
        assert np.allclose(m, [5.0, 0.0], atol=0.1)   # best probe (-7.5) + tau2 (+2.5) each cycle
    assert len(eng.state) - n_live <= 2          # probes released: no per-probe leak


def test_parallel_scheme_unchanged_by_logger(tmp_path):
    runs = []
    for logger in (None, CycleLogger(tmp_path / "p")):
        eng = Walk1D(drift=0.05)
        driver, h0 = make_driver(eng, logger=logger, sigma=0.3)
        runs.append(driver.run(h0, tau1=1, tau2=1, max_trial=6, max_cycle=50)[1])
    assert np.array_equal(runs[0], runs[1])
    cyc = read(tmp_path / "p" / "cycles.csv")
    trials = read(tmp_path / "p" / "trials.csv")
    assert all(c["scheme"] == "parallel" and c["n_trials"] == "6" for c in cyc)
    assert len(trials) == 6 * len(cyc)
    for c in cyc:
        probs = [float(t["prob"]) for t in trials if t["cycle"] == c["cycle"]]
        assert sum(probs) == pytest.approx(1.0, abs=1e-4)
        chosen = [t for t in trials if t["cycle"] == c["cycle"] and t["chosen"] == "1"]
        assert len(chosen) == 1 and float(chosen[0]["prob"]) == pytest.approx(float(c["chosen_prob"]), rel=1e-4)


def test_log_is_streamed_and_observable_columns_are_dynamic(tmp_path):
    calls = []

    def observables(coords, scale=1.0):
        calls.append(1)
        out = {"x": float(coords[0, 0]) * scale}
        if len(calls) > 3:
            out["late_key"] = len(calls)          # appears after the header was written
        return out

    logger = CycleLogger(tmp_path / "o", observables_fn=observables, observables_args={"scale": 2.0})
    eng = Walk1D()
    driver, h0 = make_driver(eng, logger=logger, scheme="serial", batch_size=1)
    orig = logger.log_cycle
    sizes = []

    def spy(row, coords=None):
        orig(row, coords)
        sizes.append((tmp_path / "o" / "cycles.csv").stat().st_size)   # flushed every cycle

    logger.log_cycle = spy
    driver.run(h0, tau1=1, tau2=1, max_trial=5, max_cycle=10)
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
    cyc = read(tmp_path / "o" / "cycles.csv")
    assert "x" in cyc[0] and "late_key" not in cyc[0]
    assert float(cyc[0]["x"]) == pytest.approx(2.0 * float(cyc[0]["committed_metric"])
                                                * np.sign(float(cyc[0]["x"])), rel=1e-4)
    extra = read(tmp_path / "o" / "cycles_extra.csv")
    assert extra and all(e["key"] == "late_key" for e in extra)
    trials = read(tmp_path / "o" / "trials.csv")
    per_cycle = {c["cycle"]: sum(t["cycle"] == c["cycle"] for t in trials) for c in cyc}
    assert len(set(per_cycle.values())) > 1        # variable number of probes per cycle


def test_scheme_options_and_config():
    from pathgennie.utils.config import AppConfig

    assert scheme_options({}) == {"scheme": "parallel"}
    opts = scheme_options({"scheme": "serial", "serial": {"max_try": 50, "batch_size": 2}})
    assert opts == dict(scheme="serial", accept="record", max_try=50, batch_size=2,
                        on_max_try="best", min_gain=0.0)
    with pytest.raises(ValueError):
        scheme_options({"scheme": "serial", "serial": {"maxtry": 5}})
    cfg = AppConfig.model_validate({
        "pathgennie": {"scheme": "serial", "serial": {"accept": "anchor", "max_try": 10}},
        "log": {"trials": False, "observables": {"module": "m", "function": "f", "cutoff": 8}},
    }).model_dump(exclude_none=True)
    assert cfg["pathgennie"]["serial"]["accept"] == "anchor"
    assert cfg["log"]["cycles"] == "cycles.csv" and cfg["log"]["trials"] is False
    with pytest.raises(Exception):
        AppConfig.model_validate({"pathgennie": {"scheme": "sequential"}})
    with pytest.raises(Exception):
        AppConfig.model_validate({"pathgennie": {"serial": {"max_tries": 3}}})
