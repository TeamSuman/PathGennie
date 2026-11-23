"""Probe mode (phaseX/TUNING_STRATEGY.md §3-4): one unselected batch of trials from an anchor, saved at
sub-frames, plus velocity-continued relaxations of the best trials. Nothing is selected and no product is needed;
the probe yields drift, spread, acceptance and retention as functions of segment length, from which the gain of a
parameter set is predicted.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathgennie.core.parallel import SerialExecutor
from pathgennie.core.probe import (
    expected_cost, predicted_gain, probe_stats, rank_parameter_sets, read_regime, run_probe, selected_gain,
)
from pathgennie.core.progress import EscapeMetric


class LineEngine:
    """1-D engine. A sampler (fresh velocities) moves x by ``v(seed)`` per step; a runner (velocities kept) moves
    it by ``relax`` per step. Sub-frames are returned every ``subframe_stride`` steps."""

    def __init__(self, v, relax=0.0, fail=()):
        self.v, self.relax, self.fail = v, relax, set(fail)
        self.state, self.next_id, self.calls, self.n_sampler = {}, 1, [], 0

    def _new(self, pos):
        h = self.next_id
        self.next_id += 1
        self.state[h] = np.asarray(pos, dtype=float).reshape(1, 3).copy()
        return h

    create_handle = _new

    def clone_anchor(self, handle):
        return self._new(self.state[handle])

    def run_segment(self, handle, n_steps, *, randomize_velocities, seed, device=None,
                    save_subframes=False, subframe_stride=1):
        self.calls.append((int(n_steps), bool(randomize_velocities), bool(save_subframes), int(subframe_stride)))
        if randomize_velocities:
            k = self.n_sampler
            self.n_sampler += 1
            if k in self.fail:
                raise RuntimeError("segment died")
            rate = self.v(k)
        else:
            rate = self.relax
        pos = self.state[handle].copy()
        frames = []
        for step in range(1, int(n_steps) + 1):
            pos = pos.copy()
            pos[0, 0] += rate
            if step % subframe_stride == 0:
                frames.append(pos)
        h = self._new(pos)
        return (h, np.array(frames)) if save_subframes else h

    def get_coords(self, handle):
        return self.state[handle]

    def release(self, handle):
        self.state.pop(handle, None)


def _progress():
    return EscapeMetric(lambda c: np.array([float(c[0, 0])]), start_cv=np.array([0.0]), escape_metric="cv0")


def _probe(engine, **kw):
    args = dict(n_trials=4, n_steps=8, stride=2, n_continue=2, seed=7, executor=SerialExecutor())
    args.update(kw)
    return run_probe(engine, _progress(), engine.create_handle([1.0, 0, 0]), **args)


def test_probe_records_every_trial_at_every_subframe_relative_to_the_anchor():
    engine = LineEngine(lambda k: 0.1 * (k - 1))            # rates -0.1, 0, 0.1, 0.2
    p = _probe(engine)
    assert list(p["steps"]) == [2, 4, 6, 8]
    assert p["anchor_metric"] == pytest.approx(1.0)
    assert p["gain"].shape == (4, 4)
    np.testing.assert_allclose(p["gain"][:, -1], [-0.8, 0.0, 0.8, 1.6])
    np.testing.assert_allclose(p["gain"][3], [0.4, 0.8, 1.2, 1.6])
    assert p["frames"].shape == (4, 4, 1, 3)


def test_probe_runs_all_trials_unselected_with_fresh_velocities_and_subframes():
    engine = LineEngine(lambda k: 0.1)
    _probe(engine)
    samplers = [c for c in engine.calls if c[1]]
    assert samplers == [(8, True, True, 2)] * 4


def test_probe_continues_the_best_trials_with_velocities_kept():
    engine = LineEngine(lambda k: 0.1 * (k - 1), relax=-0.05)
    p = _probe(engine)
    runners = [c for c in engine.calls if not c[1]]
    assert runners == [(8, False, True, 2)] * 2
    assert list(p["continued"]) == [3, 2]                   # best first
    np.testing.assert_allclose(p["cont_gain"][0], [1.5, 1.4, 1.3, 1.2])
    np.testing.assert_allclose(p["cont_gain"][1], [0.7, 0.6, 0.5, 0.4])


def test_probe_releases_every_handle_it_made():
    engine = LineEngine(lambda k: 0.1, relax=0.0)
    anchor = engine.create_handle([1.0, 0, 0])
    run_probe(engine, _progress(), anchor, n_trials=4, n_steps=8, stride=2, n_continue=2, seed=7,
              executor=SerialExecutor())
    assert list(engine.state) == [anchor]


def test_a_failed_trial_is_marked_and_left_out_of_the_statistics():
    engine = LineEngine(lambda k: 0.1, fail={1})
    p = _probe(engine)
    assert np.isnan(p["gain"][1]).all() and np.isfinite(p["gain"][[0, 2, 3]]).all()
    s = probe_stats(p)
    assert s["n"] == 3
    np.testing.assert_allclose(s["mu"], [0.2, 0.4, 0.6, 0.8])


def test_probe_is_reproducible_for_a_seed():
    seen = []
    for _ in range(2):
        engine = LineEngine(lambda k: 0.1)
        seen.append(_probe(engine)["seeds"])
    assert list(seen[0]) == list(seen[1]) and len(set(seen[0])) == 4


def test_stats_give_drift_spread_acceptance_and_retention_by_length():
    engine = LineEngine(lambda k: 0.1 * (k - 1), relax=-0.05)
    s = probe_stats(_probe(engine))
    np.testing.assert_allclose(s["mu"][-1], 0.4)
    np.testing.assert_allclose(s["s"][-1], np.std([-0.8, 0.0, 0.8, 1.6], ddof=1))
    np.testing.assert_allclose(s["a"], [0.5, 0.5, 0.5, 0.5])
    # retention = gain left after t2 over gain before, summed over the continued trials: (1.5+0.7)/(1.6+0.8)
    np.testing.assert_allclose(s["R"][0], 2.2 / 2.4)
    np.testing.assert_allclose(s["R"][-1], 1.6 / 2.4)


def _stats(gain, cont_gain=None, steps=(2, 4)):
    gain = np.asarray(gain, float)
    p = {"steps": np.array(steps), "gain": gain, "anchor_metric": 0.0,
         "continued": np.array([0]), "cont_gain": np.asarray(cont_gain if cont_gain is not None
                                                             else [gain[0, -1]] * len(steps), float)[None]}
    return probe_stats(p)


def test_predicted_gain_of_best_of_n_is_the_empirical_expected_maximum_times_retention():
    s = _stats([[1.0, 1.0], [3.0, 3.0]], cont_gain=[0.5, 0.25])      # R = 0.5, 0.25
    # n = 1: mean 2; n = 2 from two values: E[max] = 1*(1/4) + 3*(3/4) = 2.5
    assert predicted_gain(s, n=1, tau1=4, tau2=2, rule="best") == pytest.approx(2.0 * 0.5)
    assert predicted_gain(s, n=2, tau1=4, tau2=4, rule="best") == pytest.approx(2.5 * 0.25)


def test_predicted_gain_of_first_accept_is_the_mean_gaining_trial_weighted_by_the_chance_of_one():
    s = _stats([[-1.0, -1.0], [3.0, 3.0]], cont_gain=[-1.0, -1.0])
    # a = 0.5; with n = 2 the chance of a gaining trial is 0.75 and its mean gain 3; R is 1 here
    s["R"] = np.array([1.0, 1.0])
    assert predicted_gain(s, n=2, tau1=4, tau2=2, rule="first_accept") == pytest.approx(0.75 * 3.0)


def test_tau2_zero_keeps_the_whole_gain_and_lengths_off_the_grid_are_refused():
    s = _stats([[1.0, 1.0], [3.0, 3.0]], cont_gain=[0.5, 0.25])
    assert predicted_gain(s, n=1, tau1=2, tau2=0, rule="best") == pytest.approx(2.0)
    with pytest.raises(ValueError):
        predicted_gain(s, n=1, tau1=3, tau2=2, rule="best")


def test_cost_counts_force_calls_and_sequential_steps():
    s = _stats([[-1.0, -1.0], [3.0, 3.0]])                  # a = 0.5
    calls, wall = expected_cost(s, n=4, tau1=4, tau2=2, rule="best")
    assert (calls, wall) == (4 * 4 + 2, 4 + 2)
    # first-accept in batches of 2: second batch runs with probability 0.25
    calls, wall = expected_cost(s, n=4, tau1=4, tau2=2, rule="first_accept", batch=2)
    assert calls == pytest.approx(2 * 4 * 1.25 + 2) and wall == pytest.approx(4 * 1.25 + 2)


def test_ranking_orders_parameter_sets_by_predicted_gain_per_force_call():
    s = _stats([[1.0, 1.0], [3.0, 3.0]], cont_gain=[1.0, 1.0])
    sets = {"wide": dict(n=2, tau1=4, tau2=2, rule="best"), "single": dict(n=1, tau1=2, tau2=2, rule="best")}
    ranked = rank_parameter_sets(s, sets)
    assert [r["name"] for r in ranked] == ["single", "wide"]        # 2/4 per call beats 2.5/10
    assert ranked[0]["gain_per_call"] == pytest.approx(2.0 / 4)


@pytest.mark.parametrize("gain, regime", [
    ([[0.5, 1.0]] * 9 + [[-0.1, -0.1]], "downhill"),
    ([[-0.2, -0.6]] * 8 + [[0.2, 0.1]] * 2, "climb"),
    ([[0.3, 0.6], [-0.3, -0.6]] * 5, "plateau"),
])
def test_regime_is_read_from_drift_and_acceptance(gain, regime):
    assert read_regime(_stats(gain)) == regime


# --- revision of 18 Nov 2026 (phaseX/TUNE1_probe_F_MeCl/README.md, "Revision 2"): relaxation is read from the
# --- selected trials themselves, as a gain after tau2 relative to the anchor, not as a ratio


def test_relaxations_can_be_longer_than_the_sampler_segments():
    engine = LineEngine(lambda k: 0.1 * (k + 1), relax=-0.05)
    p = _probe(engine, n_steps=4, relax_steps=12)
    assert list(p["steps"]) == [2, 4]
    assert list(p["cont_steps"]) == [2, 4, 6, 8, 10, 12]
    assert p["cont_gain"].shape == (2, 6)
    # best trial: 0.4 per step for 4 steps, then -0.05 per step for 12
    assert p["cont_gain"][0, -1] == pytest.approx(1.6 - 0.6)


def test_relaxation_length_defaults_to_the_sampler_length():
    p = _probe(LineEngine(lambda k: 0.1))
    assert list(p["cont_steps"]) == list(p["steps"])


def _made(gain_end, continued, cont_end):
    """A probe of one sub-frame at step 10 and relaxations read at step 10."""
    return {"steps": np.array([10]), "cont_steps": np.array([10]),
            "gain": np.array(gain_end, dtype=float).reshape(-1, 1), "continued": np.array(continued, dtype=int),
            "cont_gain": np.array(cont_end, dtype=float).reshape(-1, 1)}


def test_first_accept_net_gain_is_the_mean_relaxed_gain_of_the_gaining_trials():
    # trials 0 and 1 gain; after relaxation they sit at +0.1 and -0.3 from the anchor. Trial 2 lost and is ignored.
    p = _made([0.4, 0.2, -0.1, -0.5], [0, 1, 2], [0.1, -0.3, -0.9])
    a = 0.5
    assert selected_gain(p, n=1, tau2=10, rule="first_accept") == pytest.approx(-0.1 * a)
    assert selected_gain(p, n=3, tau2=10, rule="first_accept") == pytest.approx(-0.1 * (1 - (1 - a) ** 3))


def test_first_accept_net_gain_without_relaxation_is_the_sampler_gain():
    p = _made([0.4, 0.2, -0.1, -0.5], [0, 1], [0.1, -0.3])
    assert selected_gain(p, n=1, tau2=0, rule="first_accept") == pytest.approx(0.3 * 0.5)


def test_first_accept_net_gain_is_zero_when_no_trial_gains():
    p = _made([-0.4, -0.2], [1, 0], [0.5, 0.5])
    assert selected_gain(p, n=8, tau2=10, rule="first_accept") == 0.0


def test_best_rule_weights_the_relaxed_gains_by_the_chance_of_being_the_maximum():
    # N = 4, n = 2: the best trial is the maximum with chance 1 - (3/4)^2 = 7/16, the second with 5/16.
    p = _made([0.4, 0.2, -0.1, -0.5], [0, 1], [0.1, -0.3])
    want = (7 * 0.1 + 5 * -0.3) / 12          # weights renormalised over the continued trials
    assert selected_gain(p, n=2, tau2=10, rule="best") == pytest.approx(want)


def test_selected_gain_refuses_a_relaxation_length_off_the_grid():
    p = _made([0.4, 0.2], [0, 1], [0.1, -0.3])
    with pytest.raises(ValueError):
        selected_gain(p, n=1, tau2=7, rule="first_accept")
