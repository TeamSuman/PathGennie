"""Weighted Ensemble free energies must be able to discard the transient.

The estimator averaged bin occupancy from iteration 0, including the relaxation
away from whatever seeded the run. Seeding is commonly one walker per bin -- a
uniform distribution, maximally unlike the Boltzmann distribution being estimated
-- so the transient flattens the profile and biases barriers low. Worse, only the
*total* weight was traced, so a user could not even correct for it afterwards.

These tests use a toy engine whose "coordinates" are the CV itself, so the exact
equilibrium answer is known and the bias is measurable rather than argued.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathgennie.sampling import WeightedEnsembleStage, build_path_ensemble


class DriftEngine:
    """1-D walkers that relax from wherever they start toward cv = 0.

    Deterministic and monotone, so "the early iterations are unrepresentative" is
    a fact about the trajectory rather than a statistical claim.
    """

    def __init__(self, decay=0.5):
        self.decay = decay
        self.state = {}
        self.next_id = 0

    def _new(self, cv):
        h = self.next_id
        self.next_id += 1
        self.state[h] = float(cv)
        return h

    def create_state(self, coords):
        return self._new(np.asarray(coords, dtype=float).ravel()[0])

    def create_handle(self, coords):
        return self.create_state(coords)

    def clone_anchor(self, handle):
        return self._new(self.state[handle])

    def run_segment(self, handle, n_steps, *, randomize_velocities, seed, device=None,
                    save_subframes=False, subframe_stride=1):
        return self._new(self.state[handle] * self.decay)

    def get_coords(self, handle):
        return np.array([[self.state[handle], 0.0, 0.0]])

    def release(self, handle):
        self.state.pop(handle, None)


def _run(**kw):
    engine = DriftEngine()
    # Seed spread over the whole range: the artificial uniform start.
    frames = np.array([[[cv, 0.0, 0.0]] for cv in np.linspace(0.1, 3.9, 8)])
    handles = [engine.create_state(f) for f in frames]
    ensemble = build_path_ensemble(
        frames=frames, metrics=np.linspace(0.1, 3.9, 8), handles=handles,
    )
    stage = WeightedEnsembleStage(
        cv_fn=lambda c: float(np.asarray(c).ravel()[0]),
        tau_steps=1, n_iterations=20, bin_edges=np.linspace(0.0, 4.0, 9),
        target_count=2, seed=3, kT=1.0, **kw,
    )
    return stage.run(ensemble, engine)


def test_per_iteration_occupancy_is_recorded():
    """Without this trace, burn-in cannot be chosen or checked after the fact."""
    res = _run()
    trace = res.metadata["bin_weight_trace"]
    assert trace.shape == (20, 8)
    assert np.all(trace >= 0)
    # Row sums are the per-iteration total weight, which WE conserves.
    assert np.allclose(trace.sum(axis=1), res.metadata["weight_trace"])


def test_default_is_unchanged_behaviour():
    res = _run()
    assert res.metadata["burn_in"] == 0
    trace = res.metadata["bin_weight_trace"]
    prob = trace.sum(axis=0) / trace.sum()
    assert np.allclose(res.metadata["bin_probability"], prob)


def test_burn_in_excludes_the_transient():
    """The estimate must come only from post-burn-in iterations."""
    res = _run(burn_in=10)
    assert res.metadata["burn_in"] == 10
    trace = res.metadata["bin_weight_trace"]
    expected = trace[10:].sum(axis=0) / trace[10:].sum()
    assert np.allclose(res.metadata["bin_probability"], expected)
    # And it is genuinely a different answer than averaging everything.
    all_iters = trace.sum(axis=0) / trace.sum()
    assert not np.allclose(expected, all_iters)


def test_burn_in_removes_a_measurable_bias():
    """Walkers relax to cv=0, so at long times all weight belongs in bin 0.

    Averaging from iteration 0 leaves weight smeared across the upper bins purely
    because of where the run was seeded.
    """
    biased = _run()
    corrected = _run(burn_in=0.75)
    tail = slice(1, None)
    assert corrected.metadata["bin_probability"][tail].sum() \
        < biased.metadata["bin_probability"][tail].sum(), \
        "burn-in did not reduce occupancy attributable to the seeded start"
    assert corrected.metadata["bin_probability"][0] > biased.metadata["bin_probability"][0]


def test_fractional_burn_in_is_a_fraction_of_the_run():
    assert _run(burn_in=0.5).metadata["burn_in"] == 10
    assert _run(burn_in=0.25).metadata["burn_in"] == 5


def test_burn_in_never_discards_every_iteration():
    """An empty estimator is worse than a biased one; clamp instead."""
    res = _run(burn_in=999)
    assert res.metadata["burn_in"] == 19
    assert np.isfinite(res.free_energy).any()


def test_negative_burn_in_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        WeightedEnsembleStage(cv_fn=lambda c: 0.0, tau_steps=1, n_iterations=5,
                              burn_in=-1)


def test_flux_trace_is_recorded_per_iteration():
    """The rate is a steady-state quantity, so it needs the same treatment."""
    res = _run()
    assert res.metadata["flux_trace"].shape == (20,)
