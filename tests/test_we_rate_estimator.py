"""The ``we`` (wepath) package never computed a rate in code.

``base.py`` logged each iteration as::

    flux_rate = flux_this_iter #/ TAU if TAU > 0 else 0
    self.flux_history.append(flux_rate)

The division by tau is commented out, and ``TAU`` was a **dead local** in
``WESS.run()`` -- assigned once at main.py:93 and never referenced again. So
``flux_history`` held per-iteration *fluxes* (dimensionless probability) under the
name ``flux_rate``, and the rate constant was reconstructed by hand afterwards,
along with the choice of averaging window. That hand-windowing is exactly the
"inspection-based window" objection raised against the pathway-kinetics work.

Two consequences this file pins down:
  1. tau must be stored, so a reported rate is reproducible rather than dependent
     on remembering dt x n_steps_per_tau. (The value has already been ambiguous
     once in this project's history, and every rate scales as 1/tau.)
  2. the average must exclude the transient, because WE flux is a rate only at
     steady state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

WE_SRC = Path(__file__).resolve().parents[1] / "we" / "src"
if str(WE_SRC) not in sys.path:
    sys.path.insert(0, str(WE_SRC))

pytest.importorskip("openmm")
wepath_base = pytest.importorskip("wepath.base")

WeightedEnsembleBase = wepath_base.WeightedEnsembleBase


class _Recorder(WeightedEnsembleBase):
    """Minimal stand-in exposing only what the logging/estimator paths touch."""

    def __init__(self, tau=None, n_iterations=10):
        self.tau = tau
        self.n_iterations = n_iterations
        self.n_total_bins = 1
        self.walkers = []
        self.flux_history = []
        self.rate_history = []
        self.total_flux_to_target = 0.0

    def _get_bin_assignments(self, walkers):
        return []


class _Files(dict):
    """File handles that accept writes and discard them."""

    def __init__(self):
        import io
        super().__init__(bin_f=io.StringIO(), walker_f=io.StringIO())


def test_rate_history_is_flux_divided_by_tau():
    """The regression: a rate must have units of 1/time, not be a bare flux."""
    r = _Recorder(tau=2.0)
    r._log_iteration_data(0, 4.0e-3, _Files())
    assert r.rate_history[-1] == pytest.approx(2.0e-3), (
        "rate_history did not divide the flux by tau -- this is the commented-out "
        "division that meant no rate was ever computed in code"
    )


def test_flux_history_still_holds_the_raw_flux():
    """Both quantities must exist and be distinguishable, not conflated."""
    r = _Recorder(tau=2.0)
    r._log_iteration_data(0, 4.0e-3, _Files())
    assert r.flux_history[-1] == pytest.approx(4.0e-3)
    assert r.flux_history[-1] != r.rate_history[-1]


def test_missing_tau_yields_nan_rather_than_a_flux_mislabelled_as_a_rate():
    r = _Recorder(tau=None)
    r._log_iteration_data(0, 4.0e-3, _Files())
    assert np.isnan(r.rate_history[-1]), "a flux was returned as if it were a rate"
    out = r.rate_estimate()
    assert np.isnan(out["rate"]) and out["n_used"] == 0


def test_burn_in_excludes_the_transient():
    """WE flux is a rate only at steady state; averaging from zero biases it."""
    r = _Recorder(tau=1.0, n_iterations=10)
    r.rate_history = [10.0] * 5 + [1.0] * 5          # transient, then steady
    assert r.rate_estimate(burn_in=0.5)["rate"] == pytest.approx(1.0)
    assert r.rate_estimate(burn_in=0)["rate"] == pytest.approx(5.5), \
        "no-burn-in must still be available, and must show the contamination"


def test_burn_in_accepts_a_count_as_well_as_a_fraction():
    r = _Recorder(tau=1.0, n_iterations=10)
    r.rate_history = [10.0] * 5 + [1.0] * 5
    assert r.rate_estimate(burn_in=5)["rate"] == pytest.approx(1.0)
    assert r.rate_estimate(burn_in=5)["burn_in"] == 5


def test_burn_in_never_discards_every_iteration():
    """An empty estimator is worse than a biased one -- but say what was used."""
    r = _Recorder(tau=1.0, n_iterations=4)
    r.rate_history = [1.0, 2.0, 3.0, 4.0]
    out = r.rate_estimate(burn_in=99)
    assert out["n_used"] >= 1 and out["burn_in"] == 3


def test_steady_state_flag_is_false_when_the_rate_is_still_drifting():
    r = _Recorder(tau=1.0, n_iterations=40)
    r.rate_history = list(np.linspace(1.0, 20.0, 40))       # monotonic drift
    assert r.rate_estimate(burn_in=0)["steady_state"] is False


def test_steady_state_flag_is_true_for_a_stationary_series():
    rng = np.random.default_rng(0)
    r = _Recorder(tau=1.0, n_iterations=40)
    r.rate_history = list(1.0 + 0.01 * rng.standard_normal(40))
    assert r.rate_estimate(burn_in=0)["steady_state"] is True


def test_estimate_reports_stderr_and_the_tau_it_used():
    r = _Recorder(tau=2.5, n_iterations=20)
    r.rate_history = list(1.0 + 0.1 * np.arange(20) % 3)
    out = r.rate_estimate(burn_in=0.5)
    assert out["tau"] == 2.5
    assert out["stderr"] > 0 and np.isfinite(out["stderr"])


def test_empty_history_does_not_raise():
    out = _Recorder(tau=1.0).rate_estimate()
    assert np.isnan(out["rate"]) and out["n_used"] == 0
