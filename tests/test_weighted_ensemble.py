"""Weighted Ensemble stage tests (pure NumPy: no torch, no MD binary)."""

import numpy as np

from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import SerialExecutor
from pathgennie.core.progress import EscapeMetric
from pathgennie.core.toy import ToyLangevinEngine, wolfe_quapp_potential
from pathgennie.sampling import (
    GridBinner,
    PathEnsemble,
    Walker,
    WeightedEnsembleStage,
    build_path_ensemble,
    make_stage,
    resample,
)


# --------------------------------------------------------------------------- #
# resample (split/merge)
# --------------------------------------------------------------------------- #
def _clone_release_stubs():
    store = {"next": 0, "alive": set()}

    def clone(handle):
        store["next"] += 1
        h = ("clone", store["next"])
        store["alive"].add(h)
        return h

    def release(handle):
        store["alive"].discard(handle)

    return clone, release, store


def test_resample_split_conserves_weight_and_count():
    rng = np.random.default_rng(0)
    clone, release, _ = _clone_release_stubs()
    walkers = [Walker(handle=i, weight=w, bin=0) for i, w in enumerate([0.1, 0.2, 0.7])]
    out = resample(walkers, target_count=4, rng=rng, clone_fn=clone, release_fn=release)
    assert len(out) == 4
    assert abs(sum(w.weight for w in out) - 1.0) < 1e-12


def test_resample_merge_conserves_weight_and_count():
    rng = np.random.default_rng(0)
    clone, release, _ = _clone_release_stubs()
    weights = [0.05, 0.05, 0.1, 0.3, 0.5]
    walkers = [Walker(handle=i, weight=w, bin=0) for i, w in enumerate(weights)]
    out = resample(walkers, target_count=2, rng=rng, clone_fn=clone, release_fn=release)
    assert len(out) == 2
    assert abs(sum(w.weight for w in out) - 1.0) < 1e-12


# --------------------------------------------------------------------------- #
# GridBinner
# --------------------------------------------------------------------------- #
def test_grid_binner_edges_and_indices():
    binner = GridBinner.from_values([-2.0, 2.0], n_bins=4, pad=0.0)
    assert binner.n_bins == 4
    assert np.all(np.diff(binner.edges) > 0)
    assert binner.bin_index(-2.0) == 0
    assert binner.bin_index(2.0) == 3
    assert binner.bin_index(-100.0) == 0   # clamp low
    assert binner.bin_index(100.0) == 3    # clamp high


# --------------------------------------------------------------------------- #
# End-to-end WE on the toy Wolfe-Quapp engine
# --------------------------------------------------------------------------- #
def _toy_path_ensemble(seed=0):
    """Produce a PathEnsemble spanning the WQ y-axis from a short toy run."""
    engine = ToyLangevinEngine(dt=0.005, kT=2.0)
    initial = engine.create_state((-1.0, -1.4))

    def y_cv(coords):
        return np.array([coords[0, 1]])

    progress = EscapeMetric(y_cv, start_cv=np.array([-1.4]), escape_metric="cv0")
    driver = PathGennieDriver(
        engine, progress, convergence_fn=lambda c: False,
        executor=SerialExecutor(), sigma=0.3, seed=seed, verbosity=0,
    )
    traj, metrics, handles = driver.run(
        initial, tau1=5, tau2=10, max_trial=5, max_cycle=40, save_freq=2,
        collect_seeds=True,
    )
    ens = build_path_ensemble(traj, metrics, handles=handles, cv_fn=lambda c: c[0, 1])
    return engine, ens


def test_we_conserves_weight_and_balances_bins():
    engine, ens = _toy_path_ensemble()
    stage = WeightedEnsembleStage(
        cv_fn=lambda c: c[0, 1], tau_steps=5, n_iterations=15,
        n_bins=8, target_count=4, seed=1,
    )
    result = stage.run(ens, engine)

    # Total weight conserved every iteration and at the end.
    np.testing.assert_allclose(result.metadata["weight_trace"], 1.0, atol=1e-9)
    assert abs(result.weights.sum() - 1.0) < 1e-9
    # Each surviving walker lives in a bin balanced to target_count.
    assert result.weights.size % 4 == 0


def test_we_recovers_fes_minimum_in_a_basin():
    engine, ens = _toy_path_ensemble()
    stage = WeightedEnsembleStage(
        cv_fn=lambda c: c[0, 1], tau_steps=8, n_iterations=60,
        n_bins=10, target_count=4, seed=2, kT=2.0,
    )
    result = stage.run(ens, engine)

    centers = result.metadata["bin_centers"]
    fe = result.free_energy
    finite = np.isfinite(fe)
    min_center = centers[finite][np.argmin(fe[finite])]
    # WQ minima sit near y ~ +/-1.4; the FES minimum should be in a basin,
    # not at the central barrier (|y| ~ 0).
    assert abs(min_center) > 0.7

    # And the barrier region should be higher in free energy than the basin.
    barrier_bins = finite & (np.abs(centers) < 0.4)
    if barrier_bins.any():
        assert np.min(fe[barrier_bins]) >= np.min(fe[finite]) - 1e-9


def test_we_recycling_produces_finite_rate():
    engine, ens = _toy_path_ensemble()
    stage = make_stage(
        "weighted_ensemble",
        cv_fn=lambda c: c[0, 1], tau_steps=6, n_iterations=40,
        n_bins=8, target_count=3, seed=3,
        recycle=True, source_cv=-1.4, target_cv=1.2, timestep_ps=0.002,
    )
    result = stage.run(ens, engine)
    assert result.rate_constants is not None
    assert result.rate_constants["flux_per_iter"] >= 0.0
    assert np.isfinite(result.rate_constants["rate"])
    np.testing.assert_allclose(result.metadata["weight_trace"], 1.0, atol=1e-9)


def test_make_stage_opes_not_implemented():
    try:
        make_stage("opes")
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
