"""Committor learning from PathGennie swarms (skipped without PyTorch/SciPy)."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")

from pathgennie.committor import BasinPair, SphereBasin, SwarmDataset, SwarmRecorder  # noqa: E402
from pathgennie.committor.basins import LABEL_NONE  # noqa: E402
from pathgennie.committor.reference import grid_committor  # noqa: E402
from pathgennie.committor.train import CommittorModel, committor_scores, train_committor  # noqa: E402
from pathgennie.core.driver import PathGennieDriver  # noqa: E402
from pathgennie.core.progress import TargetMetric  # noqa: E402
from pathgennie.core.toy import ToyLangevinEngine, wolfe_quapp_potential  # noqa: E402

A = np.array([-1.174, 1.477])
B = np.array([1.124, -1.485])
R = 0.4
BASINS = BasinPair(SphereBasin(A, R), SphereBasin(B, R))


def _xy(coords):
    return np.array([coords[0, 0], coords[0, 1]])


def _discover(seed, src, dst):
    """One PathGennie run src -> dst, keeping every swarm."""
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    rec = SwarmRecorder(featurize=_xy, basins=BASINS)
    driver = PathGennieDriver(
        engine, TargetMetric(_xy, target_cv=dst),
        lambda c: bool(np.linalg.norm(_xy(c) - dst) < 0.3),
        sigma=0.5, seed=seed, verbosity=0, swarm_recorder=rec,
    )
    driver.run(engine.create_state(src), tau1=20, tau2=40, max_trial=16, max_cycle=400, save_freq=5)
    return rec.dataset()


def _collect(seeds):
    runs = []
    for seed in seeds:
        runs.append(_discover(seed, A, B))
        runs.append(_discover(1000 + seed, B, A))
    return SwarmDataset.concatenate(runs)


@pytest.fixture(scope="module")
def reference():
    return grid_committor(
        wolfe_quapp_potential,
        lambda x, y: (x - A[0]) ** 2 + (y - A[1]) ** 2 < R * R,
        lambda x, y: (x - B[0]) ** 2 + (y - B[1]) ** 2 < R * R,
        xlim=(-2.6, 2.6), ylim=(-2.6, 2.6), n=161, kT=1.0,
    )


@pytest.fixture(scope="module")
def train_data():
    return _collect(range(6))


@pytest.fixture(scope="module")
def model(train_data):
    return train_committor(train_data, steps=2500, seed=0)


def test_committor_from_discovery_swarms_matches_exact(model, train_data, reference):
    """Milestone M1: swarms recorded during ordinary PathGennie discovery runs are
    enough to learn the committor along the discovered transition tube."""
    held_out = _collect(range(500, 504))
    points = held_out.starts[held_out.start_label == LABEL_NONE]
    exact = reference(points)
    scores = committor_scores(model.predict(points), exact)

    assert scores["n_transition"] >= 30
    assert scores["pearson"] > 0.97
    assert scores["pearson_transition"] > 0.93
    assert scores["rmse_transition"] < 0.15

    # ...and it is far better than reading progress off the geometric CV that
    # steered the runs (relative distance to the two minima).
    da = np.linalg.norm(points - A, axis=1)
    db = np.linalg.norm(points - B, axis=1)
    geometric = committor_scores(da / (da + db), exact)
    assert scores["rmse_transition"] < 0.6 * geometric["rmse_transition"]

    # The cost of the training data is a few shooting points' worth of dynamics.
    assert train_data.total_steps < 600_000


def test_selected_trials_alone_are_biased(train_data, reference):
    """Why every trial is recorded: a target built from the selected endpoint only
    is conditioned on the outcome and no longer estimates E[q(X_tau) | X_0]."""
    use = (train_data.start_label == LABEL_NONE) & train_data.mask.all(axis=1)
    data = train_data.subset(use)
    q_end = reference(data.ends.reshape(-1, 2)).reshape(data.ends.shape[:2])
    q_end = np.where(data.end_label >= 0, data.end_label, q_end)
    q_start = reference(data.starts)

    all_trials = q_end.mean(axis=1)
    chosen_only = (q_end * data.chosen).sum(axis=1)
    # Runs go both ways, so the selection bias has both signs; compare magnitudes.
    err_all = np.sqrt(np.mean((all_trials - q_start) ** 2))
    err_chosen = np.sqrt(np.mean((chosen_only - q_start) ** 2))
    assert err_all < 0.05
    assert err_chosen > 2.0 * err_all


def test_training_needs_both_basins(train_data):
    one_sided = SwarmDataset(**{name: getattr(train_data, name).copy() for name in SwarmDataset._FIELDS})
    one_sided.end_label[one_sided.end_label == 1] = LABEL_NONE
    with pytest.raises(ValueError, match="both basins"):
        train_committor(one_sided, steps=10)


def test_model_roundtrip_and_logit(model, tmp_path):
    points = np.array([[0.0, 0.0], [-1.0, 0.3], [0.9, 0.2]])
    path = tmp_path / "committor.pt"
    model.save(path)
    back = CommittorModel.load(path)
    np.testing.assert_allclose(back.predict(points), model.predict(points), rtol=1e-6)
    q = model.predict(points)
    np.testing.assert_allclose(model.logit(points), np.log(q / (1.0 - q)), rtol=1e-4, atol=1e-4)
    assert isinstance(model.logit(points[0]), float)


def test_warm_start_keeps_standardization(model, train_data):
    refreshed = train_committor(train_data.subset(slice(0, 200)), steps=100, lr=1e-4, init=model)
    np.testing.assert_array_equal(refreshed.feature_mean, model.feature_mean)
    np.testing.assert_array_equal(refreshed.feature_std, model.feature_std)
    assert len(refreshed.history) == len(model.history) + 1
    # a short, gentle refresh starts from the old model rather than from scratch
    points = np.array([[0.0, 0.0], [-1.0, 0.3], [0.9, 0.2]])
    np.testing.assert_allclose(refreshed.predict(points), model.predict(points), atol=0.1)
    # and the model handed in is left untouched (still frozen, same weights)
    assert not any(p.requires_grad for p in model.net.parameters())


def test_scores_report_transition_region_separately():
    ref = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.3, 0.5, 0.7])
    pred = ref.copy()
    pred[6:] = [0.7, 0.5, 0.3]                 # wrong exactly where it matters
    scores = committor_scores(pred, ref)
    assert scores["pearson"] > 0.8             # the easy points hide the failure
    assert scores["pearson_transition"] == pytest.approx(-1.0)
    assert scores["n_transition"] == 3


def test_early_stopping_uses_a_held_out_split(train_data):
    plain = train_committor(train_data, steps=1500, seed=0)
    stopped = train_committor(train_data, steps=1500, seed=0, val_fraction=0.2, patience=3)
    assert plain.val_history == []
    assert len(stopped.val_history) >= 1
    assert len(stopped.history) <= len(plain.history)            # may stop before `steps`
    assert len(stopped.val_history) == len(stopped.history)      # one evaluation per 100 steps
    # the returned weights are those of the best held-out evaluation, not the last
    best = int(np.argmin(stopped.val_history))
    assert len(stopped.val_history) - 1 - best <= 3
    points = np.array([[0.0, 0.0], [-1.0, 0.3], [0.9, 0.2]])
    assert np.all((stopped.predict(points) > 0) & (stopped.predict(points) < 1))


def test_validation_split_is_skipped_for_tiny_datasets(train_data):
    tiny = train_data.subset(np.flatnonzero(train_data.start_label == LABEL_NONE)[:12])
    model = train_committor(tiny, steps=100, val_fraction=0.5, require_both_basins=False)
    assert model.val_history == []
