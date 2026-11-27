"""Committor-steered selection and the closed loop (skipped without PyTorch/SciPy)."""

import numpy as np
import pytest

from pathgennie.core.selection import boltzmann_probs, selection_probs

A = np.array([-1.174, 1.477])
B = np.array([1.124, -1.485])
R = 0.4


def _xy(coords):
    return np.array([coords[0, 0], coords[0, 1]])


# ---- selection weights (NumPy only) ---------------------------------------------

def test_boltzmann_probs_with_log_committor_are_proportional_to_committor():
    q = np.array([0.02, 0.1, 0.4, 0.4, 0.9])
    np.testing.assert_allclose(boltzmann_probs(np.log(q), sigma=1.0), q / q.sum())


def test_boltzmann_weight_ratio_does_not_depend_on_the_rest_of_the_batch():
    """The min-max softmax rescales by the batch range; Boltzmann weights do not."""
    pair = np.log([0.2, 0.4])
    for extra in ([], [np.log(0.9)], [np.log(1e-3), np.log(0.99)]):
        p = boltzmann_probs(np.concatenate([pair, extra]))
        assert p[1] / p[0] == pytest.approx(2.0)
    narrow = selection_probs(np.concatenate([pair, [np.log(0.5)]]), 0.5)
    wide = selection_probs(np.concatenate([pair, [np.log(1e-3)]]), 0.5)
    assert narrow[1] / narrow[0] != pytest.approx(wide[1] / wide[0], rel=0.05)


def test_boltzmann_probs_limits_and_validation():
    m = np.array([0.0, 1.0, 3.0])
    assert boltzmann_probs(m, sigma=1e-3).argmax() == 2 and boltzmann_probs(m, sigma=1e-3).max() > 0.999
    np.testing.assert_allclose(boltzmann_probs(m, sigma=1e6), 1 / 3, atol=1e-5)
    np.testing.assert_allclose(boltzmann_probs(m + 1e4), boltzmann_probs(m))      # no overflow
    with pytest.raises(ValueError):
        boltzmann_probs(m, sigma=0.0)
    with pytest.raises(ValueError):
        boltzmann_probs(np.array([0.0, np.nan]))


def test_driver_rejects_unknown_selection():
    from pathgennie.core.driver import PathGennieDriver
    from pathgennie.core.progress import TargetMetric
    from pathgennie.core.toy import ToyLangevinEngine

    with pytest.raises(ValueError, match="selection"):
        PathGennieDriver(ToyLangevinEngine(), TargetMetric(_xy, target_cv=B), lambda c: False,
                         selection="greedy")


# ---- committor progress variable ------------------------------------------------

class _StubModel:
    """Committor that depends on x only: logit = 2 x."""

    def logit(self, features):
        return 2.0 * float(np.asarray(features)[0])


def test_committor_progress_is_log_probability_of_reaching_the_target():
    from pathgennie.committor.progress import CommittorProgress

    coords = np.array([[0.5, 0.0, 0.0]])
    q = 1.0 / (1.0 + np.exp(-1.0))
    to_b = CommittorProgress(_StubModel(), _xy, target="B", floor=0.0)
    to_a = CommittorProgress(_StubModel(), _xy, target="A", floor=0.0)
    assert to_b.metric(to_b.project(coords, cycle=3)) == pytest.approx(np.log(q))
    assert to_a.metric(to_a.project(coords)) == pytest.approx(np.log(1.0 - q))
    assert to_b.committor(coords) == pytest.approx(q)


def test_committor_progress_floor_keeps_weights_finite():
    from pathgennie.committor.progress import CommittorProgress

    far = np.array([[-500.0, 0.0, 0.0]])          # model says q = 0 exactly
    floored = CommittorProgress(_StubModel(), _xy, target="B", floor=1e-3)
    assert floored.metric(floored.project(far)) == pytest.approx(np.log(1e-3))
    with pytest.raises(ValueError):
        CommittorProgress(_StubModel(), _xy, target="C")


# ---- closed loop ------------------------------------------------------------------

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from pathgennie.committor import BasinPair, SphereBasin  # noqa: E402
from pathgennie.committor.basins import LABEL_NONE  # noqa: E402
from pathgennie.committor.loop import CommittorLoop, LoopConfig  # noqa: E402
from pathgennie.committor.reference import grid_committor  # noqa: E402
from pathgennie.committor.train import CommittorEnsemble, committor_scores  # noqa: E402
from pathgennie.core.progress import TargetMetric  # noqa: E402
from pathgennie.core.toy import ToyLangevinEngine, wolfe_quapp_potential  # noqa: E402

BASINS = BasinPair(SphereBasin(A, R), SphereBasin(B, R))


def _x_only(target):
    return TargetMetric(lambda c: np.array([c[0, 0]]),
                        target_cv=np.array([B[0] if target == "B" else A[0]]))


def _loop(seed=0, **overrides):
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    config = LoopConfig(seed=seed, **overrides)
    loop = CommittorLoop(engine, BASINS, _xy, _xy,
                         start_a=engine.create_state(A), start_b=engine.create_state(B),
                         bootstrap_progress=_x_only, config=config)
    return engine, loop


@pytest.fixture(scope="module")
def small_loop():
    engine, loop = _loop(n_models=2, train_steps=400, refresh_steps=100, episodes_per_round=6)
    loop.run(2)
    return engine, loop


def test_loop_bookkeeping(small_loop):
    engine, loop = small_loop
    cfg = loop.config
    data = loop.recorder.dataset()
    assert [r.round for r in loop.reports] == [0, 1, 2]
    assert loop.reports[-1].swarms == len(data)
    # MD cost counts every trial and every runner
    assert loop.md_steps == data.n_trials * cfg.tau1 + len(loop.recorder.runner_starts) * cfg.tau2
    assert loop.reports[-1].md_steps == loop.md_steps
    # every round's episodes are accounted for
    for report in loop.reports[1:]:
        assert report.end_in_a + report.end_in_b + report.unfinished == cfg.episodes_per_round
    # restart coordinates were kept, aligned with the swarms
    assert len(loop.recorder.coords_starts) == len(data)
    assert isinstance(loop.model, CommittorEnsemble) and len(loop.model) == 2


def test_loop_releases_every_handle(small_loop):
    engine, loop = small_loop
    assert set(engine._cache) == {loop.start_a, loop.start_b}


def test_bootstrap_episode_must_leave_its_basin_first(small_loop):
    """Runs that start inside A or B are not 'converged' at cycle 0."""
    _, loop = small_loop
    data = loop.recorder.dataset()
    assert data.start_label[0] == 0                       # first swarm starts in A
    assert (data.start_label == LABEL_NONE).sum() > 0     # ...and the run did leave it
    both = [(data.end_label == k).sum() for k in (0, 1)]
    assert min(both) > 0                                  # boundary data from both basins


def test_bootstrap_retries_until_both_basins_have_boundary_data():
    """An episode can leave its basin on the runner and end without one trial
    endpoint inside it; bootstrap then launches more instead of failing to train."""
    counts = []
    for seed in range(12):
        _, loop = _loop(seed=seed, n_models=1, train_steps=5)
        report = loop.bootstrap()
        n_a, n_b = loop._boundary_counts()
        assert n_a > 0 and n_b > 0
        counts.append(report.end_in_a + report.end_in_b + report.unfinished)
    assert min(counts) == 2                                # the usual case: one episode each


def test_bootstrap_gives_up_with_a_clear_error():
    # A tiny "product" high on the potential wall: runs slide off it at once and
    # no trial ever ends inside it again.
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    basins = BasinPair(SphereBasin(A, R), SphereBasin([2.3, 2.3], 0.05))
    far = engine.create_state([2.3, 2.3])
    loop = CommittorLoop(engine, basins, _xy, _xy, start_a=engine.create_state(A), start_b=far,
                         bootstrap_progress=lambda target: TargetMetric(_xy, target_cv=A),
                         config=LoopConfig(bootstrap_retries=1, bootstrap_max_cycles=3, n_models=1))
    with pytest.raises(RuntimeError, match="trial endpoints"):
        loop.bootstrap()


def test_restart_points_cover_the_committor_range(small_loop):
    _, loop = small_loop
    data = loop.recorder.dataset()
    picks = loop.restart_points(40)
    assert len(picks) == 40
    assert (data.start_label[picks] == LABEL_NONE).all()  # never restart inside a basin
    q = loop.model.predict(data.starts[picks])
    all_q = loop.model.predict(data.starts[data.start_label == LABEL_NONE])
    occupied = np.unique(np.minimum((all_q * 10).astype(int), 9))
    picked = np.unique(np.minimum((q * 10).astype(int), 9))
    assert set(occupied) <= set(picked)                   # every occupied bin is revisited


def test_round_requires_bootstrap():
    _, loop = _loop()
    with pytest.raises(RuntimeError, match="bootstrap"):
        loop.round()


def test_ensemble_spread_is_zero_for_one_member(small_loop):
    _, loop = small_loop
    pts = np.array([[0.0, 0.0], [1.0, 1.3]])
    assert (CommittorEnsemble(loop.model.models[:1]).std(pts) == 0).all()
    assert (loop.model.std(pts) >= 0).all()
    assert isinstance(loop.model.logit(pts[0]), float)


@pytest.mark.slow
def test_closed_loop_learns_committor_from_a_poor_cv():
    """Milestone M3 in miniature: bootstrapped with x alone, the loop resolves the
    committor over the whole landscape, both reaction channels included."""
    ref = grid_committor(
        wolfe_quapp_potential,
        lambda x, y: (x - A[0]) ** 2 + (y - A[1]) ** 2 < R * R,
        lambda x, y: (x - B[0]) ** 2 + (y - B[1]) ** 2 < R * R,
        xlim=(-2.6, 2.6), ylim=(-2.6, 2.6), n=161,
    )
    grid = np.array([[x, y] for x in np.linspace(-2, 2, 41) for y in np.linspace(-2.2, 2.2, 45)])
    grid = grid[wolfe_quapp_potential(grid[:, 0], grid[:, 1]) < -1.0]
    exact = ref(grid)

    _, loop = _loop(seed=0)
    loop.run(5)
    pred = loop.model.predict(grid)
    scores = committor_scores(pred, exact)
    assert scores["pearson_transition"] > 0.95
    assert scores["rmse_transition"] < 0.12
    ts = (exact > 0.05) & (exact < 0.95)
    for channel in (grid.sum(axis=1) < 0, grid.sum(axis=1) >= 0):
        sel = ts & channel
        assert np.sqrt(np.mean((pred[sel] - exact[sel]) ** 2)) < 0.15
    assert loop.md_steps < 250_000
