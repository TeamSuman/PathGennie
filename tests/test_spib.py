"""SPIB data-driven CV tests (skipped if PyTorch is unavailable)."""

import numpy as np
import pytest

pytest.importorskip("torch")

from pathgennie.core.driver import PathGennieDriver  # noqa: E402
from pathgennie.core.parallel import SerialExecutor  # noqa: E402
from pathgennie.core.progress import TargetMetric  # noqa: E402
from pathgennie.core.toy import ToyLangevinEngine  # noqa: E402
from pathgennie.cv.features import Featurizer  # noqa: E402
from pathgennie.cv.spib import SPIBProgress, kmeans_labels, train_spib  # noqa: E402


def _two_state_trajectory(n=600, seed=0):
    rng = np.random.default_rng(seed)
    states = [0]
    for _ in range(1, n):
        states.append(states[-1] if rng.random() < 0.95 else 1 - states[-1])
    states = np.array(states)
    centers = np.array([[-3.0, 0.0], [3.0, 0.0]])
    feats = centers[states] + 0.3 * rng.standard_normal((n, 2))
    return feats, states


def test_kmeans_separates_clusters():
    feats, gt = _two_state_trajectory()
    labels = kmeans_labels(feats, k=2, seed=0)
    acc = max((labels == gt).mean(), (labels == 1 - gt).mean())
    assert acc > 0.95


def test_spib_recovers_two_states():
    feats, gt = _two_state_trajectory()
    res = train_spib(
        feats, dt=1, n_states_init=4, latent_dim=1,
        beta=1e-3, epochs=40, n_refine=4, seed=0,
    )
    # Emergent number of metastable states should be 2 (started from 4).
    assert 2 <= res.n_states <= 4

    # The learned latent should separate the two ground-truth basins.
    import torch
    x = (feats - res.feature_mean) / res.feature_std
    with torch.no_grad():
        z = res.model.encode(torch.tensor(x, dtype=torch.float32))[0].numpy().ravel()
    sep = abs(z[gt == 0].mean() - z[gt == 1].mean())
    pooled = z.std() + 1e-9
    assert sep / pooled > 1.5

    acc = max((res.labels == gt).mean(), (res.labels == 1 - gt).mean())
    assert acc > 0.85


def test_spib_progress_drives_and_switches_to_learned_cv():
    engine = ToyLangevinEngine(dt=0.005)
    initial = engine.create_state((-1.174, 1.477))
    target = np.array([[1.124, -1.485, 0.0]])

    def xy(coords):
        return np.array([coords[0, 0], coords[0, 1]])

    bootstrap = TargetMetric(xy, target_cv=np.array([1.124, -1.485]))
    progress = SPIBProgress(
        Featurizer(funcs=[], standardize=False),
        bootstrap,
        mode="target",
        target_coords=target,
        refresh_every=15,
        min_frames=30,
        dt=1,
        train_kwargs=dict(n_states_init=3, latent_dim=1, epochs=15, n_refine=2, seed=0),
    )

    driver = PathGennieDriver(
        engine, progress, convergence_fn=lambda c: False,
        executor=SerialExecutor(), sigma=0.2, seed=1, verbosity=0,
    )
    traj, metrics = driver.run(initial, tau1=10, tau2=20, max_trial=6, max_cycle=50, save_freq=5)

    # SPIB trained on the fly and the CV is now the learned latent.
    assert progress.result is not None
    assert progress.n_states is not None and progress.n_states >= 1
    learned_cv = progress.project(engine.get_coords(initial))
    assert learned_cv.shape == (1,)  # latent_dim
    assert traj.shape[0] >= 1


def test_spib_progress_caches_features_and_bounds_buffer():
    """Incremental feature caching must equal re-featurizing, and honour max_buffer.

    This is the on-the-fly-SPIB performance/memory fix: features are cached when a
    frame is observed (O(1)) instead of re-featurizing the whole raw-coordinate
    buffer on every refresh (O(N^2) cumulatively), and ``max_buffer`` bounds host
    memory with a most-recent sliding window while preserving the start frame.
    """
    from pathgennie.core.progress import EscapeMetric
    from pathgennie.cv.features import Featurizer

    featurizer = Featurizer(funcs=[], standardize=False)  # raw(coords) == coords.ravel()
    bootstrap = EscapeMetric(lambda c: np.array([c[0, 0]]), start_cv=np.array([0.0]), escape_metric="cv0")

    prog = SPIBProgress(
        featurizer, bootstrap, mode="escape",
        refresh_every=10_000, min_frames=10_000,  # never trigger training here
        max_buffer=5,
    )
    coords = [np.array([[float(i), 0.0, 0.0]]) for i in range(8)]
    for i, c in enumerate(coords):
        prog.observe(c, cycle=i)

    # Sliding window keeps only the most recent max_buffer frames ...
    assert len(prog._features) == 5
    # ... but the original start frame is retained separately for the start latent.
    np.testing.assert_allclose(prog._feature_start, featurizer.raw(coords[0]))
    # Cached features equal re-featurizing the corresponding raw coords (equivalence
    # with the previous re-featurize-everything path).
    for cached, c in zip(prog._features, coords[-5:]):
        np.testing.assert_allclose(cached, featurizer.raw(c))
    # min_frames not reached -> no training was triggered.
    assert prog.result is None
