"""OpenPathSampling bridge tests.

The dependency-free seed preparation is fully verified; the OPS-dependent stage
is verified only to fail informatively when OpenPathSampling is absent (it is not
installed in CI).
"""

import numpy as np

from pathgennie.sampling import (
    PathSamplingStage,
    build_path_ensemble,
    make_stage,
)
from pathgennie.sampling.path_sampling import (
    CVRangeState,
    extract_transition_path,
    is_reactive,
    label_frames,
    prepare_ops_seed,
    tis_interfaces,
)


def _frames(ys):
    return np.array([[[0.0, y, 0.0]] for y in ys], dtype=float)


CV = lambda c: c[0, 1]
A = CVRangeState("A", -2.0, -1.0)
B = CVRangeState("B", 1.0, 2.0)


def test_cv_range_state_contains():
    assert A.contains(-1.5) and not A.contains(0.0)
    assert B.contains(1.5) and not B.contains(-1.5)


def test_label_frames():
    labels = label_frames(_frames([-1.5, 0.0, 1.5]), CV, A, B)
    assert list(labels) == [0, -1, 1]   # A, none, B


def test_extract_transition_path():
    frames = _frames([-1.5, -1.4, -0.5, 0.2, 0.9, 1.5, 1.6])
    span = extract_transition_path(frames, CV, A, B)
    assert span == (1, 5)               # last A before first B .. first B
    assert is_reactive(frames, CV, A, B)


def test_not_reactive_paths():
    assert extract_transition_path(_frames([-1.5, -1.4, -0.5]), CV, A, B) is None  # never reaches B
    assert extract_transition_path(_frames([0.2, 0.9, 1.5]), CV, A, B) is None      # never in A first


def test_tis_interfaces():
    lin = tis_interfaces(-1.0, 1.0, 5, spacing="linear")
    assert lin.shape == (5,)
    assert np.all(np.diff(lin) > 0) and lin[0] == -1.0 and lin[-1] == 1.0
    exp = tis_interfaces(-1.0, 1.0, 5, spacing="exp")
    assert np.all(np.diff(exp) > 0) and np.isclose(exp[0], -1.0) and np.isclose(exp[-1], 1.0)
    assert tis_interfaces(0.0, 1.0, 1).shape == (1,)
    for bad in (lambda: tis_interfaces(1.0, 0.0, 3), lambda: tis_interfaces(0.0, 1.0, 0)):
        try:
            bad(); assert False
        except ValueError:
            pass


def test_prepare_ops_seed():
    frames = _frames([-1.5, -1.4, -0.5, 0.2, 0.9, 1.5, 1.6])
    ens = build_path_ensemble(frames, np.arange(len(frames), dtype=float))
    seed = prepare_ops_seed(ens, CV, A, B, interfaces=[-0.5, 0.0, 0.5])
    assert seed["reactive"] and seed["span"] == (1, 5)
    assert seed["seed_frames"].shape == (5, 1, 3)
    assert seed["cv_trajectory"].shape == (7,)
    assert list(seed["interfaces"]) == [-0.5, 0.0, 0.5]


def test_make_stage_tps_tis():
    tps = make_stage("tps", cv_fn=CV, state_a=(-2, -1), state_b=(1, 2))
    assert isinstance(tps, PathSamplingStage) and tps.mode == "tps"
    tis = make_stage("tis", cv_fn=CV, state_a=(-2, -1), state_b=(1, 2), interfaces=[-1, 0, 1])
    assert isinstance(tis, PathSamplingStage) and tis.mode == "tis"


def test_tis_requires_interfaces():
    try:
        PathSamplingStage(CV, state_a=(-2, -1), state_b=(1, 2), mode="tis")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_run_without_ops_raises_importerror():
    frames = _frames([-1.5, -1.4, -0.5, 0.2, 0.9, 1.5, 1.6])
    ens = build_path_ensemble(frames, np.arange(len(frames), dtype=float))
    stage = PathSamplingStage(CV, state_a=(-2, -1), state_b=(1, 2), mode="tps")
    try:
        stage.run(ens, engine=None)
        assert False, "expected ImportError (OpenPathSampling not installed)"
    except ImportError as exc:
        assert "OpenPathSampling" in str(exc)
