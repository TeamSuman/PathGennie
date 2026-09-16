"""Tests for Task A4's RRT extensions: ``scale``, ``goal_test``, ``max_nodes``
and checkpoint/resume -- all exercised on the toy Wolfe-Quapp engine so no MD
binary or GPU is needed.
"""

import numpy as np
import pytest

from pathgennie.core.parallel import SerialExecutor
from pathgennie.core.toy import ToyLangevinEngine
from pathgennie.search.rrt import RRT, rrt_connect

# WQ minima sit near (-1.17, 1.48) and (1.12, -1.48) -- see tests/test_rrt.py.
BASIN_A = (-1.174, 1.477)
BASIN_B = (1.124, -1.485)

# How much the toy CV's second component is stretched to build an anisotropic
# CV space: without `scale` to compensate, Euclidean nearest-neighbour and
# goal-tolerance checks are dominated by the stretched component.
STRETCH = 20.0


def _stretched_cv(coords):
    return np.array([coords[0, 0], coords[0, 1] * STRETCH])


def test_scale_reaches_goal_on_anisotropic_landscape():
    """`scale=[1, STRETCH]` should exactly undo the CV stretch internally, so
    the search behaves like the well-tested isotropic case and reaches the
    opposite basin within the same iteration budget."""
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    start = engine.create_state(BASIN_A)
    rrt = RRT(
        engine, _stretched_cv,
        lower=[-2.0, -2.0 * STRETCH], upper=[2.0, 2.0 * STRETCH],
        tau1=5, tau2=10, n_expand=8, sigma=0.05, goal_bias=0.2,
        executor=SerialExecutor(), seed=0,
        scale=[1.0, STRETCH],
    )
    target = [BASIN_B[0], BASIN_B[1] * STRETCH]
    result = rrt.build(start, target_cv=target, max_iter=300, goal_tol=0.5)

    assert result.tree_size > 1
    assert result.success, "scaled RRT should reach the opposite basin"
    assert result.reason is None
    # Node CVs are stored in the internal (scaled-back) space, so they should
    # read back as the ordinary, unstretched physical CVs.
    assert np.linalg.norm(result.path[0].cv - np.array(BASIN_A)) < 0.8
    assert np.linalg.norm(result.path[-1].cv - np.array(BASIN_B)) <= 0.5


def test_unscaled_control_reaches_the_same_goal_slower():
    """M2 control for the test above: brief Step 1 asks for "RRT with
    scale=[1, 10] reaches a goal that unscaled RRT reaches slower" -- without
    `scale` to undo the stretch, nearest-neighbour distances and the
    goal-tolerance check are dominated by the second (x20-stretched) CV
    component. Both searches still eventually succeed here (goal-biased
    sampling occasionally targets the goal directly regardless of scale), so
    "slower" is measured the way the brief phrases it: the same seed/budget
    needs a *much larger* tree unscaled than scaled -- confirmed empirically
    stable (3x-8x) across seeds 0-4 before picking this assertion's margin."""
    target = [BASIN_B[0], BASIN_B[1] * STRETCH]

    engine_scaled = ToyLangevinEngine(dt=0.005, kT=1.0)
    scaled = RRT(
        engine_scaled, _stretched_cv,
        lower=[-2.0, -2.0 * STRETCH], upper=[2.0, 2.0 * STRETCH],
        tau1=5, tau2=10, n_expand=8, sigma=0.05, goal_bias=0.2,
        executor=SerialExecutor(), seed=0,
        scale=[1.0, STRETCH],
    ).build(engine_scaled.create_state(BASIN_A), target_cv=target, max_iter=300, goal_tol=0.5)
    assert scaled.success

    engine_unscaled = ToyLangevinEngine(dt=0.005, kT=1.0)
    unscaled = RRT(
        engine_unscaled, _stretched_cv,
        lower=[-2.0, -2.0 * STRETCH], upper=[2.0, 2.0 * STRETCH],
        tau1=5, tau2=10, n_expand=8, sigma=0.05, goal_bias=0.2,
        executor=SerialExecutor(), seed=0,
        scale=None,
    ).build(engine_unscaled.create_state(BASIN_A), target_cv=target, max_iter=300, goal_tol=0.5)
    assert unscaled.success

    assert unscaled.tree_size > 2 * scaled.tree_size, (
        f"unscaled tree_size={unscaled.tree_size} should be far larger (slower) than "
        f"scaled tree_size={scaled.tree_size} for the same seed/budget"
    )


def _xy(coords):
    return np.array([coords[0, 0], coords[0, 1]])


def test_goal_test_variant_uses_basin_membership_not_distance():
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    start = engine.create_state(BASIN_A)

    def near_basin_b(handle) -> bool:
        coords = engine.get_coords(handle)
        xy = np.array([coords[0, 0], coords[0, 1]])
        return bool(np.linalg.norm(xy - np.array(BASIN_B)) <= 0.5)

    rrt = RRT(
        engine, _xy, lower=[-2.0, -2.0], upper=[2.0, 2.0],
        tau1=5, tau2=10, n_expand=8, sigma=0.05, goal_bias=0.2,
        executor=SerialExecutor(), seed=0,
    )
    # A deliberately tiny goal_tol would make the old distance-based check
    # (almost) never succeed; goal_test must be what actually gates success
    # here, proving goal_tol is ignored when goal_test is given.
    result = rrt.build(
        start, target_cv=list(BASIN_B), max_iter=300, goal_tol=1e-9,
        goal_test=near_basin_b,
    )

    assert result.success
    assert result.reason is None
    assert near_basin_b(result.goal_node.handle)


def test_max_nodes_stops_the_search_with_a_reason():
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    start = engine.create_state(BASIN_A)
    rrt = RRT(
        engine, _xy, lower=[-2.0, -2.0], upper=[2.0, 2.0],
        tau1=4, tau2=8, n_expand=4, executor=SerialExecutor(), seed=3,
    )
    # An unreachable goal (way outside the sampling box) with a near-zero
    # tolerance guarantees the run cannot succeed by luck before the cap bites.
    result = rrt.build(
        start, target_cv=[100.0, 100.0], max_iter=1000, goal_tol=1e-9,
        max_nodes=5,
    )

    assert not result.success
    assert result.reason == "max_nodes"
    assert result.tree_size == 5


def test_checkpoint_resume_continues_a_capped_search_to_success(tmp_path):
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    start = engine.create_state(BASIN_A)
    ckpt_path = str(tmp_path / "rrt_ckpt.npz")

    rrt = RRT(
        engine, _xy, lower=[-2.0, -2.0], upper=[2.0, 2.0],
        tau1=5, tau2=10, n_expand=8, sigma=0.05, goal_bias=0.2,
        executor=SerialExecutor(), seed=0,
    )
    capped = rrt.build(
        start, target_cv=list(BASIN_B), max_iter=1000, goal_tol=0.5,
        max_nodes=6,
    )
    assert not capped.success
    assert capped.reason == "max_nodes"
    assert len(rrt.nodes) == 6

    rrt.checkpoint(ckpt_path)

    resumed = RRT.resume(
        ckpt_path, engine, _xy,
        lower=[-2.0, -2.0], upper=[2.0, 2.0],
        tau1=5, tau2=10, n_expand=8, sigma=0.05, goal_bias=0.2,
        executor=SerialExecutor(), seed=1,
    )
    assert len(resumed.nodes) == 6
    # Restored node CVs and parent links must match the checkpointed tree.
    for orig, restored in zip(rrt.nodes, resumed.nodes):
        np.testing.assert_allclose(orig.cv, restored.cv)
        assert orig.parent == restored.parent

    result = resumed.build(start, target_cv=list(BASIN_B), max_iter=300, goal_tol=0.5)
    assert result.success, "resuming should continue growing the same tree to success"
    assert result.tree_size > 6


# -- C1: checkpoint/resume must persist and validate scale/bounds ------------

def test_resume_raises_on_scale_mismatch(tmp_path):
    """C1: resuming without re-passing the original `scale` (or with a
    different one) must raise -- silently reinterpreting every restored,
    already-scaled node CV in the wrong space is exactly the failure mode
    `scale` exists to prevent elsewhere in this class."""
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    start = engine.create_state(BASIN_A)
    ckpt_path = str(tmp_path / "rrt_ckpt_scale_mismatch.npz")

    rrt = RRT(
        engine, _stretched_cv, lower=[-2.0, -2.0 * STRETCH], upper=[2.0, 2.0 * STRETCH],
        tau1=4, tau2=8, n_expand=4, executor=SerialExecutor(), seed=0,
        scale=[1.0, STRETCH],
    )
    rrt.build(start, max_iter=3, goal_tol=0.5)
    rrt.checkpoint(ckpt_path)

    # Omitting `scale` on resume (defaults to None) mismatches the checkpoint's [1.0, STRETCH].
    with pytest.raises(ValueError, match="scale"):
        RRT.resume(
            ckpt_path, engine, _stretched_cv,
            lower=[-2.0, -2.0 * STRETCH], upper=[2.0, 2.0 * STRETCH],
            tau1=4, tau2=8, n_expand=4, executor=SerialExecutor(), seed=1,
        )

    # A *different* scale (not just a missing one) must also be rejected.
    with pytest.raises(ValueError, match="scale"):
        RRT.resume(
            ckpt_path, engine, _stretched_cv,
            lower=[-2.0, -2.0 * STRETCH], upper=[2.0, 2.0 * STRETCH],
            tau1=4, tau2=8, n_expand=4, executor=SerialExecutor(), seed=1,
            scale=[1.0, 2.0 * STRETCH],
        )


def test_resume_raises_on_bounds_mismatch(tmp_path):
    """C1: mismatched lower/upper bounds on resume must also raise (bounds are
    stored, and used internally, in scaled space -- same failure mode as a
    scale mismatch)."""
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    start = engine.create_state(BASIN_A)
    ckpt_path = str(tmp_path / "rrt_ckpt_bounds_mismatch.npz")

    rrt = RRT(
        engine, _xy, lower=[-2.0, -2.0], upper=[2.0, 2.0],
        tau1=4, tau2=8, n_expand=4, executor=SerialExecutor(), seed=0,
    )
    rrt.build(start, max_iter=3, goal_tol=0.5)
    rrt.checkpoint(ckpt_path)

    with pytest.raises(ValueError, match="bounds"):
        RRT.resume(
            ckpt_path, engine, _xy,
            lower=[-3.0, -2.0], upper=[2.0, 2.0],  # lower[0] changed from -2.0 to -3.0
            tau1=4, tau2=8, n_expand=4, executor=SerialExecutor(), seed=1,
        )


def test_checkpoint_resume_round_trip_with_scale_succeeds(tmp_path):
    """C1: the matching-scale/bounds round trip (the common, correct case)
    must still work end to end."""
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    start = engine.create_state(BASIN_A)
    ckpt_path = str(tmp_path / "rrt_ckpt_scale_ok.npz")

    rrt = RRT(
        engine, _stretched_cv, lower=[-2.0, -2.0 * STRETCH], upper=[2.0, 2.0 * STRETCH],
        tau1=4, tau2=8, n_expand=4, executor=SerialExecutor(), seed=0,
        scale=[1.0, STRETCH],
    )
    rrt.build(start, max_iter=3, goal_tol=0.5)
    rrt.checkpoint(ckpt_path)

    resumed = RRT.resume(
        ckpt_path, engine, _stretched_cv,
        lower=[-2.0, -2.0 * STRETCH], upper=[2.0, 2.0 * STRETCH],
        tau1=4, tau2=8, n_expand=4, executor=SerialExecutor(), seed=1,
        scale=[1.0, STRETCH],
    )
    assert len(resumed.nodes) == len(rrt.nodes)
    for orig, restored in zip(rrt.nodes, resumed.nodes):
        np.testing.assert_allclose(orig.cv, restored.cv)


# -- I1: checkpoint/resume must persist and restore RNG state ----------------

def test_resume_restores_rng_state_for_bit_for_bit_continuation(tmp_path):
    """I1: a resumed build must reproduce an uninterrupted one -- growing the
    same tree across a checkpoint/resume boundary must draw the exact same
    random numbers (and therefore make the exact same choices) as growing it
    in one continuous, uninterrupted `build()` call would."""
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    start = engine.create_state(BASIN_A)
    ckpt_path = str(tmp_path / "rrt_ckpt_rng.npz")

    def _make(seed):
        return RRT(
            engine, _xy, lower=[-2.0, -2.0], upper=[2.0, 2.0],
            tau1=4, tau2=8, n_expand=4, executor=SerialExecutor(), seed=seed,
        )

    # Reference: grow 3 nodes, checkpoint, then keep growing the SAME
    # in-process tree for 3 more nodes without any interruption.
    reference = _make(seed=7)
    reference.build(start, max_iter=3, goal_tol=0.5)
    reference.checkpoint(ckpt_path)
    reference.build(start, max_iter=3, goal_tol=0.5)

    # Resumed: load the same checkpoint into a fresh RRT deliberately
    # constructed with a *different* `seed` kwarg (proving continuation comes
    # from the restored RNG state, not from re-deriving it from `seed`), then
    # grow it the same way.
    resumed = RRT.resume(
        ckpt_path, engine, _xy,
        lower=[-2.0, -2.0], upper=[2.0, 2.0],
        tau1=4, tau2=8, n_expand=4, executor=SerialExecutor(), seed=999,
    )
    resumed.build(start, max_iter=3, goal_tol=0.5)

    assert len(resumed.nodes) == len(reference.nodes)
    for ref_node, res_node in zip(reference.nodes, resumed.nodes):
        np.testing.assert_allclose(ref_node.cv, res_node.cv)


# -- NaN quarantine: a diverged trial must never enter the tree --------------

def test_extend_quarantines_a_nonfinite_trial_and_counts_it():
    """A cv_fn that returns a non-finite CV for exactly one of the n_expand
    swarm trials must have that trial released and excluded from selection
    (never entering the tree), and counted in RRTResult.n_diverged."""
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    start = engine.create_state(BASIN_A)

    calls = {"n": 0}

    def flaky_cv(coords):
        calls["n"] += 1
        # Call #1 is the root node (added before the loop); call #2 is the
        # first of the n_expand=4 swarm trials in the one extend() below --
        # make exactly that one diverge.
        if calls["n"] == 2:
            return np.array([np.nan, np.nan])
        return np.array([coords[0, 0], coords[0, 1]])

    rrt = RRT(
        engine, flaky_cv, lower=[-2.0, -2.0], upper=[2.0, 2.0],
        tau1=4, tau2=8, n_expand=4, executor=SerialExecutor(), seed=0,
    )
    result = rrt.build(start, max_iter=1, goal_tol=0.5)

    assert result.n_diverged == 1
    assert result.tree_size == 2  # root + the one committed node
    assert all(np.isfinite(node.cv).all() for node in rrt.nodes)


def test_extend_quarantines_a_nonfinite_tau2_runner_and_skips_the_expansion():
    """A cv_fn that returns a non-finite CV only for the tau2 runner segment
    (every swarm trial stays finite) must have that runner released and
    counted in n_diverged, and the expansion skipped entirely -- unlike a
    diverged swarm trial, there is no fallback candidate for tau2, so no node
    is added for this extend() call at all."""
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    start = engine.create_state(BASIN_A)

    calls = {"n": 0}

    def flaky_cv(coords):
        calls["n"] += 1
        # Call #1 is the root node; calls #2-#5 are the n_expand=4 swarm
        # trials (all finite); call #6 is the tau2 runner -- make only that
        # one diverge.
        if calls["n"] == 6:
            return np.array([np.nan, np.nan])
        return np.array([coords[0, 0], coords[0, 1]])

    rrt = RRT(
        engine, flaky_cv, lower=[-2.0, -2.0], upper=[2.0, 2.0],
        tau1=4, tau2=8, n_expand=4, executor=SerialExecutor(), seed=0,
    )
    result = rrt.build(start, max_iter=1, goal_tol=0.5)

    assert result.n_diverged == 1
    assert result.tree_size == 1  # only the root -- the diverged tau2 expansion added nothing
    assert all(np.isfinite(node.cv).all() for node in rrt.nodes)


def test_extend_raises_if_every_trial_diverges():
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    start = engine.create_state(BASIN_A)

    def always_nan_cv(coords):
        return np.array([np.nan, np.nan])

    rrt = RRT(
        engine, always_nan_cv, lower=[-2.0, -2.0], upper=[2.0, 2.0],
        tau1=4, tau2=8, n_expand=4, executor=SerialExecutor(), seed=0,
    )
    with pytest.raises(RuntimeError):
        rrt.build(start, max_iter=1, goal_tol=0.5)


# -- rrt_connect extensions: scale, max_nodes, connect_test ------------------

def test_rrt_connect_scale_changes_whether_the_pair_connects():
    """Same scaled-vs-unscaled comparison as the RRT.build tests above, but
    for rrt_connect: forwarding `scale` to both trees (and therefore to the
    connect-distance check) should let two stretched-landscape configurations
    join with a much smaller combined tree than the unscaled search needs for
    the same seed/budget (empirically stable, 3x-18x across seeds 1-4; seed=2
    picked here for a comfortable, non-borderline margin)."""
    engine_scaled = ToyLangevinEngine(dt=0.005, kT=1.0)
    scaled = rrt_connect(
        engine_scaled, _stretched_cv,
        engine_scaled.create_state(BASIN_A), engine_scaled.create_state(BASIN_B),
        lower=[-2.0, -2.0 * STRETCH], upper=[2.0, 2.0 * STRETCH],
        tau1=5, tau2=10, n_expand=8, sigma=0.05,
        executor=SerialExecutor(), seed=2, max_iter=300, connect_tol=0.5,
        scale=[1.0, STRETCH],
    )
    assert scaled.success, "scaled rrt_connect should join the two stretched-CV configurations"

    engine_unscaled = ToyLangevinEngine(dt=0.005, kT=1.0)
    unscaled = rrt_connect(
        engine_unscaled, _stretched_cv,
        engine_unscaled.create_state(BASIN_A), engine_unscaled.create_state(BASIN_B),
        lower=[-2.0, -2.0 * STRETCH], upper=[2.0, 2.0 * STRETCH],
        tau1=5, tau2=10, n_expand=8, sigma=0.05,
        executor=SerialExecutor(), seed=2, max_iter=300, connect_tol=0.5,
    )
    assert unscaled.success

    assert unscaled.tree_size > 2 * scaled.tree_size, (
        f"unscaled tree_size={unscaled.tree_size} should be far larger (slower) than "
        f"scaled tree_size={scaled.tree_size} for the same seed/budget"
    )


def test_rrt_connect_max_nodes_bounds_each_tree():
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    start = engine.create_state(BASIN_A)
    goal = engine.create_state(BASIN_B)

    result = rrt_connect(
        engine, _xy, start, goal,
        lower=[-2.0, -2.0], upper=[2.0, 2.0],
        tau1=4, tau2=8, n_expand=4,
        executor=SerialExecutor(), seed=3, max_iter=200, connect_tol=1e-9,
        max_nodes=3,
    )
    assert not result.success
    assert result.reason == "max_nodes"
    # Each of the two trees is capped at max_nodes, so their combined size
    # (the only thing rrt_connect's public RRTResult exposes) is bounded by
    # 2 * max_nodes -- i.e. neither tree individually exceeded the cap.
    assert result.tree_size <= 2 * 3


def test_rrt_connect_custom_connect_test_forces_exhaustion():
    """A connect_test that always returns False must override connect_tol
    entirely -- even two nodes at zero distance must not be accepted -- so
    the search runs to max_iter and reports failure."""
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    start = engine.create_state(BASIN_A)
    goal = engine.create_state(BASIN_B)

    result = rrt_connect(
        engine, _xy, start, goal,
        lower=[-2.0, -2.0], upper=[2.0, 2.0],
        tau1=4, tau2=8, n_expand=4,
        executor=SerialExecutor(), seed=1, max_iter=50, connect_tol=1e9,
        connect_test=lambda handle_a, handle_b: False,
    )
    assert not result.success
