"""Tests for Task A4's RRT extensions: ``scale``, ``goal_test``, ``max_nodes``
and checkpoint/resume -- all exercised on the toy Wolfe-Quapp engine so no MD
binary or GPU is needed.
"""

import numpy as np

from pathgennie.core.parallel import SerialExecutor
from pathgennie.core.toy import ToyLangevinEngine
from pathgennie.search.rrt import RRT

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
    # A deliberately huge goal_tol would make the old distance-based check
    # trivially succeed immediately; goal_test must be what actually gates
    # success here, proving goal_tol is ignored when goal_test is given.
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
