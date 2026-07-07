"""RRT / RRT-Connect search tests on the toy Wolfe-Quapp engine (pure NumPy)."""

import numpy as np

from pathgennie.core.parallel import SerialExecutor
from pathgennie.core.toy import ToyLangevinEngine
from pathgennie.search.rrt import RRT, rrt_connect


def _xy(coords):
    return np.array([coords[0, 0], coords[0, 1]])


# WQ minima sit near (-1.17, 1.48) and (1.12, -1.48).
BASIN_A = (-1.174, 1.477)
BASIN_B = (1.124, -1.485)


def test_rrt_reaches_target_basin():
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    start = engine.create_state(BASIN_A)
    rrt = RRT(
        engine, _xy, lower=[-2.0, -2.0], upper=[2.0, 2.0],
        tau1=5, tau2=10, n_expand=8, sigma=0.05, goal_bias=0.2,
        executor=SerialExecutor(), seed=0,
    )
    result = rrt.build(start, target_cv=list(BASIN_B), max_iter=300, goal_tol=0.5)

    assert result.tree_size > 1
    assert result.success, "RRT should reach the opposite basin"
    # Path starts in basin A and ends near basin B.
    assert np.linalg.norm(result.path[0].cv - np.array(BASIN_A)) < 0.8
    assert np.linalg.norm(result.path[-1].cv - np.array(BASIN_B)) <= 0.5
    # Path nodes form a parent chain.
    for child in result.path[1:]:
        assert child.parent is not None


def test_rrt_grows_tree_without_target():
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    start = engine.create_state(BASIN_A)
    rrt = RRT(
        engine, _xy, lower=[-2.0, -2.0], upper=[2.0, 2.0],
        tau1=4, tau2=8, n_expand=6, executor=SerialExecutor(), seed=1,
    )
    result = rrt.build(start, target_cv=None, max_iter=40)
    assert result.tree_size == 41  # root + 40 expansions
    # The tree should have explored beyond the start basin.
    cvs = np.stack([n.cv for n in rrt.nodes])
    assert cvs[:, 0].max() - cvs[:, 0].min() > 0.5


def test_rrt_connect_links_two_basins():
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    start = engine.create_state(BASIN_A)
    goal = engine.create_state(BASIN_B)
    result = rrt_connect(
        engine, _xy, start, goal,
        lower=[-2.0, -2.0], upper=[2.0, 2.0],
        tau1=5, tau2=10, n_expand=8, sigma=0.05,
        executor=SerialExecutor(), seed=2, max_iter=300, connect_tol=0.6,
    )
    assert result.success
    assert len(result.path) >= 2
    endpoints = np.array([result.path[0].cv, result.path[-1].cv])
    # One endpoint near each basin (order may vary by which tree was 'start').
    near_a = min(np.linalg.norm(endpoints - np.array(BASIN_A), axis=1))
    near_b = min(np.linalg.norm(endpoints - np.array(BASIN_B), axis=1))
    assert near_a < 0.8 and near_b < 0.8
