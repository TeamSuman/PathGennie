"""Ground-truth tools: the grid reference and the shooting committor."""

import numpy as np
import pytest

from pathgennie.committor import BasinPair, SphereBasin, shooting_committor
from pathgennie.core.toy import ToyLangevinEngine, wolfe_quapp_potential

pytest.importorskip("scipy")

from pathgennie.committor.reference import grid_committor  # noqa: E402

A = np.array([-1.174, 1.477])
B = np.array([1.124, -1.485])
R = 0.4


def _xy(coords):
    return np.array([coords[0, 0], coords[0, 1]])


def _wolfe_quapp_reference(n=161):
    return grid_committor(
        wolfe_quapp_potential,
        lambda x, y: (x - A[0]) ** 2 + (y - A[1]) ** 2 < R * R,
        lambda x, y: (x - B[0]) ** 2 + (y - B[1]) ** 2 < R * R,
        xlim=(-2.6, 2.6), ylim=(-2.6, 2.6), n=n, kT=1.0,
    )


def test_flat_potential_gives_linear_committor():
    """Free diffusion between two walls: q is exactly linear in x."""
    ref = grid_committor(
        lambda x, y: np.zeros(np.broadcast(x, y).shape),
        lambda x, y: (x <= 0.0) & (y == y),
        lambda x, y: (x >= 1.0) & (y == y),
        xlim=(0.0, 1.0), ylim=(0.0, 1.0), n=41,
    )
    pts = np.array([[0.25, 0.5], [0.5, 0.2], [0.8, 0.9]])
    np.testing.assert_allclose(ref(pts), pts[:, 0], atol=1e-6)


def test_symmetric_double_well_is_half_at_the_barrier():
    ref = grid_committor(
        lambda x, y: 5.0 * (x**2 - 1.0) ** 2 + 2.0 * y**2,
        lambda x, y: (x < -0.8) & (y == y),
        lambda x, y: (x > 0.8) & (y == y),
        xlim=(-1.6, 1.6), ylim=(-1.5, 1.5), n=81,
    )
    assert ref(np.array([[0.0, 0.0]]))[0] == pytest.approx(0.5, abs=1e-6)
    # antisymmetry about the barrier
    assert ref(np.array([[0.3, 0.2]]))[0] + ref(np.array([[-0.3, 0.2]]))[0] == pytest.approx(1.0, abs=1e-6)


def test_reference_is_converged_in_the_grid_spacing():
    pts = np.array([[0.0, 0.0], [-1.0, 0.3], [0.9, 0.2], [0.5, -0.9]])
    np.testing.assert_allclose(_wolfe_quapp_reference(121)(pts), _wolfe_quapp_reference(201)(pts), atol=5e-3)


def test_overlapping_or_empty_basins_are_rejected():
    flat = lambda x, y: np.zeros(np.broadcast(x, y).shape)  # noqa: E731
    with pytest.raises(ValueError, match="no grid point"):
        grid_committor(flat, lambda x, y: x < -5.0, lambda x, y: x > 0.5,
                       xlim=(0, 1), ylim=(0, 1), n=11)
    with pytest.raises(ValueError, match="overlap"):
        grid_committor(flat, lambda x, y: x < 0.6, lambda x, y: x > 0.4,
                       xlim=(0, 1), ylim=(0, 1), n=11)


def test_shooting_from_inside_a_basin_costs_nothing():
    engine = ToyLangevinEngine(dt=0.005)
    basins = BasinPair(SphereBasin(A, R), SphereBasin(B, R))
    res = shooting_committor(engine, engine.create_state(B), basins, _xy, n_shots=7)
    assert (res.committor, res.total_steps, res.n_open) == (1.0, 0, 0)
    assert res.stderr > 0.0                               # never reports zero uncertainty


def test_shooting_reports_uncommitted_shots():
    engine = ToyLangevinEngine(dt=0.005)
    basins = BasinPair(SphereBasin(A, R), SphereBasin(B, R))
    res = shooting_committor(engine, engine.create_state([0.0, 0.0]), basins, _xy,
                             n_shots=5, max_steps=3, check_every=1, seed=0)
    assert res.n_open == 5 and np.isnan(res.committor) and res.total_steps == 15


def test_shooting_matches_exact_committor():
    """Milestone M0: shooting agrees with the exact solution within sampling error."""
    ref = _wolfe_quapp_reference()
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    basins = BasinPair(SphereBasin(A, R), SphereBasin(B, R))
    for point in ([0.0, 0.0], [-1.0, 0.3], [0.5, -0.9]):
        start = engine.create_state(point)
        live = len(engine._cache)
        res = shooting_committor(engine, start, basins, _xy, n_shots=300,
                                 max_steps=100_000, check_every=5, seed=4)
        assert res.n_open == 0
        assert abs(res.committor - ref(np.array([point]))[0]) < 4.0 * res.stderr
        np.testing.assert_allclose(_xy(engine.get_coords(start)), point)   # start untouched
        assert len(engine._cache) == live                                  # no leaked handles
