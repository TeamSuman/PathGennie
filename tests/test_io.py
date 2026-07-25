from pathlib import Path

import pytest
import numpy as np

from pathgennie.backends.amber.utils import (
    parse_prmtop,
    read_rst7_coords,
    wrap_frames_pbc,
    write_multimodel_pdb,
    write_native_trajectory,
)

REPO = Path(__file__).resolve().parents[1]
COMMON = REPO / "examples" / "alanine_dipeptide" / "common"
PRMTOP = COMMON / "ala_dipeptide.prmtop"
RST7 = COMMON / "ala_dipeptide_equilibrated.rst7"


def test_rst7_read_shape():
    coords = read_rst7_coords(RST7)
    assert coords.ndim == 2 and coords.shape[1] == 3
    assert np.all(np.isfinite(coords))


def test_prmtop_atom_mass_consistency():
    info = parse_prmtop(PRMTOP)
    assert len(info["atom_names"]) == len(info["masses"])
    coords = read_rst7_coords(RST7)
    assert coords.shape[0] == len(info["atom_names"])


def test_multimodel_pdb_roundtrip(tmp_path):
    info = parse_prmtop(PRMTOP)
    n_atoms = len(info["atom_names"])
    frames = np.zeros((2, n_atoms, 3), dtype=float)
    frames[1] += 1.0
    out = tmp_path / "traj.pdb"
    write_multimodel_pdb(out, info, frames)
    text = out.read_text()
    assert text.count("MODEL") == 2
    assert text.count("ENDMDL") == 2
    # Coordinates should be parseable back from the fixed-width records.
    xs = [float(line[30:38]) for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))]
    assert len(xs) == 2 * n_atoms


def test_wrap_frames_pbc_no_box_is_identity():
    info = {"box_lengths": None}
    frames = np.random.default_rng(0).standard_normal((1, 5, 3))
    np.testing.assert_array_equal(wrap_frames_pbc(frames, info), frames)


# ---------------------------------------------------------------------------
# Trajectory timestep tests — verify dt and per-frame time round-trip
# ---------------------------------------------------------------------------

def test_write_trajectory_dt_xtc(tmp_path):
    """XTC files should store the correct dt and per-frame timestamps."""
    mda = pytest.importorskip("MDAnalysis")

    n_atoms, n_frames, dt = 10, 5, 0.5
    frames = np.random.default_rng(42).standard_normal((n_frames, n_atoms, 3)).astype(np.float32)
    info = {"box_lengths": None}
    out = tmp_path / "test.xtc"

    write_native_trajectory(out, info, frames, dt=dt)

    u = mda.Universe.empty(n_atoms, trajectory=True)
    u.load_new(str(out))
    assert u.trajectory.dt == pytest.approx(dt)
    for i, ts in enumerate(u.trajectory):
        assert ts.time == pytest.approx(i * dt)


def test_write_trajectory_dt_dcd(tmp_path):
    """DCD header-level dt should round-trip correctly (ps→AKMA→ps)."""
    mda = pytest.importorskip("MDAnalysis")

    n_atoms, n_frames, dt = 10, 4, 2.0
    frames = np.random.default_rng(7).standard_normal((n_frames, n_atoms, 3)).astype(np.float32)
    info = {"box_lengths": None}
    out = tmp_path / "test.dcd"

    write_native_trajectory(out, info, frames, dt=dt)

    u = mda.Universe.empty(n_atoms, trajectory=True)
    u.load_new(str(out))
    assert u.trajectory.dt == pytest.approx(dt, rel=1e-5)
    for i, ts in enumerate(u.trajectory):
        assert ts.time == pytest.approx(i * dt, rel=1e-5)


def test_write_trajectory_dt_none_still_works(tmp_path):
    """Passing dt=None (default) should not break the writer."""
    mda = pytest.importorskip("MDAnalysis")

    n_atoms, n_frames = 10, 3
    frames = np.random.default_rng(0).standard_normal((n_frames, n_atoms, 3)).astype(np.float32)
    info = {"box_lengths": None}
    out = tmp_path / "test.xtc"

    write_native_trajectory(out, info, frames)  # dt=None (default)

    u = mda.Universe.empty(n_atoms, trajectory=True)
    u.load_new(str(out))
    assert u.trajectory.n_frames == n_frames
