from pathlib import Path

import os
import shutil

import pytest
import yaml
import numpy as np

from pathgennie.backends.amber.utils import (
    parse_prmtop,
    read_native_trajectory,
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


def test_read_native_trajectory_roundtrip(tmp_path):
    """Write frames with write_native_trajectory, read back with
    read_native_trajectory, and verify shape and values match."""
    mda = pytest.importorskip("MDAnalysis")

    n_atoms, n_frames = 10, 5
    rng = np.random.default_rng(42)
    frames = rng.standard_normal((n_frames, n_atoms, 3)).astype(np.float32)
    info = {"box_lengths": None}
    out = tmp_path / "roundtrip.xtc"

    write_native_trajectory(out, info, frames, dt=2.0)
    read_back = read_native_trajectory(out)

    assert read_back.shape == frames.shape
    np.testing.assert_allclose(read_back, frames, atol=1e-2)


def _overwrite_case(tmp_path, existing_names, cfg_extra=None):
    """A case complete enough that ``run()`` reaches the overwrite guard.

    ``run()`` validates the executable and input files *before* the guard, so a
    skeleton directory fails earlier for an unrelated reason. ``/bin/echo`` stands
    in for the MD binary -- nothing is propagated, because the guard raises before
    the driver starts.
    """
    case = tmp_path / "case"
    (case / "output").mkdir(parents=True)
    shutil.copy(PRMTOP, case / "x.prmtop")
    shutil.copy(REPO / "examples/alanine_dipeptide/amber/ala_dipeptide_equilibrated.rst7",
                case / "x.rst7")
    (case / "p.py").write_text(
        "import numpy as np\n"
        "def f(coords, **kw):\n    return np.array([float(coords[0, 0])])\n"
        "def g(coords, **kw):\n    return False\n"
    )
    for name in existing_names:
        (case / "output" / name).write_text("existing content")
    cfg = {
        "amber": {"topology": "x.prmtop", "initial_restart": "x.rst7",
                  "executable": "/bin/echo"},
        "pathgennie": {"mode": "escape", "tau1_steps": 1, "tau2_steps": 1,
                       "max_trial": 1, "max_cycle": 1},
        "projection": {"module": "p", "function": "f"},
        "convergence": {"module": "p", "function": "g"},
        "workdir": ".",
    }
    cfg["pathgennie"].update(cfg_extra or {})
    (case / "input.yaml").write_text(yaml.safe_dump(cfg))
    return case


def test_existing_output_raises_without_overwrite(tmp_path):
    """The guard must come from pg_amber.run, not from the test.

    The previous version rebuilt the condition inline and then raised
    FileExistsError itself inside pytest.raises -- it would have passed with the
    feature deleted. This calls the real entrypoint.
    """
    from pathgennie.backends.amber import pg_amber

    case = _overwrite_case(tmp_path, ["reactive_path.pdb"])
    cwd = os.getcwd()
    try:
        with pytest.raises(FileExistsError, match="already exist"):
            pg_amber.run(case, "input.yaml")
    finally:
        os.chdir(cwd)


def test_overwrite_true_gets_past_the_guard(tmp_path):
    """With overwrite set the run must fail *later*, not on the guard."""
    from pathgennie.backends.amber import pg_amber

    case = _overwrite_case(tmp_path, ["reactive_path.pdb"], {"overwrite": True})
    cwd = os.getcwd()
    try:
        with pytest.raises(Exception) as info:
            pg_amber.run(case, "input.yaml")
        assert not isinstance(info.value, FileExistsError), \
            "overwrite: true must not trip the existing-output guard"
    finally:
        os.chdir(cwd)


def test_prmtop_readers_accept_str_paths():
    """parse_prmtop / read_prmtop_flag must accept a str, like the other readers.

    read_rst7_coords already coerced with Path(path), but the prmtop readers did
    not, so a plain string raised a bare
    ``AttributeError: 'str' object has no attribute 'read_text'`` far from the
    call site. Analysis scripts naturally pass strings.
    """
    from pathgennie.backends.amber.utils import parse_prmtop, read_prmtop_flag

    if not PRMTOP.exists():
        import pytest
        pytest.skip("example prmtop not available")

    prmtop = PRMTOP
    from_path = parse_prmtop(prmtop)
    from_str = parse_prmtop(str(prmtop))
    assert from_str["atom_names"] == from_path["atom_names"]
    assert len(read_prmtop_flag(str(prmtop), "ATOM_NAME")) > 0


# ---------------------------------------------------------------------------
# Triclinic unit cell — the trajectory writer must record the real cell, not a
# 90/90/90 box built from the diagonal alone.
# ---------------------------------------------------------------------------

def test_gro_box_vectors_orthorhombic_and_triclinic():
    from pathgennie.backends.gromacs.utils import gro_box_vectors

    np.testing.assert_allclose(
        gro_box_vectors([2.0, 3.0, 4.0]),
        [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
    )
    # GROMACS order: v1x v2y v3z v1y v1z v2x v2z v3x v3y (v1y=v1z=v2z=0 by convention)
    np.testing.assert_allclose(
        gro_box_vectors([5.0, 5.0, 3.5, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5]),
        [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [2.5, 2.5, 3.5]],
    )


def _dodecahedron_gro(tmp_path, n_atoms=3):
    """A minimal .gro with a rhombic-dodecahedron (triclinic) box line."""
    lines = ["triclinic test", f"{n_atoms:5d}"]
    for i in range(n_atoms):
        lines.append(f"{1:5d}{'SOL':>5s}{'OW':>5s}{i + 1:5d}{0.1 * i:8.3f}{0.2 * i:8.3f}{0.3 * i:8.3f}")
    # 5.91225 5.91225 4.18059 0 0 0 0 2.95613 2.95613 (rfah_ctd's equilibrated 300 K fold-A box)
    lines.append("   5.91225   5.91225   4.18059   0.00000   0.00000   0.00000   0.00000   2.95613   2.95613")
    p = tmp_path / "triclinic.gro"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_read_topology_info_keeps_the_full_triclinic_cell(tmp_path):
    from pathgennie.backends.gromacs.utils import read_topology_info

    info = read_topology_info(_dodecahedron_gro(tmp_path))
    # box_lengths keeps only the diagonal (Angstrom) — kept for backward compatibility
    np.testing.assert_allclose(info["box_lengths"], [59.1225, 59.1225, 41.8059], rtol=1e-6)
    # box_vectors carries the off-diagonal terms that make it a dodecahedron
    np.testing.assert_allclose(
        info["box_vectors"],
        [[59.1225, 0.0, 0.0], [0.0, 59.1225, 0.0], [29.5613, 29.5613, 41.8059]],
        rtol=1e-6,
    )


def test_unitcell_dimensions_prefers_vectors_over_the_diagonal(tmp_path):
    pytest.importorskip("MDAnalysis")
    from pathgennie.backends.amber.utils import unitcell_dimensions
    from pathgennie.backends.gromacs.utils import read_topology_info

    info = read_topology_info(_dodecahedron_gro(tmp_path))
    dims = unitcell_dimensions(info)
    assert dims is not None
    # a = b = 59.1225 A; c = |(29.5613, 29.5613, 41.8059)| = 59.1225 A for a dodecahedron
    np.testing.assert_allclose(dims[:3], [59.1225, 59.1225, 59.1225], rtol=1e-5)
    # the defining dodecahedron angles: alpha = beta = 60, gamma = 90
    np.testing.assert_allclose(dims[3:], [60.0, 60.0, 90.0], atol=1e-3)
    # falling back to the diagonal alone (the old behaviour) would have said 90/90/90
    diag_only = unitcell_dimensions({"box_lengths": info["box_lengths"]})
    np.testing.assert_allclose(diag_only[3:], [90.0, 90.0, 90.0])


@pytest.mark.parametrize("ext", ["dcd", "xtc"])
def test_write_native_trajectory_records_the_triclinic_cell(tmp_path, ext):
    """Regression: the writer used to emit [lx, ly, lz, 90, 90, 90] from box_lengths, so a
    rhombic-dodecahedron run produced a trajectory claiming an orthorhombic cell ~1.4x too
    large in volume."""
    mda = pytest.importorskip("MDAnalysis")
    from pathgennie.backends.gromacs.utils import read_topology_info

    info = read_topology_info(_dodecahedron_gro(tmp_path))
    n_atoms = 3
    frames = np.random.default_rng(3).standard_normal((2, n_atoms, 3)).astype(np.float32)
    out = tmp_path / f"traj.{ext}"
    write_native_trajectory(out, info, frames, dt=1.0)

    u = mda.Universe.empty(n_atoms, trajectory=True)
    u.load_new(str(out))
    for ts in u.trajectory:
        np.testing.assert_allclose(ts.dimensions[:3], [59.1225, 59.1225, 59.1225], rtol=1e-4)
        np.testing.assert_allclose(ts.dimensions[3:], [60.0, 60.0, 90.0], atol=1e-2)


def test_write_native_trajectory_orthorhombic_fallback_unchanged(tmp_path):
    """A topology with only box_lengths (e.g. a prmtop) still gets the 90/90/90 cell."""
    mda = pytest.importorskip("MDAnalysis")

    n_atoms = 4
    frames = np.zeros((2, n_atoms, 3), dtype=np.float32)
    out = tmp_path / "ortho.xtc"
    write_native_trajectory(out, {"box_lengths": np.array([30.0, 40.0, 50.0])}, frames, dt=1.0)

    u = mda.Universe.empty(n_atoms, trajectory=True)
    u.load_new(str(out))
    for ts in u.trajectory:
        np.testing.assert_allclose(ts.dimensions, [30.0, 40.0, 50.0, 90.0, 90.0, 90.0], rtol=1e-4)
