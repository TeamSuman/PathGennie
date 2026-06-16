from pathlib import Path

import numpy as np

from pathgennie.backends.amber.utils import (
    parse_prmtop,
    read_rst7_coords,
    wrap_frames_pbc,
    write_multimodel_pdb,
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
