"""Geometry helpers of the alanine dipeptide committor example (no MD engine)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "examples" / "committor_alanine_dipeptide" / "adp.py"
_spec = importlib.util.spec_from_file_location("committor_adp_example", _PATH)
adp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adp)


def _reference_coords():
    rows = []
    for line in adp.PDB.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[17:20] in ("ACE", "ALA", "NME"):
            rows.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(rows)


def test_atom_selections_match_the_structure():
    names = [line[12:16].strip() for line in adp.PDB.read_text().splitlines()
             if line.startswith(("ATOM", "HETATM")) and line[17:20] in ("ACE", "ALA", "NME")]
    assert len(names) == 22
    assert [names[i] for i in adp.PHI_ATOMS] == ["C", "N", "CA", "C"]
    assert [names[i] for i in adp.PSI_ATOMS] == ["N", "CA", "C", "N"]
    heavy = [names[i] for i in adp.HEAVY_ATOMS]
    assert len(heavy) == 10 and not any(n.startswith("H") for n in heavy)
    assert sum(not n.startswith("H") for n in names) == 10        # all heavy atoms are selected


def test_features_are_rigid_motion_invariant():
    coords = _reference_coords()
    feats = adp.heavy_atom_distances(coords)
    assert feats.shape == (45,)
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    moved = coords @ rot.T + np.array([3.0, -2.0, 5.0])
    np.testing.assert_allclose(adp.heavy_atom_distances(moved), feats, atol=1e-9)
    np.testing.assert_allclose(adp.phi_psi(moved), adp.phi_psi(coords), atol=1e-8)


def test_basins_wrap_across_the_dihedral_branch_cut():
    pair = adp.basins()
    assert pair.label([-150.0, 170.0]) == 0
    assert pair.label([-155.0, -182.0]) == 0          # psi = 178 degrees after wrapping
    assert pair.label([-155.0, -175.0]) == -1         # psi = 185: 15 degrees off
    assert pair.label([90.0, -50.0]) == 1
    assert pair.label([-150.0, 150.0]) == -1          # 20 degrees off in psi
    assert pair.label([0.0, 0.0]) == -1


def test_dihedral_progress_uses_the_minimum_image():
    progress = adp.dihedral_progress("A")
    near = progress.metric(np.array([-150.0, -175.0]))     # 15 degrees from C7eq
    far = progress.metric(np.array([-150.0, 120.0]))       # 50 degrees
    assert near == pytest.approx(-15.0) and near > far
