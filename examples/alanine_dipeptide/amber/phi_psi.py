"""Phi/psi collective variables for solvated ACE-ALA-NME."""

from __future__ import annotations

import numpy as np


# 0-based indices in the tleap-generated ACE-ALA-NME topology.
PHI_ATOMS = (4, 6, 8, 14)  # ACE C, ALA N, ALA CA, ALA C
PSI_ATOMS = (6, 8, 14, 16)  # ALA N, ALA CA, ALA C, NME N


def dihedral_degrees(coords: np.ndarray, atoms: tuple[int, int, int, int]) -> float:
    p0, p1, p2, p3 = (coords[index] for index in atoms)
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def phi_psi_cv(coords, phi_atoms=PHI_ATOMS, psi_atoms=PSI_ATOMS):
    coords = np.asarray(coords, dtype=float)
    return np.asarray(
        [
            dihedral_degrees(coords, tuple(int(index) for index in phi_atoms)),
            dihedral_degrees(coords, tuple(int(index) for index in psi_atoms)),
        ],
        dtype=float,
    )


def angular_delta_degrees(values, target):
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float)
    return (values - target + 180.0) % 360.0 - 180.0


def reached_phi_psi(coords, target=(-60.0, -40.0), tolerance=25.0, **kwargs):
    delta = angular_delta_degrees(phi_psi_cv(coords, **kwargs), target)
    return bool(np.linalg.norm(delta) < float(tolerance))
