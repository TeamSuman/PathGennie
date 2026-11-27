"""Alanine dipeptide in vacuum for committor milestone M2 (OpenMM).

System, basins and features follow the benchmark of Pharos (arXiv:2611.26765)
so results are comparable: 22 atoms, vacuum, 300 K, basins C7eq and C7ax
centred at (phi, psi) = (-150, 170) and (90, -50) degrees, each including
configurations within 10 degrees of the centre in both dihedrals; the network
input is the 45 pairwise distances among the ten heavy atoms.

Force field
-----------
The reference benchmark uses CHARMM27, which OpenMM does not bundle. Pass the
CHARMM files (``psf=``, ``params=[rtf/prm/str ...]``) to :func:`build_simulation`
to use it. Without them the bundled ``charmm36_2024.xml`` is used as a
**stand-in**: the same code path, but a different backbone CMAP, so numbers
obtained with it are not the M2 result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
PDB = HERE.parent / "alanine_dipeptide" / "common" / "ala_dipeptide_equilibrated.pdb"

# 0-based indices in ACE-ALA-NME as ordered in the PDB above.
PHI_ATOMS = (4, 6, 8, 14)     # ACE C, ALA N, ALA CA, ALA C
PSI_ATOMS = (6, 8, 14, 16)    # ALA N, ALA CA, ALA C, NME N
HEAVY_ATOMS = (1, 4, 5, 6, 8, 10, 14, 15, 16, 18)

C7EQ = np.array([-150.0, 170.0])
C7AX = np.array([90.0, -50.0])
BASIN_HALF_WIDTH = 10.0


def _dihedral(coords: np.ndarray, atoms) -> float:
    p0, p1, p2, p3 = (coords[i] for i in atoms)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def phi_psi(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    return np.array([_dihedral(coords, PHI_ATOMS), _dihedral(coords, PSI_ATOMS)])


def angular_delta(a, b) -> np.ndarray:
    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0


def heavy_atom_distances(coords: np.ndarray) -> np.ndarray:
    """The 45 pairwise distances (Angstrom) among the ten heavy atoms."""
    xyz = np.asarray(coords, dtype=float)[list(HEAVY_ATOMS)]
    i, j = np.triu_indices(len(HEAVY_ATOMS), k=1)
    return np.linalg.norm(xyz[i] - xyz[j], axis=1)


def basins():
    from pathgennie.committor import BasinPair, CallableBasin

    def square(center):
        return CallableBasin(lambda cv: bool(np.all(np.abs(angular_delta(cv, center)) < BASIN_HALF_WIDTH)))

    return BasinPair(A=square(C7EQ), B=square(C7AX))


def dihedral_progress(target: str):
    """Geometric progress variable toward ``"A"`` (C7eq) or ``"B"`` (C7ax)."""
    from pathgennie.core.progress import TargetMetric

    return TargetMetric(phi_psi, target_cv=C7AX if target == "B" else C7EQ, periodic=[360.0, 360.0])


def build_simulation(*, psf: Optional[str] = None, params: Optional[Sequence[str]] = None,
                     temperature: float = 300.0, friction: float = 1.0, timestep_fs: float = 1.0,
                     platform: str = "CPU", threads: int = 1):
    """Return ``(simulation, positions_nm, force_field_label)`` for the vacuum system."""
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    pdb = app.PDBFile(str(PDB))
    model = app.Modeller(pdb.topology, pdb.positions)
    model.delete([r for r in model.topology.residues() if r.name in ("WAT", "HOH")])
    model.topology.setPeriodicBoxVectors(None)

    if psf is not None:
        if not params:
            raise ValueError("CHARMM parameter files are required with a PSF")
        structure = app.CharmmPsfFile(psf)
        system = structure.createSystem(app.CharmmParameterSet(*params),
                                        nonbondedMethod=app.NoCutoff, constraints=None)
        topology, label = structure.topology, f"CHARMM files ({Path(psf).name})"
    else:
        system = app.ForceField("charmm36_2024.xml").createSystem(
            model.topology, nonbondedMethod=app.NoCutoff, constraints=None)
        topology, label = model.topology, "charmm36_2024.xml (stand-in, not CHARMM27)"

    integrator = openmm.LangevinMiddleIntegrator(
        temperature * unit.kelvin, friction / unit.picosecond, timestep_fs * unit.femtosecond)
    properties = {"Threads": str(threads)} if platform == "CPU" else {}
    simulation = app.Simulation(topology, system, integrator,
                                openmm.Platform.getPlatformByName(platform), properties)
    simulation.context.setPositions(model.positions)
    simulation.minimizeEnergy()
    positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    return simulation, positions, label
