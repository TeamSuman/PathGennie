#!/usr/bin/env python
"""Build and equilibrate the solvated ACE-ALA-NME QM/MM conformational system."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import parmed as pmd
from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform, unit
from openmm.app import AmberInpcrdFile, AmberPrmtopFile, HBonds, PME, Simulation


ROOT = Path(__file__).resolve().parent
TLEAP = shutil.which("tleap") or os.environ.get("TLEAP", "tleap")


def run_cmd(cmd: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def write_tleap_input(path: Path) -> None:
    path.write_text(
        """source leaprc.protein.ff14SB
source leaprc.water.tip3p
mol = sequence { ACE ALA NME }
solvateBox mol TIP3PBOX 8.0
addIonsRand mol Na+ 1 Cl- 1
saveamberparm mol qmmm_ala.prmtop qmmm_ala_initial.rst7
savepdb mol qmmm_ala_initial.pdb
quit
""",
        encoding="utf-8",
    )


def equilibrate(prmtop: Path, inpcrd: Path) -> tuple[np.ndarray, list[float]]:
    top = AmberPrmtopFile(str(prmtop))
    crd = AmberInpcrdFile(str(inpcrd))
    system = top.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=HBonds,
    )
    system.addForce(MonteCarloBarostat(1.0 * unit.atmosphere, 300.0 * unit.kelvin, 25))
    integrator = LangevinMiddleIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.001 * unit.picoseconds,
    )
    simulation = Simulation(top.topology, system, integrator, Platform.getPlatformByName("CPU"))
    simulation.context.setPositions(crd.positions)
    if crd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*crd.boxVectors)
    simulation.minimizeEnergy(maxIterations=200)
    simulation.context.setVelocitiesToTemperature(300.0 * unit.kelvin)
    simulation.step(500)
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    coords = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    box_vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
    box = [
        float(np.linalg.norm(box_vectors[0])),
        float(np.linalg.norm(box_vectors[1])),
        float(np.linalg.norm(box_vectors[2])),
        90.0,
        90.0,
        90.0,
    ]
    return np.asarray(coords, dtype=float), box


def main() -> None:
    write_tleap_input(ROOT / "tleap.in")
    run_cmd([TLEAP, "-f", "tleap.in"])
    coords, box = equilibrate(ROOT / "qmmm_ala.prmtop", ROOT / "qmmm_ala_initial.rst7")
    structure = pmd.load_file(str(ROOT / "qmmm_ala.prmtop"))
    structure.coordinates = coords
    structure.box = box
    structure.save(str(ROOT / "qmmm_ala_equilibrated.rst7"), overwrite=True)
    structure.save(str(ROOT / "qmmm_ala_equilibrated.pdb"), overwrite=True)
    print("Generated solvated, ionized, equilibrated ACE-ALA-NME QM/MM conformational system.")
    print(f"Atoms: {coords.shape[0]}")
    print(f"Box: {box[:3]} Angstrom")


if __name__ == "__main__":
    main()
