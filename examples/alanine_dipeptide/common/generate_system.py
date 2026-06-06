#!/usr/bin/env python
"""Build and equilibrate a solvated alanine dipeptide standard test system."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import parmed as pmd
from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform, unit
from openmm.app import AmberInpcrdFile, AmberPrmtopFile, PME, HBonds, Simulation


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "common"
AMBER = ROOT / "amber"
GROMACS = ROOT / "gromacs"
OPENMM = ROOT / "openmm"
TLEAP = shutil.which("tleap") or "/home/dm/Soft/miniconda3/envs/pg/bin/tleap"


def run_cmd(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed: "
            + " ".join(cmd)
            + "\nstdout:\n"
            + result.stdout
            + "\nstderr:\n"
            + result.stderr
        )


def write_tleap_input() -> Path:
    leap_in = COMMON / "tleap.in"
    leap_in.write_text(
        """source leaprc.protein.ff14SB
source leaprc.water.tip3p
mol = sequence { ACE ALA NME }
solvateBox mol TIP3PBOX 8.0
saveamberparm mol ala_dipeptide.prmtop ala_dipeptide_initial.rst7
savepdb mol ala_dipeptide_initial.pdb
quit
""",
        encoding="utf-8",
    )
    return leap_in


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
    platform = Platform.getPlatformByName("CPU")
    simulation = Simulation(top.topology, system, integrator, platform)
    simulation.context.setPositions(crd.positions)
    if crd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*crd.boxVectors)
    simulation.minimizeEnergy(maxIterations=200)
    simulation.context.setVelocitiesToTemperature(300.0 * unit.kelvin)
    simulation.step(500)
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    coords = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    box_vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
    lengths = [
        float(np.linalg.norm(box_vectors[0])),
        float(np.linalg.norm(box_vectors[1])),
        float(np.linalg.norm(box_vectors[2])),
        90.0,
        90.0,
        90.0,
    ]
    return np.asarray(coords, dtype=float), lengths


def write_equilibrated_files(coords: np.ndarray, box: list[float]) -> None:
    source = COMMON / "ala_dipeptide.prmtop"
    structure = pmd.load_file(str(source))
    structure.coordinates = coords
    structure.box = box

    structure.save(str(COMMON / "ala_dipeptide_equilibrated.rst7"), overwrite=True)
    structure.save(str(COMMON / "ala_dipeptide_equilibrated.pdb"), overwrite=True)

    for target in (AMBER, OPENMM):
        shutil.copy(COMMON / "ala_dipeptide.prmtop", target / "ala_dipeptide.prmtop")
        shutil.copy(COMMON / "ala_dipeptide_equilibrated.rst7", target / "ala_dipeptide_equilibrated.rst7")
        shutil.copy(COMMON / "ala_dipeptide_equilibrated.pdb", target / "ala_dipeptide_equilibrated.pdb")

    structure.save(str(GROMACS / "ala_dipeptide.top"), overwrite=True)
    structure.save(str(GROMACS / "ala_dipeptide_equilibrated.gro"), overwrite=True)
    shutil.copy(COMMON / "ala_dipeptide.prmtop", GROMACS / "ala_dipeptide.prmtop")


def main() -> None:
    for directory in (COMMON, AMBER, GROMACS, OPENMM):
        directory.mkdir(parents=True, exist_ok=True)
    leap_in = write_tleap_input()
    run_cmd([TLEAP, "-f", str(leap_in.name)], COMMON)
    coords, box = equilibrate(
        COMMON / "ala_dipeptide.prmtop",
        COMMON / "ala_dipeptide_initial.rst7",
    )
    write_equilibrated_files(coords, box)
    print("Generated solvated and equilibrated alanine dipeptide system.")
    print(f"Atoms: {coords.shape[0]}")
    print(f"Box: {box[:3]} Angstrom")


if __name__ == "__main__":
    main()
