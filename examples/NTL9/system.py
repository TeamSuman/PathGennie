# Modules
from openmm import *
from openmm.app import *
from openmm.unit import *
import os
from sys import stdout

class Simulation_obj:
    def __init__(self, device_idx = 0):

        top_path = "./ntl9.prmtop"
        pdb_path = "./reference.pdb"
        # Load AMBER topology and coordinates
        prmtop = AmberPrmtopFile(top_path)          # chignolin.prmtop
        pdb = PDBFile(pdb_path)                  # chignolin.pdb

        # Get positions from the PDB file
        pos = pdb.getPositions()


        # Create implicit solvent system
        system = prmtop.createSystem(
            nonbondedMethod= NoCutoff,          # No periodicity
            constraints= HBonds,                # Bonds to H constrained
            implicitSolvent= HCT,               # HCT GB model (igb = 1)
            soluteDielectric=1.0,
            solventDielectric=78.5
        )

        # Langevin integrator matching md.in
        integrator = LangevinMiddleIntegrator(
            300*kelvin,       # temp0
            5.0/picosecond,     # gamma_ln
            0.002*picoseconds # dt
        )

        platform = Platform.getPlatformByName("CUDA")
        platformProperties = {'Precision': "mixed", 'DeviceIndex': str(device_idx)}
        self.simulation = Simulation(prmtop.topology, system, integrator, platform, platformProperties)

