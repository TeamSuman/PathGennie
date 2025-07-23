# Modules
from openmm import *
from openmm.app import *
from openmm.unit import *
import os
from sys import stdout

class ImplicitOMM:
    def __init__(self, device_idx = 0):

        path = "/home/dibyendu/Projects/WE/NoteBooks/NTL9/Data/"
        top_path = os.path.join(path, 'ntl9.prmtop')
        pdb_path = os.path.join(path, 'reference.pdb')
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
            80.0/picosecond,     # gamma_ln
            0.002*picoseconds # dt
        )

        platform = Platform.getPlatformByName("CUDA")
        platformProperties = {'Precision': "single", 'DeviceIndex': str(device_idx)}
        self.simulation = Simulation(prmtop.topology, system, integrator, platform, platformProperties)
        #self.simulation.reporters.append(StateDataReporter(stdout, 10000, step=True))
        #self.simulation.reporters.append(DCDReporter('output.dcd', 5000))
        self.simulation.context.setPositions(pdb.positions)
        self.simulation.context.setVelocitiesToTemperature(300*kelvin)
