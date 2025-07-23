import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import joblib
import MDAnalysis as mda
from tqdm.auto import trange
from openmm import *
from openmm import LangevinMiddleIntegrator
from openmm import Platform
from openmm.app import GromacsGroFile, GromacsTopFile, HBonds, PME, Simulation
from openmm.unit import picosecond, nanometers, femtoseconds, kelvin, amu


class Simulation_obj:
    def __init__(self,device = 0):

        ffdir = "/usr/share/gromacs/top"
        platform_name = "CUDA"
        self.temperature = 300

        self.nonbondedMethod = PME
        self.nonbondedCutoff = 1.0 * nanometers
        self.ewaldErrorTolerance = 0.0005
        self.constraints = HBonds
        self.rigidWater = True
        self.constraintTolerance = 0.000001
        self.hydrogenMass = 1.5 * amu

        gro_file = GromacsGroFile("pbcmol.gro")
        top_file = GromacsTopFile("topol.top", periodicBoxVectors=gro_file.getPeriodicBoxVectors(), includeDir=ffdir)

        integrator = LangevinMiddleIntegrator(self.temperature * kelvin, 1.0 / picosecond, 2.0 * femtoseconds)
        integrator.setConstraintTolerance(self.constraintTolerance)

        system = top_file.createSystem(
            nonbondedMethod=self.nonbondedMethod,
            nonbondedCutoff=self.nonbondedCutoff,
            constraints=self.constraints,
            rigidWater=self.rigidWater,
            ewaldErrorTolerance=self.ewaldErrorTolerance,
            hydrogenMass=self.hydrogenMass
        )

        if platform_name == 'CUDA':
            platform = Platform.getPlatformByName('CUDA')
            properties = {'Precision': 'mixed', 'DeviceIndex': str(device)}
            self.simulation = Simulation(top_file.topology, system, integrator, platform, properties)
        else:
            platform = Platform.getPlatformByName('CPU')
            self.simulation = Simulation(top_file.topology, system, integrator, platform)

        self.box = np.array([list(gro_file.getPeriodicBoxVectors()._value[i]) for i in range(3)]).astype(np.float32)
        self.simulation.context.reinitialize(preserveState=True)
