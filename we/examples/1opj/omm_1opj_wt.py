import copy

from openmm import *
from openmm import LangevinMiddleIntegrator, Platform
from openmm.app import PME, GromacsGroFile, GromacsTopFile, HBonds, Simulation
from openmm.unit import amu, femtoseconds, kelvin, nanometers, picosecond


class OpenMMRunner:
    def __init__(self, device=0, gro_file=None, top_file=None, ffdir=None, dt_ps=0.002, temperature=300.0, platform="CUDA"):

        if gro_file is None or top_file is None:
            raise ValueError("OpenMMRunner requires 'gro_file' and 'top_file'.")

        platform_name = platform
        self.temperature = temperature

        self.nonbondedMethod = PME
        self.nonbondedCutoff = 1.0 * nanometers
        self.ewaldErrorTolerance = 0.0005
        self.constraints = HBonds
        self.rigidWater = True
        self.constraintTolerance = 0.000001
        self.hydrogenMass = 1.5 * amu

        gro = GromacsGroFile(gro_file)
        top_kwargs = {"periodicBoxVectors": gro.getPeriodicBoxVectors()}
        if ffdir is not None:
            top_kwargs["includeDir"] = ffdir
        self.top = GromacsTopFile(top_file, **top_kwargs)

        self.integrator = LangevinMiddleIntegrator(self.temperature * kelvin, 1.0 / picosecond, dt_ps * picosecond)
        self.integrator.setConstraintTolerance(self.constraintTolerance)

        self.system = self.top.createSystem(
            nonbondedMethod=self.nonbondedMethod,
            nonbondedCutoff=self.nonbondedCutoff,
            constraints=self.constraints,
            rigidWater=self.rigidWater,
            ewaldErrorTolerance=self.ewaldErrorTolerance,
            hydrogenMass=self.hydrogenMass
        )

        if platform_name == 'CUDA':
            self.platform = Platform.getPlatformByName('CUDA')
            self.prop = {'Precision': 'mixed', 'DeviceIndex': str(device)}
            self.simulation = Simulation(self.top.topology, self.system, self.integrator, self.platform, self.prop)
        else:
            self.platform = Platform.getPlatformByName('CPU')
            self.simulation = Simulation(self.top.topology, self.system, self.integrator, self.platform)

    def _create_simulation(self):
        """Creates the main OpenMM Simulation object and sets its initial state."""
        new_integrator = copy.copy(self.integrator)
        new_integrator.setRandomNumberSeed(0)
        self.simulation = Simulation(self.top.topology, self.system, new_integrator, self.platform, self.prop)
