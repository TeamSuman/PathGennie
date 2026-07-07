import copy
import warnings

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit


class ThreeHoleForce(mm.CustomExternalForce):
    """
    OpenMM implementation of the 2D three-hole potential.
    Includes a harmonic restraint in z (1000*z^2) to keep it quasi-2D.
    """

    def __init__(self):
        expr = """
            1000*z*z
            - 3.0*exp(-x*x) * (exp(-(y-1.6666666667)^2) - exp(-(y-0.3333333333)^2))
            - 5*exp(-y*y) * (exp(-(x-1)^2) + exp(-(x+1)^2))
        """
        super().__init__(expr)


###############################################################################
# System Object
###############################################################################
class OpenMMRunner:
    def __init__(self, device=0, temperature=200.0, platform="CUDA"):
        warnings.filterwarnings("ignore")

        # --- 1. Configuration Parameters (Slightly adjusted for better kinetics) ---
        self.nParticles = 1
        self.mass = 1.0 * unit.dalton
        self.temperature = temperature * unit.kelvin
        self.friction = 10 / unit.picosecond
        self.timestep = 2 * unit.femtosecond

        # --- 2. Build System & Force ---
        self.system = mm.System()
        self.force = ThreeHoleForce()

        # Create dummy topology
        self.topology = app.Topology()
        chain = self.topology.addChain()
        residue = self.topology.addResidue("LIG", chain)

        # Add particles to System, Force, and Topology
        for i in range(self.nParticles):
            self.system.addParticle(self.mass)
            self.force.addParticle(i, [])
            self.topology.addAtom(f"C{i}", app.Element.getBySymbol("C"), residue)

        self.system.addForce(self.force)

        # --- 3. Integrator ---
        self.integrator = mm.LangevinIntegrator(
            self.temperature, self.friction, self.timestep
        )

        # --- 4. Platform & Context ---
        try:
            self.platform = mm.Platform.getPlatformByName(platform)
            if platform.upper() == "CUDA":
                self.properties = {"Precision": "mixed", "DeviceIndex": str(device)}
            else:
                self.properties = {}
        except Exception:
            self.platform = mm.Platform.getPlatformByName("CPU")
            self.properties = {}

        # self.simulation = app.Simulation(self.topology, self.system, integrator, platform, properties)

        # Define an initial position array where all particles start in Basin A (x < -0.5)
        # Target start: x in [-2, -0.5], y in [-1, 3], z in [0, 1]
        # starting_pos = (np.random.rand(self.nParticles, 3) * np.array([1.5, 4.0, 1.0])) + np.array([-2.0, -1.0, 0.0])

        # self.simulation.context.setPositions(starting_pos)
        # self.simulation.context.setVelocitiesToTemperature(self.temperature)

    def _create_simulation(self, seed=0):
        """Creates the main OpenMM Simulation object and sets its initial state."""
        new_integrator = copy.copy(self.integrator)
        new_integrator.setRandomNumberSeed(seed)
        self.simulation = app.Simulation(
            self.topology, self.system, new_integrator, self.platform, self.properties
        )
