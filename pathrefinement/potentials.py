import warnings
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit


class Potential2D(ABC):
    """Abstract base class for 2D external potentials."""

    def __init__(self, energy_scale: float = 1.0):
        self.energy_scale = energy_scale

    @abstractmethod
    def energy(self, xy: np.ndarray) -> np.ndarray:
        """Calculate the potential energy at given points."""
        pass

    @abstractmethod
    def create_force(self) -> mm.Force:
        """Create the corresponding OpenMM CustomExternalForce."""
        pass

    @property
    @abstractmethod
    def minima(self) -> Dict[str, np.ndarray]:
        """Return a dictionary of named minima coordinates."""
        pass

    def energy_surface(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Evaluate energy over a 2D grid for contour plotting."""
        shape = X.shape
        xy = np.column_stack((X.ravel(), Y.ravel()))
        Z = self.energy(xy).reshape(shape)
        return Z

    def gradient(self, xy: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """Numerical gradient using central difference."""
        xy = np.asarray(xy, dtype=float)
        grad = np.zeros_like(xy)
        for i in range(xy.shape[1]):
            xy_plus = xy.copy()
            xy_minus = xy.copy()
            xy_plus[:, i] += h
            xy_minus[:, i] -= h
            grad[:, i] = (self.energy(xy_plus) - self.energy(xy_minus)) / (2 * h)
        return grad

    def make_bad_initial_path(
        self, start_name: str, end_name: str, n_images: int = 30, noise: float = 0.2
    ) -> np.ndarray:
        """Generate a linear interpolation path with optional noise."""
        if start_name not in self.minima or end_name not in self.minima:
            raise ValueError(f"Minima {start_name} or {end_name} not found.")

        start = self.minima[start_name]
        end = self.minima[end_name]

        # Linear interpolation
        t = np.linspace(0, 1, n_images)[:, np.newaxis]
        path = start * (1 - t) + end * t

        # Add noise
        if noise > 0:
            np.random.seed(42) # Reproducible noise
            path += np.random.normal(scale=noise, size=path.shape)
            # Pin endpoints
            path[0] = start
            path[-1] = end

        return path

    def create_simulation(
        self,
        temperature: float = 300.0,
        timestep: float = 1.0,
        friction: float = 10.0,
        mass: float = 12.0,
        seed: int = 0,
        device: int = 0,
    ) -> app.Simulation:
        """Create a generic single-particle OpenMM simulation in 2D."""
        temperature = temperature * unit.kelvin
        timestep = timestep * unit.femtosecond
        friction = friction / unit.picosecond
        mass = mass * unit.dalton

        system = mm.System()
        system.addParticle(mass)
        system.addForce(self.create_force())

        # Constrain Z to zero
        z_force = mm.CustomExternalForce("kz * z^2")
        z_force.addGlobalParameter("kz", 50000.0)
        z_force.addParticle(0, [])
        system.addForce(z_force)

        topology = app.Topology()
        chain = topology.addChain()
        res = topology.addResidue("X", chain)
        topology.addAtom("X", app.Element.getBySymbol("C"), res)

        integrator = mm.LangevinMiddleIntegrator(temperature, friction, timestep)
        integrator.setRandomNumberSeed(seed)

        try:
            platform = mm.Platform.getPlatformByName("CUDA")
            props = {"Precision": "mixed", "DeviceIndex": str(device)}
        except Exception:
            platform = mm.Platform.getPlatformByName("CPU")
            props = {}

        simulation = app.Simulation(topology, system, integrator, platform, props)
        return simulation


class MullerBrownPotential(Potential2D):
    """Müller-Brown potential."""

    def __init__(self, energy_scale: float = 0.1):
        super().__init__(energy_scale)
        self.params = {
            "A": [-200, -100, -170, 15],
            "a": [-1, -1, -6.5, 0.7],
            "b": [0, 0, 11, 0.6],
            "c": [-10, -10, -6.5, 0.7],
            "x0": [1, 0, -0.5, -1],
            "y0": [0, 0.5, 1.5, 1],
        }

    @property
    def minima(self) -> Dict[str, np.ndarray]:
        return {
            "A": np.array([-0.558224, 1.441726]),
            "B": np.array([0.623499, 0.028038]),
            "C": np.array([-0.050, 0.467]),
            "SADDLE_1": np.array([-0.822, 0.624]), # Saddle between A and B
            "SADDLE_2": np.array([0.212, 0.293]),  # Saddle between B and C
        }

    def energy(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(xy)
        x = xy[:, 0]
        y = xy[:, 1]
        V = np.zeros_like(x, dtype=float)
        for i in range(4):
            V += self.params["A"][i] * np.exp(
                self.params["a"][i] * (x - self.params["x0"][i]) ** 2
                + self.params["b"][i] * (x - self.params["x0"][i]) * (y - self.params["y0"][i])
                + self.params["c"][i] * (y - self.params["y0"][i]) ** 2
            )
        return V * self.energy_scale

    def create_force(self) -> mm.CustomExternalForce:
        expr = "scale * (" + " + ".join([
            f"A{i+1}*exp(a{i+1}*(x-x{i+1})^2 + b{i+1}*(x-x{i+1})*(y-y{i+1}) + c{i+1}*(y-y{i+1})^2)"
            for i in range(4)
        ]) + ")"
        force = mm.CustomExternalForce(expr)
        force.addGlobalParameter("scale", self.energy_scale)
        for i in range(4):
            idx = i + 1
            force.addGlobalParameter(f"A{idx}", self.params["A"][i])
            force.addGlobalParameter(f"a{idx}", self.params["a"][i])
            force.addGlobalParameter(f"b{idx}", self.params["b"][i])
            force.addGlobalParameter(f"c{idx}", self.params["c"][i])
            force.addGlobalParameter(f"x{idx}", self.params["x0"][i])
            force.addGlobalParameter(f"y{idx}", self.params["y0"][i])
        force.addParticle(0, [])
        return force


class ThreeHolePotential(Potential2D):
    """Modified Wolfe-Quapp three-hole potential."""

    def __init__(self, energy_scale: float = 1.0):
        super().__init__(energy_scale)

    @property
    def minima(self) -> Dict[str, np.ndarray]:
        # Approximate minima locations based on the function
        # A quick minimization would give exact, but these are close enough for endpoints
        return {
            "A": np.array([-1.1, 1.1]),
            "B": np.array([-1.1, -1.1]),
            "C": np.array([1.1, -1.1]),
            "SADDLE_1": np.array([-1.0, 0.0]), # Approx saddle A-B
            "SADDLE_2": np.array([0.0, -1.0]), # Approx saddle B-C
            "SADDLE_3": np.array([0.0, 0.0]),  # Approx central high region
        }

    def energy(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(xy)
        x = xy[:, 0]
        y = xy[:, 1]
        V = 2.0 * (x**4 + y**4 - 2 * x**2 - 4 * y**2 + 2 * x * y + 0.8 * x + 0.1 * y + 9.28)
        return V * self.energy_scale

    def create_force(self) -> mm.CustomExternalForce:
        expr = "scale * 2.0 * (x^4 + y^4 - 2*x^2 - 4*y^2 + 2*x*y + 0.8*x + 0.1*y + 9.28)"
        force = mm.CustomExternalForce(expr)
        force.addGlobalParameter("scale", self.energy_scale)
        force.addParticle(0, [])
        return force
