"""
common.py — shared helpers for the Muller-Brown path refinement example.
"""

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit


# ── Muller-Brown potential parameters ────────────────────────────────────────
_MB_PARAMS = {
    "A":  [-200, -100, -170,  15],
    "a":  [  -1,   -1, -6.5, 0.7],
    "b":  [   0,    0,  11,  0.6],
    "c":  [ -10,  -10, -6.5, 0.7],
    "x0": [   1,    0, -0.5,  -1],
    "y0": [   0,  0.5,  1.5,   1],
}

# Exact global minima (user-provided)
MINIMA = {
    "A": np.array([-0.558224, 1.441726]),   # deeper minimum (start)
    "B": np.array([ 0.623499, 0.028038]),   # shallow minimum (end)
}

ENERGY_SCALE = 1.0


def muller_brown_energy(xy: np.ndarray) -> np.ndarray:
    """Evaluate the Muller-Brown potential at (N, 2) points."""
    xy = np.atleast_2d(xy)
    x, y = xy[:, 0], xy[:, 1]
    V = np.zeros(len(x))
    for i in range(4):
        V += _MB_PARAMS["A"][i] * np.exp(
            _MB_PARAMS["a"][i] * (x - _MB_PARAMS["x0"][i]) ** 2
            + _MB_PARAMS["b"][i] * (x - _MB_PARAMS["x0"][i]) * (y - _MB_PARAMS["y0"][i])
            + _MB_PARAMS["c"][i] * (y - _MB_PARAMS["y0"][i]) ** 2
        )
    return V * ENERGY_SCALE


def create_mb_system(seed: int = 0, device: int = 0) -> app.Simulation:
    """
    Create an independent single-particle OpenMM simulation for the
    Muller-Brown potential.  Safe to call inside a worker process.
    """
    temperature = 300.0 * unit.kelvin
    timestep    = 1.0   * unit.femtosecond
    friction    = 10.0  / unit.picosecond
    mass        = 12.0  * unit.dalton

    system = mm.System()
    system.addParticle(mass)

    # Muller-Brown force (custom expression in x, y)
    terms = []
    for i in range(4):
        A  = _MB_PARAMS["A"][i]  * ENERGY_SCALE
        a  = _MB_PARAMS["a"][i]
        b  = _MB_PARAMS["b"][i]
        c  = _MB_PARAMS["c"][i]
        x0 = _MB_PARAMS["x0"][i]
        y0 = _MB_PARAMS["y0"][i]
        terms.append(
            f"{A}*exp({a}*(x-{x0})^2 + {b}*(x-{x0})*(y-{y0}) + {c}*(y-{y0})^2)"
        )
    mb_force = mm.CustomExternalForce(" + ".join(terms))
    mb_force.addParticle(0, [])
    system.addForce(mb_force)

    # Constrain z to zero
    z_force = mm.CustomExternalForce("50000.0 * z^2")
    z_force.addParticle(0, [])
    system.addForce(z_force)

    topology = app.Topology()
    chain = topology.addChain()
    res   = topology.addResidue("X", chain)
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


def feature_fn(coords: np.ndarray) -> np.ndarray:
    """
    Map PathGennieMD coords (N_atoms=1, 3) in Angstroms → 2D (x, y) in
    Muller-Brown units.  Module-level so it is picklable by multiprocessing.
    """
    return coords[0, :2] / 10.0


def make_linear_path(start: np.ndarray, end: np.ndarray, n_images: int = 100) -> np.ndarray:
    """Linear interpolation between two 2D points."""
    t = np.linspace(0.0, 1.0, n_images)
    return start[None, :] + t[:, None] * (end - start)[None, :]
