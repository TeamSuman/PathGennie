# import warnings
# warnings.filterwarnings('ignore')

# import os
# import numpy as np
# import joblib
# import MDAnalysis as mda
# from tqdm.auto import trange
# from openmm import *
# from openmm import LangevinMiddleIntegrator
# from openmm import Platform
# from openmm.app import GromacsGroFile, GromacsTopFile, HBonds, PME, Simulation
# from openmm.unit import picosecond, nanometers, femtoseconds, kelvin, amu

# class Unbind:
#     def __init__(self, crd_file, top_file, projection=None, kwargs=None, temperature=300):

#         self.temperature = temperature
#         self.mda_u = mda.Universe(top_file, crd_file)
#         self.projection = projection
#         self.kwargs = kwargs

#         sim = SimObj

#         self.simulation = sim.simulation
#         self.simulation.context.reinitialize()

#     def move(self, positions, relax1=5, relax2=20, save=False, save_frequency=None,
#              max_try=200, out='trajectory.xtc', adapt_interval=50,
#              , max_cycle=50000):


#         self.writer = mda.Writer(out, self.mda_u.atoms.n_atoms)
#         weight = []
#         distance_list = []

#         init_distance = 0.0
#         self.simulation.context.setPositions(positions)
#         self.simulation.minimizeEnergy()
#         self.simulation.step(50000)
#         state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox = True)
#         positions = np.array(state.getPositions(asNumpy = True), dtype=np.float32)
#         old_proj, _, _ = self.projection(10 * positions, **self.kwargs)
#         #print(pos)
#         for i in range(0, 50000):
#             count = 0
#             while True:
#                 self.simulation.context.setPositions(positions)
#                 self.simulation.context.setVelocitiesToTemperature(self.temperature * kelvin)
#                 self.simulation.step(relax1)
#                 state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox = True)
#                 pos = np.array(state.getPositions(asNumpy = True), dtype=np.float32)
#                 #print(pos)
#                 new_proj, new_dist, min_dist = self.projection(10 * pos, **self.kwargs)
#                 current_distance = np.linalg.norm(new_proj - old_proj)
#                 count += 1

#                 if (current_distance > init_distance) or (count >= max_try):
#                     self.simulation.step(relax2)
#                     state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox = True)
#                     pos = np.array(state.getPositions(asNumpy = True), dtype=np.float32)
#                     new_proj, new_dist, min_dist = self.projection(10 * pos, **self.kwargs)
#                     current_distance = np.linalg.norm(new_proj - old_proj)
#                     init_distance = current_distance
#                     positions = pos
#                     weight.append(count)
#                     distance_list.append(init_distance)
#                     new_proj = old_proj
#                     break
#             if save:
#                 if i % save_frequency == 0:
#                     self.mda_u.atoms.positions = 10 * positions
#                     self.writer.write(self.mda_u.atoms)

#             if i % 10 == 0:
#                 if new_dist > 30.0:
#                     if min_dist > 10.0:
#                         break

#         self.writer.close()
#         return weight, distance_list

import os
import logging
from collections import deque
from typing import Callable, Optional, List, Dict, Any, Tuple

import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCWriter
from openmm import unit
from openmm.app import Simulation
from tqdm.auto import trange

# Logger config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Steer:
    """
    Steers a molecular dynamics simulation by maximizing the distance from the initial projection.

    Uses a probe-and-accept algorithm: performs short simulations,
    evaluates a projection function (CV), and accepts new configurations
    if they diverge from the start.
    """
    NM_TO_ANGSTROM = 10.0

    def __init__(
        self,
        simulation_obj: Simulation,
        topology_file_path: str,
        projection_fn: Callable[..., np.ndarray],
        projection_args: Optional[Dict[str, Any]] = None,
        temperature: float = 300.0
    ):
        if not isinstance(simulation_obj, Simulation):
            raise TypeError("simulation_obj must be an OpenMM Simulation instance.")
        if not callable(projection_fn):
            raise ValueError("A callable projection_fn must be provided.")

        if not os.path.exists(topology_file_path):
            logging.error(f"Topology file not found: {topology_file_path}")
            raise FileNotFoundError(topology_file_path)

        self.simulation = simulation_obj
        self.projection_fn = projection_fn
        self.projection_args = projection_args or {}
        self.temperature_k = temperature * unit.kelvin
        self.mda_universe = mda.Universe(topology_file_path)

    def _initialize_simulation_state(self, positions: np.ndarray, steps: int = 10, minimize = False):
        """Minimize energy, assign temperature, and equilibrate velocities."""
        self.simulation.context.setPositions(positions)
        if minimize == True:
            self.simulation.minimizeEnergy()
        self.simulation.context.setVelocitiesToTemperature(self.temperature_k)
        self.simulation.step(steps)

    def _get_projection(self, positions_nm: np.ndarray) -> np.ndarray:
        """Convert positions to Å and evaluate the projection function."""
        positions_angstrom = self.NM_TO_ANGSTROM * positions_nm
        return np.asarray(self.projection_fn(positions_angstrom, **self.projection_args)[0])

    def _update_adaptive_relaxation(
        self,
        gradient_history: deque,
        window: int,
        threshold: float,
        relax1: float,
        relax2: float,
        decay: float
    ) -> Tuple[float, float]:
        if len(gradient_history) == window:
            slope = np.polyfit(np.arange(window), list(gradient_history), 1)[0]
            if abs(slope) < threshold and relax2 > 0:
                delta = min(relax2, decay)
                relax2 -= delta
                relax1 += delta
                logging.debug(f"Adjusting relaxation: relax1={relax1:.2f}, relax2={relax2:.2f}")
        return relax1, relax2

    def move(
        self,
        positions,
        relax1=5,
        relax2=20,
        save=False,
        save_frequency=10,
        max_try=200,
        output_file='trajectory.xtc',
        verbose=False,
        max_cycle=50000,
        stagnation_window=100
    ):
        self.writer = mda.Writer(output_file, self.mda_universe.atoms.n_atoms)
        projection_steps = []
        divergence_history = []
        projection_history = deque(maxlen=stagnation_window)

        previous_projection = self.projection_fn(10 * positions, **self.projection_args)

        # Initialize simulation
        self.simulation.context.setPositions(positions)
        self.simulation.minimizeEnergy()
        self.simulation.context.setVelocitiesToTemperature(self.temperature_k)
        self.simulation.step(10000)

        start_projection, _, _ = self.projection_fn(10 * positions, **self.projection_args)
        max_distance = 0.0

        state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        positions = np.array(state.getPositions(asNumpy=True), dtype=np.float32)

        for cycle in trange(max_cycle):
            retry_count = 0
            best_projection = None
            best_positions = None
            best_distance = -np.inf

            while retry_count < max_try:
                self.simulation.context.setPositions(positions)
                self.simulation.context.setVelocitiesToTemperature(self.temperature_k)
                self.simulation.step(relax1)

                state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
                current_positions = np.array(state.getPositions(asNumpy=True), dtype=np.float32)
                current_projection, _, _ = self.projection_fn(10 * current_positions, **self.projection_args)

                distance_from_start = np.linalg.norm(current_projection - start_projection)

                if distance_from_start > best_distance:
                    best_distance = distance_from_start
                    best_projection = current_projection
                    best_positions = current_positions

                retry_count += 1

                # If a new max is found, step longer to stabilize and accept the frame
                if best_distance > max_distance:
                    self.simulation.step(relax2)

                    state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
                    best_positions = np.array(state.getPositions(asNumpy=True), dtype=np.float32)
                    best_projection, com_dist, min_dist = self.projection_fn(10 * best_positions, **self.projection_args)
                    best_distance = np.linalg.norm(best_projection - start_projection)
                    break  # Accept new better state immediately

            max_distance = max(max_distance, best_distance)
            positions = best_positions
            projection_steps.append(retry_count)
            divergence_history.append(max_distance)
            previous_projection = best_projection

            # Track projection convergence
            proj_scalar = best_projection if np.isscalar(best_projection) else np.linalg.norm(best_projection)
            projection_history.append(proj_scalar)

            # Check for stagnation over recent steps
            # if len(projection_history) == stagnation_window:
            #     delta = np.abs(np.diff(projection_history))
            #     if np.all(delta < no_change_threshold):
            #         if verbose:
            #             print(f"Early stopping at cycle {cycle}: projection change < {no_change_threshold} for {stagnation_window} steps.")
            #         break

            # Save trajectory frame
            if save and ((save_frequency is None) or (cycle % save_frequency == 0)):
                self.mda_universe.atoms.positions = 10 * positions
                self.writer.write(self.mda_universe.atoms)

            if verbose and cycle % 50 == 0:
                print(f"Cycle {cycle:5d}: Current distance: {com_dist}")

            # Optional early stop condition
            if cycle > 100:
                if com_dist > 30.0:
                    if min_dist > 10.0:
                        break
            #    break

        self.writer.close()
        #return projection_steps, divergence_history
