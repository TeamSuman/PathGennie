import os
import time
import logging
from collections import deque
from typing import Callable, Optional, Dict, Any, Tuple

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

    def _get_projection(self, positions_nm: np.ndarray) -> np.ndarray:
        """Convert positions to Å and evaluate the projection function."""
        positions_angstrom = self.NM_TO_ANGSTROM * positions_nm
        return np.asarray(self.projection_fn(positions_angstrom, **self.projection_args))

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
    def _move_(
        self,
        positions,
        relax1=5,
        relax2=20,
        save=False,
        save_frequency=None,
        max_try=200,
        output_file='trajectory.xtc',
        verbose=False,
        thresh_cutoff=0.1,
        max_cycle=50000
    ):
        self.writer = mda.Writer(output_file, self.mda_universe.atoms.n_atoms)
        projection_steps = []
        divergence_history = []

        previous_projection = self.projection_fn(10 * positions, **self.projection_args)
        #initial_distance = np.linalg.norm(previous_projection - self.target_projection) if self.target_projection is not None else 0.0

        # Initialize simulation
        self.simulation.context.setPositions(positions)
        self.simulation.minimizeEnergy()
        self.simulation.context.setVelocitiesToTemperature(self.temperature_k)
        self.simulation.step(10000)

        state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        positions = np.array(state.getPositions(asNumpy=True), dtype=np.float32)

        start_projection = self.projection_fn(10 * positions, **self.projection_args)
        previous_projection = start_projection.copy()
        max_distance = 0.0
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
                current_projection = self.projection_fn(10 * current_positions, **self.projection_args)

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
                    best_projection = self.projection_fn(10 * best_positions, **self.projection_args)
                    best_distance = np.linalg.norm(best_projection - start_projection)
                    break  # accept new better state immediately

            max_distance = max(max_distance, best_distance)
            positions = best_positions
            projection_steps.append(retry_count)
            divergence_history.append(max_distance)
            previous_projection = best_projection

            if save and ((save_frequency is None) or (cycle % save_frequency == 0)):
                self.mda_universe.atoms.positions = 10 * positions
                self.writer.write(self.mda_universe.atoms)

            if verbose and cycle % 50 == 0:
                print(f"Cycle : {cycle}, Current Projection: {best_projection}")

            if cycle > 100 and best_projection[1] < thresh_cutoff:
                print(best_projection)
                break

        self.writer.close()
    def move(
        self,
        positions,
        relax1=5,
        relax2=20,
        save=False,
        save_frequency=None,
        max_try=200,
        output_file='trajectory.xtc',
        verbose=False,
        thresh_cutoff=0.1,
        max_cycle=50000,
        stagnation_window=1000,
        no_change_threshold=1e-2,
        force_unfold_cycles=1000
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

        state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        positions = np.array(state.getPositions(asNumpy=True), dtype=np.float32)

        start_projection = self.projection_fn(10 * positions, **self.projection_args)
        start_rmsd, start_q = start_projection
        previous_projection = np.copy(start_projection)
        max_distance = 0.0
        best_score = -np.inf  # for unfolding

        for cycle in trange(max_cycle):
            retry_count = 0
            best_projection = None
            best_positions = None
            best_distance = -np.inf
            best_score_in_try = -np.inf

            force_unfold = cycle < force_unfold_cycles

            while retry_count < max_try:
                self.simulation.context.setPositions(positions)
                self.simulation.context.setVelocitiesToTemperature(self.temperature_k)
                self.simulation.step(relax1)

                state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
                current_positions = np.array(state.getPositions(asNumpy=True), dtype=np.float32)
                current_projection = self.projection_fn(10 * current_positions, **self.projection_args)
                rmsd, q = current_projection

                if force_unfold:
                    score = (rmsd - start_rmsd) - (q - start_q)
                    if score > best_score_in_try:
                        best_score_in_try = score
                        best_projection = current_projection
                        best_positions = current_positions
                else:
                    distance_from_start = np.linalg.norm(current_projection - start_projection)
                    if distance_from_start > best_distance:
                        best_distance = distance_from_start
                        best_projection = current_projection
                        best_positions = current_positions

                retry_count += 1

                # Accept improvement immediately
                if force_unfold:
                    if best_score_in_try > best_score:
                        self.simulation.context.setPositions(best_positions)
                        self.simulation.step(relax2)
                        state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
                        best_positions = np.array(state.getPositions(asNumpy=True), dtype=np.float32)
                        best_projection = self.projection_fn(10 * best_positions, **self.projection_args)
                        best_score = best_score_in_try
                        break
                else:
                    if best_distance > max_distance:
                        self.simulation.context.setPositions(best_positions)
                        self.simulation.step(relax2)
                        state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
                        best_positions = np.array(state.getPositions(asNumpy=True), dtype=np.float32)
                        best_projection = self.projection_fn(10 * best_positions, **self.projection_args)
                        best_distance = np.linalg.norm(best_projection - start_projection)
                        break

            # Update
            positions = best_positions
            projection_steps.append(retry_count)
            previous_projection = best_projection
            if force_unfold:
                proj_scalar = best_score_in_try
            else:
                proj_scalar = best_distance
                max_distance = max(max_distance, best_distance)

            divergence_history.append(proj_scalar)

            # --- Stagnation Detection ---
            if best_projection is not None:
                projection_scalar = (
                    best_score_in_try if force_unfold else (
                        best_projection if np.isscalar(best_projection) else np.linalg.norm(best_projection)
                    )
                )
                projection_history.append(projection_scalar)

                if len(projection_history) == stagnation_window:
                    delta = np.abs(np.diff(list(projection_history)))
                    if np.all(delta < no_change_threshold):
                        if verbose:
                            print(f"Early stopping at cycle {cycle}: stagnation detected.")
                        break

            # Save trajectory frame
            if save and best_positions is not None and ((save_frequency is None) or (cycle % save_frequency == 0)):
                self.mda_universe.atoms.positions = 10 * best_positions
                self.writer.write(self.mda_universe.atoms)

            if verbose and cycle % 50 == 0 and best_projection is not None:
                mode = "Unfolding" if force_unfold else "Exploring"
                print(f"Cycle {cycle:5d}, Projection: {best_projection}")

            if cycle > 100 and best_projection[1] < thresh_cutoff:
                print(best_projection)
                break

        self.writer.close()
