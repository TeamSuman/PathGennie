import os
import logging
from collections import deque
from typing import Callable, Optional, List, Dict, Any, Tuple

import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.XTC import XTCWriter
from openmm import OpenMMException, unit
from openmm.app import Simulation
from tqdm.auto import trange

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Steer:
    """
    A class to steer a molecular dynamics simulation towards a target collective variable (CV) value.

    This class uses an iterative "probe-and-accept" algorithm. In each cycle, it runs
    a short simulation, checks if the system has moved closer to the target CV,
    and accepts the new state if it has improved or if a maximum number of retries
    is reached.
    """
    NM_TO_ANGSTROM = 10.0

    def __init__(
        self,
        simulation_obj: Simulation,
        topology_file_path: str,
        projection_fn: Callable[..., np.ndarray],
        target_projection: np.ndarray,
        projection_args: Optional[Dict[str, Any]] = None,
        temperature: float = 300.0,
    ):
        """
        Initializes the Steer object.

        Args:
            simulation_obj: An initialized OpenMM Simulation object.
            topology_file_path: Path to the topology file (e.g., .pdb, .gro) for MDAnalysis.
            projection_fn: A callable function that takes atomic positions (in Angstroms)
                           and returns a projection value (CV).
            target_projection: The target value for the projection (CV).
            projection_args: Optional dictionary of keyword arguments for the projection_fn.
            temperature: The temperature for velocity re-assignment (in Kelvin).

        Raises:
            FileNotFoundError: If the topology_file_path does not exist.
            ValueError: If essential arguments like projection_fn or target_projection are not provided.
        """
        if not isinstance(simulation_obj, Simulation):
            raise TypeError("simulation_obj must be an OpenMM Simulation instance.")
        if not callable(projection_fn):
            raise ValueError("A callable projection_fn must be provided.")
        if target_projection is None:
            raise ValueError("A target_projection must be provided.")

        self.simulation = simulation_obj
        self.projection_fn = projection_fn
        self.target_projection = np.asarray(target_projection)
        self.projection_args = projection_args or {}
        self.temperature_k = temperature * unit.kelvin

        try:
            self.mda_universe = mda.Universe(topology_file_path)
        except FileNotFoundError:
            logging.error(f"Topology file not found at: {topology_file_path}")
            raise

    def _initialize_simulation_state(self, positions: np.ndarray):
        """Sets initial positions, minimizes energy, and assigns velocities."""
        self.simulation.context.setPositions(positions)
        self.simulation.minimizeEnergy()
        self.simulation.context.setVelocitiesToTemperature(self.temperature_k)
        self.simulation.step(10) # A few steps to equilibrate velocities

    def _get_current_projection(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gets current positions and computes their projection."""
        state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        current_positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        # Convert to Angstrom for the projection function, as is common
        positions_angstrom = self.NM_TO_ANGSTROM * current_positions
        current_projection = self.projection_fn(positions_angstrom, **self.projection_args)
        return np.asarray(current_projection), current_positions

    def _update_adaptive_relaxation(
        self,
        gradient_history: deque,
        gradient_window: int,
        gradient_threshold: float,
        relax1: float,
        relax2: float,
        decay_rate: float
    ) -> Tuple[float, float]:
        """Adjusts relaxation steps based on convergence gradient."""
        if len(gradient_history) == gradient_window:
            # Fit a line to the recent history to get the slope (gradient)
            gradient = np.polyfit(np.arange(gradient_window), list(gradient_history), 1)[0]
            # If convergence has stalled, adjust relaxation times
            if abs(gradient) < gradient_threshold and relax2 > 0:
                decay = min(relax2, decay_rate) # Don't let relax2 go below zero
                relax2 -= decay
                relax1 += decay
                logging.debug(f"Convergence stalled. Adjusting relaxation: relax1={relax1:.2f}, relax2={relax2:.2f}")
        return relax1, relax2

    def _move_(
        self,
        initial_positions: np.ndarray,
        output_file: str = 'trajectory.xtc',
        relax1_steps: int = 1,
        relax2_steps: int = 5,
        save_frequency: Optional[int] = 10,
        max_probes_per_cycle: int = 200,
        max_cycles: int = 50000,
        convergence_threshold: float = 0.1,
        min_cycles_for_convergence: int = 1000,
        gradient_window: int = 10,
        gradient_threshold: float = 1e-5,
        adaptive_decay_rate: float = 0.01,
        verbose: bool = False
    ) -> Tuple[List[int], List[float]]:
        """
        Runs the steering simulation.

        Args:
            initial_positions: Starting positions for the simulation (in nm, as OpenMM array).
            output_file: Path to save the output trajectory.
            relax1_steps: Number of "probe" steps in each attempt.
            relax2_steps: Number of "commit" steps after a successful probe.
            save_frequency: Save frame every N cycles. If None, saves only the last frame.
            max_probes_per_cycle: Maximum number of attempts to find a better state per cycle.
            max_cycles: Total number of steering cycles to perform.
            convergence_threshold: The distance to target at which the simulation is considered converged.
            min_cycles_for_convergence: Minimum number of cycles before checking for convergence.
            gradient_window: Number of steps to consider for calculating convergence gradient.
            gradient_threshold: The gradient below which convergence is considered stalled.
            adaptive_decay_rate: Rate at which to adjust relaxation steps when stalled.
            verbose: If True, log detailed progress.

        Returns:
            A tuple containing:
            - A list of probe counts for each cycle.
            - A list of the distance to target at the end of each cycle.
        """
        probe_counts_history = []
        convergence_history = []
        gradient_history = deque(maxlen=gradient_window)
        relax1, relax2 = float(relax1_steps), float(relax2_steps)

        # Ensure the writer is properly closed on exit or error
        try:
            with mda.Writer(output_file, self.mda_universe.atoms.n_atoms) as writer:
                # Initialize simulation state
                self._initialize_simulation_state(initial_positions)

                # Get starting projection and distance
                previous_projection, positions_nm = self._get_current_projection()
                min_distance = np.linalg.norm(previous_projection - self.target_projection)
                logging.info(f"Starting steering. Initial distance to target: {min_distance:.4f}")

                for cycle in trange(max_cycles, desc="Steering Cycles"):
                    probe_count = 0
                    found_better_state = False

                    while probe_count < max_probes_per_cycle:
                        probe_count += 1
                        # Set starting point for this probe
                        self.simulation.context.setPositions(positions_nm)
                        self.simulation.context.setVelocitiesToTemperature(self.temperature_k)

                        try:
                            # 1. Probe step
                            self.simulation.step(int(relax1))
                            current_projection, _ = self._get_current_projection()
                            distance_to_target = np.linalg.norm(current_projection - self.target_projection)
                        except OpenMMException as e:
                            logging.warning(f"Simulation became unstable at cycle {cycle}, probe {probe_count}. Error: {e}")
                            # Reset to last good state and try again or break
                            continue

                        # Check if this probe brought us closer
                        if distance_to_target < min_distance:
                            found_better_state = True
                            break

                    # 2. Commit step (if better state found or max probes reached)
                    self.simulation.step(int(relax2))

                    current_projection, positions_nm = self._get_current_projection()
                    min_distance = np.linalg.norm(current_projection - self.target_projection)

                    # Record history and update adaptive parameters
                    probe_counts_history.append(probe_count)
                    convergence_history.append(min_distance)
                    gradient_history.append(min_distance)
                    relax1, relax2 = self._update_adaptive_relaxation(
                        gradient_history, gradient_window, gradient_threshold, relax1, relax2, adaptive_decay_rate
                    )

                    # Save trajectory frame
                    if save_frequency and (cycle % save_frequency == 0):
                        self.mda_universe.atoms.positions = self.NM_TO_ANGSTROM * positions_nm
                        writer.write(self.mda_universe.atoms)

                    if verbose and cycle % 100 == 0:
                        logging.info(f"Cycle: {cycle}, CV: {current_projection}, Dist: {min_distance:.4f}, Probes: {probe_count}")

                    # Check for convergence
                    if cycle > min_cycles_for_convergence and min_distance < convergence_threshold:
                        logging.info(f"Convergence criteria met at cycle {cycle}. Final distance: {min_distance:.4f}")
                        break

                # Save the final frame
                self.mda_universe.atoms.positions = self.NM_TO_ANGSTROM * positions_nm
                writer.write(self.mda_universe.atoms)

        except Exception as e:
            logging.error(f"An unexpected error occurred during the steering simulation: {e}")
            raise # Re-raise the exception after logging

        logging.info("Steering simulation finished.")
        return probe_counts_history, convergence_history


    def move(
        self,
        initial_positions: np.ndarray,
        output_file: str = 'trajectory.xtc',
        relax1_steps: int = 1,
        relax2_steps: int = 5,
        save_frequency: Optional[int] = 10,
        max_probes_per_cycle: int = 200,
        max_cycles: int = 50000,
        convergence_threshold: float = 0.1,
        stalled_patience: int = 10000,
        stalled_tolerance: float = 1e-3,
        verbose: bool = False
    ) -> Tuple[List[int], List[float]]:
        """
        Runs the steering simulation.
        """
        probe_counts_history = []
        convergence_history = []

        stalled_cycles_counter = 0

        try:
            with mda.Writer(output_file, self.mda_universe.atoms.n_atoms) as writer:
                self._initialize_simulation_state(initial_positions)

                previous_projection, positions_nm = self._get_current_projection()
                min_distance = np.linalg.norm(previous_projection - self.target_projection)
                best_distance_in_window = min_distance
                logging.info(f"Starting steering. Initial distance to target: {min_distance:.4f}")

                for cycle in trange(max_cycles, desc="Steering Cycles"):
                    probe_count = 0

                    while probe_count < max_probes_per_cycle:
                        probe_count += 1
                        self.simulation.context.setPositions(positions_nm)
                        self.simulation.context.setVelocitiesToTemperature(self.temperature_k)

                        try:
                            self.simulation.step(int(relax1_steps))
                            current_projection, _ = self._get_current_projection()
                            distance_to_target = np.linalg.norm(current_projection - self.target_projection)
                        except OpenMMException as e:
                            logging.warning(f"Simulation unstable at cycle {cycle}, probe {probe_count}. Error: {e}")
                            continue

                        if distance_to_target < min_distance:
                            break

                    self.simulation.step(int(relax2_steps))

                    current_projection, positions_nm = self._get_current_projection()
                    min_distance = np.linalg.norm(current_projection - self.target_projection)

                    probe_counts_history.append(probe_count)
                    convergence_history.append(min_distance)

                    if (best_distance_in_window - min_distance) > stalled_tolerance:
                        best_distance_in_window = min_distance
                        stalled_cycles_counter = 0
                    else:
                        stalled_cycles_counter += 1

                    if save_frequency and (cycle % save_frequency == 0):
                        self.mda_universe.atoms.positions = self.NM_TO_ANGSTROM * positions_nm
                        writer.write(self.mda_universe.atoms)

                    if verbose and cycle % 100 == 0:
                        logging.info(f"Cycle: {cycle}, CV: {current_projection}, Dist: {min_distance:.4f}, Probes: {probe_count}")

                    if min_distance < convergence_threshold:
                        logging.info(f"Convergence criteria met at cycle {cycle}. Final distance: {min_distance:.4f}")
                        break

                    if stalled_cycles_counter >= stalled_patience:
                        logging.info(f"Simulation stalled for {stalled_patience} cycles. No improvement > {stalled_tolerance:.1e}. Stopping.")
                        break

                self.mda_universe.atoms.positions = self.NM_TO_ANGSTROM * positions_nm
                writer.write(self.mda_universe.atoms)

        except Exception as e:
            logging.error(f"An unexpected error occurred during the steering simulation: {e}")
            raise

        logging.info("Steering simulation finished.")
        #eturn probe_counts_history, convergence_history
