import numpy as np
import pandas as pd
import gc
import os
import time
from openmm.unit import kelvin
#from ligand_resampler import LigandResampler
from .resampler import Resampler # Assuming your corrected Resampler is in this file
from .bin_methods import RegularBin
from .base import WeightedEnsembleBase

class WESS(WeightedEnsembleBase):
    def __init__(self, config, initial_positions, projection_fn, kwargs):
        """
        Weighted Ensemble Steady-State (WESS) simulation driver.
        ... (rest of your __init__ method) ...
        """
        # Core simulation settings
        self.config = config
        self.initial_positions = initial_positions
        self.projection_fn = projection_fn
        self.kwargs = kwargs

        # FIX: Corrected typo from 'n_walker_per_bin' to 'n_walkers_per_bin'
        self.n_walkers_per_bin = self.config.get('n_walkers_per_bin') # Target walkers per bin
        self.temperature = self.config.get('temperature') * kelvin    # Simulation temperature
        self.n_steps_per_tau = self.config.get('n_steps_per_tau')      # MD steps per WE iteration
        self.n_iterations = self.config.get('n_iterations', 10)      # Number of WE iterations
        self.dt = self.config.get('dt', 0.002)                         # Time step (ps)
        self.n_gpus = self.config.get('n_gpus', 1)                     # Number of GPUs to use
        self.survive_empty = self.config.get('survive_empty', True)    # Allow empty bins to survive
        self.warp_function = self.config.get('warp_function')          # Custom warp function (optional)
        self.warp_kwargs = self.config.get('warp_kwargs')              # Arguments for warp function

        # Output file paths
        self.flux_file = self.config.get('flux_file', 'flux.txt')
        self.bin_file = self.config.get('bin_file', 'bin.txt')
        self.walkers_file = self.config.get('walkers_file', 'walkers.txt')
        # Set up binning scheme
        if 'bin_edges' in config and config['bin_edges'] is not None:
            self.bins = RegularBin(bin_edges=config['bin_edges'], wall=True)
        else:
            self.bins = RegularBin(
                min_values=config['bin_min'],
                max_values=config['bin_max'],
                n_bins=config['nbins'],
                wall=True
            )
        self.bins.initialize_bins()
        # Set up resampler
        self.resampler = Resampler(self.bins, target_per_bin=self.n_walkers_per_bin)
        print(self.resampler.N_TARGET_PER_BIN)
        # Identify the "source" bin
        if self.config.get('source_projection_value') is not None:
            self.source_bin_index = self.bins.find_bin(self.config.get('source_projection_value'))
        else:
            self.source_bin_index = self.config.get("source_bin_index")
        # Bookkeeping
        self.n_total_bins = self.bins.all_bins
        self.walkers = []
        self.total_flux_matrix = np.zeros((self.n_total_bins, self.n_total_bins))
        self.total_flux_to_target = 0.0
        self.flux_history = []
        self.all_lineage_maps = []
        self.all_history = []
        # Trajectory storage
        self.h5_file = self.config.get('traj_file', 'trajectory.h5')
        self.h5_atom_indices = self.config.get('h5_atom_indices', None)
        self.traj_file = None


    def _debug_resampling(self, i, prev_walkers, new_walkers, map_parents, warp_ids):
        """
        Provides a detailed report on the resampling step for debugging.
        Checks for weight conservation and prints a summary of events.
        """
        print(f"\n--- [DEBUG] Resampling Report for Iteration {i} ---")
        
        # 1. Check for weight conservation
        # Note: We only sum the weights of walkers that were NOT warped.
        non_warped_indices = [idx for idx in range(len(prev_walkers)) if idx not in warp_ids]
        weight_before = np.sum([prev_walkers[idx].weight for idx in non_warped_indices])
        weight_after = np.sum([w.weight for w in new_walkers])
        
        print(f"Total weight before (non-warped): {weight_before:.10f}")
        print(f"Total weight after resampling:    {weight_after:.10f}")
        if not np.isclose(weight_before, weight_after):
            print("  /!\\ WARNING: Total weight was NOT conserved!")
        else:
            print("  (Success: Total weight was conserved)")

        # 2. Analyze events using the parent map
        # Invert the map to be parent-centric: {parent_idx: [child_indices]}
        parent_to_children = {}
        for child_idx, parent_info in map_parents.items():
            child_idx = int(child_idx)
            if isinstance(parent_info, list): # Merge event
                survivor_idx, loser_idx = parent_info
                # The survivor is the primary parent
                if survivor_idx not in parent_to_children: parent_to_children[survivor_idx] = []
                parent_to_children[survivor_idx].append(child_idx)
                # We can also note the loser for a more detailed report
                if loser_idx not in parent_to_children: parent_to_children[loser_idx] = []
            else: # Split or 1-to-1
                parent_idx = parent_info
                if parent_idx not in parent_to_children: parent_to_children[parent_idx] = []
                parent_to_children[parent_idx].append(child_idx)

        print("\nEvent Summary:")
        for parent_idx, child_indices in sorted(parent_to_children.items()):
            parent_weight = prev_walkers[parent_idx].weight
            child_weights = [new_walkers[c_idx].weight for c_idx in child_indices]
            
            # Identify the event type
            if len(child_indices) > 1:
                # SPLIT Event
                print(f"  - SPLIT: Parent {parent_idx} (w={parent_weight:.6f}) -> "
                      f"Children {child_indices} (total w={np.sum(child_weights):.6f})")
            elif len(child_indices) == 1:
                child_idx = child_indices[0]
                parent_info = map_parents[str(child_idx)]
                if isinstance(parent_info, list):
                    # MERGE Event
                    survivor_idx, loser_idx = parent_info
                    loser_weight = prev_walkers[loser_idx].weight
                    print(f"  - MERGE: Parent {survivor_idx} (w={parent_weight:.6f}) + "
                          f"Parent {loser_idx} (w={loser_weight:.6f}) -> "
                          f"Child {child_idx} (w={child_weights[0]:.6f})")
                else:
                    # 1-to-1 SURVIVAL Event
                    print(f"  - SURVIVE: Parent {parent_idx} (w={parent_weight:.6f}) -> "
                          f"Child {child_idx} (w={child_weights[0]:.6f})")
        
        print("--- [DEBUG] End of Report ---\n")


    def run(self):
        """Executes the entire WESS simulation workflow."""
        # ... (your existing setup code before the loop) ...
        try:
            start_time = time.perf_counter()
            TAU = self.dt * self.n_steps_per_tau
            self.resampling_history = {}
            self.warping_history = {}
            self.initialize_walkers()
            warp_ids = self._handle_warping()
            self.walkers = [w for j, w in enumerate(self.walkers) if j not in warp_ids] # This line is a bug, should be removed
            self.walkers, _ = self.resampler.resample(self.walkers) # Pass warp_ids here

            if self.walkers: # Check if walkers exist before normalizing
                for walker in self.walkers:
                    walker.weight = 1.0/len(self.walkers)
            self._store_backup_source()
            task_queues, result_queue, workers = self._setup_workers()
            file_handlers = {
                'flux_f': open(self.flux_file, 'w'),
                'bin_f': open(self.bin_file, 'w'),
                'walker_f': open(self.walkers_file, 'w')
            }
            n_dims = len(self.bins.n_bins)
            pc_headers = " ".join([f"pc_{j+1}" for j in range(n_dims)])
            walker_header = f"iteration walker_index {pc_headers} weight\n"
            file_handlers['walker_f'].write(walker_header)
            print(f"[WESS] Total init time: {time.perf_counter() - start_time:.2f} seconds")

            for i in range(self.n_iterations):
                if not self.walkers:
                    print(f"Iter {i+1}/{self.n_iterations}: No walkers remaining. Stopping.")
                    break
                
                new_weights = np.sum([w.weight for w in self.walkers])
                print("before_weights: ", new_weights)
                self.walkers = self._propagate_walkers(task_queues, result_queue)
                new_weights = np.sum([w.weight for w in self.walkers])
                print("after_weights: ", new_weights)
                
                #self._save_h5_data(i)

                warp_ids = self._handle_warping()
                self.warping_history[i] = warp_ids
                terminated_weights = [w.weight for j, w in enumerate(self.walkers) if j in warp_ids]
                flux_this_iter = np.sum(terminated_weights)
                
                # Store walkers *before* resampling for the debug function
                self.prev_walkers = self.walkers

                # --- RESAMPLING STEP ---
                self.walkers, map_parents = self.resampler.resample(self.walkers, warp_ids)
                self.resampling_history[i] = map_parents
                # --- INSERT DEBUG CALL HERE ---
                #self._debug_resampling(i, self.prev_walkers, self.walkers, map_parents, warp_ids)

                self._recycle_weights(terminated_weights)
                self._log_iteration_data(i, flux_this_iter, file_handlers)
                
                print("*"*50)
                gc.collect()

        finally:
            # ... (your existing cleanup code) ...
            if self.traj_file:
                self.traj_file.close()
            if 'workers' in locals():
                self._cleanup_workers(task_queues, workers)
            if 'file_handlers' in locals():
                for f in file_handlers.values():
                    f.close()
        return self._calculate_final_results(TAU)
