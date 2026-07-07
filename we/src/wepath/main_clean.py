import numpy as np
import pandas as pd
import gc
import os
import time
from openmm.unit import kelvin
from .ligand_resampler import LigandResampler
#from resampler import Resampler
from .bin_methods import RegularBin
from .base import WeightedEnsembleBase
from .util import save_json_on_the_fly

class WESS(WeightedEnsembleBase):
    def __init__(self, config, initial_positions, projection_fn, kwargs):


        """
        Weighted Ensemble Steady-State (WESS) simulation driver.
        ...
        """
        # Core simulation settings

        self.config = config
        self.initial_positions = initial_positions
        self.projection_fn = projection_fn
        self.kwargs = kwargs
        self.n_walkers_per_bin = self.config.get('n_walkers_per_bin') # Target walkers per bin
        self.temperature = self.config.get('temperature') * kelvin   # Simulation temperature
        self.n_steps_per_tau = self.config.get('n_steps_per_tau')     # MD steps per WE iteration
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

        #self.resampler = Resampler(self.bins, target_per_bin=self.n_walkers_per_bin)
        self.resampler = LigandResampler(self.bins, target_per_bin = self.n_walkers_per_bin, protein_idx = self.config.get('protein_idx'), ligand_idx = self.config.get('ligand_idx'))
        # Identify the "source" bin
        self.source_bin_indices = self.config.get("source_bin_indices")
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

    def run(self):
        """Executes the entire WESS simulation workflow with incremental disk saving."""
        try:

            start_time = time.perf_counter()
            TAU = self.dt * self.n_steps_per_tau
            self.resampling_history = {}
            self.warping_history = {}
            self.initialize_walkers()
            warp_ids = self._handle_warping()
            self.walkers = [w for j, w in enumerate(self.walkers) if j not in warp_ids]
            self.walkers, _ = self.resampler.resample(self.walkers)

            if self.walkers: # Check if walkers exist before normalizing
                 for walker in self.walkers:
                     walker.weight = 1/len(self.walkers)
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

                self.walkers = self._propagate_walkers(task_queues, result_queue)
                #self._save_h5_data(i)
                warp_ids = self._handle_warping()
                reject_ids = self._handle_cleaning()
                self._log_flux_data(i, warp_ids, file_handlers)

                reject_ids_only = reject_ids - warp_ids
                rejected_weights = [
                    w.weight for j, w in enumerate(self.walkers) if j in reject_ids_only
                ]
                terminated_weights = [
                    w.weight for j, w in enumerate(self.walkers) if j in warp_ids
                ]

                all_pruned = reject_ids | warp_ids
                all_weights_to_recycle = rejected_weights + terminated_weights

                flux_this_iter = np.sum(terminated_weights)
                self.prev_walkers = self.walkers

                self.walkers, map_parents = self.resampler.resample(self.walkers, all_pruned)
                #self.resampling_history[i] = map_parents
                #save_json_on_the_fly(self.resampling_history[i], "resampling.json")
                #save_json_on_the_fly(self.warping_history[i], "warping.json")
                self._recycle_weights(all_weights_to_recycle)
                self._log_iteration_data(i, flux_this_iter, file_handlers)

                for ids in warp_ids:
                    print(f"warped: {ids}, weight: {self.prev_walkers[ids].weight}")

                print("*"*50)
                gc.collect()

        finally:
            if self.traj_file:
                self.traj_file.close()
            if 'workers' in locals():
                self._cleanup_workers(task_queues, workers)
            if 'file_handlers' in locals():
                for f in file_handlers.values():
                    f.close()
