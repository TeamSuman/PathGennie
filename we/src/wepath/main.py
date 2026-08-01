import gc
import time

import numpy as np
from openmm.unit import kelvin

from .base import WeightedEnsembleBase
from .bin_methods import RegularBin
from .resampler import Resampler


class WESS(WeightedEnsembleBase):
    """
    Weighted Ensemble Steady-State (WESS) simulation driver.
    """

    def __init__(self, config, initial_positions, projection_fn, kwargs):
        """
        Initialize the WESS simulation.
        """
        # Core simulation settings
        self.config = config
        self.initial_positions = initial_positions
        self.projection_fn = projection_fn
        self.kwargs = kwargs

        # Simulation parameters
        self.n_walkers_per_bin = self.config.get('n_walkers_per_bin')  # Target walkers per bin
        self.temperature = self.config.get('temperature') * kelvin    # Simulation temperature
        self.n_steps_per_tau = self.config.get('n_steps_per_tau')     # MD steps per WE iteration
        self.n_iterations = self.config.get('n_iterations', 10)       # Number of WE iterations
        self.dt = self.config.get('dt', 0.002)                        # Time step (ps)
        # tau is the WE iteration length -- the time each walker is propagated
        # between resampling events. Every rate this code reports is a flux
        # DIVIDED BY TAU, so recording it explicitly is what makes the rate
        # reproducible instead of something reconstructed by hand afterwards.
        self.tau = (self.dt * self.n_steps_per_tau
                    if self.n_steps_per_tau is not None else None)
        self.n_gpus = self.config.get('n_gpus', 1)                    # Number of GPUs to use

        # Advanced WE settings
        self.survive_empty = self.config.get('survive_empty', True)   # Allow empty bins to survive
        self.warp_function = self.config.get('warp_function')         # Custom warp function
        self.warp_kwargs = self.config.get('warp_kwargs')             # Arguments for warp function
        self.should_clean = self.config.get('enable_cleaning', False) # Toggle cleaning
        self.clean_threshold = self.config.get("clean_threshold", 50.0)
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

        # Master seed for every stochastic resampling decision. Set it in the
        # config to make a run reproducible; leave it unset for fresh entropy.
        self.seed = self.config.get('seed')
        self._rng = None

        self.resampler = Resampler(
            self.bins, target_per_bin=self.n_walkers_per_bin, seed=self.seed
        )
        #self.resampler = LigandResampler(self.bins, target_per_bin = self.n_walkers_per_bin, protein_idx = self.config.get('protein_idx'), ligand_idx = self.config.get('ligand_idx'))

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
        self.flux_history = []      # per-iteration flux (probability, dimensionless)
        self.rate_history = []      # per-iteration rate (flux / tau, units 1/time)
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
            self.resampling_history = {}
            self.warping_history = {}

            # Initialization
            self.initialize_walkers()

            # Initial warping/resampling
            warp_ids = set()
            if not self.config.get("skip_initial_warping", False):
                warp_ids = self._handle_warping()
            self.walkers = [w for j, w in enumerate(self.walkers) if j not in warp_ids]
            self.walkers, _ = self.resampler.resample(self.walkers)

            if self.walkers:  # Check if walkers exist before normalizing
                for walker in self.walkers:
                    walker.weight = 1.0 / len(self.walkers)
            print(
                f"[WESS] Iteration-zero initialization: {len(self.walkers)} walkers "
                f"in {len(np.unique(self._get_bin_assignments(self.walkers)))} bins."
            )

            self._store_backup_source()
            task_queues, result_queue, workers = self._setup_workers()

            # Prepare file handlers
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

            # Main Simulation Loop
            for i in range(self.n_iterations):
                if not self.walkers:
                    print(f"Iter {i+1}/{self.n_iterations}: No walkers remaining. Stopping.")
                    break

                # Save walker state before propagation (for flux/bin tracking)
                walkers_before = [w.clone() for w in self.walkers]
                binned_before = self._get_bin_assignments(walkers_before)

                # 1. Propagate
                self.walkers = self._propagate_walkers(task_queues, result_queue)

                # 2. Handle Warping
                warp_ids = self._handle_warping()

                # 3. Handle Cleaning (Conditionally)
                clean_ids = set()
                if self.should_clean:
                    clean_ids = self._handle_cleaning(self.clean_threshold)

                # 4. Calculate Flux (only for warped walkers)
                flux_weights = [
                    w.weight for j, w in enumerate(self.walkers) if j in warp_ids
                ]
                flux_this_iter = np.sum(flux_weights)
                self._log_flux_data(i, warp_ids, file_handlers)

                # 5. Gather all dead walkers
                all_dead_ids = warp_ids.union(clean_ids)
                all_terminated_weights = [
                    w.weight for j, w in enumerate(self.walkers) if j in all_dead_ids
                ]

                self.prev_walkers = self.walkers

                # 6. Apply survivor scheme
                if self.survive_empty:
                    self.walkers = self._apply_survivor_scheme(walkers_before, binned_before)

                # 7. Resample and Recycle
                self.walkers, map_parents = self.resampler.resample(self.walkers, all_dead_ids)
                self._recycle_weights(all_terminated_weights)

                # 8. Save Data
                self._save_h5_data(i)
                self._log_iteration_data(i, flux_this_iter, file_handlers)

                if len(warp_ids) > 0 or len(clean_ids) > 0:
                    print(f"Iter {i}: Warped {len(warp_ids)}, Cleaned {len(clean_ids)}")

                print("*" * 50)
                gc.collect()

        finally:
            if self.traj_file:
                self.traj_file.close()
            if 'workers' in locals():
                self._cleanup_workers(task_queues, workers)
            if 'file_handlers' in locals():
                for f in file_handlers.values():
                    f.close()
