import multiprocessing as mp
import os
import socket

import h5py
import numpy as np
from .gpu_worker import GPUWorker
from .cpu_worker import CPUWorker
from openmm.unit import kelvin
from .util import omm2np

# Other classes (Resampler, Walker, etc.) are imported
from .walker import Walker

try:
    import torch
    def get_local_gpu_count():
        return torch.cuda.device_count()
except ImportError:
    import subprocess
    def get_local_gpu_count():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                stdout=subprocess.PIPE, encoding="utf-8", check=True
            )
            return len(result.stdout.strip().splitlines())
        except Exception:
            return 0

class WeightedEnsembleBase:
    """
    Base class for Weighted Ensemble simulations.
    Provides core functionality for walker management,
    worker setup, trajectory storage, and logging.
    """

    @property
    def rng(self):
        """Dedicated Generator for stochastic resampling decisions.

        Using NumPy's global RNG made a run impossible to reproduce even with
        identical inputs. Seeded from ``self.seed`` when the config supplies one;
        ``None`` preserves the previous non-deterministic behaviour.
        """
        if getattr(self, "_rng", None) is None:
            self._rng = np.random.default_rng(getattr(self, "seed", None))
        return self._rng

    def _log_flux_data(self, iteration, warp_ids, files):
        """Writes flux data for warped walkers to the log file."""
        if not warp_ids:
            return

        flux_f = files.get('flux_f')
        if flux_f is None:
            return

        for wid in warp_ids:
            if wid >= len(self.walkers):
                continue  # skip invalid IDs

            walker = self.walkers[wid]
            weight = walker.weight
            cvs = walker.progress_coord

            if isinstance(cvs, np.ndarray):
                pc_str = ", ".join(f"pc{j+1}={val:.2f}" for j, val in enumerate(cvs))
            else:
                pc_str = str(cvs)

            flux_f.write(
                f"Iteration {iteration}, Warp ID: {wid}, Weight: {weight:.6e}, CVs: {pc_str}\n"
            )

        flux_f.flush()

    def _log_iteration_data(self, i, flux_this_iter, files):
        """Writes simulation data for the current iteration to files."""
        self.total_flux_to_target += flux_this_iter
        # `flux_this_iter` is a probability per iteration (dimensionless). The rate
        # is that divided by tau. The division used to be commented out and TAU was
        # a dead local in main.run(), so `flux_history` held FLUXES under the name
        # `flux_rate` and no rate was ever computed in code -- it was reconstructed
        # by hand afterwards, along with the choice of averaging window.
        self.flux_history.append(flux_this_iter)
        tau = getattr(self, "tau", None)
        self.rate_history.append(flux_this_iter / tau if tau else float("nan"))

        # Bin probabilities
        bin_probabilities = np.zeros(self.n_total_bins)
        current_assignments = self._get_bin_assignments(self.walkers)
        for i_w, walker in enumerate(self.walkers):
            if i_w < len(current_assignments):
                bin_idx_flat = current_assignments[i_w]
                bin_probabilities[bin_idx_flat] += walker.weight
        files['bin_f'].write(" ".join(map(str, bin_probabilities)) + "\n")
        files['bin_f'].flush()

        # Walker coordinates & weights
        for walker_idx, walker in enumerate(self.walkers):
            pc_components = " ".join(f"{pc:.6f}" for pc in walker.progress_coord)
            files['walker_f'].write(f"{i} {walker_idx} {pc_components} {walker.weight:.6e}\n")
        files['walker_f'].flush()

        print(f"Iter {i+1}/{self.n_iterations}, Walkers: {len(self.walkers)}, "
              f"Flux: {flux_this_iter:.4e}, Total Flux: {self.total_flux_to_target:.4e}")

    def rate_estimate(self, burn_in=0.5):
        """Steady-state rate constant from the flux history.

        Weighted Ensemble flux is a rate only once the walker distribution has
        reached steady state; averaging from iteration zero mixes in the transient
        while the ensemble is still filling the bins. ``burn_in`` discards the
        leading window -- a float in (0, 1) is a fraction of the run, an int is a
        count -- matching ``WeightedEnsembleStage``'s convention.

        Returns a dict with ``rate``, ``stderr``, the window used, and a
        ``steady_state`` flag comparing the two halves of the retained window. The
        flag is a diagnostic, not a guarantee: it can only detect drift large
        relative to the scatter, and it reports ``False`` rather than raising so a
        caller still gets the number together with the warning.
        """
        import numpy as _np

        # getattr: rate_estimate is public and may be called on an object whose
        # __init__ did not complete (the bin config raises before the histories
        # are set up), and a missing attribute there should read as 'no data'.
        hist = _np.asarray(getattr(self, "rate_history", []), dtype=float)
        n = len(hist)
        if n == 0:
            return {"rate": float("nan"), "stderr": float("nan"), "n_used": 0,
                    "burn_in": 0, "steady_state": False,
                    "note": "no iterations recorded"}
        if isinstance(burn_in, float) and 0.0 < burn_in < 1.0:
            n_burn = int(round(burn_in * n))
        else:
            n_burn = int(burn_in)
        # Never discard everything: a biased estimate beats no estimate, as long
        # as the burn-in actually used is reported alongside it.
        n_burn = min(max(n_burn, 0), max(0, n - 1))
        w = hist[n_burn:]
        finite = w[_np.isfinite(w)]
        if finite.size == 0:
            return {"rate": float("nan"), "stderr": float("nan"), "n_used": 0,
                    "burn_in": n_burn, "steady_state": False,
                    "note": "tau not set, so no rate could be formed"}
        rate = float(finite.mean())
        stderr = float(finite.std(ddof=1) / _np.sqrt(finite.size)) if finite.size > 1 else float("nan")
        half = finite.size // 2
        steady = True
        if half >= 2:
            a, b = finite[:half], finite[half:]
            se = _np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
            steady = bool(se == 0 or abs(b.mean() - a.mean()) <= 2 * se)
        return {"rate": rate, "stderr": stderr, "n_used": int(finite.size),
                "burn_in": n_burn, "steady_state": steady,
                "tau": getattr(self, "tau", None)}

    def _save_h5_data(self, iteration_id: int):
        """
        Saves walker data for a specific iteration to a single HDF5 file.
        ... (rest of the function is unchanged as it was correct) ...
        """
        if self.h5_atom_indices is None:
            return
        valid_walkers = [w for w in self.walkers if hasattr(w, 'positions') and w.positions is not None]
        if not valid_walkers:
            print(f"Iteration {iteration_id}: No valid walkers with position data to save to HDF5.")
            return
        n_new_walkers = len(valid_walkers)
        n_atoms_to_save = len(self.h5_atom_indices)
        pcoord_dim = np.array(valid_walkers[0].progress_coord).shape
        pcoord_shape = (None,) + pcoord_dim if pcoord_dim else (None,)
        coords_data = np.zeros((n_new_walkers, n_atoms_to_save, 3), dtype=np.float32)
        weights_data = np.zeros(n_new_walkers, dtype=np.float64)
        pcoords_data = np.zeros((n_new_walkers,) + pcoord_dim, dtype=np.float32)
        iteration_data = np.full(n_new_walkers, iteration_id, dtype=int)
        for i, walker in enumerate(valid_walkers):
            coords_data[i] = walker.positions[self.h5_atom_indices]
            weights_data[i] = walker.weight
            pcoords_data[i] = walker.progress_coord
        with h5py.File(self.h5_file, 'a') as f:
            datasets_to_write = {
                'coordinates': {'data': coords_data, 'maxshape': (None, n_atoms_to_save, 3)},
                'weights': {'data': weights_data, 'maxshape': (None,)},
                'pcoords': {'data': pcoords_data, 'maxshape': pcoord_shape},
                'iteration': {'data': iteration_data, 'maxshape': (None,)}
            }
            is_first_write = 'coordinates' not in f
            for name, props in datasets_to_write.items():
                if is_first_write:
                    f.create_dataset(name, data=props['data'], maxshape=props['maxshape'], chunks=True)
                else:
                    dset = f[name]
                    current_size = dset.shape[0]
                    dset.resize(current_size + n_new_walkers, axis=0)
                    dset[current_size:] = props['data']

    def initialize_walkers(self):
        """Creates initial walkers from starting positions and weights."""
        states = len(self.initial_positions)
        if states == 1:
            n_initial = self.n_walkers_per_bin
            initial_weight = 1.0 / n_initial
            for _ in range(n_initial):
                w = Walker(positions=self.initial_positions[0], weight=initial_weight)
                w.progress_coord = self.projection_fn(omm2np(w.positions), self.kwargs)
                self.walkers.append(w)
        else:
            initial_weight = 1.0 / states
            for i in range(states):
                w = Walker(positions=self.initial_positions[i], weight=initial_weight)
                w.progress_coord = self.projection_fn(omm2np(w.positions), self.kwargs)
                self.walkers.append(w)

    def _setup_workers(self):
        """
        Sets up workers based on the configured platform (CPU or CUDA).
        """
        node_name = socket.gethostname()
        runner_cls = self.config.get('runner_class')

        # Check platform configuration (Default to CUDA if not specified)
        platform_mode = self.config.get('platform', 'cuda').lower()

        workers = []
        task_queues = []
        result_queue = mp.Queue()

        if runner_cls is None:
             raise ValueError("You must provide 'runner_class' in your config dictionary.")

        # --- CPU MODE ---
        if platform_mode == 'cpu':
            # Default to all available cores if not specified
            n_workers = self.config.get('n_workers', mp.cpu_count())
            threads_per_worker = self.config.get('threads_per_worker', 1)

            print(f"[{node_name}] Setting up {n_workers} CPU Workers "
                  f"({threads_per_worker} threads/worker).")

            task_queues = [mp.Queue() for _ in range(n_workers)]
            runner_kwargs = self.config.get('runner_kwargs', {})

            for worker_id in range(n_workers):
                w = CPUWorker(
                    worker_id=worker_id,
                    task_queue=task_queues[worker_id],
                    result_queue=result_queue,
                    temperature=self.temperature,
                    projection_fn=self.projection_fn,
                    kwargs=self.kwargs,
                    n_steps_per_tau=self.n_steps_per_tau,
                    runner_class=runner_cls,
                    threads=threads_per_worker,
                    runner_kwargs=runner_kwargs
                )
                w.start()
                workers.append(w)

        # --- GPU MODE ---
        else:
            local_gpus = get_local_gpu_count()
            if local_gpus == 0:
                print(f"[{node_name}] Warning: No GPUs detected, but platform is 'cuda'.")

            print(f"[{node_name}] Detected {local_gpus} GPUs.")

            task_queues = [mp.Queue() for _ in range(local_gpus)]

            runner_kwargs = self.config.get('runner_kwargs', {})

            for gpu_id in range(local_gpus):
                w = GPUWorker(
                    gpu_id=gpu_id,
                    task_queue=task_queues[gpu_id],
                    result_queue=result_queue,
                    temperature=self.temperature,
                    projection_fn=self.projection_fn,
                    kwargs=self.kwargs,
                    n_steps_per_tau=self.n_steps_per_tau,
                    runner_class=runner_cls,
                    runner_kwargs=runner_kwargs
                )
                w.start()
                workers.append(w)

        return task_queues, result_queue, workers


    def _propagate_walkers(self, task_queues, result_queue):
        """Distributes walker propagation tasks and collects results."""
        walkers_before = [w.clone() for w in self.walkers]

        for idx, walker in enumerate(walkers_before):
            #gpu_id = idx % self.n_gpus
            walker_id = idx % len(task_queues)
            task = (idx, walker.positions, walker.weight, walker.velocities)
            task_queues[walker_id].put(task)

        updated_walkers = [None] * len(walkers_before)
        for _ in range(len(walkers_before)):
            idx, new_pos, weight, new_vel, new_pc = result_queue.get()
            updated_walkers[idx] = Walker(
                positions=new_pos,
                weight=weight,
                velocities=new_vel,
                progress_coord=new_pc
            )

        return updated_walkers

    def _handle_warping(self):
        """
        Checks each walker against the warping criteria and identifies those to be
        removed/recycled.

        Efficiency Fix: The process is consolidated into a single loop,
        performing position conversion, warping check, and index gathering
        simultaneously.
        """
        remove_idxs = set()
        warpable_statuses = []  # Optional: to keep track of the statuses for debugging/logging

        for idx, walker in enumerate(self.walkers):
            # 1. Get positions and convert (omm2np handles OpenMM to NumPy conversion)
            # Note: We assume the walker.positions attribute exists and is a valid object.
            try:
                positions_np = omm2np(walker.positions)
            except AttributeError:
                # Handle case where a walker might not have positions yet (e.g., initial state)
                continue

            # 2. Check the warping condition
            is_warpable = self.warp_function(positions_np, self.warp_kwargs)
            warpable_statuses.append(is_warpable)

            # 3. Record index if the condition is met
            if is_warpable:
                remove_idxs.add(idx)

        print(f"[WESS] {len(remove_idxs)} walkers recycled in this iteration.")
        print(f"Warpable statuses: {warpable_statuses}")

        return remove_idxs

    def _recycle_weights(self, terminated_weights):
        """Distributes weights of terminated walkers to source bin walkers."""
        if not terminated_weights:
            return
        recycle_weight = sum(terminated_weights)
        current_bin_indices = self._get_bin_assignments(self.walkers)
        if len(self.source_bin_indices):
            source_walkers_indices = [
                idx for idx, bin_idx in enumerate(current_bin_indices)
                if bin_idx in self.source_bin_indices]
        else:
            source_walkers_indices = [
                idx for idx, bin_idx in enumerate(current_bin_indices)
                if bin_idx == self.source_bin_index]

        #print("source_recycling: ", source_walkers_indices)
        if source_walkers_indices:
            weight_per_source_walker = recycle_weight / len(source_walkers_indices)
            for idx in source_walkers_indices:
                self.walkers[idx].weight += weight_per_source_walker
            walkers = [self.walkers[i] for i in source_walkers_indices]
            self.redistribute_excess_weight(walkers)
        elif self.backups:
            weight_per_backup = recycle_weight / len(self.backups)
            for backup_walker in self.backups:
                new_walker = backup_walker.clone()
                new_walker.weight = weight_per_backup
                self.walkers.append(new_walker)
            print(f"No workers in source bin to recycle weights. Using {len(self.backups)} backups.")

    def redistribute_excess_weight(self, walkers, cap=0.1):
            """
            Ensure no walker exceeds the given weight cap.
            If a walker exceeds the cap, distribute the excess evenly among the others.

            Args:
                walkers (list): List of walker objects, each with a .weight attribute.
                cap (float): Maximum allowed weight per walker.
            """
            if not walkers: # Handle case with no walkers
                return

            # Weighted Ensemble is unbiased only because resampling conserves total
            # probability -- the rate constant is literally a sum of walker weights
            # arriving at the target. So weight must never be stripped from a donor
            # unless there is a recipient to receive it.
            #
            # If the walkers cannot collectively hold their own weight under the cap
            # (total > cap * n), no assignment satisfies the cap without destroying
            # probability. In that case leave the weights untouched rather than
            # silently biasing every downstream observable.
            total_weight = sum(w.weight for w in walkers)
            if total_weight > cap * len(walkers) * (1.0 + 1e-12):
                print(
                    f"[WESS] Weight cap {cap} unsatisfiable for {len(walkers)} walker(s) "
                    f"holding {total_weight:.6g}; skipping redistribution to conserve weight."
                )
                return

            max_iters = 100 # Safety break to prevent unforeseen infinite loops
            iter_count = 0

            while iter_count < max_iters:
                iter_count += 1

                # Classify BEFORE mutating anything, so excess is only ever removed
                # in the same step that hands it to recipients.
                donors = [w for w in walkers if w.weight > cap]
                recipients = [w for w in walkers if w.weight <= cap]

                if not donors or not recipients:
                    break

                total_excess = sum(w.weight - cap for w in donors)
                if total_excess <= 1e-12:
                    break

                for w in donors:
                    w.weight = cap
                share = total_excess / len(recipients)
                for r in recipients:
                    r.weight += share

            if iter_count >= max_iters:
                print("Warning: Weight redistribution reached max iterations.")

    def _store_backup_source(self):
        """Stores a backup copy of the source bin walkers."""
        current_bin_indices = self._get_bin_assignments(self.walkers)
        if len(self.source_bin_indices):
            source_walkers_indices = [
                idx for idx, bin_idx in enumerate(current_bin_indices)
                if bin_idx in self.source_bin_indices
            ]
        else:
            source_walkers_indices = [
                idx for idx, bin_idx in enumerate(current_bin_indices)
                if bin_idx == self.source_bin_index
            ]
        self.backups = []
        if source_walkers_indices:
            for idx in source_walkers_indices:
                self.backups.append(self.walkers[idx].clone())
        else:
            print("Warning: No source walkers found to create backups. Recycling may fail if source bin empties.")

    def _get_bin_assignments(self, walkers):
        """Returns bin indices for each walker."""
        if not walkers:
            return np.empty(0, dtype=int)
        assignments = [self.bins.find_bin(w.progress_coord) for w in walkers]
        return np.array(assignments)

    def _cleanup_workers(self, task_queues, workers):
        """Stops all GPU workers."""
        for q in task_queues:
            q.put("STOP")
        for w in workers:
            w.join()

    def _handle_cleaning(self, threshold = 50.0):
        remove_idxs = set()
        for idx, _ in enumerate(self.walkers):
            if np.abs(self.walkers[idx].progress_coord[1]) > threshold:
                remove_idxs.add(idx)
        print(f"[WESS] {len(remove_idxs)} walkers cleaned in this iteration.")
        return remove_idxs

    def _apply_survivor_scheme(self, walkers_before, binned_before):
        """
        Replicates walkers from occupied neighbor bins into newly emptied bins
        to maintain coverage, even when no 'excess' walkers exist.

        """
        if binned_before.size == 0:
            return self.walkers

        binned_after = self._get_bin_assignments(self.walkers)
        previously_occupied = set(binned_before)
        newly_occupied = set(binned_after)
        lost_bins = previously_occupied - newly_occupied
        if not lost_bins:
            return self.walkers

        for walker in walkers_before:
            print(walker.progress_coord)
        # Bin → list of walker indices (current step)
        walkers_per_bin_after = {b: [] for b in np.unique(binned_after)}
        for j, bin_idx in enumerate(binned_after):
            walkers_per_bin_after[bin_idx].append(j)

        # First pass: identify excess walkers
        sacrificial_candidates = []
        for _, walker_indices in walkers_per_bin_after.items():
            if len(walker_indices) > self.n_walkers_per_bin:
                sacrificial_candidates.extend(walker_indices[self.n_walkers_per_bin:])

        # If no excess, we will steal from nearest neighbor bins progressively
        for lost_bin_index in lost_bins:
            # Find donor walker index
            donor_idx = None

            # Search nearest bins for donor
            for neighbor_bin in self.bins.get_sorted_neighbors(lost_bin_index):
                candidates = walkers_per_bin_after.get(neighbor_bin, [])
                if len(candidates) > 0:
                    donor_idx = candidates.pop(0)  # steal first walker
                    break
            #print(donor_idx)
            if donor_idx is None:
                continue  # no donor found

            # Find a survivor from previous step in the nearest occupied bin
            chosen_neighbor = None
            for neighbor_bin in self.bins.get_sorted_neighbors(lost_bin_index):
                if neighbor_bin in previously_occupied:
                    chosen_neighbor = neighbor_bin
                    break
            if chosen_neighbor is None:
                continue

            survivor_idxs = np.where(binned_before == chosen_neighbor)[0]
            if len(survivor_idxs) == 0:
                continue
            #print(survivor_idxs)
            survivor_to_copy_from = self.rng.choice(survivor_idxs)
            # Replace donor walker with clone of survivor
            saved_weight = self.walkers[survivor_to_copy_from].weight
            self.walkers[survivor_to_copy_from] = walkers_before[donor_idx].clone()
            self.walkers[survivor_to_copy_from].weight = saved_weight

        return self.walkers
