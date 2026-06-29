import os
from dataclasses import dataclass

import numpy as np
import tqdm

# from pathref import EnsemblePathRefiner
from .ensemblerefiner import EnsemblePathRefinerFast
from .pathcv import PathCV
from pathgennie.backends.openmm import PathGennieMD
from .principal_curve import PrincipalCurve


@dataclass
class PathGennieConfig:
    tau1: int = 5
    tau2: int = 45
    max_trial: int = 15
    max_cycle: int = 10000
    tol_target: float = 0.01
    save_freq: int = 5
    target_sigma: float = 0.0
    temperature: float = 30.0


# Assuming PathGennieConfig, PathGennieMD, EnsemblePathRefiner, PathCV, and pc are defined elsewhere


class PathGennieIterativeLearner:
    def __init__(
        self,
        simulation,
        topology_file,
        project_fn,
        converge_fn,
        initial_coords,
        start,
        end,
        device="cpu",
        verbosity=0,
        config=None,  # Replaced default instantiation for cleaner syntax
        output_dir="./pathgennie_output",  # New parameter for saving state
        pc=None,
    ):
        self.simulation = simulation
        self.topology_file = topology_file
        self.project_fn = project_fn
        self.converge_fn = converge_fn
        self.initial_coords = initial_coords
        self.start = start
        self.end = end
        self.device = device
        self.verbosity = verbosity
        self.config = config if config is not None else PathGennieConfig()
        self.pc = pc if pc is not None else PrincipalCurve(n_images=100, lam=0.15, n_iter=80, verbose=False)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # -------------------------------------------------
    def _log(self, msg, level=1):
        if self.verbosity >= level:
            print(msg)

    # -------------------------------------------------
    def _run_single_round(self, path_cv, target_cv, n_traj):
        self._log("Running PathGennie trajectories...", 2)

        pathgennie = PathGennieMD(
            simulation=self.simulation,
            topology_file=self.topology_file,
            projection_fn=self.project_fn,
            projection_args={"path_cv": path_cv},
            mode="target",
            target_projection=target_cv,
            temperature=self.config.temperature,
            convergence_fn=self.converge_fn,
            convergence_args={
                "projection_fn": self.project_fn,
                "target": target_cv,
                "path_cv": path_cv,
            },
        )

        traj_list, trial_list = [], []

        for _ in tqdm.trange(n_traj, disable=self.verbosity < 1):
            traj, trial = pathgennie.run(
                initial_pos=self.initial_coords,
                tau1=self.config.tau1,
                tau2=self.config.tau2,
                max_trial=self.config.max_trial,
                max_cycle=self.config.max_cycle,
                tol_target=self.config.tol_target,
                save_freq=self.config.save_freq,
                verbosity=2,
                target_sigma=self.config.target_sigma,
            )
            traj_list.append(traj)
            trial_list.append(trial)

        return traj_list, trial_list

    # -------------------------------------------------
    def _smooth_trajectories(self, traj_list, pc):
        smooth_paths = []
        for traj in traj_list:
            path = pc.fit(traj.reshape(-1, 3))
            smooth_paths.append(path.reshape(-1, 1, 3))
        return smooth_paths

    # -------------------------------------------------
    def _refine_ensemble(self, smooth_paths):
        self._log("Refining ensemble paths...", 2)

        refiner = EnsemblePathRefinerFast(
            hidden_dim=512,
            device=self.device,
        )

        refiner.fit(
            trajectories=smooth_paths,
            epochs=5000,
            start=self.start,
            end=self.end,
            lr=5e-4,
            patience=20,
            verbosity=self.verbosity >= 2,
        )

        return refiner.transform(n_points=50)

    # -------------------------------------------------
    def _save_checkpoint(self, it, traj_list, smooth_paths, refined_path):
        """Saves compressed trajectories and the refined path for the current iteration."""
        self._log(f"Saving checkpoint for iteration {it}...", 2)

        # Save raw trajectories (compressed)
        traj_dict = {f"traj_{i}": t for i, t in enumerate(traj_list)}
        np.savez_compressed(os.path.join(self.output_dir, f"trajectories_iter_{it}.npz"), **traj_dict)

        # Save smoothed paths (compressed)
        smooth_dict = {f"smooth_{i}": p for i, p in enumerate(smooth_paths)}
        np.savez_compressed(os.path.join(self.output_dir, f"smooth_paths_iter_{it}.npz"), **smooth_dict)

        # Save the single refined path (standard numpy array)
        np.save(os.path.join(self.output_dir, f"refined_path_iter_{it}.npy"), refined_path)

    # -------------------------------------------------
    def run(self, path_cv_init, target_cv, n_outer=50, n_traj=5, start_iter=0):
        """
        Runs the iterative learning process.

        To restart an old run: load your last 'refined_path_iter_{X}.npy',
        initialize a PathCV with it, and pass it as `path_cv_init` along
        with `start_iter=X+1`.
        """
        path_cv = path_cv_init
        learned_paths = []

        for it in range(start_iter, n_outer):
            self._log(f"\n=== Outer iteration {it + 1}/{n_outer} ===", 1)

            traj_list, _ = self._run_single_round(
                path_cv=path_cv,
                target_cv=target_cv,
                n_traj=n_traj,
            )

            # Assuming 'pc' is a globally available or previously initialized object
            smooth_paths = self._smooth_trajectories(traj_list, pc)
            refined_path = self._refine_ensemble(smooth_paths)

            learned_paths.append(refined_path)

            # Save data securely before generating the new PathCV
            self._save_checkpoint(it, traj_list, smooth_paths, refined_path)

            path_cv = PathCV(
                refined_path,
                mass_weights=None,
                enforce_equidistance=True,
                equidistance_tol=0.50,
                normalize_output=True,
            )

        return learned_paths
