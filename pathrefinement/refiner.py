import json
import multiprocessing
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from pathgennie.backends.openmm import PathGennieMD
from .ensemblerefiner import EnsemblePathRefinerFast
from .pathcv import PathCV
from .potentials import Potential2D
from .principal_curve import PrincipalCurve


@dataclass
class PathRefinementConfig:
    n_iterations: int = 10
    n_trajectories: int = 5
    pathgennie_tau1: int = 10
    pathgennie_tau2: int = 40
    pathgennie_max_trial: int = 10
    pathgennie_max_cycle: int = 500
    pathgennie_tol_target: float = 0.05
    keep_endpoints: bool = True
    project_to_real: bool = False  # Snap refined path nodes to nearest physical trajectory frames
    seed: int = 42
    nn_epochs: int = 2000
    nn_hidden_dim: int = 128
    device: str = "cpu"
    verbosity: int = 1
    # --- Parallel walker settings ---
    n_workers: int = 1   # Number of parallel worker processes (1 = serial)
    worker_device: int = 0  # GPU DeviceIndex for each worker simulation


@dataclass
class RefinementResult:
    initial_path: np.ndarray
    refined_path: np.ndarray
    path_history: List[np.ndarray]
    converged: bool
    n_iterations_run: int
    metadata: Dict[str, Any]
    timing: Optional[Dict[str, float]] = None  # Per-iteration wall-clock seconds

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "initial_path.npy"), self.initial_path)
        np.save(os.path.join(directory, "refined_path.npy"), self.refined_path)
        np.savez_compressed(
            os.path.join(directory, "path_history.npz"),
            **{f"iter_{i}": p for i, p in enumerate(self.path_history)},
        )
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=2)
        if self.timing:
            with open(os.path.join(directory, "timing.json"), "w") as f:
                json.dump(self.timing, f, indent=2)

    def plot(self, filename: str, potential: Optional[Potential2D] = None):
        from .plotting import plot_initial_vs_refined
        plot_initial_vs_refined(self, potential, filename)


# ---------------------------------------------------------------------------
# Top-level worker function (must be module-level to be picklable by multiprocessing)
# ---------------------------------------------------------------------------

def _run_single_trajectory(args):
    """
    Worker function that runs a single PathGennie trajectory in an isolated
    process. Creates its own OpenMM Simulation from scratch so there is zero
    shared state with the parent or sibling workers.

    Args:
        args: tuple of
            (potential, config, path_cv, start_pt, worker_seed, feature_fn)

    Returns:
        np.ndarray of shape (T, D) — trajectory in feature space, or None if
        the trajectory was empty.
    """
    potential, config, path_cv, start_pt, worker_seed, feature_fn = args

    # Create a brand-new independent Simulation inside this worker process.
    # This is the key: each process gets its own system + integrator + context.
    sim = potential.create_simulation(
        seed=worker_seed,
        device=config.worker_device,
    )

    # Rebuild projection / convergence callables that capture the local sim.
    target_cv = np.array([1.0])

    def _project_fn(coords, path_cv):
        features = feature_fn(coords)
        s, z = path_cv.compute(np.atleast_2d(features))
        return np.array([s])

    def _converge_fn(coords, path_cv, target):
        features = feature_fn(coords)
        s, z = path_cv.compute(np.atleast_2d(features))
        return float(s) >= float(target[0]) - config.pathgennie_tol_target

    pathgennie = PathGennieMD(
        simulation=sim,
        projection_fn=_project_fn,
        projection_args={"path_cv": path_cv},
        mode="target",
        target_projection=target_cv,
        convergence_fn=_converge_fn,
        convergence_args={"path_cv": path_cv, "target": target_cv},
        temperature=300.0,
    )

    initial_pos = np.zeros((1, 3))
    initial_pos[0, :2] = start_pt

    traj, _ = pathgennie.run(
        initial_pos=initial_pos,
        tau1=config.pathgennie_tau1,
        tau2=config.pathgennie_tau2,
        max_trial=config.pathgennie_max_trial,
        max_cycle=config.pathgennie_max_cycle,
        verbosity=0,  # suppress per-worker output
    )

    if len(traj) == 0:
        return None

    # Convert Angstrom trajectory frames → 2D feature space
    traj_features = np.array([feature_fn(pt) for pt in traj])
    return traj_features


# ---------------------------------------------------------------------------
# Default feature function — must be module-level to be picklable
# ---------------------------------------------------------------------------

def _default_feature_fn(coords: np.ndarray) -> np.ndarray:
    """Maps (N_atoms, 3) Angstrom coords → 2D Muller-Brown feature vector."""
    return coords[0, :2] / 10.0


# ---------------------------------------------------------------------------
# PathRefiner
# ---------------------------------------------------------------------------

class PathRefiner:
    """Iterative path refiner using PathGennieMD and NN smoothing.

    Supports parallel trajectory walkers via ``config.n_workers``.
    When ``n_workers > 1``, each walker runs in a separate process with its
    own OpenMM Simulation context, giving near-linear speedup for CPU-based
    potentials (e.g. Muller-Brown).
    """

    def __init__(self, potential: Potential2D, config: PathRefinementConfig, feature_fn=None):
        self.potential = potential
        self.config = config
        # Parent process keeps one simulation for quick CV evaluation only.
        self.simulation = self.potential.create_simulation(seed=config.seed)

        # Use module-level default so feature_fn is always picklable
        if feature_fn is None:
            self.feature_fn = _default_feature_fn
        else:
            self.feature_fn = feature_fn

    def _project_fn(self, coords: np.ndarray, path_cv: PathCV) -> np.ndarray:
        # coords from PathGennieMD will be (N_atoms, 3) in Angstroms
        features = self.feature_fn(coords)
        s, z = path_cv.compute(np.atleast_2d(features))
        return np.array([s])

    def _converge_fn(self, coords: np.ndarray, path_cv: PathCV, target: np.ndarray) -> bool:
        features = self.feature_fn(coords)
        s, z = path_cv.compute(np.atleast_2d(features))
        return float(s) >= float(target[0]) - self.config.pathgennie_tol_target

    def _collect_trajectories(self, path_cv: PathCV, start_pt: np.ndarray, it_seed: int) -> List[np.ndarray]:
        """Run n_trajectories walkers, either serially or in parallel."""
        n = self.config.n_trajectories
        n_workers = min(self.config.n_workers, n)  # never spawn more than needed

        # Build argument list — each worker gets a unique seed
        worker_args = [
            (
                self.potential,
                self.config,
                path_cv,
                start_pt,
                it_seed + trj_idx,   # unique seed per walker
                self.feature_fn,
            )
            for trj_idx in range(n)
        ]

        if n_workers <= 1:
            # --- Serial fallback ---
            raw = [_run_single_trajectory(a) for a in worker_args]
        else:
            # --- Parallel: each worker creates its own Simulation after fork ---
            ctx = multiprocessing.get_context("spawn")  # 'spawn' is safe with CUDA
            with ctx.Pool(processes=n_workers) as pool:
                raw = pool.map(_run_single_trajectory, worker_args)

        return [r for r in raw if r is not None]

    def refine(self, initial_path: np.ndarray) -> RefinementResult:
        # initial_path is (N, 2)
        initial_path = np.asarray(initial_path)
        path_history = [initial_path.copy()]
        current_path = initial_path.copy()

        pc = PrincipalCurve(n_images=len(initial_path), verbose=(self.config.verbosity > 1))

        start_pt = current_path[0]
        end_pt = current_path[-1]

        target_cv = np.array([1.0])  # Normalized s reaches 1.0 at the end

        per_iter_times: Dict[str, float] = {}
        n_workers_actual = min(self.config.n_workers, self.config.n_trajectories)
        converged = False

        for it in range(self.config.n_iterations):
            if self.config.verbosity > 0:
                print(f"--- Iteration {it + 1}/{self.config.n_iterations} "
                      f"[{n_workers_actual} worker(s)] ---")

            path_cv = PathCV(
                current_path[:, np.newaxis, :],  # (N, 1, 2)
                enforce_equidistance=False,
                normalize_output=True,
            )

            # Unique seed per iteration so different iterations explore differently
            it_seed = self.config.seed + it * self.config.n_trajectories

            t0 = time.perf_counter()
            traj_list = self._collect_trajectories(path_cv, start_pt, it_seed)
            t_walkers = time.perf_counter() - t0

            if self.config.verbosity > 0:
                print(f"  Walkers: {len(traj_list)}/{self.config.n_trajectories} "
                      f"succeeded in {t_walkers:.1f}s")

            if not traj_list:
                print("Warning: No trajectories generated.")
                break

            # Smooth individual trajectories with PrincipalCurve
            smooth_paths = []
            for traj_2d in traj_list:
                smooth_path = pc.fit(traj_2d)
                smooth_paths.append(smooth_path)

            # Refine ensemble with NN
            if self.config.verbosity > 0:
                print("  Training NN refiner...")

            t1 = time.perf_counter()
            refiner = EnsemblePathRefinerFast(
                hidden_dim=self.config.nn_hidden_dim, device=self.config.device
            )

            # Format for EnsemblePathRefinerFast: (T, N, D)
            smooth_paths_3d = [p[:, np.newaxis, :] for p in smooth_paths]

            refiner.fit(
                trajectories=smooth_paths_3d,
                epochs=self.config.nn_epochs,
                start=start_pt[np.newaxis, :] if self.config.keep_endpoints else None,
                end=end_pt[np.newaxis, :] if self.config.keep_endpoints else None,
                verbosity=self.config.verbosity > 1,
                patience=50,
            )
            t_nn = time.perf_counter() - t1

            refined_path_3d = refiner.transform(
                n_points=len(initial_path),
                project_to_real=self.config.project_to_real
            )
            current_path = refined_path_3d[:, 0, :]

            if self.config.keep_endpoints:
                current_path[0] = start_pt
                current_path[-1] = end_pt

            path_history.append(current_path.copy())

            per_iter_times[f"iter_{it+1}_walkers_s"] = round(t_walkers, 2)
            per_iter_times[f"iter_{it+1}_nn_s"] = round(t_nn, 2)

            # Convergence check (Frechet distance or simple max norm)
            diff = np.max(np.linalg.norm(current_path - path_history[-2], axis=1))
            if self.config.verbosity > 0:
                print(f"  Max path update: {diff:.4f}")

            if diff < 1e-3:  # hardcoded simple tolerance
                print("Converged.")
                converged = True
                break

        return RefinementResult(
            initial_path=initial_path,
            refined_path=current_path,
            path_history=path_history,
            converged=converged,
            n_iterations_run=it + 1,
            metadata=asdict(self.config),
            timing=per_iter_times,
        )
