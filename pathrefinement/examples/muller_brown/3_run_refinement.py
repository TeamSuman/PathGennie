"""
3_run_refinement.py
───────────────────
Iterative path refinement on the Muller-Brown potential using PathGennieMD
with parallel walkers.  Mirrors the structure of CLN025/3_run_refinement.py.

Prerequisites
─────────────
Run 1_generate_initial_path.py first to produce results/initial_path/initial_path.npy.

Parallel walkers
────────────────
Set N_WORKERS > 1 to run trajectories concurrently.  Each worker spawns its own
independent OpenMM Simulation context (system + integrator + PRNG) so there is
no shared state between workers.  The expected speedup is ~N_WORKERS× for
CPU-based potentials.
"""

import argparse
import multiprocessing
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from pathrefinement.pathcv import PathCV
from pathrefinement.principal_curve import PrincipalCurve
from pathrefinement.ensemblerefiner import EnsemblePathRefinerFast
from pathgennie.backends.openmm import PathGennieMD
from common import (
    MINIMA,
    create_mb_system,
    feature_fn,
)


# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description="Iterative Path Refinement")
parser.add_argument("--config", default=os.path.join(BASE_DIR, "config.yaml"), help="Path to YAML config file")
# Parse args only when run as main script (to avoid issues when imported/spawned)
if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    config_path = args.config
else:
    config_path = os.path.join(BASE_DIR, "config.yaml")

# Load YAML
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
else:
    cfg = {}

# Extract config values with defaults
ref_cfg = cfg.get("refinement", {})
N_ITERATIONS   = ref_cfg.get("n_iterations", 5)
N_TRAJECTORIES  = ref_cfg.get("n_trajectories", 8)
N_WORKERS      = ref_cfg.get("n_workers", 8)
WORKER_DEVICE  = ref_cfg.get("worker_device", 0)
SEED           = ref_cfg.get("seed", 42)
N_PATH_IMAGES  = ref_cfg.get("n_path_images", 100)
TOL_TARGET     = ref_cfg.get("tol_target", 0.05)
LAMBDA_SCALE   = ref_cfg.get("lambda_scale", 1.0)

pg_cfg = cfg.get("pathgennie", {})
TAU1           = pg_cfg.get("tau1", 5)
TAU2           = pg_cfg.get("tau2", 10)
MAX_TRIAL      = pg_cfg.get("max_trial", 10)
MAX_CYCLE      = pg_cfg.get("max_cycle", 1000)
SAVE_FREQ      = pg_cfg.get("save_freq", 10)
TEMPERATURE    = pg_cfg.get("temperature", 300.0)
SIGMA          = pg_cfg.get("sigma", 0.01)

nn_cfg = cfg.get("refiner", {})
NN_EPOCHS      = nn_cfg.get("epochs", 5000)
NN_HIDDEN_DIM  = nn_cfg.get("hidden_dim", 128)
NN_DEVICE      = nn_cfg.get("device", "cuda")
NN_LR          = nn_cfg.get("lr", 0.005)
NN_PATIENCE    = nn_cfg.get("patience", 200)
NN_BATCH_SIZE  = nn_cfg.get("batch_size", 128)

pc_cfg = cfg.get("principal_curve", {})
PC_LAM         = pc_cfg.get("lam", 0.1)
PC_N_ITER      = pc_cfg.get("n_iter", 50)
PC_TOL         = pc_cfg.get("tol", 1e-5)
PC_VERBOSE     = pc_cfg.get("verbose", False)

paths_cfg = cfg.get("paths", {})
INITIAL_PATH_NPY = os.path.join(BASE_DIR, paths_cfg.get("initial_path_npy", "results/initial_path/initial_path.npy"))
OUTPUT_DIR       = os.path.join(BASE_DIR, paths_cfg.get("output_dir", "results/refinement"))


# ── Worker function (module-level → picklable) ────────────────────────────────

def _run_single_trajectory(args):
    """
    Run one PathGennie trajectory inside an isolated worker process.
    Creates its own OpenMM Simulation from scratch; no shared state with
    parent or sibling workers.
    """
    path_cv, start_pos_nm, worker_seed = args

    sim = create_mb_system(seed=worker_seed, device=WORKER_DEVICE)
    target_cv = np.array([1.0, 0.0])

    def project_fn(coords, path_cv):
        s, z = path_cv.compute(np.atleast_2d(coords[0, :2]))
        return np.array([s, z])

    def converge_fn(coords, path_cv, target):
        s, _ = path_cv.compute(np.atleast_2d(coords[0, :2]))
        return float(s) >= float(target[0]) - TOL_TARGET

    pathgennie = PathGennieMD(
        simulation=sim,
        projection_fn=project_fn,
        projection_args={"path_cv": path_cv},
        mode="target",
        target_projection=target_cv,
        convergence_fn=converge_fn,
        convergence_args={"path_cv": path_cv, "target": target_cv},
        temperature=TEMPERATURE,
        sigma=SIGMA
    )

    traj, _ = pathgennie.run(
        initial_pos=start_pos_nm,
        tau1=TAU1,
        tau2=TAU2,
        max_trial=MAX_TRIAL,
        max_cycle=MAX_CYCLE,
        save_freq=SAVE_FREQ,
        verbosity=1,
    )

    if len(traj) == 0:
        return None

    # Check if this trajectory actually reached target convergence
    last_frame = traj[-1]
    if not converge_fn(last_frame, path_cv, target_cv):
        return None

    # traj frames are (N_atoms, 3) in Angstroms — convert to 2D features (in Angstroms)
    return np.array([pt[0, :2] for pt in traj])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load initial path ────────────────────────────────────────────────────
    in_npy = INITIAL_PATH_NPY
    if not os.path.exists(in_npy):
        print(f"Error: {in_npy} not found.")
        print("Please run 1_generate_initial_path.py first.")
        return

    print("Loading initial path...")
    initial_path = np.load(in_npy)           # (N_images, 2)
    print(f"  Shape: {initial_path.shape}")

    # Start position for all walkers: minimum A in nm (OpenMM uses nm)
    start_pos_nm = np.zeros((1, 3))
    start_pos_nm[0, :2] = MINIMA["A"]       # already in nm-equivalent units

    out_dir = OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # ── Initialise smoothing curve ───────────────────────────────────────────
    pc = PrincipalCurve(
        n_images=N_PATH_IMAGES,
        lam=PC_LAM,
        n_iter=PC_N_ITER,
        tol=PC_TOL,
        verbose=PC_VERBOSE,
    )

    current_path = initial_path.copy() * 10.0  # Operate in Angstroms
    path_history = [initial_path.copy()]        # Store history in nanometers

    n_workers_actual = min(N_WORKERS, N_TRAJECTORIES)
    print(f"\nStarting refinement: {N_ITERATIONS} iterations × "
          f"{N_TRAJECTORIES} trajectories [{n_workers_actual} parallel workers]\n")

    # ── Iterative refinement loop ─────────────────────────────────────────────
    for it in range(N_ITERATIONS):
        print(f"=== Iteration {it + 1}/{N_ITERATIONS} ===")

        path_cv = PathCV(
            current_path[:, np.newaxis, :],   # (N, 1, 2) in Angstroms
            enforce_equidistance=False,
            normalize_output=True,
        )
        path_cv.lam *= LAMBDA_SCALE  # Scale down lambda to broaden the CV tube (prevents stuck walkers)

        # Each worker gets a unique seed so integrators diverge
        it_base_seed = SEED + it * N_TRAJECTORIES
        worker_args = [
            (path_cv, start_pos_nm, it_base_seed + trj)
            for trj in range(N_TRAJECTORIES)
        ]

        # ── Parallel or serial walker collection ─────────────────────────────
        t0 = time.perf_counter()
        if n_workers_actual <= 1:
            raw_trajs = [_run_single_trajectory(a) for a in worker_args]
        else:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=n_workers_actual) as pool:
                raw_trajs = pool.map(_run_single_trajectory, worker_args)
        t_walkers = time.perf_counter() - t0

        traj_list = [r for r in raw_trajs if r is not None]
        print(f"  Walkers: {len(traj_list)}/{N_TRAJECTORIES} succeeded "
              f"in {t_walkers:.1f}s")

        if not traj_list:
            print("  No trajectories generated. Stopping.")
            break

        # Save raw trajectories (in nanometers) for debug visualization
        traj_debug_dir = os.path.join(out_dir, "trajs", f"iter_{it + 1}")
        os.makedirs(traj_debug_dir, exist_ok=True)
        for ti, traj_feat in enumerate(traj_list):
            np.save(os.path.join(traj_debug_dir, f"traj_{ti}.npy"), traj_feat / 10.0)

        # ── Smooth each trajectory ───────────────────────────────────────────
        smooth_paths = []
        for traj_feat in traj_list:
            smooth_paths.append(pc.fit(traj_feat))

        # ── Learn consensus path with EnsembleRefiner ────────────────────────
        print("  Learning consensus path with EnsembleRefiner...")
        t1 = time.perf_counter()
        refiner = EnsemblePathRefinerFast(hidden_dim=NN_HIDDEN_DIM, device=NN_DEVICE)
        smooth_paths_3d = [p[:, np.newaxis, :] for p in smooth_paths]

        refiner.fit(
            trajectories=smooth_paths_3d,
            epochs=NN_EPOCHS,
            start=initial_path[0][np.newaxis, :] * 10.0,
            end=initial_path[-1][np.newaxis, :] * 10.0,
            verbosity=True,
            patience=NN_PATIENCE,
            batch_size=NN_BATCH_SIZE,
            lr=NN_LR,
        )
        t_nn = time.perf_counter() - t1

        refined_3d = refiner.transform(n_points=N_PATH_IMAGES)
        current_path = refined_3d[:, 0, :]
        # Pin endpoints to exact global minima in Angstroms
        current_path[0]  = initial_path[0] * 10.0
        current_path[-1] = initial_path[-1] * 10.0

        path_history.append(current_path.copy() / 10.0)

        diff = np.max(np.linalg.norm(path_history[-1] - path_history[-2], axis=1))
        print(f"  NN training: {t_nn:.1f}s  |  Max path update: {diff:.4f}\n")

        # Save per-iteration snapshot in nanometers
        np.save(os.path.join(out_dir, f"path_iter_{it + 1}.npy"), current_path / 10.0)

    # ── Save final results ────────────────────────────────────────────────────
    print("Saving final results...")
    np.save(os.path.join(out_dir, "final_path.npy"), current_path / 10.0)
    np.savez(
        os.path.join(out_dir, "path_history.npz"),
        **{f"iter_{i}": p for i, p in enumerate(path_history)},
    )
    print(f"Done. Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
