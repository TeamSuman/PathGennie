"""
1_generate_initial_path.py
─────────────────────────
Generate an initial path between the two global minima of the Muller-Brown
potential using PathGennieMD to sample the transition pathway, then resample
and save it to results/initial_path/.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

import argparse
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from pathgennie.backends.openmm import PathGennieMD
from pathrefinement.principal_curve import PrincipalCurve
from common import MINIMA, create_mb_system, muller_brown_energy


def resample_path(path: np.ndarray, n_points: int = 100) -> np.ndarray:
    """Resample a 2D path to n_points equidistant points."""
    dists = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
    cum_dists = np.concatenate(([0.0], np.cumsum(dists)))
    t = np.linspace(0, cum_dists[-1], n_points)
    x = np.interp(t, cum_dists, path[:, 0])
    y = np.interp(t, cum_dists, path[:, 1])
    return np.column_stack((x, y))


BASE_DIR = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description="Generate Initial Path")
parser.add_argument("--config", default=os.path.join(BASE_DIR, "config.yaml"), help="Path to YAML config file")
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

ref_cfg = cfg.get("refinement", {})
N_PATH_IMAGES = ref_cfg.get("n_path_images", 100)

paths_cfg = cfg.get("paths", {})
INITIAL_PATH_NPY = os.path.join(BASE_DIR, paths_cfg.get("initial_path_npy", "results/initial_path/initial_path.npy"))


def main():
    out_dir = os.path.dirname(INITIAL_PATH_NPY)
    os.makedirs(out_dir, exist_ok=True)

    print("Global minima:")
    print(f"  A = {MINIMA['A']}")
    print(f"  B = {MINIMA['B']}")

    print("\nSetting up PathGennie initial path generator...")
    # Setup OpenMM simulation for MB potential
    simulation = create_mb_system(seed=42)

    # We want to steer coordinates (in Angstroms) towards target B (in Angstroms)
    target_pos = MINIMA["B"] * 10.0  # Angstroms

    def project_fn(coords):
        # coords is (1, 3) in Angstroms
        return coords[0, :2]

    def converge_fn(coords, target):
        # target is target_pos in Angstroms
        dist = np.linalg.norm(coords[0, :2] - target)
        # Converge when within 0.15 nm (1.5 Angstroms) of B
        return float(dist) < 1.5

    pathgennie = PathGennieMD(
        simulation=simulation,
        projection_fn=project_fn,
        projection_args={},
        mode="target",
        target_projection=target_pos,
        convergence_fn=converge_fn,
        convergence_args={"target": target_pos},
        temperature=300.0,
        sigma=0.1,
    )

    # Initial position is State A in nanometers (OpenMM uses nanometers)
    start_pos = np.zeros((1, 3))
    start_pos[0, :2] = MINIMA["A"]

    print("Running PathGennie to sample transition trajectory A -> B...")
    traj, _ = pathgennie.run(
        initial_pos=start_pos,
        tau1=10,
        tau2=20,
        max_trial=15,
        max_cycle=1000,
        save_freq=1,
        verbosity=1,
    )

    if len(traj) == 0:
        raise RuntimeError("Failed to generate initial path with PathGennie.")

    # Convert trajectory to Muller-Brown units (nm)
    # traj is (N_cycles, 1, 3) in Angstroms
    raw_path_from_traj = traj[:, 0, :2] / 10.0
    # Ensure endpoints are exactly A and B in raw_path
    raw_path = np.vstack([MINIMA["A"], raw_path_from_traj, MINIMA["B"]])

    # Refine and smooth the initial path using PrincipalCurve to ensure it is equidistant
    print(f"Enforcing equidistance and smoothing path with PrincipalCurve ({N_PATH_IMAGES} images)...")
    pc = PrincipalCurve(n_images=N_PATH_IMAGES, lam=0.25, n_iter=50)
    initial_path = pc.fit(raw_path)

    out_npy = INITIAL_PATH_NPY
    np.save(out_npy, initial_path)
    print(f"Saved initial path to → {out_npy} (shape {initial_path.shape})")

    # ── Visualisation ──────────────────────────────────────────────────
    x = np.linspace(-1.8, 1.2, 300)
    y = np.linspace(-0.5, 2.2, 300)
    X, Y = np.meshgrid(x, y)
    Z = muller_brown_energy(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    Z = np.clip(Z, -200, 300)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(X, Y, Z, levels=40, cmap="RdYlGn_r", alpha=0.8)
    ax.contour(X, Y, Z, levels=40, colors="k", linewidths=0.3, alpha=0.4)
    
    # Plot raw trajectory
    ax.plot(raw_path[:, 0], raw_path[:, 1], "gray", lw=1, alpha=0.6, label="Raw PathGennie Traj")
    # Plot resampled initial path
    ax.plot(initial_path[:, 0], initial_path[:, 1], "w--", lw=2, label="Resampled Initial path")
    
    ax.scatter(*MINIMA["A"], c="lime", s=120, zorder=5, edgecolors="k", label="Minimum A")
    ax.scatter(*MINIMA["B"], c="red",  s=120, zorder=5, edgecolors="k", label="Minimum B")
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Muller-Brown — PathGennie Initial Path")
    ax.legend(fontsize=9)
    plt.tight_layout()

    plot_file = os.path.join(out_dir, "initial_path.png")
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Saved → {plot_file}")


if __name__ == "__main__":
    main()
