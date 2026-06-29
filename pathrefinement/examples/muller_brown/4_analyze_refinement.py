"""
4_analyze_refinement.py
───────────────────────
Visualize the iterative Muller-Brown path refinement results.
Mirrors the structure of CLN025/4_analyze_refinement.py.

Produces two plots:
  1. refinement_iterations.png  — all paths overlaid on the potential surface
  2. refinement_convergence.png — per-iteration MSD (convergence indicator)

Prerequisites
─────────────
Run 1_generate_initial_path.py and 3_run_refinement.py first.
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from common import MINIMA, muller_brown_energy

BASE_DIR = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description="Analyze Path Refinement")
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

paths_cfg = cfg.get("paths", {})
def resolve_path(p, default):
    val = paths_cfg.get(p, default)
    if os.path.isabs(val):
        return val
    return os.path.join(BASE_DIR, val)

init_file = resolve_path("initial_path_npy", "results/initial_path/initial_path.npy")
INIT_DIR = os.path.dirname(init_file)
REFINE_DIR = resolve_path("output_dir", "results/refinement")


def load_all_paths():
    """Load initial path, all per-iteration snapshots, and final path."""
    init_npy = init_file
    if not os.path.exists(init_npy):
        raise FileNotFoundError(f"Initial path not found: {init_npy}\n"
                                "Run 1_generate_initial_path.py first.")

    paths  = [np.load(init_npy)]
    labels = ["Initial (linear)"]

    iter_files = sorted(
        glob.glob(os.path.join(REFINE_DIR, "path_iter_*.npy")),
        key=lambda f: int(os.path.basename(f).replace("path_iter_", "").replace(".npy", "")),
    )
    for f in iter_files:
        it_num = int(os.path.basename(f).replace("path_iter_", "").replace(".npy", ""))
        paths.append(np.load(f))
        labels.append(f"Iter {it_num}")

    return paths, labels


def plot_surface_background(ax):
    """Draw the Muller-Brown energy surface as a filled contour."""
    x = np.linspace(-1.8, 1.2, 300)
    y = np.linspace(-0.5, 2.2, 300)
    X, Y = np.meshgrid(x, y)
    Z = muller_brown_energy(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    Z = np.clip(Z, -200, 300)
    ax.contourf(X, Y, Z, levels=40, cmap="RdYlGn_r", alpha=0.75)
    ax.contour(X, Y, Z, levels=40, colors="k", linewidths=0.25, alpha=0.35)
    return ax


def plot_trajectories_debug():
    """Plot all raw trajectories generated in each iteration in subplots."""
    traj_dirs = sorted(
        glob.glob(os.path.join(REFINE_DIR, "trajs", "iter_*")),
        key=lambda d: int(os.path.basename(d).replace("iter_", "")),
    )
    if not traj_dirs:
        print("No raw trajectories found to plot.")
        return

    n_iters = len(traj_dirs)
    ncols = min(3, n_iters)
    nrows = (n_iters + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    for idx, d in enumerate(traj_dirs):
        ax = axes[idx]
        it_num = int(os.path.basename(d).replace("iter_", ""))
        
        plot_surface_background(ax)
        
        if it_num == 1:
            prev_path_file = init_file
        else:
            prev_path_file = os.path.join(REFINE_DIR, f"path_iter_{it_num - 1}.npy")
            
        if os.path.exists(prev_path_file):
            prev_path = np.load(prev_path_file)
            ax.plot(prev_path[:, 0], prev_path[:, 1], color="black", lw=1.5, ls="--", label="Guiding Path", alpha=0.7)

        traj_files = glob.glob(os.path.join(d, "traj_*.npy"))
        first_traj = True
        for tf in traj_files:
            traj = np.load(tf)
            label = "Raw Trajs" if first_traj else None
            ax.plot(traj[:, 0], traj[:, 1], color="gray", lw=0.5, alpha=0.4, label=label)
            first_traj = False

        curr_path_file = os.path.join(REFINE_DIR, f"path_iter_{it_num}.npy")
        if os.path.exists(curr_path_file):
            curr_path = np.load(curr_path_file)
            ax.plot(curr_path[:, 0], curr_path[:, 1], color="blue", lw=2, label="Refined Path")

        ax.scatter(*MINIMA["A"], c="lime", s=60, zorder=10, edgecolors="k")
        ax.scatter(*MINIMA["B"], c="red",  s=60, zorder=10, edgecolors="k")
        
        ax.set_title(f"Iteration {it_num}", fontsize=11, fontweight="bold")
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    for idx in range(n_iters, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle("Raw Walkers Trajectories per Refinement Iteration", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    out3 = os.path.join(REFINE_DIR, "refinement_trajectories.png")
    plt.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out3}")




def plot_sz_progress():
    """Plot the trajectories of each iteration in (s, z) space."""
    traj_dirs = sorted(
        glob.glob(os.path.join(REFINE_DIR, "trajs", "iter_*")),
        key=lambda d: int(os.path.basename(d).replace("iter_", "")),
    )
    if not traj_dirs:
        print("No raw trajectories found to plot in s,z space.")
        return

    n_iters = len(traj_dirs)
    ncols = min(3, n_iters)
    nrows = (n_iters + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    for idx, d in enumerate(traj_dirs):
        ax = axes[idx]
        it_num = int(os.path.basename(d).replace("iter_", ""))
        
        if it_num == 1:
            prev_path_file = init_file
        else:
            prev_path_file = os.path.join(REFINE_DIR, f"path_iter_{it_num - 1}.npy")
            
        if not os.path.exists(prev_path_file):
            print(f"Guiding path for iteration {it_num} not found. Skipping.")
            continue
            
        prev_path = np.load(prev_path_file)
        from pathrefinement.pathcv import PathCV
        path_cv = PathCV(
            prev_path[:, np.newaxis, :],
            enforce_equidistance=False,
            normalize_output=True,
        )

        traj_files = glob.glob(os.path.join(d, "traj_*.npy"))
        first_traj = True
        for tf in traj_files:
            traj = np.load(tf)
            
            s_vals = []
            z_vals = []
            for pt in traj:
                s, z = path_cv.compute(pt[np.newaxis, :])
                s_vals.append(s)
                z_vals.append(z)
                
            label = "Walkers" if first_traj else None
            ax.plot(s_vals, z_vals, color="purple", lw=0.6, alpha=0.4, label=label)
            first_traj = False

        curr_path_file = os.path.join(REFINE_DIR, f"path_iter_{it_num}.npy")
        if os.path.exists(curr_path_file):
            curr_path = np.load(curr_path_file)
            s_path = []
            z_path = []
            for pt in curr_path:
                s, z = path_cv.compute(pt[np.newaxis, :])
                s_path.append(s)
                z_path.append(z)
            ax.plot(s_path, z_path, color="red", lw=2, label="Refined Path")

        ax.axhline(0, color="black", lw=1.5, ls="--", label="Guiding Path (z=0)")
        
        ax.set_title(f"Iteration {it_num}", fontsize=11, fontweight="bold")
        ax.set_xlabel("s (progress)", fontsize=9)
        ax.set_ylabel("z (distance)", fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    for idx in range(n_iters, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle("Walkers and Refined Paths in (s, z) Space", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    out4 = os.path.join(REFINE_DIR, "refinement_sz.png")
    plt.savefig(out4, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out4}")


def main():
    print("Loading paths...")
    paths, labels = load_all_paths()
    print(f"  Loaded {len(paths)} paths  ({labels[0]} → {labels[-1]})")

    n_paths = len(paths)
    colors  = plt.cm.cool(np.linspace(0.0, 1.0, n_paths))

    # ── Plot 1: All paths overlaid on the surface ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_surface_background(ax)

    for i, (path, label) in enumerate(zip(paths, labels)):
        is_first = (i == 0)
        is_last  = (i == n_paths - 1)
        lw    = 2.5 if is_last  else (1.5 if is_first else 1.0)
        alpha = 1.0 if is_last  else (0.8 if is_first else 0.45 + 0.4 * i / n_paths)
        ls    = "--" if is_first else "-"
        color = "white" if is_first else colors[i]

        ax.plot(path[:, 0], path[:, 1],
                color=color, lw=lw, alpha=alpha, ls=ls,
                label=label, zorder=3 + i)

    # Endpoints (global minima)
    ax.scatter(*MINIMA["A"], c="lime", s=120, zorder=10, edgecolors="k", label="Min A")
    ax.scatter(*MINIMA["B"], c="red",  s=120, zorder=10, edgecolors="k", label="Min B")

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Muller-Brown Path Refinement — All Iterations", fontsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, framealpha=0.9)
    plt.tight_layout()

    out1 = os.path.join(REFINE_DIR, "refinement_iterations.png")
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out1}")

    # ── Plot 2: Convergence (MSD between consecutive iterations) ──────────────
    if n_paths > 1:
        msd_vals   = []
        max_d_vals = []
        for i in range(1, n_paths):
            diff = paths[i] - paths[i - 1]
            msd_vals.append(np.mean(np.sum(diff ** 2, axis=1)))
            max_d_vals.append(np.max(np.linalg.norm(diff, axis=1)))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(range(1, len(msd_vals) + 1), msd_vals,
                     marker="o", lw=2, color="steelblue")
        axes[0].set_title("Mean Squared Displacement per Iteration")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("MSD")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(range(1, len(msd_vals) + 1))

        axes[1].plot(range(1, len(max_d_vals) + 1), max_d_vals,
                     marker="s", lw=2, color="tomato")
        axes[1].set_title("Max Path Point Displacement per Iteration")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Max ‖Δpath‖")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(range(1, len(max_d_vals) + 1))

        plt.suptitle("Path Refinement Convergence", fontsize=13, fontweight="bold")
        plt.tight_layout()

        out2 = os.path.join(REFINE_DIR, "refinement_convergence.png")
        plt.savefig(out2, dpi=200)
        plt.close()
        print(f"Saved → {out2}")

    # ── Plot 3: Debug trajectories per iteration ─────────────────────────────
    plot_trajectories_debug()

    # ── Plot 4: s,z space progress ───────────────────────────────────────────
    plot_sz_progress()

    print("Done.")


if __name__ == "__main__":
    main()


