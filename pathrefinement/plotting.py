import matplotlib.pyplot as plt
import numpy as np

from .potentials import Potential2D
from .refiner import RefinementResult


def plot_potential_surface(potential: Potential2D, ax: plt.Axes, xlim=(-2, 2), ylim=(-2, 2)):
    """Plot contour map of the potential energy surface."""
    X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    Z = potential.energy_surface(X, Y)
    
    # Clip Z for better visualization if there are steep walls
    Z_min = np.min(Z)
    levels = np.linspace(Z_min, Z_min + 50, 20)
    
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.8, extend="max")
    ax.contour(X, Y, Z, levels=levels, colors="white", alpha=0.3, linewidths=0.5)
    return cf


def plot_path(path: np.ndarray, ax: plt.Axes, **kwargs):
    """Plot a path overlay on the surface."""
    if path.shape[1] != 2:
        raise ValueError("Path must be 2D")
    ax.plot(path[:, 0], path[:, 1], marker="o", markersize=4, **kwargs)


def plot_initial_vs_refined(result: RefinementResult, potential: Potential2D, filename: str):
    """Save a side-by-side plot of initial and refined paths."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Estimate limits from paths
    all_pts = np.vstack(result.path_history)
    xlim = (np.min(all_pts[:, 0]) - 0.5, np.max(all_pts[:, 0]) + 0.5)
    ylim = (np.min(all_pts[:, 1]) - 0.5, np.max(all_pts[:, 1]) + 0.5)
    
    for ax, title, path in zip(
        axes,
        ["Initial Path", "Refined Path"],
        [result.initial_path, result.refined_path]
    ):
        cf = plot_potential_surface(potential, ax, xlim=xlim, ylim=ylim)
        plot_path(path, ax, color="red" if "Initial" in title else "white", 
                  label="Path", zorder=5)
        
        # Plot minima
        for name, pt in potential.minima.items():
            ax.plot(pt[0], pt[1], "r*", markersize=10, label=name if title == "Initial Path" else "")
            
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
    # Shrink current axis by 10% and put legend outside
    box = axes[0].get_position()
    axes[0].set_position([box.x0, box.y0, box.width * 0.9, box.height])
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.colorbar(cf, ax=axes, orientation="vertical", shrink=0.8, label="Energy")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_refinement_iterations(result: RefinementResult, potential: Potential2D, filename: str):
    """Save a plot showing the evolution of the path across iterations."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    all_pts = np.vstack(result.path_history)
    xlim = (np.min(all_pts[:, 0]) - 0.5, np.max(all_pts[:, 0]) + 0.5)
    ylim = (np.min(all_pts[:, 1]) - 0.5, np.max(all_pts[:, 1]) + 0.5)
    
    cf = plot_potential_surface(potential, ax, xlim=xlim, ylim=ylim)
    fig.colorbar(cf, ax=ax, label="Energy")
    
    n_iters = len(result.path_history)
    colors = plt.cm.copper(np.linspace(0, 1, n_iters))
    
    for i, path in enumerate(result.path_history):
        alpha = 0.3 + 0.7 * (i / max(1, n_iters - 1))
        label = "Evolution" if i == n_iters // 2 else ""
        ax.plot(path[:, 0], path[:, 1], color=colors[i], alpha=alpha, label=label)
        
    ax.plot(result.initial_path[:, 0], result.initial_path[:, 1], "r--", label="Initial")
    ax.plot(result.refined_path[:, 0], result.refined_path[:, 1], "w-o", markersize=4, label="Final")
    
    ax.legend()
    ax.set_title("Path Refinement Iterations")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
