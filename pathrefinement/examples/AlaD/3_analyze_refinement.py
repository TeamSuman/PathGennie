import os
import sys
import glob
import numpy as np
import yaml
import matplotlib.pyplot as plt

# Allow running this example out-of-the-box without installing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from common import load_coords, phi_psi_cv

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    config_file = os.path.join(base_dir, "config.yaml")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
    refine_dir = os.path.join(base_dir, config["paths"]["output_dir"])
    initial_npy = os.path.join(base_dir, config["paths"]["initial_path_npy"])
    initial_dcd = os.path.join(base_dir, config["paths"]["initial_path_dcd"])
    prmtop_file = os.path.join(base_dir, "ala_dipeptide.prmtop")
    
    if not os.path.exists(initial_npy):
        print(f"Error: Initial path features file {initial_npy} not found. Please run 1_generate_initial_path.py first.")
        return
        
    print("Loading path features (2D dihedrals in degrees)...")
    initial_path = np.load(initial_npy)
    if np.max(np.abs(initial_path)) < 2 * np.pi:
        initial_path = np.degrees(initial_path)
    
    path_files = sorted(glob.glob(os.path.join(refine_dir, "path_iter_*.npy")))
    paths = [initial_path]
    labels = ["Initial Path"]
    
    for f in path_files:
        path = np.load(f)
        if np.max(np.abs(path)) < 2 * np.pi:
            path = np.degrees(path)
        paths.append(path)
        it_num = f.split("_iter_")[-1].split(".npy")[0]
        labels.append(f"Iteration {int(it_num)+1}")
        
    final_file = os.path.join(refine_dir, "final_path.npy")
    if os.path.exists(final_file) and final_file not in path_files:
        path = np.load(final_file)
        if np.max(np.abs(path)) < 2 * np.pi:
            path = np.degrees(path)
        paths.append(path)
        labels.append("Final Path")
        
    print(f"Loaded {len(paths)} paths.")
    
    # Load raw coordinates from initial trajectory for background plotting
    raw_dihedrals = None
    if os.path.exists(initial_dcd):
        print("Loading raw initial trajectory for background plotting...")
        traj_nm = load_coords(initial_dcd, topology_file=prmtop_file)
        traj_angstrom = traj_nm * 10.0
        raw_dihedrals = phi_psi_cv(traj_angstrom)
        
    plt.figure(figsize=(9, 8))
    
    # Plot background raw trajectory in phi/psi space if available
    if raw_dihedrals is not None:
        plt.scatter(raw_dihedrals[:, 0], raw_dihedrals[:, 1], color='lightgray', alpha=0.6, s=15, label='Raw Trajectory')
        
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(paths)))
    
    for i, (path, label) in enumerate(zip(paths, labels)):
        phi = path[:, 0]
        psi = path[:, 1]
        
        lw = 2.5 if i == 0 or i == len(paths)-1 else 1.5
        alpha = 1.0 if i == 0 or i == len(paths)-1 else 0.7
        ls = '--' if i == 0 else '-'
        marker = 'o' if i == len(paths)-1 else None
        
        plt.plot(
            phi, psi,
            color=colors[i], label=label,
            linewidth=lw, alpha=alpha, linestyle=ls,
            marker=marker, markersize=3
        )
        
        # Mark endpoints
        if i == len(paths) - 1:
            plt.scatter(phi[0], psi[0], color='green', marker='s', s=100, zorder=10, label='Start State ($C_{eq}$)')
            plt.scatter(phi[-1], psi[-1], color='red', marker='^', s=100, zorder=10, label='Target State ($\\alpha_L$)')
        elif i == 0:
            plt.scatter(phi[0], psi[0], color='green', marker='s', s=60, zorder=5)
            plt.scatter(phi[-1], psi[-1], color='red', marker='^', s=60, zorder=5)
            
    plt.title("AlaD Path Refinement in 2D $(\\phi, \\psi)$ Dihedral Space", fontsize=14)
    plt.xlabel("$\\phi$ (degrees)", fontsize=12)
    plt.ylabel("$\\psi$ (degrees)", fontsize=12)
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.grid(True, alpha=0.3)
    plt.axvline(0, color='black', lw=0.5, alpha=0.5)
    plt.axhline(0, color='black', lw=0.5, alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    out_img = os.path.join(refine_dir, "refinement_ramachandran.png")
    plt.savefig(out_img, dpi=300)
    print(f"Saved Ramachandran visualization to {out_img}")
    plt.close()
    
    # Plot Mean Squared Displacement Update between iterations (in degrees^2)
    if len(paths) > 1:
        plt.figure(figsize=(8, 5))
        msd_vals = []
        for i in range(1, len(paths)):
            phi_prev = paths[i-1][:, 0]
            psi_prev = paths[i-1][:, 1]
            phi_curr = paths[i][:, 0]
            psi_curr = paths[i][:, 1]
            
            diff_phi = (phi_curr - phi_prev + 180.0) % 360.0 - 180.0
            diff_psi = (psi_curr - psi_prev + 180.0) % 360.0 - 180.0
            
            diff = np.column_stack([diff_phi, diff_psi])
            msd = np.mean(np.sum(diff**2, axis=1))
            msd_vals.append(msd)
            
        plt.plot(range(1, len(msd_vals)+1), msd_vals, marker='o', linewidth=2, color='darkblue')
        plt.title("AlaD Path Convergence (MSD in Dihedral Space)")
        plt.xlabel("Iteration")
        plt.ylabel("Mean Squared Displacement Update (degrees$^2$)")
        plt.xticks(range(1, len(msd_vals)+1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_conv = os.path.join(refine_dir, "refinement_convergence.png")
        plt.savefig(out_conv, dpi=300)
        print(f"Saved convergence plot to {out_conv}")
        plt.close()

if __name__ == "__main__":
    main()
