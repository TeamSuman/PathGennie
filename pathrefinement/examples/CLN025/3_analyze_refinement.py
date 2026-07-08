import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from pathrefinement.principal_curve import PrincipalCurve
from common import load_coords, get_calpha_indices, compute_pairwise_distances

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # portable: this scripts dir
    prmtop_file = os.path.join(base_dir, "chignolin.prmtop")
    refine_dir = os.path.join(base_dir, "results", "refinement")
    init_pdb = os.path.join(base_dir, "results", "initial_path", "initial_path.pdb")
    
    if not os.path.exists(init_pdb):
        print(f"Error: Initial path {init_pdb} not found.")
        return

    print("Loading initial trajectory features...")
    calpha_indices = get_calpha_indices(prmtop_file)
    def feature_fn(coords):
        return compute_pairwise_distances(coords, calpha_indices)

    traj_nm = load_coords(init_pdb)
    raw_features = np.array([feature_fn(pt * 10.0) for pt in traj_nm])
    
    # Reconstruct the 20-node initial reference path
    pc = PrincipalCurve(n_images=100, verbose=False)
    initial_path = pc.fit(raw_features)
    initial_path[0] = raw_features[0]
    initial_path[-1] = raw_features[-1]

    # Load iterative paths
    path_files = sorted(glob.glob(os.path.join(refine_dir, "path_iter_*.npy")))
    paths = [initial_path]
    labels = ["Initial Path"]
    
    for f in path_files:
        paths.append(np.load(f))
        it_num = f.split("_iter_")[-1].split(".npy")[0]
        labels.append(f"Iteration {int(it_num)+1}")
        
    final_file = os.path.join(refine_dir, "final_path.npy")
    if os.path.exists(final_file) and final_file not in path_files:
        paths.append(np.load(final_file))
        labels.append("Final Path")

    print(f"Loaded {len(paths)} paths for visualization.")

    # Load folded state features for dRMSD
    import MDAnalysis.analysis.rms as rms
    folded_coords = load_coords(os.path.join(base_dir, "chignolin_folded.pdb"))
    # feature_fn extracts calpha internally and expects coords in Angstroms
    target_features = feature_fn(folded_coords * 10.0)

    def calc_rg_drmsd(features_array):
        # features_array is shape (N_frames, 45) in Angstroms
        N_atoms = len(calpha_indices)
        # Rg = sqrt( 1/N^2 * sum(d_ij^2) )
        rgs = np.sqrt(np.sum(features_array**2, axis=1) / (N_atoms**2))
        
        # dRMSD to target
        diff = features_array - target_features
        drmsds = np.sqrt(np.mean(diff**2, axis=1))
        
        return drmsds, rgs

    print("Computing Rg and dRMSD for all paths...")
    
    plt.figure(figsize=(10, 8))
    
    raw_drmsds, raw_rgs = calc_rg_drmsd(raw_features)
    
    # Plot the background raw trajectory
    plt.plot(raw_drmsds, raw_rgs, color='lightgray', alpha=0.5, label='Raw Initial Trajectory')
    plt.scatter(raw_drmsds[0], raw_rgs[0], color='black', marker='s', s=100, label='Unfolded', zorder=5)
    plt.scatter(raw_drmsds[-1], raw_rgs[-1], color='gold', marker='*', s=200, label='Folded Target', zorder=5)

    # Plot the paths
    colors = plt.cm.jet(np.linspace(0, 1, len(paths)))
    for i, (path, label) in enumerate(zip(paths, labels)):
        p_drmsds, p_rgs = calc_rg_drmsd(path)
        
        lw = 2 if i == 0 or i == len(paths)-1 else 1.5
        alpha = 1.0 if i == 0 or i == len(paths)-1 else 0.7
        ls = '--' if i == 0 else '-'
        
        plt.plot(p_drmsds, p_rgs, color=colors[i], label=label, 
                 linewidth=lw, alpha=alpha, linestyle=ls, marker='o', markersize=4)

    plt.title("PathGennie Refinement Iterations (dRMSD vs Rg)")
    plt.xlabel("Distance RMSD to Folded State ($\AA$)")
    plt.ylabel("Radius of Gyration, $R_g$ ($\AA$)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_img = os.path.join(refine_dir, "refinement_pca.png")
    plt.savefig(out_img, dpi=300)
    print(f"Saved visualization to {out_img}")
    
    # Also plot MSD between iterations to show convergence
    if len(paths) > 1:
        plt.figure(figsize=(8, 5))
        msd_vals = []
        for i in range(1, len(paths)):
            diff = paths[i] - paths[i-1]
            msd = np.mean(np.sum(diff**2, axis=1))
            msd_vals.append(msd)
            
        plt.plot(range(1, len(msd_vals)+1), msd_vals, marker='o', linewidth=2)
        plt.title("Path Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Mean Squared Displacement Update ($\AA^2$)")
        plt.xticks(range(1, len(msd_vals)+1))
        plt.grid(True, alpha=0.3)
        
        out_conv = os.path.join(refine_dir, "refinement_convergence.png")
        plt.savefig(out_conv, dpi=300)
        print(f"Saved convergence plot to {out_conv}")

if __name__ == "__main__":
    main()
