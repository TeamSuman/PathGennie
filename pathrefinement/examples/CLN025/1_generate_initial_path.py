import os
import sys
import numpy as np

# Allow running this example out-of-the-box without installing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from pathrefinement.pathcv import PathCV
from pathgennie.backends.openmm import PathGennieMD
from common import load_coords, create_chignolin_system, get_calpha_indices, compute_pairwise_distances, save_path_pdb

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # portable: this scripts dir
    prmtop_file = os.path.join(base_dir, "chignolin.prmtop")
    unfolded_file = os.path.join(base_dir, "chignolin_unfolded.pdb")
    folded_file = os.path.join(base_dir, "chignolin_folded.pdb")
    
    out_dir = os.path.join(os.path.dirname(__file__), "results", "initial_path")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading structure and topology...")
    unfolded_coords = load_coords(unfolded_file)
    folded_coords = load_coords(folded_file)
    calpha_indices = get_calpha_indices(prmtop_file)
    
    def feature_fn(coords):
        return compute_pairwise_distances(coords, calpha_indices)

    simulation = create_chignolin_system(prmtop_file)
    
    print("Setting up PathGennie initial path generator...")
    
    import MDAnalysis.analysis.rms as rms
    
    # Target coordinates for RMSD (C-alpha only)
    target_calpha = folded_coords[calpha_indices] * 10.0 # Angstroms
    target_calpha_centered = target_calpha - np.mean(target_calpha, axis=0)
    target_rg = np.sqrt(np.mean(np.sum(target_calpha_centered**2, axis=1)))

    def project_fn(coords):
        # coords are in Angstroms inside project_fn
        calpha = coords[calpha_indices]
        # Use MDAnalysis to compute optimal RMSD
        rmsd_val = rms.rmsd(calpha, target_calpha, center=True, superposition=True)
        
        calpha_centered = calpha - np.mean(calpha, axis=0)
        rg = np.sqrt(np.mean(np.sum(calpha_centered**2, axis=1)))
        return np.array([rmsd_val, rg])

    def converge_fn(coords, target):
        rmsd = project_fn(coords)[0]
        # Converged if RMSD is less than 1.5 Angstroms
        return float(rmsd) < 1.5
        
    target_cv = np.array([0.0, target_rg])
    
    pathgennie = PathGennieMD(
        simulation=simulation,
        projection_fn=project_fn,
        projection_args={},
        mode="target",
        target_projection=target_cv,
        convergence_fn=converge_fn,
        convergence_args={"target": target_cv},
        temperature=275.0,
    )
    
    print("Running PathGennie to reach target state...")
    traj, _ = pathgennie.run(
        initial_pos=unfolded_coords,
        tau1=5,
        tau2=10,
        max_trial=10,
        max_cycle=5000,
        verbosity=1,
        save_freq=1,
    )
    
    if len(traj) == 0:
        raise RuntimeError("Failed to generate initial path with PathGennie.")
        
    print(f"Generated PathGennie trajectory of length {len(traj)} frames.")
    
    # Convert trajectory to nanometers for PDB saving
    if traj.ndim == 4 and traj.shape[1] == 1:
        traj_nm = traj[:, 0, :, :] / 10.0
    else:
        traj_nm = traj / 10.0
        
    out_pdb = os.path.join(out_dir, "initial_path.pdb")
    save_path_pdb(unfolded_file, traj_nm, out_pdb)
    print(f"Saved initial trajectory to {out_pdb}")

if __name__ == "__main__":
    main()
