import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from pathrefinement.pathcv import PathCV
from pathrefinement.principal_curve import PrincipalCurve
from pathrefinement.ensemblerefiner import EnsemblePathRefinerFast
from pathgennie.backends.openmm import PathGennieMD
from common import load_coords, create_chignolin_system, get_calpha_indices, compute_pairwise_distances

def main():
    base_dir = "/home/dm/Dibyendu/Projects/GitHub/PathGennie/pathrefinement/examples/CLN025/"
    prmtop_file = os.path.join(base_dir, "chignolin.prmtop")
    
    in_pdb = os.path.join(os.path.dirname(__file__), "results", "initial_path", "initial_path.pdb")
    if not os.path.exists(in_pdb):
        print(f"Error: {in_pdb} not found. Please run 1_generate_initial_path.py first.")
        return

    print("Loading initial trajectory...")
    traj_nm = load_coords(in_pdb)
    
    # PathGennie generated coords. PathGennie expects Angstroms inside projection_fn
    # so we convert nm back to Angstroms for feature_fn
    traj_angstrom = traj_nm * 10.0
    
    calpha_indices = get_calpha_indices(prmtop_file)
    def feature_fn(coords):
        return compute_pairwise_distances(coords, calpha_indices)

    print("Smoothing raw trajectory into 20 reference nodes...")
    raw_features = np.array([feature_fn(pt) for pt in traj_angstrom])
    pc = PrincipalCurve(n_images=100, verbose=False)
    initial_path_features = pc.fit(raw_features)
    
    # Pin endpoints to the true start and end of the trajectory
    initial_path_features[0] = raw_features[0]
    initial_path_features[-1] = raw_features[-1]

    simulation = create_chignolin_system(prmtop_file)
    
    path_cv = PathCV(
        initial_path_features[:, np.newaxis, :],
        enforce_equidistance=True,
        normalize_output=True,
    )
    
    def project_fn(coords, path_cv):
        features = feature_fn(coords)
        s, z = path_cv.compute(np.atleast_2d(features))
        return np.array([s])

    def converge_fn(coords, path_cv, target):
        features = feature_fn(coords)
        s, z = path_cv.compute(np.atleast_2d(features))
        return float(s) >= float(target[0]) - 0.05
        
    target_cv = np.array([1.0])
    
    n_iterations = 10
    n_trajectories = 10
    current_path_features = initial_path_features
    path_history = [current_path_features]
    
    out_dir = os.path.join(os.path.dirname(__file__), "results", "refinement")
    os.makedirs(out_dir, exist_ok=True)
    
    unfolded_coords = traj_nm[0] # Use the exact start from trajectory

    for it in range(n_iterations):
        print(f"\n=== Refinement Iteration {it + 1}/{n_iterations} ===")
        
        pathgennie = PathGennieMD(
            simulation=simulation,
            projection_fn=project_fn,
            projection_args={"path_cv": path_cv},
            mode="target",
            target_projection=target_cv,
            convergence_fn=converge_fn,
            convergence_args={"path_cv": path_cv, "target": target_cv},
            temperature=275.0,
            sigma=0.05,
        )
        
        traj_list = []
        for _ in range(n_trajectories):
            print("  Running PathGennie trajectory...")
            traj, _ = pathgennie.run(
                initial_pos=unfolded_coords,
                tau1=20,
                tau2=20,
                max_trial=10,
                max_cycle=5000,
                verbosity=1,
            )
            if len(traj) > 0:
                print(f"  Generated trajectory of length {len(traj)}")
                traj_features = np.array([feature_fn(pt) for pt in traj])
                traj_list.append(traj_features)
                
        if not traj_list:
            print("No trajectories generated. Stopping.")
            break
            
        print("Smoothing trajectories...")
        smooth_paths = []
        for traj_feat in traj_list:
            smooth_path = pc.fit(traj_feat)
            smooth_paths.append(smooth_path)
            
        print("Learning consensus path with EnsembleRefiner...")
        refiner = EnsemblePathRefinerFast(hidden_dim=256, device="cuda")
        smooth_paths_3d = [p[:, np.newaxis, :] for p in smooth_paths]
        
        refiner.fit(
            trajectories=smooth_paths_3d,
            epochs=1000,
            start=current_path_features[0][np.newaxis, :],
            end=current_path_features[-1][np.newaxis, :],
            verbosity=1,
            patience=50,
        )
        
        refined_features_3d = refiner.transform(n_points=100)
        current_path_features = refined_features_3d[:, 0, :]
        current_path_features[0] = initial_path_features[0]
        current_path_features[-1] = initial_path_features[-1]
        
        path_history.append(current_path_features.copy())
        
        path_cv = PathCV(
            current_path_features[:, np.newaxis, :],
            enforce_equidistance=True,
            normalize_output=True,
        )
        
        diff = np.max(np.linalg.norm(current_path_features - path_history[-2], axis=1))
        print(f"Max path feature update: {diff:.4f}")
        
        np.save(os.path.join(out_dir, f"path_iter_{it}.npy"), current_path_features)

    print("Saving final results...")
    np.save(os.path.join(out_dir, "final_path.npy"), current_path_features)
    print("Done.")

if __name__ == "__main__":
    main()
