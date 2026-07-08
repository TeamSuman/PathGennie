import os
import sys
import numpy as np
import yaml
import openmm.unit as unit

# Allow running this example out-of-the-box without installing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from pathrefinement.pathcv import PathCV
from pathrefinement.principal_curve import PrincipalCurve
from pathrefinement.ensemblerefiner import EnsemblePathRefinerFast
from pathgennie.backends.openmm import PathGennieMD
from common import (
    load_coords,
    create_alad_system,
    phi_psi_cv,
    reached_phi_psi,
)

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    config_file = os.path.join(base_dir, "config.yaml")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
    gro_file = os.path.join(base_dir, "start.gro")
    top_file = os.path.join(base_dir, "topol.top")
    
    initial_npy = os.path.join(base_dir, config["paths"]["initial_path_npy"])
    if not os.path.exists(initial_npy):
        print(f"Error: {initial_npy} not found. Please run 1_generate_initial_path.py first.")
        return
        
    print("Loading initial path features...")
    initial_path_features = np.load(initial_npy) # shape (100, 2)
    # Convert path features from degrees to radians for a physical scale of z
    initial_path_features = np.radians(initial_path_features)
    print(f"Loaded path features with shape {initial_path_features.shape}")
    
    # Refine initial path using neural network refiner to ensure it is smooth
    print("Refining initial path using neural network refiner...")
    nn_conf = config["refiner"]
    initial_refiner = EnsemblePathRefinerFast(
        hidden_dim=nn_conf["hidden_dim"],
        device=nn_conf["device"]
    )
    initial_path_3d = initial_path_features[:, np.newaxis, :]
    initial_refiner.fit(
        trajectories=[initial_path_3d],
        epochs=nn_conf["epochs"],
        lr=nn_conf["lr"],
        start=initial_path_features[0][np.newaxis, :],
        end=initial_path_features[-1][np.newaxis, :],
        verbosity=1,
        patience=nn_conf["patience"],
        batch_size=nn_conf["batch_size"]
    )
    refined_initial_3d = initial_refiner.transform(n_points=config["principal_curve"]["n_images"])
    initial_path_features = refined_initial_3d[:, 0, :]
    # Pin endpoints exactly to the original initial path endpoints in radians
    initial_path_features[0] = np.radians(np.load(initial_npy)[0])
    initial_path_features[-1] = np.radians(np.load(initial_npy)[-1])
    print("Initial path refinement completed.")
    
    # Initialize PathCV in 2D space (radians)
    path_cv = PathCV(
        initial_path_features[:, np.newaxis, :], # shape (100, 1, 2)
        enforce_equidistance=False,
        normalize_output=True,
    )
    
    def feature_fn(coords):
        return np.radians(phi_psi_cv(coords))
        
    def project_fn(coords, path_cv):
        features = feature_fn(coords)
        s, z = path_cv.compute(np.atleast_2d(features))
        return np.array([s, z])
        
    def converge_fn(coords, path_cv, target):
        features = feature_fn(coords)
        s, z = path_cv.compute(np.atleast_2d(features))
        return float(s) >= float(target[0]) - 0.05
        
    target_cv = np.array([1.0, 0.0])
    
    print("Creating OpenMM simulation for walkers...")
    simulation = create_alad_system(gro_file, top_file)
    
    # Starting coordinates in nm
    start_coords = load_coords(gro_file)
    
    pc_conf = config["principal_curve"]
    pc = PrincipalCurve(
        n_images=pc_conf["n_images"],
        lam=pc_conf["lam"],
        n_iter=pc_conf["n_iter"],
        tol=pc_conf["tol"],
        verbose=pc_conf["verbose"]
    )
    
    n_iterations = config["refinement"]["n_iterations"]
    n_trajectories = config["refinement"]["n_trajectories"]
    current_path_features = initial_path_features
    path_history = [current_path_features]
    
    out_dir = os.path.join(base_dir, config["paths"]["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    
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
            temperature=config["pathgennie"]["temperature"],
            sigma=config["pathgennie"]["sigma"],
        )
        
        traj_list = []
        for traj_idx in range(n_trajectories):
            print(f"  Running PathGennie trajectory {traj_idx + 1}/{n_trajectories}...")
            traj, _ = pathgennie.run(
                initial_pos=start_coords,
                tau1=config["pathgennie"]["tau1"],
                tau2=config["pathgennie"]["tau2"],
                max_trial=config["pathgennie"]["max_trial"],
                max_cycle=config["pathgennie"]["max_cycle"],
                verbosity=1,
            )
            if len(traj) > 0:
                print(f"    Success: trajectory length = {len(traj)} frames.")
                traj_features = np.radians(phi_psi_cv(traj)) # shape (N_frames, 2)
                traj_list.append(traj_features)
            else:
                print(f"    Failed.")
                
        if not traj_list:
            print("No successful trajectories generated in this iteration. Stopping.")
            break
            
        print("Smoothing successful trajectories...")
        smooth_paths = []
        for traj_feat in traj_list:
            smooth_path = pc.fit(traj_feat)
            smooth_paths.append(smooth_path)
            
        print("Learning consensus path with EnsemblePathRefinerFast...")
        nn_conf = config["refiner"]
        refiner = EnsemblePathRefinerFast(
            hidden_dim=nn_conf["hidden_dim"],
            device=nn_conf["device"]
        )
        smooth_paths_3d = [p[:, np.newaxis, :] for p in smooth_paths]
        
        refiner.fit(
            trajectories=smooth_paths_3d,
            epochs=nn_conf["epochs"],
            lr=nn_conf["lr"],
            start=current_path_features[0][np.newaxis, :],
            end=current_path_features[-1][np.newaxis, :],
            verbosity=1,
            patience=nn_conf["patience"],
            batch_size=nn_conf["batch_size"]
        )
        
        refined_features_3d = refiner.transform(n_points=pc_conf["n_images"])
        current_path_features = refined_features_3d[:, 0, :]
        current_path_features[0] = initial_path_features[0]
        current_path_features[-1] = initial_path_features[-1]
        
        path_history.append(current_path_features.copy())
        
        # Update PathCV
        path_cv = PathCV(
            current_path_features[:, np.newaxis, :],
            enforce_equidistance=False,
            normalize_output=True,
        )
        
        # Calculate MSD difference accounting for periodic wrapping in diffs (in radians)
        phi_prev = path_history[-2][:, 0]
        psi_prev = path_history[-2][:, 1]
        phi_curr = current_path_features[:, 0]
        psi_curr = current_path_features[:, 1]
        
        diff_phi = (phi_curr - phi_prev + np.pi) % (2.0 * np.pi) - np.pi
        diff_psi = (psi_curr - psi_prev + np.pi) % (2.0 * np.pi) - np.pi
        diff = np.column_stack([diff_phi, diff_psi])
        
        diff_deg = np.degrees(diff)
        max_update = np.max(np.linalg.norm(diff_deg, axis=1))
        print(f"Max path update (degrees): {max_update:.4f}")
        
        np.save(os.path.join(out_dir, f"path_iter_{it}.npy"), current_path_features)
        
    print("Saving final refined path features...")
    np.save(os.path.join(out_dir, "final_path.npy"), current_path_features)
    print("Refinement completed successfully.")

if __name__ == "__main__":
    main()
