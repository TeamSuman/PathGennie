import os
import sys
import numpy as np
import yaml

# Allow running this example out-of-the-box without installing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from common import load_coords, phi_psi_cv
from pathrefinement.principal_curve import PrincipalCurve

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    config_file = os.path.join(base_dir, "config.yaml")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
    prmtop_file = os.path.join(base_dir, "ala_dipeptide.prmtop")
    in_dcd = os.path.join(base_dir, config["paths"]["initial_path_dcd"])
    out_npy = os.path.join(base_dir, config["paths"]["initial_path_npy"])
    
    if not os.path.exists(in_dcd):
        raise FileNotFoundError(f"Initial trajectory DCD not found: {in_dcd}")
        
    print("Loading initial trajectory coordinates from DCD...")
    # load_coords returns shape (N_frames, N_atoms, 3) in nm
    traj_nm = load_coords(in_dcd, topology_file=prmtop_file)
    traj_angstrom = traj_nm * 10.0 # nm to Angstroms
    
    print("Computing 2D (phi, psi) dihedrals for each frame...")
    raw_features = phi_psi_cv(traj_angstrom) # shape (N_frames, 2)
    print(f"Loaded {len(raw_features)} frames of 2D coordinates.")
    
    print("Smoothing raw trajectory into 100 reference nodes using PrincipalCurve...")
    pc_conf = config["principal_curve"]
    pc = PrincipalCurve(
        n_images=pc_conf["n_images"],
        lam=pc_conf["lam"],
        n_iter=pc_conf["n_iter"],
        tol=pc_conf["tol"],
        verbose=pc_conf["verbose"]
    )
    initial_path_features = pc.fit(raw_features)
    
    # Pin endpoints
    initial_path_features[0] = raw_features[0]
    initial_path_features[-1] = raw_features[-1]
    
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, initial_path_features)
    print(f"Saved initial 2D path features to {out_npy}")

if __name__ == "__main__":
    main()
