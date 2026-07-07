import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from pathrefinement.pathcv import PathCV
from pathrefinement.principal_curve import PrincipalCurve
from common import load_coords, get_calpha_indices, compute_pairwise_distances

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # portable: this scripts dir
    prmtop_file = os.path.join(base_dir, "chignolin.prmtop")
    
    in_pdb = os.path.join(base_dir, "results", "initial_path", "initial_path.pdb")
    
    print("Loading initial trajectory...")
    traj_nm = load_coords(in_pdb)
    traj_angstrom = traj_nm * 10.0
    
    calpha_indices = get_calpha_indices(prmtop_file)
    def feature_fn(coords):
        return compute_pairwise_distances(coords, calpha_indices)

    print("Computing features...")
    raw_features = np.array([feature_fn(pt) for pt in traj_angstrom])
    
    print("Smoothing into 20 reference nodes...")
    pc = PrincipalCurve(n_images=20, verbose=False)
    ref_features = pc.fit(raw_features)
    ref_features[0] = raw_features[0]
    ref_features[-1] = raw_features[-1]

    path_cv = PathCV(
        ref_features[:, np.newaxis, :],
        enforce_equidistance=True,
        normalize_output=True,
    )
    
    print("\nEvaluating (s, z) along the initial trajectory...")
    s_vals = []
    z_vals = []
    # Sample every 50th frame to keep output clean
    for i in range(0, len(raw_features), 50):
        feat = raw_features[i]
        s, z = path_cv.compute(feat[np.newaxis, :])
        s_vals.append(s)
        z_vals.append(z)
        print(f"Frame {i:4d}: s = {s:.3f}, z = {z:.3f}")

    print(f"\nAverage z: {np.mean(z_vals):.3f}")
    print(f"Max z:     {np.max(z_vals):.3f}")
    print(f"Min z:     {np.min(z_vals):.3f}")

if __name__ == "__main__":
    main()
