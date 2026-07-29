#!/usr/bin/env python3
import multiprocessing as mp
import sys
import warnings

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.align import rotation_matrix
from omm_1opj_mut import OpenMMRunner

from wepath import WESS

warnings.filterwarnings("ignore")

class PathCV:
    """
    Robust Path Collective Variables (s, z) for MD trajectory analysis.

    Implements Eqs. (8) and (9) of:
    Branduardi, Gervasio, Parrinello, JCP 126, 054103 (2007).

    Definitions:
      s = ( sum(k * exp(-lambda * MSD_k)) ) / sum(exp(-lambda * MSD_k))
      z = - (1/lambda) * ln( sum(exp(-lambda * MSD_k)) )

    where k is the frame index (1-based).

    IMPORTANT:
    - Frames MUST be aligned before passing to this class.
    - Frames MUST be approximately equidistant in MSD.
    - z is NOT a geometric distance; it is a 'soft' MSD distance.
    """

    def __init__(
        self,
        frames,
        *,
        mass_weights=None,
        enforce_equidistance=True,
        equidistance_tol=0.25,
        lambda_mode="auto",
        lambda_value=None,
        normalize_output=False,
    ):
        """
        Parameters
        ----------
        frames : list[np.ndarray]
            List of aligned frames, shape (N_atoms, 3)
        mass_weights : np.ndarray or None
            Optional mass weighting, shape (N_atoms,)
        enforce_equidistance : bool
            If True, raise error when path nodes are not equidistant
        equidistance_tol : float
            Relative tolerance on MSD spacing (std/mean)
        lambda_mode : {"auto", "manual"}
        lambda_value : float
            Required if lambda_mode == "manual"
        normalize_output : bool
            If False (default), s ranges from [1.0, N_frames].
            If True, s is scaled to [0.0, 1.0].
        """

        self.frames = frames
        self.P = len(frames)
        self.normalize_output = normalize_output

        if self.P < 2:
            raise ValueError("Path must contain at least two frames.")

        self.n_atoms = frames[0].shape[0]

        for f in frames:
            if f.shape != (self.n_atoms, 3):
                raise ValueError("All frames must have identical shape.")

        # Mass weighting
        if mass_weights is not None:
            if mass_weights.shape != (self.n_atoms,):
                raise ValueError("mass_weights must have shape (N_atoms,)")
            self.mass_weights = mass_weights / np.mean(mass_weights)
        else:
            self.mass_weights = None

        # Flatten reference path
        self.reference_path = np.array([f.reshape(-1) for f in frames])

        # Check equidistance
        self._check_equidistance(
            enforce=enforce_equidistance,
            tol=equidistance_tol,
        )

        # Lambda
        if lambda_mode == "auto":
            self.lam = self._compute_lambda()
        elif lambda_mode == "manual":
            if lambda_value is None or lambda_value <= 0:
                raise ValueError("lambda_value must be positive.")
            self.lam = float(lambda_value)
        else:
            raise ValueError("lambda_mode must be 'auto' or 'manual'.")

    # -----------------------------
    # Internal utilities
    # -----------------------------

    def _msd(self, diff):
        """
        Mean square displacement with optional mass weighting.
        """
        diff = diff.reshape(self.n_atoms, 3)
        sq = np.sum(diff ** 2, axis=1)
        if self.mass_weights is not None:
            sq *= self.mass_weights
        return np.mean(sq)

    def _check_equidistance(self, *, enforce, tol):
        diffs = np.diff(self.reference_path, axis=0)
        msds = np.array([self._msd(d) for d in diffs])

        mean = np.mean(msds)
        std = np.std(msds)

        if mean == 0:
            raise ValueError("Degenerate path: identical frames.")

        rel = std / mean

        if rel > tol:
            msg = (
                f"Path nodes are not equidistant (std/mean = {rel:.2f}). "
                "This violates assumptions of Branduardi et al."
            )
            if enforce:
                raise ValueError(msg)
            else:
                print("WARNING:", msg)

        self._mean_segment_msd = mean

    def _compute_lambda(self):
        """
        Compute auto-lambda based on the PLUMED definition:
        lambda = 2.3 * (N-1) / sum(|Xi - Xi+1|)
        """
        return 2.3 / self._mean_segment_msd

    # -----------------------------
    # Public API
    # -----------------------------

    def compute(self, coords):
        """
        Compute (s, z) for a configuration.

        Parameters
        ----------
        coords : np.ndarray, shape (N_atoms, 3)

        Returns
        -------
        s : float
            Position on path.
            Range [1.0, P] if normalize_output=False.
            Range [0.0, 1.0] if normalize_output=True.
        z : float
            'Soft' distance from path.
        """

        if coords.shape != (self.n_atoms, 3):
            raise ValueError("coords shape mismatch.")

        R = coords.reshape(-1)

        # MSD to each node: d(R, R_k)
        msd = np.array([
            self._msd(R - Ri) for Ri in self.reference_path
        ])

        # Exponentials: exp(-lambda * MSD)
        exponents = -self.lam * msd
        max_exp = np.max(exponents)
        weights = np.exp(exponents - max_exp)

        Z_partition = np.sum(weights)

        # s(R) calculation
        # 1-based indices: 1, 2, ..., P
        indices = np.arange(1, self.P + 1)
        s = np.sum(indices * weights) / Z_partition

        # Normalization (optional)
        if self.normalize_output:
            s = (s - 1.0) / (self.P - 1.0)

        # z(R) calculation
        z = -(1.0 / self.lam) * (np.log(Z_partition) + max_exp)

        return float(s), float(z)

    def compute_rms_distance(self, coords):
        """
        Convenience: RMS distance to path (approximate).
        """
        _, z = self.compute(coords)
        return np.sqrt(max(z, 0.0))

# ---------------------------------------------------------
# Helper function (Unchanged from your base code)
# ---------------------------------------------------------

def extract_coords_from_array(
    coords,
    *,
    ref_bb_coords,
    bb_indices,
    lig_indices,
    bb_masses=None,
):
    """
    Align a single coordinate frame to reference and extract ligand coords.

    Parameters
    ----------
    coords : np.ndarray
        Shape (N_atoms_total, 3)
    ref_bb_coords : np.ndarray
        Reference backbone coordinates, shape (N_bb, 3)
    bb_indices : np.ndarray
        Indices of backbone atoms in coords
    lig_indices : np.ndarray
        Indices of ligand atoms in coords
    bb_masses : np.ndarray or None
        Masses for backbone atoms (for weighted alignment)

    Returns
    -------
    lig_coords_aligned : np.ndarray
        Aligned ligand coordinates, shape (N_lig, 3)
    """

    # Extract backbone coordinates
    mob_bb = coords[bb_indices]

    # Center both
    mob_com = np.average(mob_bb, axis=0, weights=bb_masses)
    ref_com = np.average(ref_bb_coords, axis=0, weights=bb_masses)

    mob_bb_c = mob_bb - mob_com
    ref_bb_c = ref_bb_coords - ref_com

    # Compute rotation matrix (Kabsch, mass-weighted)
    R, rmsd = rotation_matrix(
        mob_bb_c,
        ref_bb_c,
        weights=bb_masses
    )

    # Apply rotation + translation to ALL atoms
    coords_aligned = (coords - mob_com) @ R.T + ref_com

    # Extract ligand
    lig_coords_aligned = coords_aligned[lig_indices]

    return lig_coords_aligned


def project(positions, kwargs):

    path_cv = kwargs['path_cv']
    ref_bb_coords = kwargs['ref_bb_coords']
    bb_indices = kwargs['bb_indices']
    lig_indices = kwargs['lig_indices']
    bb_masses = kwargs['bb_masses']

    lig_coords_aligned = extract_coords_from_array(
        10.0*positions,
        ref_bb_coords=ref_bb_coords,
        bb_indices=bb_indices,
        lig_indices=lig_indices,
        bb_masses=bb_masses,
    )

    s, z = path_cv.compute(lig_coords_aligned)
    return np.array([s, z])

def warp_criteria(positions, kwargs):
    # Unpack pre-calculated objects
    protein_indices = kwargs['protein_idx']
    ligand_indices = kwargs['ligand_idx']
    box = kwargs['box']
    positions = 10.0 * positions
    # Extract positions for protein and ligand
    prot_pos = positions[protein_indices]
    lig_pos = positions[ligand_indices]

    # Calculate pairwise distances and flatten to create a feature vector
    my_distance = distances.distance_array(prot_pos, lig_pos, box)

    min_dist = np.min(my_distance)
    return min_dist > 10.0

# -----------------------
# Main
# -----------------------
if __name__ == '__main__':

    mp.set_start_method('spawn', force=True)

    # file paths
    ref_file = "../../example_inputs/1OPJ/N368S/md.gro"

    initial_positions = []
    for i in range(135):
        u = mda.Universe(f"../../example_inputs/1OPJ/N368S/1OPJ_N368S_ac_path/frame_{i}.gro")
        initial_positions.append(0.1 * u.atoms.positions.copy())  # (n_atoms,3) in nm
    projection_fn = project

    u = mda.Universe(ref_file)
    # Define atom selections based on command-line arguments
    protein_selection_str = 'around 20 resname STI'
    ligand_selection_str = 'resname STI and not type H'
    protein_ca_indices = u.select_atoms("name CA").indices
    protein_nearby_indices = u.select_atoms(protein_selection_str).select_atoms("name CA").indices
    ligand_indices = u.select_atoms(ligand_selection_str).indices

    print(f"Found {len(protein_nearby_indices)} protein atoms and {len(ligand_indices)} ligand atoms.")

    points = np.load("../../example_inputs/1OPJ/N368S/1OPJ_N368S_ac_path/ac_path.npy")

    path_cv = PathCV(
        points,
        mass_weights=None,
        enforce_equidistance=True,
        equidistance_tol=1.50,
        normalize_output=True
    )
    ref_backbone = u.select_atoms("backbone")
    ref_ligand   = u.select_atoms("resname STI and not type H")

    # Reference coordinates
    ref_bb_coords = ref_backbone.positions.copy()
    ref_lig_coords = ref_ligand.positions.copy()

    # Masses for weighted alignment
    bb_masses = ref_backbone.masses
    box = u.dimensions
    bb_indices  = u.select_atoms("backbone").indices
    lig_indices = u.select_atoms("resname STI and not type H").indices
    # Prepare arguments for the projection function
    projection_kwargs = {
            "path_cv" : path_cv,
            "ref_bb_coords" : ref_bb_coords,
            "bb_indices": bb_indices,
            "lig_indices":lig_indices,
            "bb_masses" : bb_masses}

    warp_kwargs = {
        "protein_idx": protein_nearby_indices,
        "ligand_idx": ligand_indices,
        "box":box,
    }
    bin_edges = [list(np.linspace(0, 1.0, 45)), [0.0, 1000.0]]

    config = {
        'n_gpus': 2,
        'runner_class': OpenMMRunner,
        'enable_cleaning': False,   # results were produced WITHOUT cleaning; debugging aid only
        'clean_threshold': 75.0,
        'source_bin_indices': np.array([[0], [1], [2], [3]]),
        'temperature': 300.0,
        'bin_edges' :bin_edges,
        'protein_idx' : protein_ca_indices,
        'ligand_idx' : ligand_indices,
        #'bin_edges': bin_edges,
        'n_walkers_per_bin': 3,
        'dt': 0.004,
        'n_steps_per_tau': 5000,
        'n_iterations': 50000,
        'flux_file': 'n368s_flux_ac_algebric_clean5.txt',
        'bin_file': 'n368s_bin_ac_algebric_clean5.txt',
        'walkers_file': 'n368s_walker_ac_algebric_clean5.txt',
        'survive_empty': False,
        'warp_function': warp_criteria,
        'warp_kwargs': warp_kwargs,
        'h5_atom_indices' : list(range(0, 4682)),
        'traj_file' : 'full_space_n368s_ac5_algebric_clean.h5'
    }

    we_sim = WESS(
        config=config,
        initial_positions=initial_positions,
        projection_fn=projection_fn,
        kwargs=projection_kwargs
    )
    we_sim.run()
