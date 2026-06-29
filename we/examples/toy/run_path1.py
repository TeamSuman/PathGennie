#!/usr/bin/env python3
import multiprocessing as mp
import sys
import warnings

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.align import rotation_matrix

from toy import OpenMMRunner

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
        sq = np.sum(diff**2, axis=1)
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
        msd = np.array([self._msd(R - Ri) for Ri in self.reference_path])

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


def warp_criteria(positions, kwargs):
    center = np.array([1.0, 0.0, 0.0])
    dist = np.linalg.norm(positions - center)
    return dist < 0.5


def project(positions, kwargs):

    path_cv = kwargs["path_cv"]
    positions = positions.reshape(1, 3)
    s, z = path_cv.compute(positions)
    return np.array([s, z])


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    # file paths

    initial_positions = [np.array([-1.0, 0.0, 0.0])]
    projection_fn = project

    points = np.array(
        [
            [-1.0, 0.0, 0.0],
            [-0.95986304, 0.52879259, 0.0],
            [-0.71059753, 0.97750856, 0.0],
            [-0.50723938, 1.26035861, 0.0],
            [0.0, 1.537082, 0.0],
            [0.50723938, 1.26035861, 0.0],
            [0.71059753, 0.97750856, 0.0],
            [0.95986304, 0.52879259, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    lower_points = np.load("points_upper_equispaced.npy").reshape(-1, 1, 3)
    path_cv = PathCV(lower_points, equidistance_tol=0.5, normalize_output=True)

    projection_kwargs = {
        "path_cv": path_cv,
    }

    warp_kwargs = {}

    bin_edges = [list(np.linspace(0, 1.0, 45)), [0.0, 1.0]]

    config = {
        "n_gpus": 2,
        "runner_class": OpenMMRunner,
        "enable_cleaning": True,
        "clean_threshold": 0.25,
        "source_bin_indices": np.array([[0], [1], [2], [3]]),
        "temperature": 200.0,
        "bin_edges": bin_edges,
        #'bin_edges': bin_edges,
        "n_walkers_per_bin": 4,
        "dt": 0.002,
        "n_steps_per_tau": 50,
        "n_iterations": 5000,
        "flux_file": "flux_file_lower_3_0.1_T200.txt",
        "bin_file": "bin_file_lower_3__0.1_T200.txt",
        "walkers_file": "walker_lower_3_0.1_T200.txt",
        "survive_empty": False,
        "warp_function": warp_criteria,
        "warp_kwargs": warp_kwargs,
        "h5_atom_indices": list(range(0, 1)),
        "traj_file": "lower_3.h5",
    }

    we_sim = WESS(
        config=config,
        initial_positions=initial_positions,
        projection_fn=projection_fn,
        kwargs=projection_kwargs,
    )
    we_sim.run()
