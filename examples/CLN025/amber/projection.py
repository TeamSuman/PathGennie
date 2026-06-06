import numpy as np

_ref_coords = None


def load_reference(ref_pdb, topology):
    """
    Load reference structure once.

    Parameters
    ----------
    ref_pdb : str
        Reference PDB or rst7
    topology : str
        Amber topology
    """

    global _ref_coords

    coords = []
    with open(ref_pdb, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")):
                coords.append(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )
    if not coords:
        raise ValueError(f"No PDB coordinates found in {ref_pdb}")

    _ref_coords = np.asarray(coords, dtype=float)


def end_to_end_cv(coords, start_index=4, end_index=132):
    """
    CA-to-CA end-to-end distance projection for CLN025.

    Defaults are 0-based atom indices for residue 1 CA and residue 10 CA in
    chignolin_folded.pdb.
    """

    distance = np.linalg.norm(coords[int(end_index)] - coords[int(start_index)])
    return np.array([distance])


def end_to_end_escaped(coords, start_index=4, end_index=132, threshold=15.0):
    """Converged when the terminal CA-to-CA distance exceeds threshold."""

    return end_to_end_cv(coords, start_index=start_index, end_index=end_index)[0] < threshold


def rmsd_cv(
    coords,
    atom_indices=None,
):
    """
    Simple RMSD projection.

    Parameters
    ----------
    coords : ndarray
        (n_atoms,3) coordinates in Å

    Returns
    -------
    ndarray
        1D CV vector
    """

    global _ref_coords

    if _ref_coords is None:
        raise RuntimeError("Reference not loaded.")

    xyz = coords

    if atom_indices is not None:
        xyz = xyz[atom_indices]
        ref = _ref_coords[atom_indices]
    else:
        ref = _ref_coords

    diff = xyz - ref

    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return np.array([rmsd])


def escaped(
    coords,
    atom_indices=None,
    threshold=5.0,
):
    """
    Convergence check: has structure escaped (unfolded) from reference?

    Convergence occurs when RMSD from folded reference exceeds threshold.
    This targets the unfolded state (high RMSD = unfolded).

    Parameters
    ----------
    coords : ndarray
        (n_atoms,3) coordinates in Å
    atom_indices : ndarray, optional
        Indices of atoms to include (typically heavy atoms)
    threshold : float, optional
        RMSD threshold in Å. Default 5.0 Å.

    Returns
    -------
    bool
        True if structure has escaped (RMSD > threshold), False otherwise
    """

    global _ref_coords

    if _ref_coords is None:
        raise RuntimeError("Reference not loaded.")

    xyz = coords

    if atom_indices is not None:
        xyz = xyz[atom_indices]
        ref = _ref_coords[atom_indices]
    else:
        ref = _ref_coords

    diff = xyz - ref
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return rmsd > threshold
