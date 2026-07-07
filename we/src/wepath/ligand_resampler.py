import numpy as np
from .resampler import Resampler


def _get_kabsch_transformation(P, Q):
    """
    Calculates the optimal translation and rotation to align Q onto P.

    Args:
        P (np.ndarray): Reference coordinates, shape (N, 3).
        Q (np.ndarray): Target coordinates, shape (N, 3).

    Returns:
        tuple: A tuple (U, p_center, q_center) containing:
            - U (np.ndarray): The optimal rotation matrix (3, 3).
            - p_center (np.ndarray): The centroid of P (3,).
            - q_center (np.ndarray): The centroid of Q (3,).
    """
    # 1. Center the coordinate sets
    p_center = P.mean(axis=0)
    q_center = Q.mean(axis=0)
    P_cent = P - p_center
    Q_cent = Q - q_center

    # 2. Compute the covariance matrix
    # Using @ operator for matrix multiplication is more readable
    C = Q_cent.T @ P_cent

    # 3. Singular Value Decomposition (SVD)
    V, S, Wt = np.linalg.svd(C)

    # 4. Handle reflections to ensure a pure rotation
    # A reflection occurs if the determinant of the rotation matrix is -1.
    d = np.sign(np.linalg.det(V @ Wt))
    # Create a diagonal matrix to flip the sign of the last singular vector if needed
    diag_fix = np.diag([1, 1, d])

    # 5. Compute the optimal rotation matrix
    U = V @ diag_fix @ Wt

    return U, p_center, q_center


def kabsch_superpose(P, Q):
    """
    Finds the optimal rotation to align Q onto P and returns the aligned Q.
    P and Q must be (N, 3) arrays with N corresponding atoms.

    Args:
        P (np.ndarray): Reference coordinates, shape (N, 3).
        Q (np.ndarray): Target coordinates to align, shape (N, 3).

    Returns:
        np.ndarray: The coordinates of Q after alignment, shape (N, 3).
    """
    rotation, p_center, q_center = _get_kabsch_transformation(P, Q)

    # Apply transformation: center Q, rotate, then translate to P's center
    Q_cent = Q - q_center
    Q_rot = Q_cent @ rotation.T # Apply rotation to row vectors
    Q_aligned = Q_rot + p_center

    return Q_aligned


def ligand_rmsd(pos_ref, pos_target, protein_idx, ligand_idx):
    """
    Fit protein from pos_target to pos_ref, then calculate RMSD of the ligand.

    Args:
        pos_ref (np.ndarray): Reference coordinates (N_atoms, 3).
        pos_target (np.ndarray): Target coordinates (N_atoms, 3).
        protein_idx (list[int]): Indices of protein atoms for fitting.
        ligand_idx (list[int]): Indices of ligand atoms for RMSD.

    Returns:
        float: RMSD of ligand atoms after protein-based alignment.
    """
    # Extract protein coordinates for fitting
    P_prot_ref = pos_ref[protein_idx]
    Q_prot_tgt = pos_target[protein_idx]

    # Get the transformation based on the protein alignment
    rotation, p_center, q_center = _get_kabsch_transformation(P_prot_ref, Q_prot_tgt)

    # Apply the same transformation to the target ligand
    ligand_tgt = pos_target[ligand_idx]
    ligand_aligned = (ligand_tgt - q_center) @ rotation.T + p_center

    # Calculate RMSD between the reference ligand and the aligned target ligand
    diff = ligand_aligned - pos_ref[ligand_idx]
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return rmsd


def rmsd_matrix(walkers, protein_idx, ligand_idx):
    """
    Compute the pairwise ligand RMSD matrix between a list of walkers.
    This function is already efficient and correct.
    """
    n = len(walkers)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rmsd = ligand_rmsd(
                walkers[i].positions,
                walkers[j].positions,
                protein_idx,
                ligand_idx
            )
            M[i, j] = M[j, i] = rmsd
    return M


def ligand_com_distance(pos_ref, pos_target, protein_idx, ligand_idx):
    """
    Fit protein from pos_target to pos_ref, then calculate COM distance of the ligand.

    Args:
        pos_ref (np.ndarray): Reference coordinates (N_atoms, 3).
        pos_target (np.ndarray): Target coordinates (N_atoms, 3).
        protein_idx (list[int]): Indices of protein atoms for fitting.
        ligand_idx (list[int]): Indices of ligand atoms for COM calculation.

    Returns:
        float: Distance between ligand centers of mass after protein-based alignment.
    """
    # Extract protein coordinates for fitting
    P_prot_ref = pos_ref[protein_idx]
    Q_prot_tgt = pos_target[protein_idx]

    # Get the transformation based on the protein alignment
    rotation, p_center, q_center = _get_kabsch_transformation(P_prot_ref, Q_prot_tgt)

    # Apply the same transformation to the target ligand
    ligand_tgt = pos_target[ligand_idx]
    ligand_aligned = (ligand_tgt - q_center) @ rotation.T + p_center

    # Compute ligand COMs
    com_ref = pos_ref[ligand_idx].mean(axis=0)
    com_aligned = ligand_aligned.mean(axis=0)

    # Euclidean distance between COMs
    dist = np.linalg.norm(com_aligned - com_ref)

    return dist

class LigandResamplerBack(Resampler):
    """
    A resampler that merges walkers within a bin based on ligand RMSD similarity.
    """
    def __init__(self, bins, target_per_bin, weight_threshold=1e-249, protein_idx=None, ligand_idx=None):
        super().__init__(bins, target_per_bin, weight_threshold)
        self.protein_idx = protein_idx
        self.ligand_idx = ligand_idx

    def merge_walkers(self, walker_indices, weights, all_walkers):

        assert self.protein_idx is not None and self.ligand_idx is not None, \
            "Need protein_idx and ligand_idx for RMSD-based merging"

        walkers = [all_walkers[i].clone() for i in walker_indices]

        # Assign the correct weights for this merging event
        for walker, weight in zip(walkers, weights):
            walker.weight = weight

        # If the number of walkers is already at or below the target, no merging is needed
        if len(walkers) <= self.N_TARGET_PER_BIN:
            return walkers

        # Compute the initial pairwise RMSD matrix
        M = rmsd_matrix(walkers, self.protein_idx, self.ligand_idx)

        # Greedy merging loop: repeatedly find the closest pair and merge them
        while len(walkers) > self.N_TARGET_PER_BIN:

            # Find the closest pair of walkers (i, j)
            # Set the diagonal to infinity to ensure we only find off-diagonal minimums
            np.fill_diagonal(M, np.inf)
            i, j = np.unravel_index(np.argmin(M), M.shape)

            # To simplify indexing, always merge the walker with the larger index
            # into the one with the smaller index.
            if i > j:
                i, j = j, i

            # Merge walker j into walker i by adding their weights.
            # The conformation of walker i is kept.
            if walkers[i].weight > walkers[j].weight:
                walkers[i].weight += walkers[j].weight
                # Remove walker j from the list of walkers
                walkers.pop(j)
                M = np.delete(M, j, axis=0)
                M = np.delete(M, j, axis=1)
            else:
                walkers[j].weight += walkers[i].weight
                # Remove walker i from the list of walkers
                walkers.pop(i)
                M = np.delete(M, i, axis=0)
                M = np.delete(M, i, axis=1)
        return walkers

class LigandResampler(Resampler):
    """
    A resampler that merges walkers within a bin based on ligand RMSD similarity.
    """
    def __init__(self, bins, target_per_bin, weight_threshold=1e-6, protein_idx=None, ligand_idx=None):
        super().__init__(bins, target_per_bin, weight_threshold)
        self.protein_idx = protein_idx
        self.ligand_idx = ligand_idx

    def merge_walkers(self, walker_indices, weights, all_walkers):
        assert self.protein_idx is not None and self.ligand_idx is not None, \
            "Need protein_idx and ligand_idx for RMSD-based merging"

        walkers = [all_walkers[i].clone() for i in walker_indices]

        # Assign weights for this merging event
        for walker, weight in zip(walkers, weights):
            walker.weight = weight

        if len(walkers) <= self.N_TARGET_PER_BIN:
            return walkers, [{"survivors": [idx], "merged": []} for idx in walker_indices]

        # Compute RMSD matrix
        M = rmsd_matrix(walkers, self.protein_idx, self.ligand_idx)

        merge_events = []
        current_indices = list(walker_indices)  # track mapping back to original

        # Greedy merging loop
        step = 0
        while len(walkers) > self.N_TARGET_PER_BIN:
            step += 1
            #print(f"\n--- Merge step {step} ---")
            #print("RMSD matrix:\n", M)

            np.fill_diagonal(M, np.inf)
            i, j = np.unravel_index(np.argmin(M), M.shape)

            if i > j:
                i, j = j, i

            rmsd_pair = M[i, j]
            print(f"Chosen pair to merge: (walker {current_indices[i]}, walker {current_indices[j]}) "
                  f"with RMSD = {rmsd_pair:.4f}")
            com_distance = ligand_com_distance(walkers[i].positions, walkers[j].positions,
                                                       self.protein_idx, self.ligand_idx)
            print(f"Ligand COM distance after protein alignment: {com_distance:.4f} Å")
            # Merge walker j into i or i into j depending on weights
            if walkers[i].weight >= walkers[j].weight:
                survivor, loser = i, j
            else:
                survivor, loser = j, i

            # Record event
            merge_events.append({
                "survivor": int(current_indices[survivor]),
                "merged": int(current_indices[loser]),
                "rmsd": float(rmsd_pair)
            })

            # Transfer weight
            walkers[survivor].weight += walkers[loser].weight

            # Remove loser
            walkers.pop(loser)
            current_indices.pop(loser)
            M = np.delete(M, loser, axis=0)
            M = np.delete(M, loser, axis=1)

        # Build final mapping list
        survivor_events = []
        surviving_indices = current_indices
        for idx in surviving_indices:
            merged = [e["merged"] for e in merge_events if e["survivor"] == idx]
            survivor_events.append({
                "survivor": int(idx),
                "merged": merged
            })

        return walkers, survivor_events
