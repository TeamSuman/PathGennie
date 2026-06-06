import numpy as np


def center_of_mass(coords, atom_indices, masses):
    atom_indices = np.asarray(atom_indices, dtype=int)
    masses = np.asarray(masses, dtype=float)
    group_masses = masses[atom_indices]
    return np.sum(coords[atom_indices] * group_masses[:, None], axis=0) / np.sum(group_masses)


def com_com_distance_cv(coords, group_a_indices, group_b_indices, masses):
    """Mass-weighted COM-COM distance between two atom groups."""

    com_a = center_of_mass(coords, group_a_indices, masses)
    com_b = center_of_mass(coords, group_b_indices, masses)
    return np.array([np.linalg.norm(com_a - com_b)])


def dissociated(coords, group_a_indices, group_b_indices, masses, threshold=10.0):
    """Converged when COM-COM distance exceeds the dissociation threshold."""

    distance = com_com_distance_cv(
        coords,
        group_a_indices=group_a_indices,
        group_b_indices=group_b_indices,
        masses=masses,
    )[0]
    return distance > threshold
