"""Reaction coordinate for the identity SN2  Cl(-) + CH3Cl -> ClCH3 + Cl(-).

Atom order in sn2.prmtop (0-based): 0=C, 1=Cl_leaving, 2..4=H, 5=Cl_attacking.

xi = d(C-Cl_leaving) - d(C-Cl_attacking), the standard antisymmetric stretch:
  reactant  xi ~ -1.2 A   (leaving Cl bonded at 1.78, attacking Cl far at 3.0)
  TS        xi =  0       (symmetric, both ~2.3 A)
  product   xi ~ +1.2 A   (roles swapped)
Non-periodic, so no `periodic` key is needed in input.yaml.
"""
import numpy as np

C, CL_LEAVE, CL_ATTACK = 0, 1, 5


def sn2_cv(coords, **kwargs):
    coords = np.asarray(coords, dtype=float)
    d_leave = np.linalg.norm(coords[C] - coords[CL_LEAVE])
    d_attack = np.linalg.norm(coords[C] - coords[CL_ATTACK])
    return np.array([d_leave - d_attack])


def reacted(coords, threshold=1.0, **kwargs):
    """Converged once the new C-Cl bond is clearly formed and the old one broken.

    NOTE: xi is a *difference* of distances, so it is also satisfied by the leaving
    group departing while the nucleophile never bonds. That cannot happen for
    methyl chloride -- substitution is the only route to large xi here -- but it
    does happen on tertiary substrates. See `reacted_product_specific` below and
    the README section "A caution on convergence criteria".
    """
    return bool(sn2_cv(coords)[0] > threshold)


def reacted_product_specific(coords, d_formed=2.1, d_broken=3.0, **kwargs):
    """Stop on the *product*, not on progress: both bonds must be right.

    Prefer this form when adapting the example to a substrate where a competing
    channel (elimination, ionisation) could also drive xi up.
    """
    coords = np.asarray(coords, dtype=float)
    return bool(np.linalg.norm(coords[C] - coords[CL_ATTACK]) < d_formed
                and np.linalg.norm(coords[C] - coords[CL_LEAVE]) > d_broken)


# --------------------------------------------------------------------------- #
# The 2-D feature space used by refinement (stage 2), the free energy (stage 3)
# and the plot (stage 4). Defined once here so the three stages cannot disagree
# on component order -- they did during development, and because this particular
# reaction is symmetric the resulting mirrored axes were nearly invisible.
# --------------------------------------------------------------------------- #
def path_features(coords):
    """(n_atoms, 3) -> [d(C-Cl_attacking), d(C-Cl_leaving)]."""
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    return np.array([np.linalg.norm(coords[C] - coords[CL_ATTACK]),
                     np.linalg.norm(coords[C] - coords[CL_LEAVE])])


def path_features_traj(traj):
    """(T, n_atoms, 3) -> (T, 2), the same pair for a whole trajectory."""
    traj = np.asarray(traj, dtype=float)
    return np.column_stack([
        np.linalg.norm(traj[:, C] - traj[:, CL_ATTACK], axis=1),
        np.linalg.norm(traj[:, C] - traj[:, CL_LEAVE], axis=1),
    ])
