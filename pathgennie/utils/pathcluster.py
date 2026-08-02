import glob

import numpy as np

try:
    import MDAnalysis as mda
    from dtaidistance import dtw_ndim
    from sklearn.cluster import AgglomerativeClustering
except ImportError as _exc:  # pragma: no cover - exercised via the guard test
    # DTW path clustering needs dtaidistance and scikit-learn, which are not core
    # PathGennie dependencies. dtaidistance in particular was never declared
    # anywhere, so this module could not run at all as shipped.
    _which = f" (missing: {_exc.name})" if getattr(_exc, "name", None) else ""
    raise ImportError(
        "Path clustering (pathgennie.utils.pathcluster) needs dtaidistance, "
        f"scikit-learn and MDAnalysis{_which}.\n"
        "Install them with:  pip install 'pathgennie[analysis]'\n"
        f"Underlying import error: {_exc}"
    ) from _exc

trajectories = []
conf = "conf.gro"
trajs = glob.glob("../data/3PTB/5_10_25/*xtc")


def make_conf(u, model_file):
    # Define atom selections based on command-line arguments
    protein_selection_str = 'around 20.0 resname BEN'
    ligand_selection_str = 'resname BEN and not type H'

    protein_nearby_indices = u.select_atoms(protein_selection_str).select_atoms("name CA").indices
    ligand_indices = u.select_atoms(ligand_selection_str).indices

    print(f"Found {len(protein_nearby_indices)} protein atoms and {len(ligand_indices)} ligand atoms.")

    # Load the pre-trained PCA model
    model = joblib.load(model_file)
    print(f"Loaded PCA model from '{model_file}'.")

    # Prepare arguments for the projection function
    return {
        "protein_indices": protein_nearby_indices,
        "ligand_indices": ligand_indices,
        "model": model
    }


def projection(positions, **kwargs):
    """
    Projects the atomic positions onto the principal components.

    This function calculates the distance features between the protein and ligand,
    and then uses a pre-trained PCA model to reduce the dimensionality.

    Args:
        positions (np.ndarray): Array of atomic positions from the simulation frame.
        **kwargs: Keyword arguments containing:
            protein_indices (np.ndarray): Indices of the protein atoms.
            ligand_indices (np.ndarray): Indices of the ligand atoms.
            model (object): The loaded PCA model object.

    Returns:
        tuple: A tuple containing:
            - The first 4 principal components of the transformed features.
            - The distance between the center of mass of the protein and ligand.
            - The minimum distance between any protein and ligand atom.
    """
    protein_indices = kwargs.get('protein_indices')
    ligand_indices = kwargs.get('ligand_indices')
    model = kwargs.get('model')

    if protein_indices is None or ligand_indices is None or model is None:
        raise ValueError("Missing required arguments in projection function: protein_indices, ligand_indices, or model.")

    # Extract positions for protein and ligand
    prot_pos = positions[protein_indices]
    lig_pos = positions[ligand_indices]

    # Calculate pairwise distances and flatten to create a feature vector
    my_distance = distance_array(prot_pos, lig_pos)
    feat = my_distance.ravel()

    # Calculate center of mass distance
    prot_com = prot_pos.mean(axis=0)
    lig_com = lig_pos.mean(axis=0)
    com_dist = calc_bonds(prot_com, lig_com)

    # Transform features using the PCA model and return key metrics
    # The original code returns the first 4 components, so we slice with [:4]
    transformed_features = model.transform(feat.reshape(1, -1))
    return transformed_features[0], com_dist, np.min(my_distance)


kwargs = make_conf(ref, "../data/3PTB/pca_minmax_pipeline.pkl")

def getPC(u):
    pc_list = []
    for _ in u.trajectory:
        pc_list.append(projection(u.atoms.positions, **kwargs)[0])
    return np.array(pc_list)


for traj in trajs:
    u = mda.Universe(conf, traj)
    pc = getPC(u)
    trajectories.append(pc)

def dtw_trajectory_distance(traj1, traj2):
    """
    Calculates the multi-dimensional DTW distance between two trajectories.
    """
    # Using a window can dramatically speed up computation.
    # A 10-20% window is common.
    window_size = int(max(len(traj1), len(traj2)) * 0.2)

    # This is the line you had commented out, but it's the right one
    return dtw_ndim.distance_fast(traj1, traj2, window=window_size)

# --- Compute symmetric distance matrix ---
n = len(trajectories)
D = np.zeros((n, n))

print("Calculating DTW distance matrix...")
for i in range(n):
    for j in range(i + 1, n):
        # DTW is symmetric, so D[i,j] == D[j,i]
        # We only need to calculate it once.
        dist = dtw_trajectory_distance(trajectories[i], trajectories[j])
        D[i, j] = dist
        D[j, i] = dist
print("Matrix calculation complete.")

agg_cluster = AgglomerativeClustering(n_clusters=4,
                                      metric='precomputed',
                                      linkage='average')
labels = agg_cluster.fit_predict(D)
