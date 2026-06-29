import warnings
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import joblib
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from ligconfgen import ConformationGenerator, load_system
warnings.filterwarnings('ignore')


class LigPCGen:
    """Analyzes protein-ligand conformations using PCA and distance metrics."""

    def __init__(self, structure_path: str, protein_selection: str = 'protein',
                 ligand_selection: str = 'resname LIG'):
        """
        Initialize the analyzer with system information.

        Args:
            structure_path: Path to the structure file (e.g., .gro)
            protein_selection: MDAnalysis selection string for protein
            ligand_selection: MDAnalysis selection string for ligand
        """
        self.structure_path = structure_path
        self.protein_selection = protein_selection
        self.ligand_selection = ligand_selection
        self.u = mda.Universe(structure_path)
        self._setup_indices()

    def _setup_indices(self) -> None:
        """Setup atom indices mapping for analysis with robust fallbacks."""
        # Find protein/receptor atoms with fallback
        protein_atoms = self.u.select_atoms(self.protein_selection)
        if len(protein_atoms) == 0:
            # Fallback: everything except the ligand
            protein_atoms = self.u.select_atoms(f"not ({self.ligand_selection})")
            if len(protein_atoms) == 0:
                raise ValueError("No receptor/host atoms found in the system.")
            print(f"Warning: Selection '{self.protein_selection}' returned no atoms. Using fallback: 'not ({self.ligand_selection})'")
            self.protein_selection = f"not ({self.ligand_selection})"

        self.protein_coords = protein_atoms.positions
        
        ligand_atoms = self.u.select_atoms(self.ligand_selection)
        if len(ligand_atoms) == 0:
            raise ValueError(f"Ligand selection '{self.ligand_selection}' returned no atoms.")
        self.lig_ref_coords = ligand_atoms.positions

        # Create mapping between atom indices and their positions
        self.indices = ligand_atoms.indices
        
        # Robust hydrogen filtering (wrapped in try-except to avoid topology lookup failures on certain formats like .gro)
        try:
            non_h_atoms = ligand_atoms.select_atoms('not (type H or name H* or element H or type h or name h* or element h)')
        except Exception:
            try:
                non_h_atoms = ligand_atoms.select_atoms('not (name H* or name h*)')
            except Exception:
                non_h_atoms = ligand_atoms
        if len(non_h_atoms) == 0:
            non_h_atoms = ligand_atoms
            
        non_h_indices = non_h_atoms.indices
        self.my_map = {idx: i for i, idx in enumerate(self.indices)}
        self.filter_indices = [self.my_map[idx] for idx in non_h_indices]

    def generate_conformations(self, num_conformations: int = 10000, max_radius: float = 50.0,
                             clash_cutoff: float = 1.0, max_distance: float = 10.0,
                             rmsd_threshold: float = 1.0, max_attempts_factor: int = 10,
                             n_workers: Optional[int] = None) -> List[np.ndarray]:
        """
        Generate ligand conformations using ConformationGenerator.

        Args:
            num_conformations: Number of conformations to generate
            max_radius: Maximum radius for conformation generation
            clash_cutoff: Distance cutoff for clashes
            max_distance: Maximum distance from reference
            rmsd_threshold: RMSD threshold for filtering
            max_attempts_factor: Multiplier for max attempts
            n_workers: Number of parallel workers (defaults to min(cpu_count, 16) to avoid thrashing)

        Returns:
            List of generated conformations
        """
        import multiprocessing
        if n_workers is None:
            n_workers = min(multiprocessing.cpu_count(), 16)

        generator = ConformationGenerator(
            protein_coords=self.protein_coords,
            lig_ref_coords=self.lig_ref_coords,
            max_radius=max_radius,
            clash_cutoff=clash_cutoff,
            max_distance=max_distance,
            rmsd_threshold=rmsd_threshold
        )

        conformations = generator.generate_conformations(
            num_conformations=num_conformations,
            max_attempts_factor=max_attempts_factor,
            n_workers=n_workers
        )

        # Append reference conformation
        conformations.append(self.lig_ref_coords)
        return conformations

    def calculate_distances(self, conformations: List[np.ndarray],
                          around_distance: float = 20.0) -> np.ndarray:
        """
        Calculate distances between protein/receptor and ligand conformations.
        Implements robust target selection and vectorized batched distance calculation.

        Args:
            conformations: List of ligand conformations
            around_distance: Distance cutoff for nearby protein atoms

        Returns:
            Array of flattened distance matrices of shape (num_conformations, N_prot * N_lig_filtered)
        """
        # Find nearby receptor/host atoms
        receptor = self.u.select_atoms(self.protein_selection)
        nearby_receptor = self.u.select_atoms(f"({self.protein_selection}) and (around {around_distance} ({self.ligand_selection}))")
        
        if len(nearby_receptor) == 0:
            nearby_receptor = receptor

        # Robust reference selection: Try standard CA, then BB, then backbone, then heavy, then all
        ref_atoms = nearby_receptor.select_atoms("name CA")
        if len(ref_atoms) == 0:
            ref_atoms = nearby_receptor.select_atoms("name BB")
        if len(ref_atoms) == 0:
            try:
                ref_atoms = nearby_receptor.select_atoms("backbone")
            except Exception:
                pass
        if len(ref_atoms) == 0:
            try:
                ref_atoms = nearby_receptor.select_atoms("not (name H* or name h* or type H or type h or element H or element h)")
            except Exception:
                try:
                    ref_atoms = nearby_receptor.select_atoms("not (name H* or name h*)")
                except Exception:
                    ref_atoms = nearby_receptor
        if len(ref_atoms) == 0:
            ref_atoms = nearby_receptor

        protein_nearby = ref_atoms.positions
        print(f"Selected {protein_nearby.shape[0]} receptor reference atoms for distance calculation.")
        
        if protein_nearby.shape[0] == 0:
            raise ValueError("No receptor reference atoms found for distance calculation.")

        # Batch-vectorized distance calculation for high speed and safe memory usage
        M = len(conformations)
        N_prot = len(protein_nearby)
        N_filtered = len(self.filter_indices)

        # Convert conformations to a single numpy array and pre-filter coordinates to avoid loop indexing overhead
        confs_arr = np.array(conformations) # (M, N_lig, 3)
        confs_filtered = confs_arr[:, self.filter_indices, :] # (M, N_filtered, 3)

        batch_size = 2000
        all_distances = []
        for i in range(0, M, batch_size):
            batch_confs = confs_filtered[i:i+batch_size] # (B, N_filtered, 3)
            # Broadcast batch configurations and receptor coordinates:
            # batch_confs[:, np.newaxis, :, :] has shape (B, 1, N_filtered, 3)
            # protein_nearby[np.newaxis, :, np.newaxis, :] has shape (1, N_prot, 1, 3)
            diff = batch_confs[:, np.newaxis, :, :] - protein_nearby[np.newaxis, :, np.newaxis, :]
            # Compute Euclidean norm over the xyz coordinate axis
            dist = np.linalg.norm(diff, axis=-1) # (B, N_prot, N_filtered)
            # Reshape to (B, N_prot * N_filtered) to match .ravel() in C-order
            all_distances.append(dist.reshape(len(batch_confs), -1))

        return np.vstack(all_distances)

    def analyze_pca(self, distances: np.ndarray, variance_threshold: float = 0.95) -> Tuple[np.ndarray, PCA]:
        """
        Perform PCA analysis on distance matrices.
        Automatically scales the number of PCA components to fit small systems without crashing.

        Args:
            distances: Distance matrices array
            variance_threshold: Cumulative variance threshold for PCA

        Returns:
            Tuple of (transformed data, PCA object)
        """
        # Determine n_components dynamically to prevent crash on systems with very few features or samples
        n_components = min(10, distances.shape[0], distances.shape[1])
        if n_components < 1:
            raise ValueError(f"Invalid distances matrix dimensions for PCA: {distances.shape}")

        pca = PCA(n_components=n_components)
        pc_ = pca.fit_transform(distances)

        # Determine number of components needed
        fve = pca.explained_variance_ratio_.cumsum()
        min_dim = min(np.where(fve < variance_threshold)[0].shape[0] + 1, n_components)
        print(f'Required {min_dim} dimensions (captures {100*fve[min_dim-1]:.2f}% variance)')

        return pc_[:, :min_dim], pca

    def find_max_separation_dimension(self, data: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Find dimension with maximum separation between last point and others.
        Robustly avoids divide-by-zero errors on constant columns.

        Args:
            data: Input data (n_samples, n_features)

        Returns:
            Tuple of (dimension index, minimum distances per dimension)
        """
        # Scale data to [0, 1] range manually to robustly handle zero-variance columns
        variance = np.var(data, axis=0)
        non_constant_cols = variance > 1e-8
        
        if not np.any(non_constant_cols):
            return 0, np.zeros(data.shape[1])

        scaled_data = np.zeros_like(data)
        for col in range(data.shape[1]):
            col_min = np.min(data[:, col])
            col_max = np.max(data[:, col])
            if col_max - col_min > 1e-8:
                scaled_data[:, col] = (data[:, col] - col_min) / (col_max - col_min)
            else:
                scaled_data[:, col] = 0.0

        last_point = scaled_data[-1]
        distances = np.abs(scaled_data[:-1] - last_point)
        min_distances = np.min(distances, axis=0)
        max_sep_dim = np.argmax(min_distances)

        return max_sep_dim, min_distances

    def save_pca(self, pca: PCA, filename: str) -> None:
        """Save PCA object to file."""
        joblib.dump(pca, filename)
        print(f'Saved PCA to {filename}')


if __name__ == "__main__":
    # Example usage
    path = "/home/dm/Dibyendu/Projects/CVSpacePathGen/Data/LigUnbind/2YKI"
    analyzer = LigPCGen(os.path.join(path, 'pbcmol.gro'))

    # Generate conformations
    conformations = analyzer.generate_conformations(num_conformations=10000)

    # Calculate distances
    distances = analyzer.calculate_distances(conformations)

    # Perform PCA analysis
    pc_, pca = analyzer.analyze_pca(distances)

    # Find dimension with maximum separation
    max_sep_dim, min_distances = analyzer.find_max_separation_dimension(pc_)

    print(f"Dimension with maximum separation: {max_sep_dim}")
    print(f"Minimum distances per dimension: {', '.join(f'{d:.4f}' for d in min_distances)}")

    # Save PCA model
    analyzer.save_pca(pca, "2yki_pca.pkl")
