import warnings
import os
from typing import Dict, List, Tuple

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
        """Setup atom indices mapping for analysis."""
        self.protein_coords = self.u.select_atoms(self.protein_selection).positions
        self.lig_ref_coords = self.u.select_atoms(self.ligand_selection).positions

        # Create mapping between atom indices and their positions
        self.indices = self.u.select_atoms(self.ligand_selection).indices
        non_h_indices = self.u.select_atoms(f'{self.ligand_selection} and not type H').indices
        self.my_map = {idx: i for i, idx in enumerate(self.indices)}
        self.filter_indices = [self.my_map[idx] for idx in non_h_indices]

    def generate_conformations(self, num_conformations: int = 10000, max_radius: float = 50.0,
                             clash_cutoff: float = 1.0, max_distance: float = 10.0,
                             rmsd_threshold: float = 1.0, max_attempts_factor: int = 10,
                             n_workers: int = 20) -> List[np.ndarray]:
        """
        Generate ligand conformations using ConformationGenerator.

        Args:
            num_conformations: Number of conformations to generate
            max_radius: Maximum radius for conformation generation
            clash_cutoff: Distance cutoff for clashes
            max_distance: Maximum distance from reference
            rmsd_threshold: RMSD threshold for filtering
            max_attempts_factor: Multiplier for max attempts
            n_workers: Number of parallel workers

        Returns:
            List of generated conformations
        """
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
        Calculate distances between protein and ligand conformations.

        Args:
            conformations: List of ligand conformations
            around_distance: Distance cutoff for nearby protein atoms

        Returns:
            Array of distance matrices
        """
        protein_nearby = self.u.select_atoms(f'around {around_distance} {self.ligand_selection}')
        protein_nearby = protein_nearby.select_atoms('name CA').positions
        print(protein_nearby.shape)
        all_distances = []
        for conf in conformations:
            dist = distance_array(protein_nearby, conf[self.filter_indices])
            all_distances.append(dist.ravel())

        return np.array(all_distances)

    def analyze_pca(self, distances: np.ndarray, variance_threshold: float = 0.95) -> Tuple[np.ndarray, PCA]:
        """
        Perform PCA analysis on distance matrices.

        Args:
            distances: Distance matrices array
            variance_threshold: Cumulative variance threshold for PCA

        Returns:
            Tuple of (transformed data, PCA object)
        """
        pca = PCA(n_components=10)
        pc_ = pca.fit_transform(distances)

        # Determine number of components needed
        fve = pca.explained_variance_ratio_.cumsum()
        min_dim = np.where(fve < variance_threshold)[0].shape[0] + 1
        print(f'Required {min_dim} dimensions (captures {100*fve[min_dim-1]:.2f}% variance)')

        return pc_[:, :min_dim], pca

    def find_max_separation_dimension(self, data: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Find dimension with maximum separation between last point and others.

        Args:
            data: Input data (n_samples, n_features)

        Returns:
            Tuple of (dimension index, minimum distances per dimension)
        """
        # Scale data to [0, 1] range
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

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
