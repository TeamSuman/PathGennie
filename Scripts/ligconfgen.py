import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.transformations import rotation_matrix
import concurrent
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import multiprocessing
import logging
from typing import Optional, Tuple, List


# ------------------------
# Standalone Helper
# ------------------------

def generate_single_conformer(protein_coords, lig_ref_coords, max_radius, clash_cutoff, max_distance) -> Optional[np.ndarray]:
    def random_rotation_matrix():
        theta = np.random.uniform(0, 2 * np.pi)
        vec = np.random.normal(size=3)
        vec /= np.linalg.norm(vec)
        return rotation_matrix(theta, vec)[:3, :3]

    def random_translation():
        while True:
            vec = np.random.uniform(-max_radius, max_radius, 3)
            if np.linalg.norm(vec) <= max_radius:
                return vec

    def check_clash(ligand_coords):
        dists = np.linalg.norm(
            protein_coords[:, np.newaxis, :] - ligand_coords[np.newaxis, :, :],
            axis=2
        )
        return np.any(dists < clash_cutoff)

    def check_too_far(ligand_coords):
        min_dists = np.min(np.linalg.norm(
            protein_coords[:, np.newaxis, :] - ligand_coords[np.newaxis, :, :],
            axis=2), axis=0)
        return np.all(min_dists > max_distance)

    new_coords = deepcopy(lig_ref_coords)
    rot_mat = random_rotation_matrix()
    centroid = new_coords.mean(axis=0)
    new_coords -= centroid
    new_coords = new_coords @ rot_mat.T
    new_coords += centroid
    new_coords += random_translation()

    if check_clash(new_coords) or check_too_far(new_coords):
        return None
    return new_coords


class ConformationGenerator:
    def __init__(self, protein_coords, lig_ref_coords,
                 max_radius=20.0, clash_cutoff=2.0,
                 max_distance=15.0, rmsd_threshold=5.0):
        self.protein_coords = protein_coords
        self.lig_ref_coords = lig_ref_coords
        self.max_radius = max_radius
        self.clash_cutoff = clash_cutoff
        self.max_distance = max_distance
        self.rmsd_threshold = rmsd_threshold

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _rmsd(a, b):
        return np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))

    def _is_unique(self, new_coords, existing_conformations):
        return all(self._rmsd(new_coords, existing) >= self.rmsd_threshold
                   for existing in existing_conformations)

    def generate_conformations(self, num_conformations: int,
                               max_attempts_factor: int = 10,
                               n_workers: Optional[int] = None) -> List[np.ndarray]:
        if n_workers is None:
            n_workers = multiprocessing.cpu_count()

        conformations = []
        total_attempts = num_conformations * max_attempts_factor
        attempts = 0

        self.logger.info(f"Generating {num_conformations} conformations (up to {total_attempts} attempts)...")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = set()

            # Initial batch
            initial_batch = min(total_attempts, n_workers * 2)
            for _ in range(initial_batch):
                fut = executor.submit(generate_single_conformer,
                                      self.protein_coords, self.lig_ref_coords,
                                      self.max_radius, self.clash_cutoff, self.max_distance)
                futures.add(fut)
                attempts += 1

            while futures and len(conformations) < num_conformations and attempts <= total_attempts:
                done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

                for future in done:
                    result = future.result()
                    if result is not None and self._is_unique(result, conformations):
                        conformations.append(result)
                        self.logger.info(f"Found conformation {len(conformations)}/{num_conformations}")

                    if attempts < total_attempts and len(conformations) < num_conformations:
                        fut = executor.submit(generate_single_conformer,
                                              self.protein_coords, self.lig_ref_coords,
                                              self.max_radius, self.clash_cutoff, self.max_distance)
                        futures.add(fut)
                        attempts += 1

        self.logger.info(f"Finished with {len(conformations)} conformations.")
        return conformations


def load_system(protein_selection: str, ligand_selection: str,
                structure_file: str) -> Tuple[np.ndarray, np.ndarray]:
    u = mda.Universe(structure_file)
    protein = u.select_atoms(protein_selection)
    ligand = u.select_atoms(ligand_selection)
    return protein.positions.copy(), ligand.positions.copy()


if __name__ == "__main__":
    protein_coords, lig_ref_coords = load_system(
        protein_selection='protein',
        ligand_selection='resname BNZ',
        structure_file="../Data/T4_Lyso_Benzene/4w52/config.gro"
    )

    generator = ConformationGenerator(
        protein_coords=protein_coords,
        lig_ref_coords=lig_ref_coords,
        max_radius=20.0,
        clash_cutoff=2.0,
        max_distance=20.0,
        rmsd_threshold=5.0
    )

    conformations = generator.generate_conformations(
        num_conformations=10000,
        max_attempts_factor=10,
        n_workers=20
    )
