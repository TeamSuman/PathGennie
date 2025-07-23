#!/usr/bin/env python3
import os
import argparse
from ligcvgen import LigPCGen

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Analyze protein-ligand conformations and find dimension with maximum separation."
    )

    parser.add_argument(
        "structure_file",
        type=str,
        help="Path to the structure file (e.g., .gro)"
    )

    parser.add_argument(
        "-n", "--num_conformations",
        type=int,
        default=10000,
        help="Number of conformations to generate (default: 10000)"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output_pca.pkl",
        help="Output filename for PCA model (default: output_pca.pkl)"
    )

    parser.add_argument(
        "--protein_sel",
        type=str,
        default="protein",
        help="MDAnalysis selection string for protein (default: 'protein')"
    )

    parser.add_argument(
        "--ligand_sel",
        type=str,
        default="resname LIG",
        help="MDAnalysis selection string for ligand (default: 'resname LIG')"
    )

    parser.add_argument(
        "-v", "--variance_threshold",
        type=float,
        default=0.95,
        help="Variance threshold for PCA (default: 0.95)"
    )

    parser.add_argument(
        "--around_distance",
        type=float,
        default=20.0,
        help="Distance cutoff for nearby protein atoms (default: 20.0 Å)"
    )

    args = parser.parse_args()

    # Run the analysis
    try:
        print(f"Starting analysis for structure: {args.structure_file}")

        # Initialize analyzer
        analyzer = LigPCGen(
            args.structure_file,
            protein_selection=args.protein_sel,
            ligand_selection=args.ligand_sel
        )

        # Generate conformations
        print(f"Generating {args.num_conformations} conformations...")
        conformations = analyzer.generate_conformations(
            num_conformations=args.num_conformations
        )

        # Calculate distances
        print("Calculating protein-ligand distances...")
        distances = analyzer.calculate_distances(
            conformations,
            around_distance=args.around_distance
        )

        # Perform PCA analysis
        print("Performing PCA analysis...")
        pc_, pca = analyzer.analyze_pca(
            distances,
            variance_threshold=args.variance_threshold
        )

        # Find dimension with maximum separation
        max_sep_dim, min_distances = analyzer.find_max_separation_dimension(pc_)

        print("\nResults:")
        print(f"Dimension with maximum separation: {max_sep_dim}")
        print(f"Minimum distances per dimension: {', '.join(f'{d:.4f}' for d in min_distances)}")

        # Save PCA model
        analyzer.save_pca(pca, args.output)
        print(f"\nPCA model saved to {args.output}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
