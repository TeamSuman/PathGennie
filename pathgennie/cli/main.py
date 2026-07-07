import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        prog="pathgennie",
        description="PathGennie: Adaptive short-burst molecular dynamics path generation.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # Subcommand: amber
    parser_amber = subparsers.add_parser(
        "amber",
        help="Run PathGennie with Amber backend"
    )
    parser_amber.add_argument("--case", type=Path, default=Path.cwd(), help="Directory containing config (default: current directory)")
    parser_amber.add_argument("--config", default="input.yaml", help="YAML config name inside the case directory (default: input.yaml)")

    # Subcommand: gromacs
    parser_gmx = subparsers.add_parser(
        "gromacs",
        help="Run PathGennie with GROMACS backend"
    )
    parser_gmx.add_argument("--case", type=Path, default=Path.cwd(), help="Directory containing config (default: current directory)")
    parser_gmx.add_argument("--config", default="input.yaml", help="YAML config name inside the case directory (default: input.yaml)")

    # Subcommand: openmm
    parser_omm = subparsers.add_parser(
        "openmm",
        help="Run PathGennie with OpenMM backend"
    )
    parser_omm.add_argument("--case", type=Path, default=Path.cwd(), help="Directory containing config (default: current directory)")
    parser_omm.add_argument("--config", default="input.yaml", help="YAML config name inside the case directory (default: input.yaml)")

    # Subcommand: pcagen
    parser_pca = subparsers.add_parser(
        "pcagen",
        help="Analyze protein-ligand conformations and construct a robust PCA distance CV space"
    )
    parser_pca.add_argument("structure_file", type=str, help="Path to the structure file (e.g., .gro, .pdb)")
    parser_pca.add_argument("-n", "--num_conformations", type=int, default=10000, help="Number of conformations to generate (default: 10000)")
    parser_pca.add_argument("-o", "--output", type=str, default="output_pca.pkl", help="Output filename for PCA model (default: output_pca.pkl)")
    parser_pca.add_argument("--protein_sel", type=str, default="protein", help="MDAnalysis selection string for protein (default: 'protein')")
    parser_pca.add_argument("--ligand_sel", type=str, default="resname LIG", help="MDAnalysis selection string for ligand (default: 'resname LIG')")
    parser_pca.add_argument("-v", "--variance_threshold", type=float, default=0.95, help="Variance threshold for PCA (default: 0.95)")
    parser_pca.add_argument("--around_distance", type=float, default=20.0, help="Distance cutoff for nearby protein atoms (default: 20.0 Å)")

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "amber":
        from pathgennie.backends.amber.pg_amber import run
        run(args.case, args.config)
    elif args.command == "gromacs":
        from pathgennie.backends.gromacs.pg_gmx import run
        run(args.case, args.config)
    elif args.command == "openmm":
        from pathgennie.backends.openmm.pg_openmm import run
        run(args.case, args.config)
    elif args.command == "pcagen":
        from pathgennie.utils.ligcvgen import LigPCGen
        try:
            print(f"Starting analysis for structure: {args.structure_file}")
            analyzer = LigPCGen(
                args.structure_file,
                protein_selection=args.protein_sel,
                ligand_selection=args.ligand_sel
            )
            print(f"Generating {args.num_conformations} conformations...")
            conformations = analyzer.generate_conformations(
                num_conformations=args.num_conformations
            )
            print("Calculating protein-ligand distances...")
            distances = analyzer.calculate_distances(
                conformations,
                around_distance=args.around_distance
            )
            print("Performing PCA analysis...")
            pc_, pca = analyzer.analyze_pca(
                distances,
                variance_threshold=args.variance_threshold
            )
            max_sep_dim, min_distances = analyzer.find_max_separation_dimension(pc_)
            print("\nResults:")
            print(f"Dimension with maximum separation: {max_sep_dim}")
            print(f"Minimum distances per dimension: {', '.join(f'{d:.4f}' for d in min_distances)}")
            analyzer.save_pca(pca, args.output)
            print(f"\nPCA model saved to {args.output}")
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    main()
