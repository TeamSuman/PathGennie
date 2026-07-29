"""Utility functions for PathGennie GROMACS cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pathgennie.backends.amber.utils import (
    enrich_args,
    load_function,
    parse_prmtop,
    resolve_case_path,
    write_metrics_csv,
    write_multimodel_pdb,
    write_trajectory,
)

__all__ = [
    "enrich_args",
    "load_function",
    "read_gro_coords",
    "read_masses_from_topology",
    "read_topology_info",
    "resolve_case_path",
    "write_gro_coords",
    "write_metrics_csv",
    "write_multimodel_pdb",
    "write_trajectory",
]


def read_masses_from_topology(
    topology_path: str | Path, include_dir: str | Path | None = None
) -> np.ndarray | None:
    """Return per-atom masses (amu) from a GROMACS topology, or ``None``.

    A ``.gro``/``.pdb`` coordinate file carries no masses, so the metadata readers
    below can only supply placeholders. Any mass-weighted collective variable (a
    centre-of-mass distance, say) silently degrades to an unweighted centroid when
    fed those placeholders, so the real masses are recovered from the topology here.

    ParmEd is tried first (purpose-built for GROMACS topologies and handles
    ``#include`` via ``include_dir``), then MDAnalysis. ``None`` is returned when
    neither can parse the file — callers must treat that as "masses unknown" rather
    than substituting ones.
    """

    topology_path = Path(topology_path)
    if not topology_path.exists():
        return None

    try:  # ParmEd: best GROMACS .top support
        import parmed as pmd

        kwargs = {}
        if include_dir is not None:
            kwargs["includeDir"] = str(include_dir)
        structure = pmd.load_file(str(topology_path), **kwargs)
        masses = np.asarray([atom.mass for atom in structure.atoms], dtype=float)
        if masses.size and np.all(np.isfinite(masses)) and masses.max() > 0:
            return masses
    except Exception:  # noqa: BLE001 - fall through to the next parser
        pass

    try:  # MDAnalysis is a hard dependency, so this is the reliable fallback
        import warnings

        import MDAnalysis as mda

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            universe = mda.Universe(str(topology_path), topology_format="ITP")
            masses = np.asarray(universe.atoms.masses, dtype=float)
        if masses.size and np.all(np.isfinite(masses)) and masses.max() > 0:
            return masses
    except Exception:  # noqa: BLE001 - masses simply unavailable
        pass

    return None


def read_topology_info(path: str | Path) -> dict[str, object]:
    """Read atom/residue metadata from PDB, GRO, or AMBER PRMTOP files."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".prmtop", ".parm7", ".top"} and _looks_like_prmtop(path):
        return parse_prmtop(path)
    if suffix == ".pdb":
        return read_pdb_topology_info(path)
    if suffix == ".gro":
        return read_gro_topology_info(path)
    raise ValueError(
        f"Unsupported metadata file for PDB output: {path}. "
        "Use a .pdb, .gro, or AMBER .prmtop/.parm7 file."
    )


def read_gro_coords(path: str | Path) -> np.ndarray:
    """Read GROMACS .gro coordinates and return Angstrom coordinates."""

    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid GRO file: {path}")

    natom = int(lines[1].strip())
    atom_lines = lines[2 : 2 + natom]
    if len(atom_lines) != natom:
        raise ValueError(f"GRO file has too few atom lines: {path}")

    coords = []
    for line in atom_lines:
        coords.append(
            [
                float(line[20:28]) * 10.0,
                float(line[28:36]) * 10.0,
                float(line[36:44]) * 10.0,
            ]
        )
    return np.asarray(coords, dtype=float)


def write_gro_coords(template_gro: str | Path, out_gro: str | Path, coords: np.ndarray) -> None:
    """Write a new .gro file using header/atom info from template_gro and updated coordinates (Ångström)."""
    template_path = Path(template_gro)
    lines = template_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid GRO file: {template_path}")

    natom = int(lines[1].strip())
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    if coords.shape[0] != natom:
        raise ValueError(f"Coordinate count mismatch: expected {natom}, got {coords.shape[0]}")

    out_lines = [lines[0], lines[1]]
    for i, line in enumerate(lines[2 : 2 + natom]):
        # gro format: positions in nm, formatted as %8.3f in columns 21-44 (1-indexed) -> [20:28], [28:36], [36:44]
        prefix = line[:20]
        suffix = line[44:] if len(line) > 44 else ""
        x_nm, y_nm, z_nm = coords[i] / 10.0
        pos_str = f"{x_nm:8.3f}{y_nm:8.3f}{z_nm:8.3f}"
        out_lines.append(f"{prefix}{pos_str}{suffix}")

    # Append box line if present
    if len(lines) > 2 + natom:
        out_lines.extend(lines[2 + natom :])

    Path(out_gro).write_text("\n".join(out_lines) + "\n", encoding="utf-8")



def read_gro_topology_info(path: str | Path) -> dict[str, object]:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid GRO file: {path}")

    natom = int(lines[1].strip())
    atom_names: list[str] = []
    atom_residue_names: list[str] = []
    atom_residue_numbers: list[int] = []
    residue_indices: dict[str, list[np.ndarray]] = {}

    residue_atom_map: dict[tuple[int, str], list[int]] = {}
    for atom_index, line in enumerate(lines[2 : 2 + natom]):
        residue_number = int(line[0:5])
        residue_name = line[5:10].strip()
        atom_name = line[10:15].strip()
        atom_names.append(atom_name)
        atom_residue_names.append(residue_name)
        atom_residue_numbers.append(residue_number)
        residue_atom_map.setdefault((residue_number, residue_name), []).append(atom_index)

    for (_residue_number, residue_name), indices in residue_atom_map.items():
        residue_indices.setdefault(residue_name, []).append(np.asarray(indices, dtype=int))

    box_lengths = None
    if len(lines) > 2 + natom:
        box_values = [float(value) for value in lines[2 + natom].split()]
        if len(box_values) >= 3:
            box_lengths = np.asarray(box_values[:3], dtype=float) * 10.0

    return {
        "atom_names": atom_names,
        "atom_residue_names": atom_residue_names,
        "atom_residue_numbers": atom_residue_numbers,
        # A .gro carries no masses. This is a PLACEHOLDER: read_masses_from_topology()
        # must supply the real values before any mass-weighted CV uses them.
        "masses": np.ones(natom, dtype=float),
        "masses_are_placeholder": True,
        "box_lengths": box_lengths,
        "residue_indices": residue_indices,
    }


def read_pdb_topology_info(path: str | Path) -> dict[str, object]:
    path = Path(path)
    atom_names: list[str] = []
    atom_residue_names: list[str] = []
    atom_residue_numbers: list[int] = []
    residue_atom_map: dict[tuple[int, str], list[int]] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("ENDMDL") and atom_names:
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_index = len(atom_names)
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        residue_number_text = line[22:26].strip()
        residue_number = int(residue_number_text) if residue_number_text else 1
        atom_names.append(atom_name)
        atom_residue_names.append(residue_name)
        atom_residue_numbers.append(residue_number)
        residue_atom_map.setdefault((residue_number, residue_name), []).append(atom_index)

    if not atom_names:
        raise ValueError(f"No ATOM/HETATM records found in PDB file: {path}")

    residue_indices: dict[str, list[np.ndarray]] = {}
    for (_residue_number, residue_name), indices in residue_atom_map.items():
        residue_indices.setdefault(residue_name, []).append(np.asarray(indices, dtype=int))

    return {
        "atom_names": atom_names,
        "atom_residue_names": atom_residue_names,
        "atom_residue_numbers": atom_residue_numbers,
        # A .pdb carries no masses -- placeholder, see read_masses_from_topology().
        "masses": np.ones(len(atom_names), dtype=float),
        "masses_are_placeholder": True,
        "residue_indices": residue_indices,
    }


def _looks_like_prmtop(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8") as handle:
            for _ in range(20):
                line = handle.readline()
                if not line:
                    return False
                if line.startswith("%FLAG"):
                    return True
    except UnicodeDecodeError:
        return False
    return False
