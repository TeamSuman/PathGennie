#!/usr/bin/env python
"""Utility functions for PathGennie Amber cases."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

try:
    from tqdm.auto import trange  # type: ignore
except ModuleNotFoundError:

    def trange(*args, **kwargs):
        return range(*args)


def resolve_case_path(case_dir: Path, path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return case_dir / path


def load_function(case_dir: Path, module_name: str, function_name: str):
    sys.path.insert(0, str(case_dir))
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def read_rst7_coords(path: str | Path) -> np.ndarray:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid Amber restart file: {path}")

    natom = int(lines[1].split()[0])
    values = []
    for line in lines[2:]:
        for start in range(0, len(line), 12):
            token = line[start : start + 12].strip()
            if not token:
                continue
            if "*" in token:
                raise ValueError(
                    f"Amber restart {path} contains overflow coordinates ({token!r}). "
                    "The MD segment likely became unstable."
                )
            values.append(float(token))
        if len(values) >= natom * 3:
            break

    if len(values) < natom * 3:
        raise ValueError(f"Amber restart file has too few coordinates: {path}")
    return np.asarray(values[: natom * 3], dtype=float).reshape(natom, 3)


def write_rst7_coords(path: str | Path, coords: np.ndarray) -> None:
    """Write a minimal AMBER rst7 restart file from ``(n_atoms, 3)`` coordinates.

    Velocities are set to zero; the file uses the standard Amber restart
    format (12.7f, 6 values per line).  This is intended for checkpoint
    restart — the next cycle's tau1 will randomize velocities anyway.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    natom = coords.shape[0]
    flat = coords.ravel()
    lines = [f"Checkpoint restart", f"{natom:5d}"]
    row: list[str] = []
    for val in flat:
        row.append(f"{val:12.7f}")
        if len(row) == 6:
            lines.append("".join(row))
            row = []
    if row:
        lines.append("".join(row))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")



def read_prmtop_flag(path: Path, flag: str) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip() == f"%FLAG {flag}":
            values = []
            cursor = index + 2
            while cursor < len(lines) and not lines[cursor].startswith("%FLAG"):
                values.append(lines[cursor])
                cursor += 1
            return values
    raise KeyError(f"Missing %FLAG {flag} in {path}")


def parse_prmtop(path: Path) -> dict[str, object]:
    atom_name_text = "".join(read_prmtop_flag(path, "ATOM_NAME"))
    residue_label_text = "".join(read_prmtop_flag(path, "RESIDUE_LABEL"))
    residue_pointer_lines = read_prmtop_flag(path, "RESIDUE_POINTER")
    mass_lines = read_prmtop_flag(path, "MASS")
    bonds_inc_h_lines = read_prmtop_flag(path, "BONDS_INC_HYDROGEN")
    bonds_without_h_lines = read_prmtop_flag(path, "BONDS_WITHOUT_HYDROGEN")

    atom_names = [
        atom_name_text[index : index + 4].strip()
        for index in range(0, len(atom_name_text), 4)
        if atom_name_text[index : index + 4].strip()
    ]
    residue_labels = [
        residue_label_text[index : index + 4].strip()
        for index in range(0, len(residue_label_text), 4)
        if residue_label_text[index : index + 4].strip()
    ]
    residue_pointers = [int(value) for line in residue_pointer_lines for value in line.split()]
    masses = np.asarray([float(value) for line in mass_lines for value in line.split()], dtype=float)
    box_lengths = None
    try:
        box_values = [float(value) for line in read_prmtop_flag(path, "BOX_DIMENSIONS") for value in line.split()]
        if len(box_values) >= 4:
            box_lengths = np.asarray(box_values[1:4], dtype=float)
    except KeyError:
        box_lengths = None

    residue_indices: dict[str, list[np.ndarray]] = {}
    atom_residue_names = []
    atom_residue_numbers = []
    for residue_index, residue_name in enumerate(residue_labels):
        start = residue_pointers[residue_index] - 1
        end = residue_pointers[residue_index + 1] - 1 if residue_index + 1 < len(residue_pointers) else len(atom_names)
        indices = np.arange(start, end, dtype=int)
        residue_indices.setdefault(residue_name, []).append(indices)
        atom_residue_names.extend([residue_name] * len(indices))
        atom_residue_numbers.extend([residue_index + 1] * len(indices))

    if len(atom_names) != len(masses):
        raise ValueError(f"Topology atom/mass count mismatch in {path}")

    bond_values = [
        int(value) for lines in (bonds_inc_h_lines, bonds_without_h_lines) for line in lines for value in line.split()
    ]
    bonds = []
    for index in range(0, len(bond_values), 3):
        if index + 1 >= len(bond_values):
            break
        bonds.append((bond_values[index] // 3, bond_values[index + 1] // 3))
    molecule_indices = connected_components(len(atom_names), bonds)

    return {
        "atom_names": atom_names,
        "atom_residue_names": atom_residue_names,
        "atom_residue_numbers": atom_residue_numbers,
        "masses": masses,
        "box_lengths": box_lengths,
        "residue_indices": residue_indices,
        "molecule_indices": molecule_indices,
    }


def connected_components(n_atoms: int, bonds: list[tuple[int, int]]) -> list[np.ndarray]:
    neighbors = [[] for _ in range(n_atoms)]
    for atom_i, atom_j in bonds:
        if 0 <= atom_i < n_atoms and 0 <= atom_j < n_atoms:
            neighbors[atom_i].append(atom_j)
            neighbors[atom_j].append(atom_i)

    seen = np.zeros(n_atoms, dtype=bool)
    components = []
    for atom_index in range(n_atoms):
        if seen[atom_index]:
            continue
        stack = [atom_index]
        seen[atom_index] = True
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in neighbors[current]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(np.asarray(component, dtype=int))
    return components


def atom_group_for_resname(topology_info: dict[str, object], resname: str) -> np.ndarray:
    groups = topology_info["residue_indices"].get(resname, [])  # type: ignore
    if len(groups) != 1:
        raise ValueError(f"Expected exactly one residue named {resname}, found {len(groups)}")
    return groups[0]


def enrich_args(args: dict[str, object], topology_info: dict[str, object]) -> dict[str, object]:
    args = dict(args)
    group_a_resname = args.pop("group_a_resname", None)
    group_b_resname = args.pop("group_b_resname", None)
    if group_a_resname is not None and group_b_resname is not None:
        args["group_a_indices"] = atom_group_for_resname(topology_info, str(group_a_resname))
        args["group_b_indices"] = atom_group_for_resname(topology_info, str(group_b_resname))
        args["masses"] = topology_info["masses"]
    return args


STANDARD_PROTEIN_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


def infer_element(atom_name: str, residue_name: str = "") -> str:
    stripped = "".join(char for char in atom_name.strip() if char.isalpha())
    if not stripped:
        return "X"
    if residue_name in STANDARD_PROTEIN_RESIDUES:
        return stripped[0].upper()
    if len(stripped) >= 2 and stripped[:2].title() in {"Cl", "Br", "Na", "Ca", "Mg", "Zn"}:
        return stripped[:2].title()
    return stripped[0].upper()


def write_multimodel_pdb(path: Path, topology_info: dict[str, object], frames: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atom_names = topology_info["atom_names"]
    residue_names = topology_info["atom_residue_names"]
    residue_numbers = topology_info["atom_residue_numbers"]

    with path.open("w", encoding="utf-8") as handle:
        for model_index, frame in enumerate(frames, start=1):
            handle.write(f"MODEL     {model_index:4d}\n")
            for atom_index, (atom_name, residue_name, residue_number) in enumerate(  # type: ignore
                zip(atom_names, residue_names, residue_numbers),  # type: ignore
                start=1,
            ):
                x, y, z = frame[atom_index - 1]
                element = infer_element(atom_name, str(residue_name))
                record = "ATOM  " if residue_name in STANDARD_PROTEIN_RESIDUES else "HETATM"
                handle.write(
                    f"{record}{atom_index:5d} {atom_name[:4]:>4s} {residue_name[:3]:>3s} "
                    f"A{int(residue_number) % 10000:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
                )
            handle.write("TER\n")
            handle.write("ENDMDL\n")
        handle.write("END\n")


def write_native_trajectory(
    path: Path, topology_info: dict[str, object], frames: np.ndarray, *, dt: float | None = None,
) -> None:
    """Write a binary trajectory selected by file extension using MDAnalysis.

    Parameters
    ----------
    dt : float, optional
        Time between saved frames in picoseconds.  When provided, the writer
        header (DCD) and per-frame timestamps (XTC/NetCDF) are set so that
        downstream tools see the correct physical time instead of a default
        iteration index.  Note: the last frame of a PathGennie run may be
        closer in time than *dt* if convergence triggered mid-interval.
    """

    try:
        import MDAnalysis as mda
        from MDAnalysis.coordinates import core
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime environment
        raise ModuleNotFoundError("MDAnalysis is required to write native trajectory formats") from exc

    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError("Trajectory frames must have shape (n_frames, n_atoms, 3)")
    if frames.shape[0] == 0:
        raise ValueError("Cannot write an empty trajectory")

    path.parent.mkdir(parents=True, exist_ok=True)
    n_atoms = frames.shape[1]
    universe = mda.Universe.empty(n_atoms, trajectory=True)
    box_lengths = topology_info.get("box_lengths")
    dimensions = None
    if box_lengths is not None:
        box = np.asarray(box_lengths, dtype=np.float32)
        if box.shape == (3,) and np.all(box > 0.0):
            dimensions = np.array([box[0], box[1], box[2], 90.0, 90.0, 90.0], dtype=np.float32)

    writer_kwargs: dict = {"n_atoms": n_atoms}
    if dt is not None:
        writer_kwargs["dt"] = dt
    writer = core.writer(str(path), **writer_kwargs)
    try:
        for i, frame in enumerate(frames):
            universe.atoms.positions = frame  # type: ignore
            universe.trajectory.ts.frame = i
            if dt is not None:
                universe.trajectory.ts.time = i * dt
            if dimensions is not None:
                universe.dimensions = dimensions
            writer.write(universe.atoms)
    finally:
        writer.close()


def write_trajectory(
    path: Path, topology_info: dict[str, object], frames: np.ndarray, *, dt: float | None = None,
) -> None:
    """Write PDB or a native trajectory based on the output extension.

    *dt* (picoseconds between saved frames) is forwarded to
    :func:`write_native_trajectory` for binary formats.  PDB MODEL records
    have no standard per-frame time field, so *dt* is a no-op for ``.pdb``.
    """

    if path.suffix.lower() == ".pdb":
        write_multimodel_pdb(path, topology_info, frames)
    else:
        write_native_trajectory(path, topology_info, frames, dt=dt)


def read_native_trajectory(path: Path, topology: Optional[Path | str] = None) -> np.ndarray:
    """Read a binary trajectory and return all frames as ``(n_frames, n_atoms, 3)`` in Ångström.

    Supports any format MDAnalysis can open (``.xtc``, ``.nc``, ``.dcd``, ``.trr``, etc.).
    Passes optional ``topology`` if available so NetCDF formats have full atom metadata.
    """

    try:
        import MDAnalysis as mda
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("MDAnalysis is required to read native trajectory formats") from exc

    if topology is not None and Path(topology).exists():
        u = mda.Universe(str(topology), str(path))
    else:
        u = mda.Universe(str(path))
    frames = np.array([ts.positions.copy() for ts in u.trajectory], dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError(f"Expected 3-D frames array from {path}, got shape {frames.shape}")
    return frames


def wrap_frames_pbc(frames: np.ndarray, topology_info: dict[str, object]) -> np.ndarray:
    """Wrap whole molecules into an orthorhombic AMBER box when available."""

    box_lengths = topology_info.get("box_lengths")
    if box_lengths is None:
        return frames
    box = np.asarray(box_lengths, dtype=float)
    if box.shape != (3,) or np.any(box <= 0.0):
        return frames
    molecule_indices = topology_info.get("molecule_indices")
    if not molecule_indices:
        return np.mod(frames, box)

    wrapped = np.asarray(frames, dtype=float).copy()
    for frame in wrapped:
        for indices in molecule_indices:  # type: ignore
            coords = frame[indices]
            center = coords.mean(axis=0)
            shift = -np.floor(center / box) * box
            frame[indices] = coords + shift
    return wrapped


def write_metrics_csv(path: Path, metrics: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("cycle,metric\n")
        for cycle, metric in enumerate(metrics):
            handle.write(f"{cycle},{float(metric):.8f}\n")


def default_mdin_controls(system_kind: str) -> dict[str, object]:
    if system_kind == "implicit":
        return {
            "dt": 0.002,
            "ntc": 2,
            "ntf": 2,
            "ntt": 3,
            "gamma_ln": 5.0,
            "ntb": 0,
            "igb": 1,
            "cut": 999.0,
            "ntpr": 100,
            "ntwx": 0,
            "ntwr": 10000,
            "ioutfm": 0,
            "ntxo": 1,
        }
    if system_kind == "vacuum":
        return {
            "dt": 0.001,
            "ntc": 1,
            "ntf": 1,
            "ntt": 3,
            "gamma_ln": 5.0,
            "ntb": 0,
            "cut": 99.0,
            "ntpr": 100,
            "ntwx": 0,
            "ntwr": 10000,
            "ioutfm": 0,
            "ntxo": 1,
        }
    return {
        "dt": 0.002,
        "ntc": 2,
        "ntf": 2,
        "ntt": 3,
        "gamma_ln": 5.0,
        "ntb": 1,
        "ntp": 0,
        "cut": 9.0,
        "iwrap": 0,
        "nscm": 100,
        "ntpr": 100,
        "ntwx": 0,
        "ntwr": 10000,
        "ioutfm": 0,
        "ntxo": 1,
    }


def write_mdin(
    path: Path,
    nstlim: int,
    temperature: float,
    controls: dict[str, object],
    *,
    continue_velocities: bool,
    random_seed: int,
    extra_text: str = "",
) -> None:
    values = dict(controls)
    values.update(
        {
            "imin": 0,
            "irest": 1 if continue_velocities else 0,
            "ntx": 5 if continue_velocities else 1,
            "nstlim": int(nstlim),
            "temp0": float(temperature),
            "ig": int(random_seed),
        }
    )
    if not continue_velocities:
        values.setdefault("tempi", float(temperature))
    else:
        values.pop("tempi", None)

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["PathGennie Amber MD", " &cntrl"]
    for key, value in values.items():
        if isinstance(value, str):
            lines.append(f"    {key} = '{value}',")
        elif isinstance(value, float):
            lines.append(f"    {key} = {value:.6g},")
        else:
            lines.append(f"    {key} = {value},")
    lines.append(" /")
    if extra_text:
        lines.append(extra_text.rstrip())
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
