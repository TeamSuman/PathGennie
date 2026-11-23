"""Per-segment adaptive QM region for AMBER QM/MM.

A reaction in water usually needs a few *specific* water molecules inside the QM region -- the
nucleophile, the base that takes a proton, the waters that solvate the leaving group -- but
which ones changes as the system moves. A fixed ``qmmask`` either freezes the wrong waters in
or keeps all of them out (then a proton has nowhere to go and the chemistry is an artefact of
the partition).

This module re-selects the QM solvent at the start of every segment:

* the **core** residues (solute, nucleophile) are always QM;
* plus the ``n_solvent`` whole solvent molecules whose heavy atoms are nearest to the **centre**
  atoms (e.g. the reacting carbon and the leaving group);
* plus a **proton-transfer guard**: any further solvent molecule with an H--heavy-atom contact
  across the QM/MM boundary shorter than ``guard_cutoff``, so a proton being shared is never
  split between the two regions.

Two properties make this safe inside a selection-based sampler:

* The region is fixed *within* a segment, so each segment is ordinary QM/MM dynamics on one
  Hamiltonian. The region changes only at segment boundaries, where PathGennie compares
  geometric progress, never energies.
* Selection is a deterministic function of the starting frame and of the region its parent
  used. Every trial of a cycle starts from the same anchor, so all of them see the same QM
  region and their scores are comparable. The parent's region travels with the restart as a
  ``.qmsel`` sidecar file, so no shared mutable state exists between concurrent segments.

**Hysteresis.** Incumbent waters get a distance bonus of ``hysteresis`` (Angstrom) when ranked.
An outsider displaces one only if it is closer by more than that. This stops two near-equidistant
waters from flickering in and out from one segment to the next.

Only whole molecules (residues) are ever moved across the boundary, so ``qmcharge`` stays
constant and no link atoms appear. This holds as long as the solvent residues are neutral, as
water is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "AdaptiveQMRegion",
    "format_residue_mask",
    "read_qm_selection",
    "selection_path",
    "write_qm_selection",
]

_QMMASK_RE = re.compile(r"""qmmask\s*=\s*(['"]).*?\1""", re.IGNORECASE | re.DOTALL)
HYDROGEN_MAX_MASS = 1.2


def format_residue_mask(residues: Iterable[int]) -> str:
    """Amber residue mask for 1-based residue numbers, consecutive runs collapsed: ``:1-2,57``."""
    res = sorted({int(r) for r in residues})
    if not res:
        raise ValueError("empty QM residue selection")
    runs, start, prev = [], res[0], res[0]
    for r in res[1:]:
        if r == prev + 1:
            prev = r
            continue
        runs.append((start, prev))
        start = prev = r
    runs.append((start, prev))
    return ":" + ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def selection_path(handle) -> Path:
    """Sidecar that records the QM residues a restart was produced with."""
    return Path(handle).with_suffix(".qmsel")


def read_qm_selection(handle) -> Optional[tuple[int, ...]]:
    path = selection_path(handle)
    if not path.exists():
        return None
    text = path.read_text().split()
    return tuple(int(t) for t in text) if text else None


def write_qm_selection(handle, residues: Sequence[int]) -> None:
    selection_path(handle).write_text(" ".join(str(int(r)) for r in residues) + "\n")


def _min_image(delta: np.ndarray, box) -> np.ndarray:
    if box is None:
        return delta
    b = np.asarray(box, dtype=float).ravel()[:3]
    if b.shape != (3,) or not np.all(b > 0):
        return delta
    return delta - b * np.round(delta / b)


@dataclass
class AdaptiveQMRegion:
    """Choose the QM residues for one segment from its starting coordinates.

    Residue numbers are 1-based (Amber convention); atom indices are 0-based.
    """

    residue_atoms: list            # residue_atoms[r - 1] -> 0-based atom indices of residue r
    core_residues: tuple
    centre_atoms: tuple
    solvent_residues: np.ndarray   # 1-based residue numbers eligible to enter the QM region
    is_hydrogen: np.ndarray        # bool per atom
    n_solvent: int
    hysteresis: float = 0.3
    guard_cutoff: float = 1.3
    _solvent_atom_index: np.ndarray = field(init=False, repr=False)
    _solvent_atom_owner: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.core_residues = tuple(sorted({int(r) for r in self.core_residues}))
        self.centre_atoms = tuple(int(a) for a in self.centre_atoms)
        self.solvent_residues = np.asarray(sorted(set(int(r) for r in self.solvent_residues)), dtype=int)
        overlap = set(self.core_residues) & set(self.solvent_residues.tolist())
        if overlap:
            raise ValueError(f"residues {sorted(overlap)} are both core and solvent")
        if not self.centre_atoms:
            raise ValueError("at least one centre atom is required")
        if self.n_solvent < 0 or self.n_solvent > len(self.solvent_residues):
            raise ValueError(f"n_solvent={self.n_solvent} outside [0, {len(self.solvent_residues)}]")
        # Flattened solvent atoms and the position (in solvent_residues) of the residue owning
        # each, so one vectorised distance call ranks every solvent molecule.
        idx, own = [], []
        for k, r in enumerate(self.solvent_residues):
            atoms = np.asarray(self.residue_atoms[r - 1], dtype=int)
            idx.append(atoms)
            own.append(np.full(len(atoms), k, dtype=int))
        self._solvent_atom_index = np.concatenate(idx) if idx else np.zeros(0, dtype=int)
        self._solvent_atom_owner = np.concatenate(own) if own else np.zeros(0, dtype=int)

    @classmethod
    def from_topology(cls, topology_info: dict, *, core_residues: Sequence[int],
                      centre_atoms: Sequence[int], n_solvent: int,
                      solvent_resname: str = "WAT", hysteresis: float = 0.3,
                      guard_cutoff: float = 1.3) -> "AdaptiveQMRegion":
        """Build from :func:`pathgennie.backends.amber.utils.parse_prmtop` output."""
        numbers = np.asarray(topology_info["atom_residue_numbers"], dtype=int)
        names = topology_info["atom_residue_names"]
        n_res = int(numbers.max())
        residue_atoms = [[] for _ in range(n_res)]
        for atom, r in enumerate(numbers):
            residue_atoms[r - 1].append(atom)
        residue_atoms = [np.asarray(a, dtype=int) for a in residue_atoms]
        solvent = sorted({int(numbers[a]) for a in range(len(numbers)) if names[a] == solvent_resname}
                         - {int(r) for r in core_residues})
        if not solvent:
            raise ValueError(f"no residues named {solvent_resname!r} in the topology")
        masses = np.asarray(topology_info["masses"], dtype=float)
        return cls(residue_atoms=residue_atoms, core_residues=tuple(core_residues),
                   centre_atoms=tuple(centre_atoms), solvent_residues=np.asarray(solvent),
                   is_hydrogen=masses < HYDROGEN_MAX_MASS, n_solvent=int(n_solvent),
                   hysteresis=float(hysteresis), guard_cutoff=float(guard_cutoff))

    # ------------------------------------------------------------------ selection
    def solvent_distances(self, coords: np.ndarray, box=None) -> np.ndarray:
        """Closest approach of each solvent residue's HEAVY atoms to any centre atom (minimum image).

        Heavy atoms only: water H librate on a ~10 fs timescale, so ranking on them would
        reshuffle near-equidistant waters every segment. Short H contacts are the guard's job.
        """
        x = np.asarray(coords, dtype=float).reshape(-1, 3)
        centres = x[list(self.centre_atoms)]
        heavy = ~self.is_hydrogen[self._solvent_atom_index]
        sol = x[self._solvent_atom_index[heavy]]
        d = np.linalg.norm(_min_image(sol[:, None, :] - centres[None, :, :], box), axis=-1).min(axis=1)
        out = np.full(len(self.solvent_residues), np.inf)
        np.minimum.at(out, self._solvent_atom_owner[heavy], d)
        return out

    def select(self, coords: np.ndarray, *, box=None,
               previous: Optional[Sequence[int]] = None) -> tuple[int, ...]:
        """Sorted 1-based QM residue numbers for a segment starting at ``coords``."""
        x = np.asarray(coords, dtype=float).reshape(-1, 3)
        dist = self.solvent_distances(x, box)
        score = dist.copy()
        if previous is not None and self.hysteresis > 0:
            incumbent = np.isin(self.solvent_residues, np.asarray(list(previous), dtype=int))
            score[incumbent] -= self.hysteresis
        # stable: equal scores resolve by residue number, so selection is reproducible
        order = np.lexsort((self.solvent_residues, score))
        chosen = set(self.solvent_residues[order[: self.n_solvent]].tolist())
        chosen |= self._guard(x, box, set(self.core_residues) | chosen)
        return tuple(sorted(set(self.core_residues) | chosen))

    def _guard(self, x: np.ndarray, box, qm: set) -> set:
        """Solvent residues sharing a short H--heavy contact with the QM region (one pass)."""
        if self.guard_cutoff <= 0:
            return set()
        qm_atoms = np.concatenate([self.residue_atoms[r - 1] for r in sorted(qm)])
        outside = ~np.isin(self.solvent_residues, np.asarray(sorted(qm), dtype=int))
        mm_mask = outside[self._solvent_atom_owner]
        mm_atoms = self._solvent_atom_index[mm_mask]
        mm_owner = self.solvent_residues[self._solvent_atom_owner[mm_mask]]
        if len(mm_atoms) == 0:
            return set()
        qh = self.is_hydrogen[qm_atoms]
        mh = self.is_hydrogen[mm_atoms]
        d = np.linalg.norm(_min_image(x[mm_atoms][:, None, :] - x[qm_atoms][None, :, :], box), axis=-1)
        # an H on one side and a heavy atom on the other
        cross = (mh[:, None] != qh[None, :])
        hit = ((d < self.guard_cutoff) & cross).any(axis=1)
        return set(mm_owner[hit].tolist())

    # ------------------------------------------------------------------ mdin text
    def mask(self, residues: Sequence[int]) -> str:
        return format_residue_mask(residues)

    def n_atoms(self, residues: Sequence[int]) -> int:
        return int(sum(len(self.residue_atoms[r - 1]) for r in residues))

    @staticmethod
    def check_template(extra_text: str) -> None:
        if not _QMMASK_RE.search(extra_text or ""):
            raise ValueError("adaptive QM needs a qmmask='...' entry in md.extra_text to replace")

    def apply(self, extra_text: str, residues: Sequence[int]) -> str:
        """``extra_text`` with its ``qmmask`` replaced by the selection."""
        self.check_template(extra_text)
        return _QMMASK_RE.sub(f"qmmask='{self.mask(residues)}'", extra_text, count=1)
