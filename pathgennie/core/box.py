"""Periodic box of each frame, carried with its coordinates.

Many collective variables need the *current* simulation box -- minimum-image
distances, cluster analysis across periodic boundaries, densities -- and under a
barostat that box changes from frame to frame. Every backend already knows it
(OpenMM ``State`` box vectors, the last line of a ``.gro``, the box line of an
``rst7``); this module carries it from the engine to the user's functions without
changing their call signature for anyone who does not ask for it.

Convention: a box is a ``(3, 3)`` float array of box vectors in Ångström, one
vector per row (``box[0]`` is ``a``), or ``None`` for a non-periodic system. An
orthorhombic box is ``np.diag([lx, ly, lz])``; use ``box.diagonal()`` for edge
lengths.

How it flows:

* An engine *may* implement ``get_box(handle)``. :func:`frame_coords` asks the
  engine for coordinates and box together and returns a :class:`FrameCoords` --
  an ``ndarray`` that also carries ``.box`` -- so the box follows the
  coordinates through the driver's anchor/trial bookkeeping, storage and
  checkpoints without separate plumbing.
* :func:`call_with_box` calls a projection or convergence function with
  ``box=...`` **only if the function declares a parameter named ``box``** (the
  same opt-in rule as ``cycle``); it then receives the coordinates as a plain
  array. A box from the engine overrides a static ``box`` given in the config;
  when the engine has none (``None``), a static value in the config is left in
  place. A function that does not declare ``box`` receives the
  :class:`FrameCoords` itself, an ``ndarray`` like before, so wrappers (backend
  convergence closures, the downstream scalar CV) can pass the box on.
* An engine *may* accept ``create_handle(coords, box=...)``; :func:`create_handle`
  passes the box only to engines that do, so a checkpoint resume restores the
  anchor's own box (essential under NPT) instead of the starting box.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

import numpy as np

__all__ = [
    "FrameCoords",
    "normalize_box",
    "box_from_lengths_angles",
    "lengths_angles",
    "engine_box",
    "frame_coords",
    "accepts_box",
    "call_with_box",
    "create_handle",
]


def lengths_angles(box: Any) -> Optional[np.ndarray]:
    """``[a, b, c, alpha, beta, gamma]`` (Å, degrees) of a box, or ``None``."""
    m = normalize_box(box)
    if m is None:
        return None
    a, b, c = np.linalg.norm(m, axis=1)
    ang = lambda u, v: np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
    return np.array([a, b, c, ang(m[1], m[2]), ang(m[0], m[2]), ang(m[0], m[1])])


def box_from_lengths_angles(a: float, b: float, c: float,
                            alpha: float = 90.0, beta: float = 90.0, gamma: float = 90.0) -> np.ndarray:
    """Box vectors (rows, Å) from edge lengths and angles in degrees (``a`` along
    x, ``b`` in the xy plane), as AMBER/CHARMM/PDB unit cells are given."""
    al, be, ga = np.radians([alpha, beta, gamma])
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    box = np.array([va, vb, [cx, cy, cz]])
    box[np.abs(box) < 1e-9] = 0.0
    return box


def normalize_box(box: Any) -> Optional[np.ndarray]:
    """Return ``box`` as a ``(3, 3)`` array of row vectors in Å, or ``None``.

    Accepts ``None``; three edge lengths (orthorhombic); six values (three
    lengths and three angles in degrees); or a ``(3, 3)`` matrix of row vectors.
    """
    if box is None:
        return None
    arr = np.asarray(box, dtype=float)
    if arr.shape == (3, 3):
        return arr.copy()
    flat = arr.ravel()
    if flat.size == 3:
        return np.diag(flat)
    if flat.size == 6:
        return box_from_lengths_angles(*flat)
    raise ValueError(f"box must be None, 3 lengths, 3 lengths + 3 angles, or a (3, 3) matrix; got shape {arr.shape}")


class FrameCoords(np.ndarray):
    """``(n_atoms, 3)`` coordinates in Å that also carry the frame's ``.box``.

    A plain ``ndarray`` subclass: anything that accepts coordinates accepts it.
    Copies and slices keep the box; ``np.asarray`` drops it, which is how
    :func:`call_with_box` hands user functions a plain array.
    """

    def __new__(cls, coords, box=None):
        obj = np.asarray(coords).view(cls)
        obj.box = normalize_box(box)
        return obj

    def __array_finalize__(self, obj):
        self.box = getattr(obj, "box", None)

    def __reduce__(self):
        # Keep the box through pickling (process pools, multiprocessing).
        state = super().__reduce__()
        return state[0], state[1], (state[2], self.box)

    def __setstate__(self, state):
        nd_state, box = state
        super().__setstate__(nd_state)
        self.box = box


def engine_box(engine: Any, handle: Any) -> Optional[np.ndarray]:
    """``engine.get_box(handle)`` normalised, or ``None`` if the engine has no box."""
    get_box = getattr(engine, "get_box", None)
    if get_box is None:
        return None
    return normalize_box(get_box(handle))


def frame_coords(engine: Any, handle: Any) -> FrameCoords:
    """Coordinates of ``handle`` with their box attached."""
    return FrameCoords(engine.get_coords(handle), engine_box(engine, handle))


_ACCEPTS: dict = {}


def _accepts(fn: Callable, name: str) -> bool:
    key = (getattr(fn, "__func__", fn), name)
    try:
        hit = _ACCEPTS.get(key)
    except TypeError:  # unhashable callable
        hit, key = None, None
    if hit is None:
        try:
            hit = name in inspect.signature(fn).parameters
        except (TypeError, ValueError):  # builtins without a signature
            hit = False
        if key is not None:
            _ACCEPTS[key] = hit
    return hit


def accepts_box(fn: Callable) -> bool:
    """True if ``fn`` declares a parameter named ``box`` (``**kwargs`` alone does not count)."""
    return _accepts(fn, "box")


def call_with_box(fn: Callable, coords, *args, **kwargs):
    """Call ``fn(coords, *args, **kwargs)``, adding ``box=`` from ``coords.box``
    when ``fn`` declares ``box`` and a box is known (``fn`` then gets a plain
    array); otherwise ``coords`` is passed through unchanged."""
    box = getattr(coords, "box", None)
    if box is not None and accepts_box(fn):
        kwargs["box"] = box
        return fn(np.asarray(coords), *args, **kwargs)
    return fn(coords, *args, **kwargs)


def create_handle(engine: Any, coords, box: Any = None):
    """``engine.create_handle(coords, box=box)`` if the engine takes a box (and one
    is given, here or on ``coords``), else ``engine.create_handle(coords)``."""
    box = normalize_box(box if box is not None else getattr(coords, "box", None))
    if box is not None and _accepts(engine.create_handle, "box"):
        return engine.create_handle(np.asarray(coords), box=box)
    return engine.create_handle(np.asarray(coords))
