"""Backend-independent engine protocol for PathGennie.

The OpenMM, AMBER and GROMACS backends each implement the same adaptive cycle
but with engine-specific state handling.  This module defines the thin contract
the shared driver (:mod:`pathgennie.core.driver`) needs from any backend so the
cycle/selection logic only has to be written once.

A *handle* is an opaque, engine-defined token that names a stored simulation
state.  For the subprocess backends it is typically a restart-file path
(``rst7``/``gro``); for the in-process OpenMM worker pool it is an id into a
per-worker state cache.  The driver never inspects a handle, it only passes it
back to the engine.
"""

from __future__ import annotations

from typing import Hashable, Optional, Protocol, runtime_checkable

import numpy as np

Handle = Hashable

__all__ = ["Handle", "Engine"]


@runtime_checkable
class Engine(Protocol):
    """Minimal MD-engine interface used by :class:`PathGennieDriver`."""

    def clone_anchor(self, handle: Handle) -> Handle:
        """Return a fresh, independent handle initialised from ``handle``.

        Used to seed each sampler trial from the current anchor without the
        trials clobbering one another's state/scratch files.
        """
        ...

    def run_segment(
        self,
        handle: Handle,
        n_steps: int,
        *,
        randomize_velocities: bool,
        seed: int,
        device: Optional[int] = None,
        save_subframes: bool = False,
        subframe_stride: int = 1,
    ) -> "Handle | tuple[Handle, np.ndarray]":
        """Propagate ``handle`` for ``n_steps`` and return the resulting handle.

        ``randomize_velocities`` selects samplers (τ1, fresh Maxwell-Boltzmann
        velocities) vs runners (τ2, continued velocities).  ``seed`` makes the
        segment reproducible; ``device`` is the GPU index when a device pool is
        in use (``None`` lets the engine choose).

        When ``save_subframes`` is True, intermediate positions are captured
        every ``subframe_stride`` integrator steps and the return changes to
        ``(Handle, subframes)`` where ``subframes`` is an
        ``(n_subframes, n_atoms, 3)`` array in Ångström.

        The tuple is returned **whenever ``save_subframes`` is True**, even if no
        frames were captured; ``subframes`` is then empty with a valid shape.
        Callers unpack unconditionally, so returning a bare handle here would make
        that unpack raise -- on the subprocess backends a handle is a file path,
        which unpacks as characters or not at all.

        Engines may differ on the tail: with a stride longer than the segment,
        OpenMM steps in ``min(stride, remaining)`` chunks and captures the segment
        end, while the toy engine's strict modulo captures nothing. Both are
        acceptable; do not rely on the exact frame count.
        """
        ...

    def get_coords(self, handle: Handle) -> np.ndarray:
        """Return positions for ``handle`` as an ``(n_atoms, 3)`` array in Angstrom."""
        ...

    def release(self, handle: Handle) -> None:
        """Release any resources (scratch files / cache entry) for ``handle``.

        Implementations must tolerate being called with an already-released or
        anchor handle (no-op) so the driver can release trials unconditionally.
        """
        ...

    def create_handle(self, coords: np.ndarray) -> Handle:
        """Create a new handle from raw coordinates (Ångström, ``(n_atoms, 3)``).

        Used by checkpoint restart to re-create an anchor handle from the saved
        coordinates.  Velocities do not need to be restored because the next
        cycle always starts with ``randomize_velocities=True`` for τ1.
        """
        ...
