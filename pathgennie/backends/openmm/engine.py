"""OpenMM in-process engine adapter for the PathGennie core driver.

Wraps a single ``openmm.app.Simulation`` (one Context, one device) behind the
:class:`pathgennie.core.engine.Engine` protocol so the shared
:class:`~pathgennie.core.driver.PathGennieDriver` can run the OpenMM backend.

A *handle* is an integer id into a cache of immutable OpenMM ``State`` snapshots.
Because a ``State`` is immutable, ``clone_anchor`` can hand out a new id that
shares the same snapshot — ``run_segment`` always ``setState``s, steps, and then
stores a *new* snapshot, so trials never alias mutable state.  ``setState``/
``getState`` round-trip positions, velocities **and periodic box vectors**, which
fixes the box-desync issue in the original loop that only set the box at build
time.

For multi-GPU in-process execution, construct one ``OpenMMEngine`` per worker
process (each with a Simulation pinned to its ``CudaDeviceIndex``); OpenMM
Contexts are not thread-safe, so a process pool — not threads — is required.

Reproducibility note: the driver's selection draws and the per-segment velocity
randomisation are seeded and reproducible.  A *stochastic* integrator's noise
stream (e.g. Langevin thermostat) is initialised at context creation and is not
reliably re-seedable per segment in OpenMM, so bit-identical trajectories are
only guaranteed with a deterministic integrator (e.g. Verlet).  The best-effort
``setRandomNumberSeed`` below helps engines/platforms that honour it.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from openmm import State, unit
from openmm.app import Simulation

from pathgennie.core.engine import Handle
from pathgennie.core.progress import ProgressVariable

NM_TO_ANG = 10.0

__all__ = ["OpenMMEngine"]


class OpenMMEngine:
    def __init__(self, simulation: Simulation, temperature: float):
        self.sim = simulation
        self.temperature = temperature * unit.kelvin  # type: ignore
        self._cache: Dict[int, State] = {}
        self._next_id = 0

    def _store(self, state: State) -> int:
        handle = self._next_id
        self._next_id += 1
        self._cache[handle] = state
        return handle

    def _snapshot(self) -> State:
        return self.sim.context.getState(getPositions=True, getVelocities=True)

    def create_state(self, positions, box_vectors=None) -> int:
        """Initialise the cache from positions (and optional box); return a handle."""
        self.sim.context.setPositions(positions)
        if box_vectors is not None:
            self.sim.context.setPeriodicBoxVectors(*box_vectors)
        self.sim.context.setVelocitiesToTemperature(self.temperature)
        return self._store(self._snapshot())

    def clone_anchor(self, handle: Handle) -> Handle:
        # State is immutable; sharing the snapshot is safe.
        assert isinstance(handle, int)
        return self._store(self._cache[handle])

    def run_segment(
        self,
        handle: Handle,
        n_steps: int,
        *,
        randomize_velocities: bool,
        seed: int,
        device: Optional[int] = None,
    ) -> Handle:
        assert isinstance(handle, int)
        self.sim.context.setState(self._cache[handle])
        # Seed the integrator's own RNG (e.g. Langevin thermostat noise) so a
        # segment is reproducible; setVelocitiesToTemperature's seed only covers
        # the initial velocity draw, not the noise injected during stepping.
        try:
            self.sim.integrator.setRandomNumberSeed(int(seed))
        except AttributeError:  # integrator without an RNG (e.g. Verlet)
            pass
        if randomize_velocities:
            self.sim.context.setVelocitiesToTemperature(self.temperature, int(seed))
        self.sim.step(int(n_steps))
        return self._store(self._snapshot())

    def get_coords(self, handle: Handle) -> np.ndarray:
        assert isinstance(handle, int)
        pos = self._cache[handle].getPositions(asNumpy=True).value_in_unit(unit.nanometer)  # type: ignore
        coords = np.asarray(pos, dtype=float) * NM_TO_ANG
        if not np.all(np.isfinite(coords)):
            raise ValueError("OpenMM segment produced non-finite coordinates (unstable dynamics?)")
        return coords

    def release(self, handle: Handle) -> None:
        assert isinstance(handle, int)
        self._cache.pop(handle, None)
