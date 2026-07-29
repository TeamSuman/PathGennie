"""OpenMM in-process engine adapter for the PathGennie core driver.

Wraps one or more ``openmm.Context`` objects — all sharing a single ``System`` on
a **single GPU** — behind the :class:`pathgennie.core.engine.Engine` protocol so
the shared :class:`~pathgennie.core.driver.PathGennieDriver` can run the OpenMM
backend and *saturate that one card*.

Single-GPU saturation (not multi-GPU)
-------------------------------------
Path generation is lightweight, so rather than spreading across GPUs the OpenMM
backend keeps **one GPU busy** by running several swarm walkers concurrently on
it. Each concurrent walker needs its own ``Context`` (a Context is not
thread-safe), but every Context is built from the *same* ``System``, and an
OpenMM ``State`` snapshot is portable across Contexts of the same System. So the
engine keeps:

* a pool of ``n_workers`` Contexts on the one device, handed out one-per-thread
  through a thread-safe queue (OpenMM releases the GIL during ``step()``, so the
  threads genuinely overlap on the GPU); and
* a single immutable-``State`` cache (handles are ints into it), guarded by a
  lock, so any Context can ``setState`` any snapshot.

``n_workers`` can be sized adaptively (``resolve_worker_count`` + a
free-GPU-memory check while the pool is built): grow until either the core count
or the GPU memory budget is reached. With ``n_workers=1`` the engine is exactly
the original single-Context adapter — no extra Contexts are built.

A *handle* is an integer id into the State cache. ``setState``/``getState``
round-trip positions, velocities **and periodic box vectors**.

Reproducibility note: the driver's selection draws and per-segment velocity
randomisation are seeded, so with a deterministic integrator (e.g. Verlet) a run
is reproducible regardless of which pooled Context executes a segment (the State
fully determines it).

A *stochastic* integrator needs more care. Its RNG stream is created together with
the Context, so ``integrator.setRandomNumberSeed()`` alone has no effect on an
already-built Context — two runs with an identical seed were measured diverging
from the very first cycle. Passing ``reproducible=True`` makes the engine
reinitialise the Context (preserving state) after re-seeding, which does control
the noise stream. That costs a reinitialise per segment, so it is opt-in and is
enabled automatically only when the case supplies a ``seed``.
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
from typing import Dict, List, Optional

import numpy as np
from openmm import Context, State, XmlSerializer, unit
from openmm.app import Simulation

from pathgennie.core.engine import Handle

NM_TO_ANG = 10.0

__all__ = ["OpenMMEngine", "resolve_worker_count"]


# --------------------------------------------------------------------------- #
# Adaptive worker-count helpers
# --------------------------------------------------------------------------- #
def _cpu_worker_cap() -> int:
    """Upper bound on concurrent workers from the scheduler's core allocation."""
    for var in ("SLURM_CPUS_PER_TASK", "NCPUS"):
        v = os.environ.get(var)
        if v and v.strip().isdigit() and int(v) > 0:
            return int(v)
    return os.cpu_count() or 1


def _gpu_free_total_bytes(device: Optional[int] = None):
    """Return ``(free, total)`` GPU memory in bytes, or ``None`` if unknown.

    Tries torch, then pynvml, then ``nvidia-smi``. ``device`` is a logical index
    interpreted within the current ``CUDA_VISIBLE_DEVICES`` mask (each probe uses
    the same numbering the process sees).
    """
    idx = 0 if device is None else int(device)
    try:  # torch respects CUDA_VISIBLE_DEVICES numbering
        import torch

        if torch.cuda.is_available():
            probe = idx if idx < torch.cuda.device_count() else 0
            free, total = torch.cuda.mem_get_info(probe)
            return int(free), int(total)
    except Exception:  # noqa: BLE001 - best-effort probe
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(mem.free), int(mem.total)
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits", "-i", str(idx)],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()
        if out:
            free_mb, total_mb = (int(x.strip()) for x in out[0].split(","))
            return free_mb * 1024 * 1024, total_mb * 1024 * 1024
    except Exception:  # noqa: BLE001
        pass
    return None


def resolve_worker_count(requested, device: Optional[int] = None) -> int:
    """Upper bound on concurrent OpenMM Contexts to *attempt* to build.

    ``"auto"`` → the CPU-core cap; an int → that many; ``None``/≤1 → 1. The engine
    further caps this by free GPU memory while it builds the pool, so ``"auto"``
    is safe even on a busy card.
    """
    if isinstance(requested, str):
        if requested.strip().lower() == "auto":
            return max(1, _cpu_worker_cap())
        requested = int(requested)
    if requested is None:
        return 1
    return max(1, int(requested))


class OpenMMEngine:
    def __init__(
        self,
        simulation: Simulation,
        temperature: float,
        *,
        n_workers=1,
        device: Optional[int] = None,
        mem_safety: float = 0.15,
        verbose: bool = False,
        reproducible: bool = False,
    ):
        self.sim = simulation
        # Re-seed the integrator per segment (needs a Context reinitialise).
        # Only meaningful when the case asked for a reproducible run.
        self.reproducible = bool(reproducible)
        self.temperature = temperature * unit.kelvin  # type: ignore
        self._cache: Dict[int, State] = {}
        self._next_id = 0
        self._lock = threading.Lock()

        # Build the Context pool on the single device. Context 0 is the passed
        # Simulation's own Context; extras are clones of the same System.
        cap = resolve_worker_count(n_workers, device)
        self._contexts: List[Context] = [simulation.context]
        self._build_pool(cap, mem_safety, device, verbose)
        self.n_workers = len(self._contexts)

        self._pool: "queue.Queue[Context]" = queue.Queue()
        for ctx in self._contexts:
            self._pool.put(ctx)

    # -- pool construction ---------------------------------------------------
    def _clone_context(self) -> Context:
        # A Context owns its integrator, so each clone needs a fresh copy.
        integrator = XmlSerializer.deserialize(XmlSerializer.serialize(self.sim.integrator))
        platform = self.sim.context.getPlatform()
        props = {}
        try:  # pin extras to the same physical device as the primary Context
            name = platform.getName()
            key = {"CUDA": "DeviceIndex", "OpenCL": "OpenCLDeviceIndex"}.get(name)
            if key is not None:
                idx = platform.getPropertyValue(self.sim.context, key)
                if idx not in (None, ""):
                    props[key] = idx
        except Exception:  # noqa: BLE001 - properties are best-effort
            props = {}
        if props:
            return Context(self.sim.system, integrator, platform, props)
        return Context(self.sim.system, integrator, platform)

    def _build_pool(self, cap: int, mem_safety: float, device: Optional[int], verbose: bool) -> None:
        while len(self._contexts) < cap:
            free_total = _gpu_free_total_bytes(device)
            if free_total is not None and free_total[0] < mem_safety * free_total[1]:
                break  # memory-limited: stop growing the pool
            try:
                self._contexts.append(self._clone_context())
            except Exception as exc:  # noqa: BLE001 - fall back to what we have
                if verbose:
                    print(f"OpenMMEngine: stopped context pool at {len(self._contexts)} ({exc})")
                break
        if verbose:
            print(f"OpenMMEngine: {len(self._contexts)} concurrent context(s) on the GPU")

    # -- cache helpers (thread-safe) -----------------------------------------
    def _store(self, state: State) -> int:
        with self._lock:
            handle = self._next_id
            self._next_id += 1
            self._cache[handle] = state
        return handle

    def create_state(self, positions, box_vectors=None) -> int:
        """Initialise the cache from positions (and optional box); return a handle.

        Called once before the concurrent loop, so it uses the primary Context.
        """
        ctx = self.sim.context
        ctx.setPositions(positions)
        if box_vectors is not None:
            ctx.setPeriodicBoxVectors(*box_vectors)
        ctx.setVelocitiesToTemperature(self.temperature)
        return self._store(ctx.getState(getPositions=True, getVelocities=True))

    def clone_anchor(self, handle: Handle) -> Handle:
        # State is immutable; sharing the snapshot is safe.
        assert isinstance(handle, int)
        with self._lock:
            state = self._cache[handle]
        return self._store(state)

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
        assert isinstance(handle, int)
        with self._lock:
            state = self._cache[handle]
        ctx = self._pool.get()
        try:
            ctx.setState(state)
            integrator = ctx.getIntegrator()
            seeded = False
            try:
                integrator.setRandomNumberSeed(int(seed))
                seeded = True
            except AttributeError:  # integrator without an RNG (e.g. Verlet)
                pass
            if seeded and self.reproducible:
                # A stochastic integrator's RNG stream is created together with the
                # Context, so changing the seed afterwards has no effect until the
                # Context is reinitialised. Without this a "seeded" run is NOT
                # reproducible on Langevin dynamics: two runs with an identical seed
                # were measured diverging from the very first cycle. reinitialize()
                # is costly, so it is opt-in via `reproducible` (set when the case
                # supplies a seed) rather than paid on every segment by default.
                ctx.reinitialize(preserveState=True)
            if randomize_velocities:
                ctx.setVelocitiesToTemperature(self.temperature, int(seed))

            if save_subframes:
                subframes: list[np.ndarray] = []
                remaining = int(n_steps)
                while remaining > 0:
                    chunk = min(subframe_stride, remaining)
                    integrator.step(chunk)
                    remaining -= chunk
                    snap = ctx.getState(getPositions=True)
                    pos = snap.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                    subframes.append(np.asarray(pos, dtype=float) * NM_TO_ANG)
                new_state = ctx.getState(getPositions=True, getVelocities=True)
            else:
                integrator.step(int(n_steps))
                new_state = ctx.getState(getPositions=True, getVelocities=True)
        finally:
            self._pool.put(ctx)

        result_handle = self._store(new_state)
        if save_subframes:
            return result_handle, np.array(subframes, dtype=np.float32)
        return result_handle

    def get_coords(self, handle: Handle) -> np.ndarray:
        assert isinstance(handle, int)
        with self._lock:
            state = self._cache[handle]
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)  # type: ignore
        coords = np.asarray(pos, dtype=float) * NM_TO_ANG
        if not np.all(np.isfinite(coords)):
            raise ValueError("OpenMM segment produced non-finite coordinates (unstable dynamics?)")
        return coords

    def release(self, handle: Handle) -> None:
        assert isinstance(handle, int)
        with self._lock:
            self._cache.pop(handle, None)

    def create_handle(self, coords: np.ndarray) -> int:
        """Re-create a handle from an ``(n_atoms, 3)`` coordinate array (Å).

        Used by checkpoint restart. Uses the primary context to build a State
        with fresh Maxwell–Boltzmann velocities.
        """
        positions = (np.asarray(coords, dtype=float) / NM_TO_ANG) * unit.nanometers
        ctx = self.sim.context
        ctx.setPositions(positions)
        ctx.setVelocitiesToTemperature(self.temperature)
        return self._store(ctx.getState(getPositions=True, getVelocities=True))
