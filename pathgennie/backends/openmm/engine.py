"""OpenMM in-process engine adapter for the PathGennie core driver.

Wraps one or more ``openmm.Context`` objects — all sharing a single ``System``,
on one GPU or several — behind the :class:`pathgennie.core.engine.Engine`
protocol so the shared :class:`~pathgennie.core.driver.PathGennieDriver` can run
the OpenMM backend and *saturate the cards it is given*.

GPU saturation, on one card or several
--------------------------------------
Path generation is lightweight, so the OpenMM backend keeps each GPU busy by
running several swarm walkers concurrently on it. Each concurrent walker needs its own ``Context`` (a Context is not
thread-safe), but every Context is built from the *same* ``System``, and an
OpenMM ``State`` snapshot is portable across Contexts of the same System. So the
engine keeps:

* a pool of ``n_workers`` Contexts per device, handed out one-per-thread
  through a thread-safe queue (OpenMM releases the GIL during ``step()``, so the
  threads genuinely overlap on the GPU); and
* a single immutable-``State`` cache (handles are ints into it), guarded by a
  lock, so any Context can ``setState`` any snapshot.

``n_workers`` can be sized adaptively (``resolve_worker_count`` + a
free-GPU-memory check while the pool is built): grow until either the core count
or the GPU memory budget is reached. With ``n_workers=1`` the engine is exactly
the original single-Context adapter — no extra Contexts are built.

Several GPUs (``devices=[0, 1, ...]``): the engine builds one pool per device
(``n_workers`` Contexts each, pinned with the platform's device-index property;
indices are logical, within ``CUDA_VISIBLE_DEVICES``), and ``run_segment`` takes
its Context from the pool of the ``device`` the executor assigned. Because a
``State`` is a host-side snapshot, a segment may run on any card regardless of
where its input State was produced, so the driver and its selection logic are
unchanged. The primary Simulation Context joins the pool of the device it was
created on, if that device is listed.

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
from typing import Dict, List, Optional, Sequence

import numpy as np
# --- optional-backend guard -------------------------------------------------
# OpenMM is an optional extra as of this release: it is a large conda-forge
# package and AMBER/GROMACS-only users were forced to install it. This check runs
# BEFORE the real openmm imports below so the failure names the install command
# instead of surfacing a bare "No module named 'openmm'". The CLI dispatches this
# module lazily, so only `pathgennie openmm` is affected.
try:
    import openmm as _openmm_check  # noqa: F401
except ImportError as _exc:  # pragma: no cover - exercised via the guard test
    raise ImportError(
        "The OpenMM backend requires the 'openmm' package, which is no longer a "
        "core PathGennie dependency.\n"
        "Install it with:  pip install 'pathgennie[openmm]'\n"
        "             or:  conda install -c conda-forge openmm\n"
        f"Underlying import error: {_exc}"
    ) from _exc
# ---------------------------------------------------------------------------
from openmm import Context, State, Vec3, XmlSerializer, unit
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
        devices: Optional[Sequence[int]] = None,
        mem_safety: float = 0.15,
        verbose: bool = False,
        reproducible: bool = False,
    ):
        """``n_workers`` is per device. ``devices`` (two or more logical GPU indices)
        enables one Context pool per device; otherwise ``device`` names the single
        card (``None``: wherever the Simulation's Context lives)."""
        self.sim = simulation
        # Re-seed the integrator per segment (needs a Context reinitialise).
        # Only meaningful when the case asked for a reproducible run.
        self.reproducible = bool(reproducible)
        self.temperature = temperature * unit.kelvin  # type: ignore
        self._cache: Dict[int, State] = {}
        self._next_id = 0
        self._lock = threading.Lock()

        devs = [int(d) for d in devices] if devices else []
        if len(set(devs)) != len(devs):
            raise ValueError(f"devices must be distinct, got {list(devices)}")
        if len(devs) <= 1:
            # Single device. Context 0 is the passed Simulation's own Context;
            # extras are clones of the same System on the same card.
            device = devs[0] if devs else device
            cap = resolve_worker_count(n_workers, device)
            self._contexts: List[Context] = [simulation.context]
            self._build_pool(cap, mem_safety, device, verbose)
            self.devices: List[Optional[int]] = [device]
            self._device_contexts: Dict[Optional[int], List[Context]] = {device: self._contexts}
        else:
            # One pool per device, each pinned explicitly. The Simulation's own
            # Context joins the pool of the device it lives on, if listed.
            home = self._context_device(simulation.context)
            self.devices = devs
            self._device_contexts = {}
            for d in devs:
                cap = resolve_worker_count(n_workers, d)
                self._contexts = [simulation.context] if home == d else []
                self._build_pool(cap, mem_safety, d, verbose, pin=True)
                if not self._contexts:
                    raise RuntimeError(f"OpenMMEngine: could not build a Context on device {d}")
                self._device_contexts[d] = self._contexts
            self._contexts = [c for d in devs for c in self._device_contexts[d]]
        self.n_workers = len(self._contexts)
        self.workers_by_device = {d: len(c) for d, c in self._device_contexts.items()}

        self._pools: Dict[Optional[int], "queue.Queue[Context]"] = {}
        for d, ctxs in self._device_contexts.items():
            q: "queue.Queue[Context]" = queue.Queue()
            for ctx in ctxs:
                q.put(ctx)
            self._pools[d] = q
        self._pool = self._pools[self.devices[0]]
        if verbose and len(self.devices) > 1:
            print("OpenMMEngine: Contexts per device " + ", ".join(
                f"{d}: {n}" for d, n in self.workers_by_device.items()))

    # -- pool construction ---------------------------------------------------
    @staticmethod
    def _device_key(platform) -> Optional[str]:
        return {"CUDA": "DeviceIndex", "OpenCL": "OpenCLDeviceIndex"}.get(platform.getName())

    def _context_device(self, ctx: Context) -> Optional[int]:
        """Logical device index a Context runs on (``None`` on CPU/Reference)."""
        platform = ctx.getPlatform()
        key = self._device_key(platform)
        if key is None:
            return None
        try:
            idx = platform.getPropertyValue(ctx, key)
            return int(str(idx).split(",")[0]) if idx not in (None, "") else None
        except Exception:  # noqa: BLE001 - properties are best-effort
            return None

    def _clone_context(self, device: Optional[int] = None) -> Context:
        """A new Context of the same System; on ``device`` if given, else on the
        primary Context's device. On CPU/Reference ``device`` is only a label."""
        # A Context owns its integrator, so each clone needs a fresh copy.
        integrator = XmlSerializer.deserialize(XmlSerializer.serialize(self.sim.integrator))
        platform = self.sim.context.getPlatform()
        props = {}
        key = self._device_key(platform)
        if key is not None and device is not None:
            props[key] = str(int(device))
        elif key is not None:
            try:  # pin extras to the same physical device as the primary Context
                idx = platform.getPropertyValue(self.sim.context, key)
                if idx not in (None, ""):
                    props[key] = idx
            except Exception:  # noqa: BLE001 - properties are best-effort
                props = {}
        if key == "CUDA" and props:
            try:  # keep the primary Context's precision on every card
                props["Precision"] = platform.getPropertyValue(self.sim.context, "Precision")
            except Exception:  # noqa: BLE001
                pass
        if props:
            return Context(self.sim.system, integrator, platform, props)
        return Context(self.sim.system, integrator, platform)

    def _build_pool(self, cap: int, mem_safety: float, device: Optional[int], verbose: bool,
                    pin: bool = False) -> None:
        """Grow ``self._contexts`` to ``cap`` (memory permitting); ``pin`` builds
        the extras explicitly on ``device`` rather than beside the primary Context."""
        while len(self._contexts) < cap:
            free_total = _gpu_free_total_bytes(device)
            if free_total is not None and free_total[0] < mem_safety * free_total[1]:
                break  # memory-limited: stop growing the pool
            try:
                self._contexts.append(self._clone_context(device if pin else None))
            except Exception as exc:  # noqa: BLE001 - fall back to what we have
                if verbose:
                    print(f"OpenMMEngine: stopped context pool at {len(self._contexts)} ({exc})")
                break
        if verbose:
            where = "the GPU" if device is None else f"device {device}"
            print(f"OpenMMEngine: {len(self._contexts)} concurrent context(s) on {where}")

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
        pool = self._pools.get(device, self._pool)
        ctx = pool.get()
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
            pool.put(ctx)

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

    def get_box(self, handle: Handle) -> Optional[np.ndarray]:
        """Box vectors of ``handle`` as a ``(3, 3)`` array of rows in Å, or ``None``
        for a non-periodic System. Under a barostat each State has its own box."""
        if not self.sim.system.usesPeriodicBoundaryConditions():
            return None
        assert isinstance(handle, int)
        with self._lock:
            state = self._cache[handle]
        vecs = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)  # type: ignore
        return np.asarray(vecs, dtype=float) * NM_TO_ANG

    def release(self, handle: Handle) -> None:
        assert isinstance(handle, int)
        with self._lock:
            self._cache.pop(handle, None)

    def create_handle(self, coords: np.ndarray, box: Optional[np.ndarray] = None) -> int:
        """Re-create a handle from an ``(n_atoms, 3)`` coordinate array (Å).

        Used by checkpoint restart. Uses the primary context to build a State
        with fresh Maxwell–Boltzmann velocities. ``box`` (``(3, 3)`` rows in Å)
        restores the frame's own box; without it the State keeps whatever box the
        primary context has (the starting box), which is wrong under a barostat.
        """
        positions = (np.asarray(coords, dtype=float) / NM_TO_ANG) * unit.nanometers
        ctx = self.sim.context
        if box is not None:
            b = np.asarray(box, dtype=float).reshape(3, 3) / NM_TO_ANG
            ctx.setPeriodicBoxVectors(*[Vec3(*row) * unit.nanometer for row in b])
        ctx.setPositions(positions)
        ctx.setVelocitiesToTemperature(self.temperature)
        return self._store(ctx.getState(getPositions=True, getVelocities=True))
