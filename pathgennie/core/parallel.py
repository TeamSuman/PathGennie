"""Parallel execution of swarm trials across devices.

The driver evaluates ``N`` independent sampler trials per cycle (and ``K*N`` once
beam/RRT policies are added).  Historically the AMBER/GROMACS backends ran these
through a bare ``ThreadPoolExecutor`` with **no device assignment**, so every
worker contended for GPU 0.  This module replaces that with a small executor
abstraction that round-robins a *pool of devices* across the trials.

An executor maps a ``worker_fn(item, device)`` over a list of work items, where
``device`` is the GPU index (or ``None``) the trial should run on.  Keeping the
device-assignment policy here means the selection/cycle logic in
:mod:`pathgennie.core.driver` is identical regardless of how many GPUs are
present, and all three backends share one code path.

Two executors are provided:

* :class:`SerialExecutor` — single device, no threads; the reference path that
  ``ProcessDevicePool``/``ThreadDevicePool`` must match for a fixed seed.
* :class:`ThreadDevicePool` — a thread per (device x workers_per_device) slot,
  suitable for the subprocess backends (AMBER ``pmemd.cuda`` / GROMACS
  ``gmx mdrun``) where the GIL is released during ``subprocess.run`` and each
  worker exports ``CUDA_VISIBLE_DEVICES`` for its device.

The in-process OpenMM process pool (one long-lived CUDA Context per GPU) is a
specialisation built on the same ``ParallelExecutor`` contract; see
``pathgennie/backends/openmm`` for that wiring.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Protocol, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

__all__ = ["ParallelExecutor", "SerialExecutor", "ThreadDevicePool", "resolve_devices"]


def resolve_devices(devices: Optional[Sequence[int]]) -> List[Optional[int]]:
    """Normalise a device spec into a list usable for round-robin assignment.

    ``None`` or an empty sequence means "let the engine choose" (single slot,
    device ``None``).
    """

    if not devices:
        return [None]
    return [int(d) for d in devices]


class ParallelExecutor(Protocol):
    """Map ``worker_fn(item, device)`` over ``items`` preserving input order."""

    devices: List[Optional[int]]

    def map(self, worker_fn: Callable[[T, Optional[int]], R], items: Sequence[T]) -> List[R]:
        ...

    def shutdown(self) -> None:
        ...


class SerialExecutor:
    """Run all trials sequentially on a single device (reference implementation)."""

    def __init__(self, device: Optional[int] = None):
        self.devices: List[Optional[int]] = [device]

    def map(self, worker_fn: Callable[[T, Optional[int]], R], items: Sequence[T]) -> List[R]:
        device = self.devices[0]
        return [worker_fn(item, device) for item in items]

    def shutdown(self) -> None:  # pragma: no cover - trivial
        pass


class ThreadDevicePool:
    """Round-robin trials over ``devices`` using a thread pool.

    Each work item ``i`` is assigned ``devices[i % len(devices)]`` so the swarm
    is spread evenly across GPUs.  ``workers_per_device`` allows more than one
    concurrent segment per GPU (useful for small systems that under-fill a GPU).
    Output order matches input order.
    """

    def __init__(self, devices: Optional[Sequence[int]] = None, workers_per_device: int = 1):
        self.devices = resolve_devices(devices)
        self.workers_per_device = max(1, int(workers_per_device))
        self._max_workers = len(self.devices) * self.workers_per_device

    def map(self, worker_fn: Callable[[T, Optional[int]], R], items: Sequence[T]) -> List[R]:
        items = list(items)
        if not items:
            return []
        n_workers = min(self._max_workers, len(items))
        if n_workers == 1:
            device = self.devices[0]
            return [worker_fn(item, device) for item in items]

        def run(indexed_item):
            index, item = indexed_item
            device = self.devices[index % len(self.devices)]
            return worker_fn(item, device)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            return list(pool.map(run, enumerate(items)))

    def shutdown(self) -> None:  # pragma: no cover - trivial
        pass
