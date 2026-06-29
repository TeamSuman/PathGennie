"""Asynchronous streaming storage for PathGennie trajectories.

Streams arrays (trajectory frames, metrics, CVs) to HDF5 format in a background thread 
so the main loop isn't blocked by I/O.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np

__all__ = ["HDF5Storage"]


class HDF5Storage:
    """Streams trajectory frames and metrics to an HDF5 file asynchronously."""

    def __init__(self, filepath: Path | str, chunk_size: int = 1):
        self.filepath = Path(filepath)
        self.chunk_size = chunk_size
        self._queue: queue.Queue[Tuple[str, np.ndarray]] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def _writer_loop(self) -> None:
        """Background thread that pops items from the queue and writes to HDF5."""
        with h5py.File(self.filepath, "a") as f:
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    # Timeout so we can check _stop_event
                    dataset_name, data = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if dataset_name not in f:
                    # Create extendable dataset
                    shape = (0,) + data.shape
                    maxshape = (None,) + data.shape
                    f.create_dataset(
                        dataset_name, 
                        shape=shape, 
                        maxshape=maxshape, 
                        dtype=data.dtype, 
                        chunks=(self.chunk_size,) + data.shape,
                        compression="gzip"
                    )

                dset = f[dataset_name]
                curr_size = dset.shape[0]
                dset.resize(curr_size + 1, axis=0)
                dset[curr_size] = data
                self._queue.task_done()

    def append(self, dataset_name: str, data: np.ndarray) -> None:
        """Queue an array to be appended to the specified dataset."""
        self._queue.put((dataset_name, np.asarray(data)))

    def close(self) -> None:
        """Wait for pending writes and close the storage."""
        self._stop_event.set()
        self._thread.join()

