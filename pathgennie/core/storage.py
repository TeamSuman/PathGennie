"""Asynchronous streaming storage for PathGennie trajectories.

Streams arrays (trajectory frames, metrics, CVs) to an HDF5 file in a background
thread so the main adaptive loop is not blocked by I/O.

Failures in the writer thread (a full disk, a bad path, an HDF5 error) are
captured and re-raised on the next :meth:`append` or on :meth:`close`, so a
long HPC run fails loudly instead of silently losing every frame after the
writer thread dies.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np

__all__ = ["HDF5Storage"]


class HDF5Storage:
    """Streams trajectory frames and metrics to an HDF5 file asynchronously."""

    def __init__(self, filepath: Path | str, chunk_size: int = 1):
        self.filepath = Path(filepath)
        self.chunk_size = max(1, int(chunk_size))
        self._queue: "queue.Queue[Tuple[str, np.ndarray]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._error: Optional[BaseException] = None
        # Signals that the file has been opened (or failed to open) so append()
        # cannot race ahead of a writer that never started.
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def _writer_loop(self) -> None:
        """Background thread: pop items from the queue and append them to HDF5."""
        try:
            with h5py.File(self.filepath, "a") as f:
                self._ready.set()
                while not self._stop_event.is_set() or not self._queue.empty():
                    try:
                        dataset_name, data = self._queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    try:
                        if dataset_name not in f:
                            shape = (0,) + data.shape
                            maxshape = (None,) + data.shape
                            f.create_dataset(
                                dataset_name,
                                shape=shape,
                                maxshape=maxshape,
                                dtype=data.dtype,
                                chunks=(self.chunk_size,) + data.shape,
                                compression="gzip",
                            )
                        dset = f[dataset_name]
                        curr_size = dset.shape[0]
                        dset.resize(curr_size + 1, axis=0)
                        dset[curr_size] = data
                    finally:
                        self._queue.task_done()
        except BaseException as exc:  # noqa: BLE001 - surfaced to the main thread
            self._error = exc
            self._stop_event.set()
        finally:
            # Unblock any waiter even if the file failed to open.
            self._ready.set()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"HDF5 storage writer failed: {self._error}") from self._error

    def append(self, dataset_name: str, data: np.ndarray) -> None:
        """Queue an array to be appended to the named dataset."""
        self._raise_if_failed()
        self._queue.put((dataset_name, np.asarray(data)))

    def close(self) -> None:
        """Wait for pending writes to flush, then close; re-raise any writer error."""
        self._stop_event.set()
        self._thread.join()
        self._raise_if_failed()
