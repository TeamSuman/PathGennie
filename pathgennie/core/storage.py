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
import time
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

    def _drain(self, poll: float = 0.01) -> None:
        """Wait for queued writes to land, without hanging on a dead writer.

        ``Queue.join()`` waits for ``task_done()`` on every queued item, and the
        writer thread only calls that for items it has *popped*. If the writer died
        with a backlog -- failed file open, full disk -- those calls never come and
        an unguarded ``join()`` blocks forever, turning a reportable error into a
        hung job at exactly the moment checkpointing is meant to protect the run.
        """
        while not self._queue.empty():
            self._raise_if_failed()
            if not self._thread.is_alive():
                raise RuntimeError(
                    "HDF5 storage writer thread is not running; "
                    f"{self._queue.qsize()} queued write(s) will never be flushed"
                )
            time.sleep(poll)
        self._raise_if_failed()
        # Safe now: at most the one in-flight item remains, and the writer is alive
        # to finish it.
        if self._thread.is_alive():
            self._queue.join()
        self._raise_if_failed()

    def append(self, dataset_name: str, data: np.ndarray) -> None:
        """Queue an array to be appended to the named dataset."""
        self._raise_if_failed()
        self._queue.put((dataset_name, np.asarray(data)))

    def close(self) -> None:
        """Wait for pending writes to flush, then close; re-raise any writer error."""
        self._stop_event.set()
        self._thread.join()
        self._raise_if_failed()

    # -- checkpoint save / load -----------------------------------------------

    def save_checkpoint(
        self,
        cycle: int,
        rng_state: dict,
        anchor_coords: np.ndarray,
        anchor_cv: np.ndarray,
        anchor_metric: float,
        metric_history: Optional[List[float] | np.ndarray] = None,
    ) -> None:
        """Flush pending writes then synchronously save restart metadata.

        This blocks the caller (main thread) while the writer thread drains its
        queue, then writes the checkpoint group directly.  The brief pause is
        acceptable because checkpoints are infrequent (every ``checkpoint_freq``
        cycles, typically much larger than ``save_freq``).
        """
        import json

        # Drain the async writer queue so trajectory/metric datasets are current.
        self._drain()

        with h5py.File(self.filepath, "a") as f:
            grp = f.require_group("checkpoint")
            # Scalar / small metadata.
            grp.attrs["cycle"] = int(cycle)
            grp.attrs["anchor_metric"] = float(anchor_metric)
            grp.attrs["rng_state_json"] = json.dumps(_rng_state_to_serialisable(rng_state))
            # Arrays (overwrite on every checkpoint).
            datasets_to_save = [
                ("anchor_coords", anchor_coords),
                ("anchor_cv", anchor_cv),
            ]
            if metric_history is not None:
                datasets_to_save.append(("metric_history", np.asarray(metric_history)))

            for name, arr in datasets_to_save:
                if name in grp:
                    del grp[name]
                grp.create_dataset(name, data=np.asarray(arr))

    @classmethod
    def load_checkpoint(cls, filepath: Path | str) -> "dict | None":
        """Load the last saved checkpoint from *filepath*, or ``None``.

        Returns a dict with keys: ``cycle``, ``rng_state``, ``anchor_coords``,
        ``anchor_cv``, ``anchor_metric``, ``trajectory``, ``metric_history``.
        """
        import json

        filepath = Path(filepath)
        if not filepath.exists():
            return None
        with h5py.File(filepath, "r") as f:
            if "checkpoint" not in f:
                return None
            grp = f["checkpoint"]
            result: dict = {
                "cycle": int(grp.attrs["cycle"]),
                "anchor_metric": float(grp.attrs["anchor_metric"]),
                "rng_state": _rng_state_from_serialised(
                    json.loads(grp.attrs["rng_state_json"])
                ),
                "anchor_coords": np.array(grp["anchor_coords"]),
                "anchor_cv": np.array(grp["anchor_cv"]),
            }
            if "metric_history" in grp:
                result["metric_history"] = np.array(grp["metric_history"]).reshape(-1)
            elif "metric" in f:
                result["metric_history"] = np.array(f["metric"]).reshape(-1)
            else:
                result["metric_history"] = np.empty((0,), dtype=float)

            # Also load the accumulated streaming trajectory data.
            if "trajectory" in f:
                result["trajectory"] = np.array(f["trajectory"])
            else:
                result["trajectory"] = np.empty((0,), dtype=np.float32)
        return result


# -- numpy RNG state serialisation helpers ------------------------------------
# numpy's public Generator state is exposed through ``bit_generator.state``.
# Some NumPy versions return ``None`` from Generator.__getstate__(), so callers
# should avoid that private pickle hook when storing restart metadata.

def _rng_state_to_serialisable(state: dict) -> dict:
    """Convert numpy RNG state dict to JSON-safe nested dicts/lists."""
    if state is None:
        raise ValueError("rng_state cannot be None; use rng.bit_generator.state")
    out: dict = {}
    for key, val in state.items():
        if isinstance(val, np.ndarray):
            out[key] = {"__ndarray__": True, "data": val.tolist(), "dtype": str(val.dtype)}
        elif isinstance(val, dict):
            out[key] = _rng_state_to_serialisable(val)
        else:
            # int, str, bool, etc.
            out[key] = val
    return out


def _rng_state_from_serialised(obj: dict) -> dict:
    """Restore a numpy RNG state dict from its JSON-safe representation."""
    out: dict = {}
    for key, val in obj.items():
        if isinstance(val, dict) and val.get("__ndarray__"):
            out[key] = np.array(val["data"], dtype=val["dtype"])
        elif isinstance(val, dict):
            out[key] = _rng_state_from_serialised(val)
        else:
            out[key] = val
    return out
