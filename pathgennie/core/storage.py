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

# Queue sentinel: routes a checkpoint write onto the writer thread rather than
# opening the file a second time from the caller.
_CHECKPOINT = "__pathgennie_checkpoint__"


class HDF5Storage:
    """Streams trajectory frames and metrics to an HDF5 file asynchronously."""

    def __init__(self, filepath: Path | str, chunk_size: int = 1):
        self.filepath = Path(filepath)
        self.chunk_size = max(1, int(chunk_size))
        self._queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()
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
                        if dataset_name == _CHECKPOINT:
                            self._write_checkpoint(f, data)
                        else:
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

    @staticmethod
    def _write_checkpoint(f: "h5py.File", payload: dict) -> None:
        """Write the checkpoint group. Runs **on the writer thread**, into its own
        open file handle, so the file is never open twice at once."""
        import json

        grp = f.require_group("checkpoint")
        grp.attrs["cycle"] = int(payload["cycle"])
        grp.attrs["anchor_metric"] = float(payload["anchor_metric"])
        # How much trajectory belongs to this checkpoint. Frames streamed after it
        # are from cycles the resumed run will re-execute, so without this the
        # resume both loads and regenerates them. Nothing else on disk identifies
        # where to cut -- frames carry no cycle index. Because the queue is FIFO,
        # every frame appended before this item has already been written, so the
        # count is exact.
        grp.attrs["n_frames"] = int(f["trajectory"].shape[0]) if "trajectory" in f else 0
        grp.attrs["rng_state_json"] = json.dumps(
            _rng_state_to_serialisable(payload["rng_state"])
        )
        datasets_to_save = [
            ("anchor_coords", payload["anchor_coords"]),
            ("anchor_cv", payload["anchor_cv"]),
        ]
        if payload.get("metric_history") is not None:
            datasets_to_save.append(("metric_history", np.asarray(payload["metric_history"])))
        for name, arr in datasets_to_save:
            if name in grp:
                del grp[name]
            grp.create_dataset(name, data=np.asarray(arr))
        # A checkpoint that is still in a buffer does not survive the crash it
        # exists to survive.
        f.flush()

    def save_checkpoint(
        self,
        cycle: int,
        rng_state: dict,
        anchor_coords: np.ndarray,
        anchor_cv: np.ndarray,
        anchor_metric: float,
        metric_history: Optional[List[float] | np.ndarray] = None,
    ) -> None:
        """Queue restart metadata behind pending writes, then wait for it to land.

        The checkpoint is written **by the writer thread**, through the file handle
        it already holds. Opening the same HDF5 file a second time while that
        thread has it open is not a supported pattern -- it happens to work because
        HDF5 caches file identifiers within a process, which is not a guarantee to
        build on.

        Routing it through the queue also makes ``n_frames`` exact for free: the
        queue is FIFO, so every frame appended before this call is on disk by the
        time the checkpoint is written.

        The caller blocks until it lands. That is deliberate -- a checkpoint that
        has not been written is not a checkpoint -- and cheap, because checkpoints
        are infrequent relative to ``save_freq``.
        """
        self._raise_if_failed()
        self._queue.put((_CHECKPOINT, {
            "cycle": cycle,
            "rng_state": rng_state,
            "anchor_coords": anchor_coords,
            "anchor_cv": anchor_cv,
            "anchor_metric": anchor_metric,
            "metric_history": metric_history,
        }))
        self._drain()

    @staticmethod
    def _truncate_to_checkpoint(filepath: Path) -> None:
        """Drop streamed rows written after the last checkpoint.

        A no-op for checkpoints written before ``n_frames`` was recorded, so old
        files still load (with the previous, duplicating behaviour) rather than
        failing.
        """
        try:
            with h5py.File(filepath, "a") as f:
                if "checkpoint" not in f:
                    return
                n = f["checkpoint"].attrs.get("n_frames")
                if n is None:
                    return
                n = int(n)
                for name in ("trajectory", "metric"):
                    if name in f and f[name].shape[0] > n:
                        f[name].resize(n, axis=0)
        except OSError:
            # Read-only location, or a file another process holds open: loading a
            # slightly-too-long trajectory is better than refusing to resume at all.
            pass

    @classmethod
    def load_checkpoint(cls, filepath: Path | str) -> "dict | None":
        """Load the last saved checkpoint from *filepath*, or ``None``.

        Returns a dict with keys: ``cycle``, ``rng_state``, ``anchor_coords``,
        ``anchor_cv``, ``anchor_metric``, ``trajectory``, ``metric_history``.

        Frames streamed *after* the checkpoint are dropped, from the returned
        arrays and from the file. They belong to cycles the resumed run will
        execute again, so keeping them would duplicate those frames -- and they
        come from a discarded branch, so they are wrong rather than merely
        redundant. Truncating the file too matters: otherwise the resumed run
        appends after the stale rows and the next resume repeats the problem.

        This runs before the caller constructs an :class:`HDF5Storage`, so there
        is no writer thread to race with.
        """
        import json

        filepath = Path(filepath)
        if not filepath.exists():
            return None

        cls._truncate_to_checkpoint(filepath)

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
