"""A dead HDF5 writer must surface as an error, not a hung job.

``save_checkpoint`` drained the queue with ``self._queue.join()`` and only then
called ``_raise_if_failed()``. ``Queue.join()`` waits for ``task_done()`` on every
queued item, and the writer thread only calls that for items it has *popped*. So
if the writer dies with a backlog -- a failed file open, a full disk, a permissions
problem -- those calls never come and ``join()`` blocks forever.

That turns a reportable error into a silent hang, and it hangs at exactly the
moment checkpointing exists to protect: a long run trying to save its restart.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pathgennie.core.storage import HDF5Storage  # noqa: E402


def _call_with_watchdog(fn, timeout=10.0):
    """Run ``fn`` in a thread; report whether it returned, raised, or hung."""
    box = {}

    def target():
        try:
            fn()
            box["outcome"] = "returned"
        except BaseException as exc:  # noqa: BLE001
            box["outcome"] = "raised"
            box["exc"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return "hung", None
    return box.get("outcome"), box.get("exc")


def _kill_writer(storage, pending=3):
    """Simulate a writer that died with items still queued."""
    storage._stop_event.set()
    storage._thread.join(timeout=5)
    storage._error = OSError("simulated writer failure (e.g. disk full)")
    for _ in range(pending):
        storage._queue.put(("trajectory", np.zeros((2, 3))))


def test_save_checkpoint_raises_instead_of_hanging(tmp_path):
    storage = HDF5Storage(tmp_path / "ckpt.h5")
    _kill_writer(storage)

    outcome, exc = _call_with_watchdog(
        lambda: storage.save_checkpoint(
            cycle=7,
            rng_state=np.random.default_rng(0).bit_generator.state,
            anchor_coords=np.zeros((2, 3)),
            anchor_cv=np.zeros(1),
            anchor_metric=0.0,
        )
    )
    assert outcome != "hung", "save_checkpoint blocked forever on a dead writer"
    assert outcome == "raised"
    assert "writer" in str(exc).lower()


def test_the_error_names_the_pending_writes(tmp_path):
    """A bare failure is hard to act on; say how much was lost."""
    storage = HDF5Storage(tmp_path / "ckpt.h5")
    _kill_writer(storage, pending=5)
    with pytest.raises(RuntimeError) as info:
        storage.save_checkpoint(
            cycle=1, rng_state=np.random.default_rng(0).bit_generator.state,
            anchor_coords=np.zeros((2, 3)), anchor_cv=np.zeros(1), anchor_metric=0.0,
        )
    assert "writer" in str(info.value).lower()


def test_checkpointing_still_works_on_a_healthy_writer(tmp_path):
    """The guard must not break the normal path."""
    path = tmp_path / "ok.h5"
    storage = HDF5Storage(path)
    for _ in range(4):
        storage.append("trajectory", np.ones((2, 3)))
    rng = np.random.default_rng(3)
    storage.save_checkpoint(
        cycle=11, rng_state=rng.bit_generator.state,
        anchor_coords=np.ones((2, 3)), anchor_cv=np.array([0.5]), anchor_metric=1.25,
    )
    storage.close()

    ckpt = HDF5Storage.load_checkpoint(path)
    assert ckpt is not None
    assert ckpt["cycle"] == 11
    assert np.isclose(ckpt["anchor_metric"], 1.25)
    assert len(ckpt["trajectory"]) == 4, "queued frames were not flushed before the checkpoint"


def test_repeated_checkpoints_do_not_accumulate_state(tmp_path):
    storage = HDF5Storage(tmp_path / "multi.h5")
    rng = np.random.default_rng(1)
    for cycle in (5, 10, 15):
        storage.append("trajectory", np.zeros((2, 3)))
        storage.save_checkpoint(
            cycle=cycle, rng_state=rng.bit_generator.state,
            anchor_coords=np.zeros((2, 3)), anchor_cv=np.zeros(1), anchor_metric=float(cycle),
        )
    storage.close()
    ckpt = HDF5Storage.load_checkpoint(tmp_path / "multi.h5")
    assert ckpt["cycle"] == 15
    assert np.isclose(ckpt["anchor_metric"], 15.0)
