"""The checkpoint must be written through the writer thread's own file handle.

``save_checkpoint`` used to open the HDF5 file a *second* time while the writer
thread still had it open. That is not a supported pattern; it happened to work
because HDF5 caches file identifiers within a process, which is an implementation
detail rather than a guarantee, and it is exactly the sort of thing that breaks
under a different HDF5 build or with file locking enabled.

Routing the write onto the writer thread also makes ``n_frames`` exact for free:
the queue is FIFO, so every frame appended before the checkpoint is already on
disk when it is written.
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pathgennie.core.storage import HDF5Storage  # noqa: E402


def test_no_second_open_while_the_writer_holds_the_file(tmp_path, monkeypatch):
    """The defect: a concurrent second handle on the same file."""
    path = tmp_path / "one.h5"
    storage = HDF5Storage(path)
    for i in range(3):
        storage.append("trajectory", np.full((2, 3), float(i)))

    real_open = h5py.File
    opens: list[str] = []

    def counting_open(name, mode="r", *a, **kw):
        opens.append(str(mode))
        return real_open(name, mode, *a, **kw)

    monkeypatch.setattr(h5py, "File", counting_open)
    storage.save_checkpoint(
        cycle=3, rng_state=np.random.default_rng(0).bit_generator.state,
        anchor_coords=np.zeros((2, 3)), anchor_cv=np.zeros(1), anchor_metric=0.0,
    )
    monkeypatch.undo()
    storage.close()

    assert opens == [], (
        f"save_checkpoint opened the file {len(opens)} time(s) ({opens}) while the "
        "writer thread already held it open")


def test_the_checkpoint_still_round_trips(tmp_path):
    path = tmp_path / "rt.h5"
    storage = HDF5Storage(path)
    for i in range(4):
        storage.append("trajectory", np.full((2, 3), float(i)))
        storage.append("metric", np.array([float(i)]))
    rng = np.random.default_rng(11)
    storage.save_checkpoint(
        cycle=9, rng_state=rng.bit_generator.state,
        anchor_coords=np.full((2, 3), 7.0), anchor_cv=np.array([0.25]),
        anchor_metric=-1.5, metric_history=np.arange(4, dtype=float),
    )
    storage.close()

    ckpt = HDF5Storage.load_checkpoint(path)
    assert ckpt["cycle"] == 9
    assert np.isclose(ckpt["anchor_metric"], -1.5)
    assert np.allclose(ckpt["anchor_coords"], 7.0)
    assert np.allclose(ckpt["anchor_cv"], [0.25])
    assert len(ckpt["metric_history"]) == 4


def test_frames_queued_before_the_checkpoint_are_counted(tmp_path):
    """FIFO ordering is what makes n_frames exact -- assert it, don't assume it."""
    path = tmp_path / "order.h5"
    storage = HDF5Storage(path)
    for i in range(6):
        storage.append("trajectory", np.full((2, 3), float(i)))
    storage.save_checkpoint(
        cycle=6, rng_state=np.random.default_rng(0).bit_generator.state,
        anchor_coords=np.zeros((2, 3)), anchor_cv=np.zeros(1), anchor_metric=0.0,
    )
    for i in range(6, 10):                       # streamed after the checkpoint
        storage.append("trajectory", np.full((2, 3), float(i)))
    storage.close()

    with h5py.File(path, "r") as f:
        assert int(f["checkpoint"].attrs["n_frames"]) == 6
        assert f["trajectory"].shape[0] == 10
    assert len(HDF5Storage.load_checkpoint(path)["trajectory"]) == 6


def test_a_dead_writer_is_still_reported(tmp_path):
    """The queued-checkpoint path must not reintroduce the hang."""
    storage = HDF5Storage(tmp_path / "dead.h5")
    storage._stop_event.set()
    storage._thread.join(timeout=5)
    storage._error = OSError("simulated writer failure")

    with pytest.raises(RuntimeError, match="writer"):
        storage.save_checkpoint(
            cycle=1, rng_state=np.random.default_rng(0).bit_generator.state,
            anchor_coords=np.zeros((2, 3)), anchor_cv=np.zeros(1), anchor_metric=0.0,
        )


def test_repeated_checkpoints_overwrite_cleanly(tmp_path):
    path = tmp_path / "multi.h5"
    storage = HDF5Storage(path)
    rng = np.random.default_rng(2)
    for cycle in (4, 8, 12):
        storage.append("trajectory", np.zeros((2, 3)))
        storage.save_checkpoint(
            cycle=cycle, rng_state=rng.bit_generator.state,
            anchor_coords=np.full((2, 3), float(cycle)), anchor_cv=np.zeros(1),
            anchor_metric=float(cycle),
        )
    storage.close()
    ckpt = HDF5Storage.load_checkpoint(path)
    assert ckpt["cycle"] == 12
    assert np.allclose(ckpt["anchor_coords"], 12.0)
