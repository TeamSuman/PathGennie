"""Robustness regressions: HDF5 checkpoint error propagation and lazy imports."""

from __future__ import annotations

import importlib.util

import h5py
import numpy as np
import pytest

from pathgennie.core.storage import HDF5Storage


def test_hdf5_roundtrip(tmp_path):
    path = tmp_path / "run.h5"
    store = HDF5Storage(path)
    for i in range(3):
        store.append("trajectory", np.full((2, 3), float(i)))
        store.append("metric", np.array([float(i)]))
    store.close()

    with h5py.File(path, "r") as f:
        assert f["trajectory"].shape == (3, 2, 3)
        assert f["metric"].shape == (3, 1)
        assert f["trajectory"][2][0, 0] == 2.0


def test_hdf5_writer_error_is_surfaced(tmp_path):
    # Parent directory does not exist -> h5py.File(..., 'a') fails in the writer
    # thread. The failure must propagate, not vanish silently.
    bad = tmp_path / "does_not_exist" / "run.h5"
    store = HDF5Storage(bad)
    with pytest.raises(RuntimeError):
        store.close()


def test_pathrefinement_base_import_without_optional_deps():
    # `import pathrefinement` must succeed even without torch/openmm, exposing
    # the dependency-free primitives; the refiner is only imported on access.
    import pathrefinement

    assert pathrefinement.PathCV is not None
    assert pathrefinement.PrincipalCurve is not None

    if importlib.util.find_spec("torch") is None:
        with pytest.raises(ImportError):
            _ = pathrefinement.PathRefiner


def test_checkpoint_metadata_roundtrip(tmp_path):
    path = tmp_path / "ckpt.h5"
    store = HDF5Storage(path)

    # Append streaming data
    store.append("trajectory", np.ones((2, 3)))
    store.append("metric", np.array([1.5]))

    rng = np.random.default_rng(12345)
    _ = rng.standard_normal(5)
    rng_state = rng.bit_generator.state
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    cv = np.array([0.5, 0.8])
    metric = -1.25

    store.save_checkpoint(
        cycle=10,
        rng_state=rng_state,
        anchor_coords=coords,
        anchor_cv=cv,
        anchor_metric=metric,
    )
    store.close()

    ckpt = HDF5Storage.load_checkpoint(path)
    assert ckpt is not None
    assert ckpt["cycle"] == 10
    assert ckpt["anchor_metric"] == pytest.approx(-1.25)
    np.testing.assert_allclose(ckpt["anchor_coords"], coords)
    np.testing.assert_allclose(ckpt["anchor_cv"], cv)
    assert ckpt["trajectory"].shape == (1, 2, 3)
    assert len(ckpt["metric_history"]) == 1

    # Verify RNG state can be restored and produces identical next numbers
    restored_rng = np.random.default_rng()
    restored_rng.bit_generator.state = ckpt["rng_state"]
    np.testing.assert_allclose(rng.standard_normal(5), restored_rng.standard_normal(5))
