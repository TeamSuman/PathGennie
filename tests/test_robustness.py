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
