"""AMBER restarts for periodic systems must carry the box.

``write_rst7_coords`` emitted coordinates only. For an explicit-solvent system
(``ntb=1``/``ntb=2``) sander then refuses the file:

    peek_ewald_inpcrd: Box info not found in inpcrd

``CoreAmberEngine.create_handle`` builds restarts through that function, and the
driver calls ``create_handle`` when resuming from a checkpoint, as does the
Weighted Ensemble stage when seeding walkers from raw frames. So **checkpoint
resume and WE seeding were both broken for any solvated AMBER system** — while
gas-phase runs were unaffected, because ``ntb=0`` needs no box. That is why it
survived the whole gas-phase campaign and only surfaced on the first solvated one.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathgennie.backends.amber.engine import CoreAmberEngine
from pathgennie.backends.amber.utils import (
    read_rst7_box,
    read_rst7_coords,
    write_rst7_coords,
)

BOX = [29.8812185, 30.1597478, 35.3328564, 90.0, 90.0, 90.0]


def test_box_is_written_when_supplied(tmp_path):
    path = tmp_path / "solv.rst7"
    write_rst7_coords(path, np.arange(9, dtype=float).reshape(3, 3), box=BOX)
    got = read_rst7_box(path)
    assert got is not None, "no box line written -- sander rejects this for ntb!=0"
    assert np.allclose(got, BOX)


def test_coordinates_survive_the_box_line(tmp_path):
    coords = np.arange(9, dtype=float).reshape(3, 3)
    path = tmp_path / "solv.rst7"
    write_rst7_coords(path, coords, box=BOX)
    assert np.allclose(read_rst7_coords(path), coords), \
        "the box line must not be mistaken for coordinates on read-back"


def test_no_box_stays_the_gas_phase_format(tmp_path):
    """ntb=0 runs must be untouched -- a spurious box line would be read as atoms."""
    path = tmp_path / "vac.rst7"
    write_rst7_coords(path, np.arange(9, dtype=float).reshape(3, 3))
    assert read_rst7_box(path) is None
    assert len(read_rst7_coords(path)) == 3


def test_three_length_box_is_accepted(tmp_path):
    """Callers often have only the lengths; angles default to 90."""
    path = tmp_path / "lengths.rst7"
    write_rst7_coords(path, np.zeros((2, 3)), box=[10.0, 11.0, 12.0])
    got = read_rst7_box(path)
    assert np.allclose(got[:3], [10.0, 11.0, 12.0])
    assert np.allclose(got[3:], [90.0, 90.0, 90.0])


def test_create_handle_propagates_the_engine_box(tmp_path):
    """The path that actually broke: checkpoint resume and WE seeding."""
    engine = CoreAmberEngine(
        topology=tmp_path / "sys.prmtop", executable=tmp_path / "sander",
        scratch_dir=tmp_path / "scratch", temperature=300.0,
        mdin_controls=dict(dt=0.002, ntb=1, cut=9.0), box=BOX,
    )
    handle = engine.create_handle(np.arange(9, dtype=float).reshape(3, 3))
    got = read_rst7_box(handle)
    assert got is not None, "create_handle produced a boxless restart for a periodic run"
    assert np.allclose(got, BOX)


def test_create_handle_without_a_box_is_unchanged(tmp_path):
    engine = CoreAmberEngine(
        topology=tmp_path / "sys.prmtop", executable=tmp_path / "sander",
        scratch_dir=tmp_path / "scratch", temperature=300.0,
        mdin_controls=dict(dt=0.002, ntb=0, cut=999.0),
    )
    handle = engine.create_handle(np.zeros((3, 3)))
    assert read_rst7_box(handle) is None


def test_periodic_engine_without_a_box_warns(tmp_path):
    """Silence here means a run that dies much later inside sander."""
    with pytest.warns(RuntimeWarning, match="periodic"):
        CoreAmberEngine(
            topology=tmp_path / "sys.prmtop", executable=tmp_path / "sander",
            scratch_dir=tmp_path / "scratch", temperature=300.0,
            mdin_controls=dict(dt=0.002, ntb=1, cut=9.0),
        )
