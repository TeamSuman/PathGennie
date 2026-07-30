"""AMBER restarts written by ``create_handle`` carry no velocities.

``sander`` aborts with "I could not find enough velocities" when told to restart
(``irest=1``, ``ntx=5``) from a coordinates-only file. The Weighted Ensemble stage
seeds walkers from raw frames via ``create_handle`` and defaults to
``continue_velocities=True``, so every AMBER WE run died on its first segment.
``run_segment`` now detects the missing block and generates velocities instead.

No AMBER binary is needed: the outcome is decided entirely by what goes into the
mdin, which these tests read back directly.
"""

from __future__ import annotations

import subprocess
import warnings

import numpy as np
import pytest

from pathgennie.backends.amber.engine import CoreAmberEngine
from pathgennie.backends.amber.utils import rst7_has_velocities, write_rst7_coords


def _write_rst7(path, coords, *, velocities=None, box=None):
    """Write an rst7 by hand so the tests control exactly which blocks exist."""
    body = []
    for block in ([coords] if velocities is None else [coords, velocities]):
        flat = np.asarray(block, dtype=float).ravel()
        for i in range(0, len(flat), 6):
            body.append("".join(f"{v:12.7f}" for v in flat[i:i + 6]))
    if box is not None:
        body.append("".join(f"{v:12.7f}" for v in box))
    path.write_text(f"title\n{len(coords):5d}\n" + "\n".join(body) + "\n")
    return path


def _engine(tmp_path, **controls):
    base = dict(dt=0.002, ntb=0, cut=999.0)
    base.update(controls)
    return CoreAmberEngine(
        topology=tmp_path / "sys.prmtop",
        executable=tmp_path / "sander",
        scratch_dir=tmp_path / "scratch",
        temperature=300.0,
        mdin_controls=base,
    )


@pytest.fixture
def captured_mdin(monkeypatch):
    """Intercept the sander launch; yield a dict that gains the mdin text."""
    seen = {}

    def fake_run(cmd, **_kw):
        seen["text"] = open(cmd[cmd.index("-i") + 1]).read()
        # run_segment reads the output restart afterwards, so it must exist.
        write_rst7_coords(cmd[cmd.index("-r") + 1], np.zeros((3, 3)))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return seen


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #
def test_created_restart_has_no_velocity_block(tmp_path):
    path = tmp_path / "ckpt.rst7"
    write_rst7_coords(path, np.arange(9, dtype=float).reshape(3, 3))
    assert not rst7_has_velocities(path)


def test_detector_finds_a_real_velocity_block(tmp_path):
    coords = np.arange(9, dtype=float).reshape(3, 3)
    path = _write_rst7(tmp_path / "full.rst7", coords, velocities=coords * 0.1)
    assert rst7_has_velocities(path)


def test_detector_does_not_mistake_a_box_line_for_velocities(tmp_path):
    """With two atoms a box line is the same width as a one-line velocity block."""
    coords = np.arange(6, dtype=float).reshape(2, 3)
    path = _write_rst7(tmp_path / "boxed.rst7", coords,
                       box=[30.0, 30.0, 30.0, 90.0, 90.0, 90.0])
    assert not rst7_has_velocities(path, has_box=True)


def test_detector_survives_a_malformed_file(tmp_path):
    path = tmp_path / "junk.rst7"
    path.write_text("title\nnot-a-number\n")
    assert not rst7_has_velocities(path)


# --------------------------------------------------------------------------- #
# The engine behaviour that was broken
# --------------------------------------------------------------------------- #
def test_continue_velocities_falls_back_on_a_coords_only_restart(tmp_path, captured_mdin):
    """The case that killed every AMBER Weighted Ensemble run."""
    engine = _engine(tmp_path)
    handle = engine.create_handle(np.arange(9, dtype=float).reshape(3, 3))

    with pytest.warns(RuntimeWarning, match="coordinates-only rst7"):
        engine.run_segment(handle, 10, randomize_velocities=False, seed=1)

    assert "irest = 0," in captured_mdin["text"]
    assert "ntx = 1," in captured_mdin["text"]


def test_continue_velocities_is_honoured_when_the_restart_has_them(tmp_path, captured_mdin):
    """The fallback must not fire on a genuine restart -- doing so would silently
    decorrelate every tau2 runner in the driver."""
    coords = np.arange(9, dtype=float).reshape(3, 3)
    handle = _write_rst7(tmp_path / "full.rst7", coords, velocities=coords * 0.1)

    engine = _engine(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.run_segment(str(handle), 10, randomize_velocities=False, seed=1)

    assert not [w for w in caught if "coordinates-only" in str(w.message)]
    assert "irest = 1," in captured_mdin["text"]
    assert "ntx = 5," in captured_mdin["text"]


def test_randomize_velocities_is_unaffected(tmp_path, captured_mdin):
    engine = _engine(tmp_path)
    handle = engine.create_handle(np.arange(9, dtype=float).reshape(3, 3))
    engine.run_segment(handle, 10, randomize_velocities=True, seed=1)
    assert "irest = 0," in captured_mdin["text"]
    assert "ntx = 1," in captured_mdin["text"]


def test_the_warning_is_emitted_once_not_once_per_segment(tmp_path, captured_mdin):
    """A swarm issues thousands of segments; one warning is informative, thousands
    are noise that buries everything else."""
    engine = _engine(tmp_path)
    handle = engine.create_handle(np.arange(9, dtype=float).reshape(3, 3))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            engine.run_segment(handle, 10, randomize_velocities=False, seed=1)
    assert len([w for w in caught if "coordinates-only" in str(w.message)]) == 1
