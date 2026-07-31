"""``run_segment`` must return a tuple whenever ``save_subframes`` is True.

The ``Engine`` protocol states it plainly:

    When ``save_subframes`` is True, intermediate positions are captured every
    ``subframe_stride`` integrator steps and the return changes to
    ``(Handle, subframes)``.

The driver relies on that and unpacks unconditionally::

    tau1_replay_handle, tau1_subframes = tau1_result

The AMBER and GROMACS backends only returned a tuple when the trajectory file
*existed*, falling through to a bare handle otherwise. A handle there is a file
path, so unpacking it raises ``ValueError: too many values to unpack`` and kills
the run. The trigger is mundane -- a stride longer than the segment, so the MD
engine writes no frames and may not create the file at all.

The toy and OpenMM engines already got this right, which is why it survived: the
two backends that can actually fail to produce a file are the two that broke the
contract.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from pathgennie.backends.amber.engine import CoreAmberEngine
from pathgennie.backends.gromacs.pg_gmx import CoreGromacsEngine
from pathgennie.backends.amber.utils import write_rst7_coords


def _amber(tmp_path):
    return CoreAmberEngine(
        topology=tmp_path / "sys.prmtop", executable=tmp_path / "sander",
        scratch_dir=tmp_path / "scratch", temperature=300.0,
        mdin_controls=dict(dt=0.002, ntb=0, cut=99.0),
    )


def _fake_sander(monkeypatch, write_traj: bool):
    """Pretend sander ran. Optionally omit the trajectory file, which is the case."""
    def fake_run(cmd, **_kw):
        out = cmd[cmd.index("-r") + 1]
        write_rst7_coords(out, np.zeros((3, 3)))
        if write_traj and "-x" in cmd:
            # A real .nc is not needed; the missing-file branch is what's under test.
            open(cmd[cmd.index("-x") + 1], "wb").close()
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", fake_run)


def test_amber_returns_a_tuple_when_no_trajectory_was_written(tmp_path, monkeypatch):
    """The defect: a bare path string here makes the driver's unpack raise."""
    engine = _amber(tmp_path)
    handle = engine.create_handle(np.zeros((3, 3)))
    _fake_sander(monkeypatch, write_traj=False)

    result = engine.run_segment(handle, 10, randomize_velocities=True, seed=1,
                                save_subframes=True, subframe_stride=100)
    assert isinstance(result, tuple), \
        f"contract violated: got {type(result).__name__}, which the driver cannot unpack"
    assert len(result) == 2
    new_handle, subframes = result           # the exact operation the driver performs
    assert isinstance(new_handle, str)
    assert len(subframes) == 0


def test_amber_empty_subframes_have_a_usable_shape(tmp_path, monkeypatch):
    """(0, n_atoms, 3), so concatenating with real blocks stays legal."""
    engine = _amber(tmp_path)
    handle = engine.create_handle(np.zeros((3, 3)))
    _fake_sander(monkeypatch, write_traj=False)
    _, subframes = engine.run_segment(handle, 10, randomize_velocities=True, seed=1,
                                      save_subframes=True, subframe_stride=100)
    assert subframes.ndim == 3
    assert subframes.shape[0] == 0
    real = np.zeros((4,) + subframes.shape[1:])
    assert np.concatenate([subframes, real], axis=0).shape[0] == 4


def test_amber_without_subframes_returns_a_bare_handle(tmp_path, monkeypatch):
    """The other half of the contract must not regress."""
    engine = _amber(tmp_path)
    handle = engine.create_handle(np.zeros((3, 3)))
    _fake_sander(monkeypatch, write_traj=False)
    result = engine.run_segment(handle, 10, randomize_velocities=True, seed=1)
    assert isinstance(result, str)


def test_gromacs_returns_a_tuple_when_no_trajectory_was_written(tmp_path, monkeypatch):
    engine = CoreGromacsEngine(
        topology=tmp_path / "sys.top", executable=tmp_path / "gmx",
        scratch_dir=tmp_path / "scratch", temperature=300.0,
        mdp_controls={"integrator": "md", "dt": 0.002},
        template_gro=tmp_path / "start.gro",
    )
    (tmp_path / "start.gro").write_text(
        "t\n 1\n    1LIG      C    1   0.000   0.000   0.000\n   1.0 1.0 1.0\n")

    def fake_run(cmd, **_kw):
        if "-c" in cmd:                       # mdrun writes the output .gro
            out = cmd[cmd.index("-c") + 1]
            if str(out).endswith(".gro"):
                open(out, "w").write(
                    "t\n 1\n    1LIG      C    1   0.000   0.000   0.000\n   1.0 1.0 1.0\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(engine, "_run", lambda cmd, stage, env: fake_run(cmd))

    result = engine.run_segment(str(tmp_path / "start.gro"), 10,
                                randomize_velocities=True, seed=1,
                                save_subframes=True, subframe_stride=100)
    assert isinstance(result, tuple), \
        f"contract violated: got {type(result).__name__}"
    _, subframes = result
    assert len(subframes) == 0


@pytest.mark.parametrize("engine_name", ["toy", "openmm"])
def test_in_process_engines_already_honour_the_contract(engine_name):
    """Regression guard for the two that were already correct.

    Only the *shape* of the return is asserted, not the frame count. The engines
    legitimately differ on what a stride longer than the segment means: OpenMM
    steps in ``min(stride, remaining)`` chunks and so always captures the segment
    end, while the toy engine uses a strict modulo and captures nothing. Both
    satisfy the protocol; the driver filters empty blocks either way.
    """
    if engine_name == "toy":
        from pathgennie.core.toy import ToyLangevinEngine
        engine = ToyLangevinEngine(dt=0.002, kT=1.0)
        handle = engine.create_state([0.0, 0.0])
    else:
        openmm = pytest.importorskip("openmm")
        from openmm import unit
        from openmm.app import Element, Simulation, Topology

        from pathgennie.backends.openmm.engine import OpenMMEngine

        system = openmm.System()
        system.addParticle(12.0)
        f = openmm.CustomExternalForce("x*x+y*y+z*z")
        f.addParticle(0, [])
        system.addForce(f)
        top = Topology()
        res = top.addResidue("LIG", top.addChain())
        top.addAtom("C", Element.getBySymbol("C"), res)
        sim = Simulation(top, system,
                         openmm.LangevinMiddleIntegrator(300 * unit.kelvin,
                                                         1 / unit.picosecond,
                                                         0.002 * unit.picoseconds),
                         openmm.Platform.getPlatformByName("CPU"))
        engine = OpenMMEngine(sim, temperature=300.0)
        handle = engine.create_state([[0.0, 0.0, 0.0]] * unit.nanometer)

    # Stride longer than the segment -- the case that broke the subprocess
    # backends. A tuple is required regardless of how many frames land.
    result = engine.run_segment(handle, 4, randomize_velocities=True, seed=1,
                                save_subframes=True, subframe_stride=100)
    assert isinstance(result, tuple)
    _, subframes = result
    assert isinstance(subframes, np.ndarray)
    assert subframes.ndim == 3, "subframes must be (n_subframes, n_atoms, 3)"
