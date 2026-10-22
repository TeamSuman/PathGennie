"""The per-frame periodic box reaches projection/convergence functions that ask for it.

Under a barostat every frame has its own box, and CVs that use minimum images or
periodic clusters need the box of *that* frame. The engines know it; these tests
pin down how it is carried (``pathgennie.core.box``), that only functions declaring
``box`` receive it, that it is stored per frame and in the checkpoint, and that a
resume restores the anchor's own box instead of the starting one.
"""

import pickle

import numpy as np
import pytest

from pathgennie.core.box import (
    FrameCoords,
    call_with_box,
    create_handle,
    lengths_angles,
    normalize_box,
)
from pathgennie.core.progress import CallableProjection


# --------------------------------------------------------------------------- #
# Box representation
# --------------------------------------------------------------------------- #

def test_normalize_box_accepts_lengths_angles_and_matrix():
    assert normalize_box(None) is None
    np.testing.assert_allclose(normalize_box([10, 20, 30]), np.diag([10, 20, 30]))
    np.testing.assert_allclose(normalize_box([10, 20, 30, 90, 90, 90]), np.diag([10, 20, 30]), atol=1e-9)
    m = np.array([[10, 0, 0], [3, 9, 0], [1, 2, 8.0]])
    np.testing.assert_allclose(normalize_box(m), m)
    with pytest.raises(ValueError):
        normalize_box([1, 2])


def test_lengths_angles_round_trip_triclinic():
    la = np.array([31.0, 33.0, 29.0, 70.0, 80.0, 60.0])
    np.testing.assert_allclose(lengths_angles(normalize_box(la)), la, atol=1e-9)


def test_framecoords_keeps_box_through_copy_slice_and_pickle_but_not_asarray():
    box = np.diag([5.0, 6.0, 7.0])
    c = FrameCoords(np.zeros((4, 3)), box)
    np.testing.assert_allclose(c.copy().box, box)
    np.testing.assert_allclose(c[:2].box, box)
    np.testing.assert_allclose(pickle.loads(pickle.dumps(c)).box, box)
    assert type(np.asarray(c)) is np.ndarray


# --------------------------------------------------------------------------- #
# Opt-in by signature
# --------------------------------------------------------------------------- #

def test_call_with_box_only_for_functions_that_declare_box():
    coords = FrameCoords(np.ones((2, 3)), [4.0, 4.0, 4.0])
    seen = {}

    def wants(c, box=None, scale=1.0):
        seen["wants"] = (type(c), None if box is None else box.copy(), scale)
        return 0

    def ignores(c, **kwargs):
        seen["ignores"] = (type(c), dict(kwargs))
        return 0

    call_with_box(wants, coords, scale=2.0)
    call_with_box(ignores, coords, scale=2.0)
    assert seen["wants"][0] is np.ndarray
    np.testing.assert_allclose(seen["wants"][1], np.diag([4.0, 4.0, 4.0]))
    # **kwargs alone does not opt in; the coordinates arrive unchanged (FrameCoords)
    assert seen["ignores"][1] == {"scale": 2.0}
    assert seen["ignores"][0] is FrameCoords


def test_static_config_box_kept_when_frame_has_none_and_overridden_when_it_has_one():
    def cv(c, box=None):
        return np.asarray(box, dtype=float).ravel()[:1]

    static = np.diag([9.0, 9.0, 9.0])
    assert call_with_box(cv, np.zeros((1, 3)), box=static)[0] == 9.0
    assert call_with_box(cv, FrameCoords(np.zeros((1, 3)), None), box=static)[0] == 9.0
    assert call_with_box(cv, FrameCoords(np.zeros((1, 3)), [3.0, 3.0, 3.0]), box=static)[0] == 3.0


def test_callable_projection_passes_box_and_cycle():
    got = {}

    def cv(c, cycle=None, box=None):
        got.update(cycle=cycle, box=box)
        return np.array([1.0])

    CallableProjection(cv).project(FrameCoords(np.zeros((1, 3)), [2.0, 2.0, 2.0]), cycle=7)
    assert got["cycle"] == 7
    np.testing.assert_allclose(got["box"], np.diag([2.0, 2.0, 2.0]))


# --------------------------------------------------------------------------- #
# OpenMM: real barostat on the CPU platform
# --------------------------------------------------------------------------- #

openmm = pytest.importorskip("openmm")
from openmm import (  # noqa: E402
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    NonbondedForce,
    Platform,
    System,
    Vec3,
    unit,
)
from openmm.app import Simulation, Topology  # noqa: E402

from pathgennie.backends.openmm.engine import OpenMMEngine  # noqa: E402
from pathgennie.backends.openmm.pg_omm import PathGennieMD  # noqa: E402


def _lj_npt(n_side=4, spacing=0.40, pressure_bar=200.0):
    """A small periodic Lennard-Jones fluid with a Monte Carlo barostat that fires
    every step, so the box changes within a few steps."""
    n = n_side ** 3
    L = n_side * spacing
    system = System()
    system.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L))
    nb = NonbondedForce()
    nb.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(0.6 * unit.nanometer)
    for _ in range(n):
        system.addParticle(40.0 * unit.amu)
        nb.addParticle(0.0, 0.34 * unit.nanometer, 0.9 * unit.kilojoule_per_mole)
    system.addForce(nb)
    system.addForce(MonteCarloBarostat(pressure_bar * unit.bar, 120 * unit.kelvin, 1))
    topology = Topology()
    chain = topology.addChain()
    res = topology.addResidue("AR", chain)
    for _ in range(n):
        topology.addAtom("Ar", None, res)
    topology.setPeriodicBoxVectors([Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L)] * unit.nanometer)
    integ = LangevinMiddleIntegrator(120 * unit.kelvin, 5.0 / unit.picosecond, 0.004 * unit.picoseconds)
    sim = Simulation(topology, system, integ, Platform.getPlatformByName("CPU"))
    g = np.arange(n_side) * spacing + spacing / 2
    pos = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    sim.context.setPositions(pos * unit.nanometer)
    return sim, pos * unit.nanometer, L


def test_openmm_engine_reports_each_frames_box_and_restores_it():
    sim, pos, L = _lj_npt()
    engine = OpenMMEngine(sim, temperature=120.0)
    h0 = engine.create_state(pos)
    np.testing.assert_allclose(engine.get_box(h0), np.diag([L * 10] * 3), rtol=1e-6)
    h1 = engine.run_segment(engine.clone_anchor(h0), 200, randomize_velocities=True, seed=3)
    b1 = engine.get_box(h1)
    assert not np.allclose(b1, engine.get_box(h0))  # the barostat moved the box
    # A handle rebuilt from coordinates + box carries that box, not the context's.
    h2 = create_handle(engine, FrameCoords(engine.get_coords(h1), b1))
    np.testing.assert_allclose(engine.get_box(h2), b1, rtol=1e-9)


def test_openmm_engine_without_periodic_boundaries_has_no_box():
    from openmm import CustomExternalForce
    system = System()
    system.addParticle(12.0 * unit.amu)
    f = CustomExternalForce("x*x+y*y+z*z")
    f.addParticle(0, [])
    system.addForce(f)
    top = Topology()
    top.addAtom("C", None, top.addResidue("X", top.addChain()))
    sim = Simulation(top, system, LangevinMiddleIntegrator(300, 1, 0.002), Platform.getPlatformByName("CPU"))
    engine = OpenMMEngine(sim, temperature=300.0)
    assert engine.get_box(engine.create_state([[0, 0, 0]] * unit.nanometer)) is None


def test_driver_passes_trial_boxes_and_stores_and_restores_them(tmp_path, monkeypatch):
    from pathgennie.core.storage import HDF5Storage

    seen = []

    def cv(coords, box=None):
        seen.append(None if box is None else float(box[0, 0]))
        return np.array([coords[0, 0]])

    def run(max_cycle):
        sim, pos, _ = _lj_npt()
        runner = PathGennieMD(
            simulation=sim, projection_fn=cv, mode="escape",
            convergence_fn=lambda c, box=None: False,
            temperature=120.0, sigma=0.1, seed=11, checkpoint_freq=2,
        )
        runner.run(pos, tau1=40, tau2=10, max_trial=3, max_cycle=max_cycle, save_freq=1,
                   verbosity=0, checkpoint_path=str(tmp_path / "ck.h5"))
        return runner

    run(4)
    assert seen and all(b is not None for b in seen)
    assert len(set(np.round(seen, 6))) > 1          # trials scored in their own boxes

    ck = HDF5Storage.load_checkpoint(tmp_path / "ck.h5")
    assert ck["anchor_box"].shape == (3, 3)
    assert ck["boxes"] is not None and len(ck["boxes"]) == len(ck["trajectory"])
    assert not np.allclose(ck["anchor_box"], np.diag([16.0] * 3))  # the barostat moved it

    # Resume: the anchor is rebuilt with the saved anchor box, not the starting box.
    restored = []
    orig = OpenMMEngine.create_handle

    def spy(self, coords, box=None):
        restored.append(None if box is None else np.array(box))
        return orig(self, coords, box=box)

    monkeypatch.setattr(OpenMMEngine, "create_handle", spy)
    run(6)
    assert restored and restored[0] is not None
    np.testing.assert_allclose(restored[0], ck["anchor_box"])


def test_runs_without_a_box_store_none():
    from pathgennie.core.box import frame_coords
    from pathgennie.core.toy import ToyLangevinEngine
    eng = ToyLangevinEngine()
    h = eng.create_state([0.0, 0.0])
    assert frame_coords(eng, h).box is None


def test_old_checkpoint_without_box_loads_with_none(tmp_path):
    from pathgennie.core.storage import HDF5Storage
    st = HDF5Storage(tmp_path / "old.h5")
    st.append("trajectory", np.zeros((2, 3)))
    st.append("metric", np.array([0.0]))
    st.save_checkpoint(cycle=0, rng_state=np.random.default_rng(0).bit_generator.state,
                       anchor_coords=np.zeros((2, 3)), anchor_cv=np.zeros(1), anchor_metric=0.0)
    st.close()
    ck = HDF5Storage.load_checkpoint(tmp_path / "old.h5")
    assert ck["anchor_box"] is None and ck["boxes"] is None


# --------------------------------------------------------------------------- #
# GROMACS / AMBER file boxes
# --------------------------------------------------------------------------- #

def _gro(path, box_line):
    path.write_text("t\n    2\n    1SOL     OW    1   0.100   0.200   0.300\n"
                    "    1SOL    HW1    2   0.150   0.200   0.300\n" + box_line + "\n")


def test_gro_box_rectangular_and_triclinic(tmp_path):
    from pathgennie.backends.gromacs.utils import read_gro_box
    _gro(tmp_path / "r.gro", "   3.00000   4.00000   5.00000")
    np.testing.assert_allclose(read_gro_box(tmp_path / "r.gro"), np.diag([30.0, 40.0, 50.0]))
    # triclinic order: v1x v2y v3z v1y v1z v2x v2z v3x v3y
    _gro(tmp_path / "t.gro", "   3.00000   4.00000   5.00000   0.00000   0.00000   1.00000   0.00000   0.50000   1.50000")
    np.testing.assert_allclose(read_gro_box(tmp_path / "t.gro"),
                               [[30, 0, 0], [10, 40, 0], [5, 15, 50]])


def test_write_gro_uses_the_frames_box(tmp_path):
    from pathgennie.backends.gromacs.utils import read_gro_box, write_gro_coords
    _gro(tmp_path / "tpl.gro", "   3.00000   3.00000   3.00000")
    for box in (np.diag([31.0, 32.0, 33.0]), np.array([[30.0, 0, 0], [10, 40, 0], [5, 15, 50]])):
        write_gro_coords(tmp_path / "tpl.gro", tmp_path / "o.gro", np.zeros((2, 3)), box=box)
        np.testing.assert_allclose(read_gro_box(tmp_path / "o.gro"), box, atol=1e-4)
    write_gro_coords(tmp_path / "tpl.gro", tmp_path / "o.gro", np.zeros((2, 3)))
    np.testing.assert_allclose(read_gro_box(tmp_path / "o.gro"), np.diag([30.0] * 3))


def test_rst7_box_written_from_a_frame_box_reads_back(tmp_path):
    from pathgennie.backends.amber.utils import read_rst7_box, write_rst7_coords
    box = normalize_box([30.0, 31.0, 32.0, 90.0, 90.0, 90.0])
    write_rst7_coords(tmp_path / "x.rst7", np.zeros((3, 3)), box=lengths_angles(box))
    np.testing.assert_allclose(normalize_box(read_rst7_box(tmp_path / "x.rst7")), box, atol=1e-5)
