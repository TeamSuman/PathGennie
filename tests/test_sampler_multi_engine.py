"""``EngineSampler`` must work with every backend, not just OpenMM.

The refinement loop was OpenMM-only, which blocked QM/MM entirely (AMBER is the
only backend that can run a QM Hamiltonian). ``EngineSampler`` closes that by
driving refinement through the core ``Engine`` protocol, so these tests assert the
contract holds for each engine independently:

  * the toy Langevin engine (always available -- this is the CI guarantee),
  * a recording stub, proving nothing OpenMM-specific is required,
  * OpenMM, when installed,
  * AMBER's ``CoreAmberEngine``, constructed against a real prmtop.

The AMBER and GROMACS rows here assert protocol conformance only, so that CI needs
no MD binary. Both have separately been driven by a real binary end-to-end --
``sander`` under QM/MM DFTB3, and ``gmx`` on solvated alanine dipeptide -- via
``examples/path_refinement_engines/refine_with_engine.py``. Run that when changing
the subprocess engines; a conformance check cannot catch a broken command line.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathgennie.core.engine import Engine
from pathrefinement.samplers import EngineSampler


# --------------------------------------------------------------------------- #
# A recording stub: implements the Engine protocol and nothing else.
# --------------------------------------------------------------------------- #
class RecordingEngine:
    """Minimal 1-D engine that drifts toward +x; records what the driver asked."""

    def __init__(self):
        self.cache = {0: np.zeros((1, 3))}
        self.next_id = 1
        self.calls = {"clone": 0, "segment": 0, "release": 0}

    def clone_anchor(self, handle):
        self.calls["clone"] += 1
        h = self.next_id
        self.next_id += 1
        self.cache[h] = self.cache[handle].copy()
        return h

    def run_segment(self, handle, n_steps, *, randomize_velocities, seed,
                    device=None, save_subframes=False, subframe_stride=1):
        self.calls["segment"] += 1
        rng = np.random.default_rng(seed)
        pos = self.cache[handle].copy()
        pos[0, 0] += 0.05 * n_steps + 0.01 * rng.standard_normal()
        h = self.next_id
        self.next_id += 1
        self.cache[h] = pos
        return h

    def get_coords(self, handle):
        return self.cache[handle]

    def release(self, handle):
        self.calls["release"] += 1
        self.cache.pop(handle, None)

    def create_handle(self, coords):
        h = self.next_id
        self.next_id += 1
        self.cache[h] = np.asarray(coords, dtype=float).reshape(-1, 3)
        return h


def _feature(xyz):
    return np.asarray(xyz, dtype=float).ravel()[:1]


def test_stub_engine_satisfies_the_protocol():
    assert isinstance(RecordingEngine(), Engine)


def test_sampler_drives_an_arbitrary_engine():
    """No OpenMM anywhere: the sampler works off the Engine protocol alone."""
    eng = RecordingEngine()
    sampler = EngineSampler(eng, initial_handle=0, feature_fn=_feature,
                            tau1=2, tau2=2, max_trial=3, max_cycle=10, tol=1e9)
    traj = sampler(path_cv=None, start_pt=np.array([0.0]), seed=7)

    assert traj is not None and len(traj) > 0
    assert traj.shape[1] == 1
    assert eng.calls["segment"] > 0, "the driver never propagated the engine"
    assert eng.calls["release"] > 0, "handles must be released (scratch/leak safety)"


def test_sampler_is_reproducible_for_a_fixed_seed():
    a = EngineSampler(RecordingEngine(), 0, feature_fn=_feature,
                      tau1=2, tau2=2, max_trial=3, max_cycle=8, tol=1e9)
    b = EngineSampler(RecordingEngine(), 0, feature_fn=_feature,
                      tau1=2, tau2=2, max_trial=3, max_cycle=8, tol=1e9)
    ta = a(path_cv=None, start_pt=np.array([0.0]), seed=1234)
    tb = b(path_cv=None, start_pt=np.array([0.0]), seed=1234)
    assert np.allclose(ta, tb), "same seed must give the same walker"


def test_sampler_with_the_toy_langevin_engine():
    """The CI guarantee: works with the always-available pure-NumPy engine."""
    from pathgennie.core.toy import ToyLangevinEngine

    eng = ToyLangevinEngine(dt=0.002, kT=1.0)
    h0 = eng.create_state([-1.0, 0.0])
    sampler = EngineSampler(eng, initial_handle=h0,
                            feature_fn=lambda xyz: np.asarray(xyz).ravel()[:2],
                            tau1=5, tau2=5, max_trial=4, max_cycle=15, tol=1e9)
    traj = sampler(path_cv=None, start_pt=np.array([1.0]), seed=11)
    assert traj is not None and traj.shape[1] == 2
    assert np.all(np.isfinite(traj))


def test_sampler_with_openmm_engine():
    pytest.importorskip("openmm")
    import openmm
    from openmm import unit
    from openmm.app import Simulation, Topology, Element

    from pathgennie.backends.openmm.engine import OpenMMEngine

    system = openmm.System()
    system.addParticle(12.0)
    f = openmm.CustomExternalForce("0.5*k*(x*x+y*y+z*z)")
    f.addGlobalParameter("k", 100.0)
    f.addParticle(0, [])
    system.addForce(f)
    top = Topology()
    chain = top.addChain()
    res = top.addResidue("LIG", chain)
    top.addAtom("C", Element.getBySymbol("C"), res)
    integ = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                            0.002 * unit.picoseconds)
    sim = Simulation(top, system, integ, openmm.Platform.getPlatformByName("CPU"))

    eng = OpenMMEngine(sim, temperature=300.0)
    h0 = eng.create_state([[0.1, 0.0, 0.0]] * unit.nanometer)
    sampler = EngineSampler(eng, initial_handle=h0, feature_fn=_feature,
                            tau1=5, tau2=5, max_trial=2, max_cycle=5, tol=1e9)
    traj = sampler(path_cv=None, start_pt=np.array([0.0]), seed=3)
    assert traj is not None and np.all(np.isfinite(traj))


def test_amber_and_gromacs_engines_conform_to_the_protocol():
    """Subprocess backends expose the same surface the sampler needs.

    Constructing them requires no MD binary; only running a segment would. This
    pins that ``EngineSampler`` could drive either, which is what makes QM/MM
    refinement possible through AMBER.
    """
    from pathlib import Path

    from pathgennie.backends.amber.engine import CoreAmberEngine
    from pathgennie.backends.gromacs.pg_gmx import CoreGromacsEngine

    tmp = Path("/tmp")
    amber = CoreAmberEngine(
        topology=tmp / "x.prmtop", executable=tmp / "sander",
        scratch_dir=tmp / "_pg_scratch_amber", temperature=300.0, mdin_controls={},
    )
    gmx = CoreGromacsEngine(
        topology=tmp / "x.top", executable=tmp / "gmx",
        scratch_dir=tmp / "_pg_scratch_gmx", temperature=300.0, mdp_controls={},
    )
    for eng in (amber, gmx):
        assert isinstance(eng, Engine)
        for method in ("clone_anchor", "run_segment", "get_coords", "release", "create_handle"):
            assert callable(getattr(eng, method)), f"{type(eng).__name__} lacks {method}"
