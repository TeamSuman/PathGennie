"""Real OpenMM (CPU platform) validation of the engine adapter + core driver.

Builds a tiny system (a few particles in a harmonic well) so the full in-process
path runs in well under a second on CPU, with no GPU required.  Skipped if
OpenMM is not installed.
"""

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import (  # noqa: E402
    CustomExternalForce,
    LangevinMiddleIntegrator,
    Platform,
    System,
    VerletIntegrator,
    unit,
)
from openmm.app import Simulation, Topology  # noqa: E402

from pathgennie.backends.openmm.engine import OpenMMEngine, resolve_worker_count  # noqa: E402
from pathgennie.backends.openmm.pg_omm import PathGennieMD  # noqa: E402


def _build_simulation(n_particles=3, integrator="langevin"):
    system = System()
    for _ in range(n_particles):
        system.addParticle(12.0 * unit.amu)
    # Soft harmonic well centred at the origin so dynamics stay bounded.
    force = CustomExternalForce("0.5*k*(x*x + y*y + z*z)")
    force.addGlobalParameter("k", 10.0)
    for i in range(n_particles):
        force.addParticle(i, [])
    system.addForce(force)

    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue("X", chain)
    for _ in range(n_particles):
        topology.addAtom("C", None, residue)

    if integrator == "verlet":
        integ = VerletIntegrator(0.002 * unit.picoseconds)
    else:
        integ = LangevinMiddleIntegrator(
            300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
        )
    sim = Simulation(topology, system, integ, Platform.getPlatformByName("CPU"))
    return sim, n_particles


def _positions(n):
    return [[0.0, 0.0, 0.0] for _ in range(n)] * unit.nanometer


def test_openmm_engine_segment_roundtrip():
    sim, n = _build_simulation()
    engine = OpenMMEngine(sim, temperature=300.0)
    h0 = engine.create_state(_positions(n))
    coords0 = engine.get_coords(h0)
    assert coords0.shape == (n, 3)

    h1 = engine.clone_anchor(h0)
    h2 = engine.run_segment(h1, 5, randomize_velocities=True, seed=1)
    coords2 = engine.get_coords(h2)
    assert coords2.shape == (n, 3)
    assert np.all(np.isfinite(coords2))
    # Original snapshot is untouched by stepping the clone.
    np.testing.assert_allclose(engine.get_coords(h0), coords0)


def test_openmm_engine_run_reproducible():
    # Deterministic integrator (Verlet): all remaining randomness (velocity
    # draws + selection) is driven by the driver's seeded RNG, so the run is
    # bit-reproducible. NOTE: with a stochastic Langevin integrator the
    # thermostat noise stream is not per-segment reseedable in OpenMM, so only
    # the driver-level decisions are reproducible there.
    def run_once():
        sim, n = _build_simulation(integrator="verlet")
        runner = PathGennieMD(
            simulation=sim,
            projection_fn=lambda c: np.array([c[0, 0]]),
            mode="escape",
            convergence_fn=lambda c, **k: abs(c[0, 0]) > 50.0,  # never converges here
            temperature=300.0,
            sigma=0.1,
            seed=2024,
        )
        return runner.run(_positions(n), tau1=3, tau2=5, max_trial=6, max_cycle=20, save_freq=2, verbosity=0)

    traj_a, metrics_a = run_once()
    traj_b, metrics_b = run_once()
    assert traj_a.shape[0] >= 1
    np.testing.assert_allclose(metrics_a, metrics_b)
    np.testing.assert_allclose(traj_a, traj_b)


# --------------------------------------------------------------------------- #
# Single-GPU saturation: concurrent Context pool
# --------------------------------------------------------------------------- #

def test_resolve_worker_count():
    assert resolve_worker_count(1) == 1
    assert resolve_worker_count(None) == 1
    assert resolve_worker_count(8) == 8
    assert resolve_worker_count("auto") >= 1  # CPU-core cap, always >= 1


def test_openmm_engine_builds_concurrent_context_pool():
    # On the CPU platform there is no GPU-memory cap, so all requested Contexts
    # are built. State snapshots are portable across the pooled Contexts.
    sim, n = _build_simulation()
    engine = OpenMMEngine(sim, temperature=300.0, n_workers=4)
    assert engine.n_workers == 4
    assert engine._pool.qsize() == 4

    h0 = engine.create_state(_positions(n))
    coords0 = engine.get_coords(h0)
    # Run several segments; each borrows a Context from the pool.
    handles = [engine.run_segment(engine.clone_anchor(h0), 5, randomize_velocities=True, seed=s)
               for s in range(6)]
    for h in handles:
        assert engine.get_coords(h).shape == (n, 3)
    # The shared immutable-State cache is intact and the original is untouched.
    np.testing.assert_allclose(engine.get_coords(h0), coords0)


def test_openmm_concurrent_matches_serial_verlet():
    # A deterministic integrator makes each segment a pure function of (state,
    # seed), independent of which pooled Context runs it, so saturating the GPU
    # with several concurrent walkers must reproduce the serial result exactly.
    def run_once(workers):
        sim, n = _build_simulation(integrator="verlet")
        runner = PathGennieMD(
            simulation=sim,
            projection_fn=lambda c: np.array([c[0, 0]]),
            mode="escape",
            convergence_fn=lambda c, **k: abs(c[0, 0]) > 50.0,
            temperature=300.0, sigma=0.1, seed=2024,
            workers_per_device=workers,
        )
        out = runner.run(_positions(n), tau1=3, tau2=5, max_trial=6,
                         max_cycle=15, save_freq=2, verbosity=0)
        return out, runner.engine.n_workers

    (traj1, m1), nw1 = run_once(1)
    (traj4, m4), nw4 = run_once(4)
    assert nw1 == 1 and nw4 == 4
    np.testing.assert_allclose(m1, m4)
    np.testing.assert_allclose(traj1, traj4)


# --------------------------------------------------------------------------- #
# Several GPUs: one Context pool per device
# --------------------------------------------------------------------------- #

def test_openmm_engine_builds_one_pool_per_device():
    # On CPU the device index is only a label, which is enough to check that each
    # device gets its own pool of n_workers Contexts and that the Simulation's
    # Context (on no listed device) is not handed out for segments.
    sim, n = _build_simulation()
    engine = OpenMMEngine(sim, temperature=300.0, n_workers=2, devices=[0, 1])
    assert engine.devices == [0, 1]
    assert engine.workers_by_device == {0: 2, 1: 2}
    assert engine.n_workers == 4
    assert all(engine._pools[d].qsize() == 2 for d in (0, 1))
    assert sim.context not in engine._contexts


def test_openmm_engine_routes_segments_to_the_assigned_device():
    import queue

    class CountingQueue(queue.Queue):
        def __init__(self):
            super().__init__()
            self.gets = 0

        def get(self, *a, **k):
            self.gets += 1
            return super().get(*a, **k)

    sim, n = _build_simulation()
    engine = OpenMMEngine(sim, temperature=300.0, n_workers=1, devices=[0, 1])
    for d in (0, 1):
        q = CountingQueue()
        while not engine._pools[d].empty():
            q.put(engine._pools[d].get())
        engine._pools[d] = q
    h0 = engine.create_state(_positions(n))
    for s in range(3):
        engine.run_segment(engine.clone_anchor(h0), 5, randomize_velocities=True, seed=s, device=1)
    engine.run_segment(engine.clone_anchor(h0), 5, randomize_velocities=True, seed=9, device=0)
    assert engine._pools[1].gets == 3 and engine._pools[0].gets == 1
    assert engine._pools[0].qsize() == 1 and engine._pools[1].qsize() == 1  # Contexts returned


def test_openmm_engine_rejects_repeated_devices():
    sim, _ = _build_simulation()
    with pytest.raises(ValueError):
        OpenMMEngine(sim, temperature=300.0, devices=[0, 0])


def test_openmm_multi_device_matches_single_device_verlet():
    # Which device (Context) runs a segment must not change a deterministic run:
    # two devices x 2 walkers reproduce one device x 1 walker exactly.
    def run_once(devices, workers):
        sim, n = _build_simulation(integrator="verlet")
        runner = PathGennieMD(
            simulation=sim,
            projection_fn=lambda c: np.array([c[0, 0]]),
            mode="escape",
            convergence_fn=lambda c, **k: abs(c[0, 0]) > 50.0,
            temperature=300.0, sigma=0.1, seed=2024,
            workers_per_device=workers, devices=devices,
        )
        out = runner.run(_positions(n), tau1=3, tau2=5, max_trial=6,
                         max_cycle=15, save_freq=2, verbosity=0)
        return out, runner.engine, runner.executor

    (traj1, m1), eng1, _ = run_once(None, 1)
    (traj2, m2), eng2, ex2 = run_once([0, 1], 2)
    assert eng1.n_workers == 1
    assert eng2.workers_by_device == {0: 2, 1: 2} and ex2.devices == [0, 1]
    np.testing.assert_allclose(m1, m2)
    np.testing.assert_allclose(traj1, traj2)


def test_pathgenniemd_single_listed_device_is_single_card_path():
    sim, n = _build_simulation()
    runner = PathGennieMD(
        simulation=sim, projection_fn=lambda c: np.array([c[0, 0]]), mode="escape",
        convergence_fn=lambda c, **k: False, temperature=300.0, devices=[0],
    )
    assert runner.devices is None and runner.device == 0
