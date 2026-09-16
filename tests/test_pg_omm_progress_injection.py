"""Tests for Task A4's `PathGennieMD(..., progress=)` injection point.

Uses a tiny real OpenMM (CPU platform) system, same pattern as
tests/test_openmm_engine.py, so this stays fast with no GPU required; skipped
outright if OpenMM is not installed.
"""

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import CustomExternalForce, Platform, System, VerletIntegrator, unit  # noqa: E402
from openmm.app import Simulation, Topology  # noqa: E402

from pathgennie.backends.openmm.pg_omm import PathGennieMD  # noqa: E402


def _build_simulation(n_particles=3):
    system = System()
    for _ in range(n_particles):
        system.addParticle(12.0 * unit.amu)
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

    integ = VerletIntegrator(0.002 * unit.picoseconds)
    sim = Simulation(topology, system, integ, Platform.getPlatformByName("CPU"))
    return sim, n_particles


def _positions(n):
    return [[0.0, 0.0, 0.0] for _ in range(n)] * unit.nanometer


class RecordingProgress:
    """A minimal, self-contained ProgressVariable (not EscapeMetric/TargetMetric)
    that records every call so the test can prove it -- not the built-ins --
    drove the run."""

    def __init__(self):
        self.project_calls = 0
        self.metric_calls = 0

    def project(self, coords, cycle=None):
        self.project_calls += 1
        return np.array([float(coords[0, 0])])

    def metric(self, cv):
        self.metric_calls += 1
        return -abs(float(cv[0]) - 5.0)


def test_custom_progress_is_used_instead_of_builtin_metrics():
    sim, n = _build_simulation()
    progress = RecordingProgress()
    runner = PathGennieMD(
        simulation=sim,
        projection_fn=lambda c: np.array([c[0, 0]]),  # unused: progress overrides it
        mode="escape",
        convergence_fn=lambda c, **k: False,
        temperature=300.0, sigma=0.1, seed=1,
        progress=progress,
    )
    traj, metrics = runner.run(
        _positions(n), tau1=3, tau2=5, max_trial=4, max_cycle=4, save_freq=1, verbosity=0,
    )
    assert traj.shape[0] >= 1
    assert progress.project_calls > 0
    assert progress.metric_calls > 0


def test_progress_with_default_mode_args_works_standalone():
    """`progress` alone, with every mode/target_projection/escape_metric/periodic
    argument left at its default, must not raise and must drive the run (this is
    the intended, non-conflicting way to use `progress`)."""
    sim, n = _build_simulation()
    progress = RecordingProgress()
    runner = PathGennieMD(
        simulation=sim,
        projection_fn=lambda c: np.array([c[0, 0]]),
        convergence_fn=lambda c, **k: False,
        temperature=300.0, seed=1,
        progress=progress,
    )
    assert runner.progress is progress
    traj, metrics = runner.run(
        _positions(n), tau1=2, tau2=2, max_trial=3, max_cycle=2, save_freq=1, verbosity=0,
    )
    assert traj.shape[0] >= 1


@pytest.mark.parametrize(
    "conflicting_kwargs",
    [
        pytest.param({"mode": "target", "target_projection": None}, id="non-default-mode"),
        pytest.param({"target_projection": np.array([1.0])}, id="target_projection"),
        pytest.param({"escape_metric": "cv0"}, id="escape_metric"),
        pytest.param({"periodic": [360.0]}, id="periodic"),
    ],
)
def test_progress_with_conflicting_builtin_args_raises(conflicting_kwargs):
    """`progress` is supposed to entirely replace the built-in escape/target
    metric; a caller who also passes a non-default mode/target_projection/
    escape_metric/periodic (e.g. leftover kwargs from migrating off the
    built-in metric) gets a loud ValueError instead of having that argument
    silently discarded -- this used to bypass validation instead (I3)."""
    sim, n = _build_simulation()
    progress = RecordingProgress()
    with pytest.raises(ValueError):
        PathGennieMD(
            simulation=sim,
            projection_fn=lambda c: np.array([c[0, 0]]),
            convergence_fn=lambda c, **k: False,
            temperature=300.0, seed=1,
            progress=progress,
            **conflicting_kwargs,
        )


def test_mode_still_validated_without_progress():
    sim, n = _build_simulation()
    with pytest.raises(ValueError):
        PathGennieMD(
            simulation=sim,
            projection_fn=lambda c: np.array([c[0, 0]]),
            mode="target",
            target_projection=None,
            convergence_fn=lambda c, **k: False,
        )
