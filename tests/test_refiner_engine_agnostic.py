"""``PathRefiner`` must not be welded to OpenMM.

The refinement mathematics -- ``PrincipalCurve``, the NN consensus refiner and ``PathCV`` --
is pure NumPy/torch and engine-independent, but ``refiner.py`` used to import
``pathgennie.backends.openmm`` at module scope and build a ``PathGennieMD`` walker inside the
trajectory worker. That made the whole iterative protocol unavailable to the AMBER and GROMACS
backends, and therefore to QM/MM reactive chemistry, where OpenMM is not an option.

These tests pin the two properties that fix:

1. ``pathrefinement.refiner`` imports with no OpenMM installed.
2. An injected ``sampler`` is used instead of the OpenMM walker, so any engine can drive the
   exploration step.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_refiner_imports_without_openmm(monkeypatch):
    """Module import must not require OpenMM *or* torch.

    torch arrives via ``ensemblerefiner`` and is an optional extra (``[ml]``), so
    CI does not install it. Blocking it here is what keeps the module-scope import
    from creeping back: the guarantee is that constructing a PathRefiner and
    driving it with an injected sampler needs neither heavy dependency.
    """
    import builtins
    import importlib
    import sys

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if (name.startswith("openmm") or name.startswith("pathgennie.backends.openmm")
                or name == "torch" or name.startswith("torch.")):
            raise ImportError(f"blocked for test: {name}")
        return real_import(name, *args, **kwargs)

    for mod in [m for m in sys.modules if "pathrefinement.refiner" in m]:
        del sys.modules[mod]
    monkeypatch.setattr(builtins, "__import__", blocked)

    for mod in [m for m in sys.modules if m.startswith("pathrefinement")]:
        del sys.modules[mod]
    module = importlib.import_module("pathrefinement.refiner")
    assert hasattr(module, "PathRefiner")
    # And it must be usable, not merely importable.
    cfg = module.PathRefinementConfig()
    cfg.n_trajectories = 2
    r = module.PathRefiner(potential=None, config=cfg,
                           sampler=lambda pcv, sp, seed: np.linspace([0.0, 0.0], [1.0, 1.0], 4))
    assert r.simulation is None
    assert len(r._collect_trajectories(None, np.zeros(2), 0)) == 2


def test_injected_sampler_replaces_the_openmm_walker():
    """A sampler is called for each walker, and OpenMM is never touched."""
    from pathrefinement.refiner import PathRefinementConfig, PathRefiner

    calls = []

    def fake_sampler(path_cv, start_pt, seed):
        calls.append(seed)
        # A trajectory in feature space: a short straight run.
        return np.linspace([0.0, 0.0], [1.0, 1.0], 8)

    cfg = PathRefinementConfig()
    cfg.n_trajectories = 4
    cfg.n_workers = 4

    # potential=None is safe precisely because the sampler short-circuits the
    # OpenMM branch -- if it did not, constructing PathRefiner would fail here.
    refiner = PathRefiner(potential=None, config=cfg, sampler=fake_sampler)
    assert refiner.simulation is None, "no OpenMM Simulation should be built when a sampler is given"

    trajs = refiner._collect_trajectories(path_cv=None, start_pt=np.zeros(2), it_seed=100)

    assert len(calls) == 4, f"sampler should run once per walker, got {len(calls)}"
    assert len(set(calls)) == 4, "each walker must get a distinct seed"
    assert len(trajs) == 4
    assert all(t.shape == (8, 2) for t in trajs)


def test_sampler_failures_are_dropped_not_fatal():
    """A walker that produces nothing is skipped rather than crashing the iteration."""
    from pathrefinement.refiner import PathRefinementConfig, PathRefiner

    def flaky_sampler(path_cv, start_pt, seed):
        if seed % 2 == 0:
            return None                       # walker failed
        return np.linspace([0.0, 0.0], [1.0, 1.0], 5)

    cfg = PathRefinementConfig()
    cfg.n_trajectories = 6
    cfg.n_workers = 1

    refiner = PathRefiner(potential=None, config=cfg, sampler=flaky_sampler)
    trajs = refiner._collect_trajectories(path_cv=None, start_pt=np.zeros(2), it_seed=0)

    assert 0 < len(trajs) < 6, "failed walkers should be dropped, successful ones kept"
    assert all(t is not None and len(t) > 0 for t in trajs)


def test_refinement_maths_is_engine_independent():
    """PrincipalCurve and PathCV work on plain arrays with no engine at all."""
    from pathrefinement import PathCV, PrincipalCurve

    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 60)
    noisy = np.stack([t, 0.4 * np.sin(np.pi * t)], axis=1) + rng.normal(0, 0.02, (60, 2))

    curve = PrincipalCurve(n_images=12)
    fitted = curve.fit(noisy)
    assert fitted.shape[1] == 2 and len(fitted) == 12

    # NOTE: PathCV.normalize_output defaults to False, so `s` is Branduardi's raw
    # index-weighted average in [1, P] -- not the [0, 1] progress coordinate the
    # manuscripts define. Both conventions are checked here so the default cannot
    # silently change.
    cv_raw = PathCV.from_2d_path(fitted)
    s_raw, z = cv_raw.compute(np.atleast_2d(noisy[0]))
    assert np.isfinite(s_raw) and np.isfinite(z)
    assert 1.0 - 1e-6 <= float(s_raw) <= 12.0 + 1e-6, "raw s must lie in [1, P]"

    cv_norm = PathCV.from_2d_path(fitted, normalize_output=True)
    s_norm, _ = cv_norm.compute(np.atleast_2d(noisy[0]))
    assert 0.0 - 1e-6 <= float(s_norm) <= 1.0 + 1e-6, "normalised s must lie in [0, 1]"
    assert abs(float(s_norm) - (float(s_raw) - 1.0) / 11.0) < 1e-9
