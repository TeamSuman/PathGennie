"""Toy-engine coverage for the downstream-stage glue used by the backends."""

import numpy as np

from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import SerialExecutor
from pathgennie.core.progress import EscapeMetric
from pathgennie.core.toy import ToyLangevinEngine
from pathgennie.sampling.runner import make_scalar_cv, run_downstream


def test_make_scalar_cv_reduces_vector():
    def proj(coords, scale=1.0):
        return np.array([coords[0, 0] * scale, coords[0, 1]])

    f0 = make_scalar_cv(proj, {"scale": 2.0}, component=0)
    f1 = make_scalar_cv(proj, {"scale": 2.0}, component=1)
    coords = np.array([[3.0, -4.0, 0.0]])
    assert f0(coords) == 6.0
    assert f1(coords) == -4.0


def test_run_downstream_weighted_ensemble_writes_outputs(tmp_path):
    engine = ToyLangevinEngine(dt=0.005, kT=2.0)
    initial = engine.create_state((-1.0, -1.4))
    scalar_cv = make_scalar_cv(lambda c: np.array([c[0, 1]]), {}, 0)
    progress = EscapeMetric(lambda c: np.array([c[0, 1]]), start_cv=np.array([-1.4]), escape_metric="cv0")
    driver = PathGennieDriver(engine, progress, lambda c: False,
                              executor=SerialExecutor(), sigma=0.3, seed=0, verbosity=0)
    traj, metrics, seeds = driver.run(initial, tau1=5, tau2=10, max_trial=5,
                                      max_cycle=20, save_freq=2, collect_seeds=True)

    result = run_downstream(
        "weighted_ensemble",
        {"tau_steps": 5, "n_iterations": 10, "n_bins": 8, "target_count": 4, "seed": 1},
        engine=engine, traj=traj, metrics=metrics, seed_handles=seeds,
        scalar_cv_fn=scalar_cv, output_dir=tmp_path,
    )
    assert result.free_energy is not None
    assert (tmp_path / "free_energy.csv").exists()
    # Header + one row per bin center.
    lines = (tmp_path / "free_energy.csv").read_text().strip().splitlines()
    assert lines[0] == "cv,free_energy"
    assert len(lines) == 1 + result.free_energy.size
