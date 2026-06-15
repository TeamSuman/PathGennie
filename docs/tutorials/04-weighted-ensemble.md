# Tutorial 04 — Free energies & rates with Weighted Ensemble

A discovered path tells you *a* route; Weighted Ensemble (WE) turns it into
quantitative numbers — a free-energy profile and, with recycling, a rate
constant. WE runs unbiased MD and resamples weighted walkers, so it reuses the
same engine and device pool.

## Discover, then run WE

```python
import numpy as np
from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import SerialExecutor
from pathgennie.core.progress import EscapeMetric
from pathgennie.core.toy import ToyLangevinEngine
from pathgennie.sampling import WeightedEnsembleStage, build_path_ensemble

kT = 2.0
engine = ToyLangevinEngine(dt=0.005, kT=kT)
initial = engine.create_state((-1.0, -1.4))

# 1) Discover a path along y, KEEPING restartable seeds (collect_seeds=True).
y = lambda c: np.array([c[0, 1]])
progress = EscapeMetric(y, start_cv=np.array([-1.4]), escape_metric="cv0")
driver = PathGennieDriver(engine, progress, lambda c: False,
                          executor=SerialExecutor(), sigma=0.3, seed=0, verbosity=0)
traj, metrics, handles = driver.run(initial, tau1=5, tau2=10, max_trial=6,
                                    max_cycle=60, save_freq=1, collect_seeds=True)
ens = build_path_ensemble(traj, metrics, handles=handles, cv_fn=lambda c: c[0, 1])

# 2) Run WE along y.
stage = WeightedEnsembleStage(cv_fn=lambda c: c[0, 1], tau_steps=10,
                              n_iterations=400, n_bins=16, target_count=6, seed=1, kT=kT)
result = stage.run(ens, engine)

centers = result.metadata["bin_centers"]
for c, f in zip(centers, result.free_energy):
    print(f"y={c:+.2f}  F={f:.2f}")
print("total weight (should be 1):", result.weights.sum())
```

The free-energy minima land in the two Wolfe–Quapp basins (`y ≈ ±1.4`), above the
central barrier — compare against the analytic marginal with
`python benchmarks/we_fes.py`.

## Rate constants via recycling

```python
stage = WeightedEnsembleStage(
    cv_fn=lambda c: c[0, 1], tau_steps=6, n_iterations=400, n_bins=16,
    recycle=True, source_cv=-1.4, target_cv=1.2, timestep_ps=0.002, kT=kT,
)
result = stage.run(ens, engine)
print(result.rate_constants)   # {"flux_per_iter": ..., "rate": ...}
```

## From a real backend

Set `pathgennie.downstream: weighted_ensemble` plus a `weighted_ensemble:` block
in `input.yaml`; the backend builds the `PathEnsemble` and runs WE automatically,
writing `free_energy.csv` (+ `rate_constants.json`) — see
[weighted-ensemble.md](../weighted-ensemble.md).
