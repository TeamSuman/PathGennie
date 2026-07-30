# Weighted Ensemble (free energies & rates)

Weighted Ensemble (WE) keeps a population of trajectories ("walkers"), each
carrying a statistical *weight*, propagates them with **unbiased** dynamics, and
periodically *resamples* (splits over-weight walkers, merges under-weight ones)
so walkers stay spread across bins of a progress coordinate. No bias force is ever
applied — the weights keep the ensemble unbiased — so WE reuses PathGennie's exact
`Engine` and `ParallelExecutor` (multi-GPU for free).

PathGennie's stage is **path-informed**: it seeds walkers from a discovered
`PathEnsemble` and bins along that path's CV range, so WE does not have to first
find the transition. With recycling it yields a steady-state flux (rate constant);
the time-averaged bin weights give a free-energy profile along the CV.

## Core pieces (`pathgennie/sampling/weighted_ensemble.py`)

- `Walker(handle, weight, bin, cv)` — a weighted trajectory.
- `GridBinner` — uniform 1-D bins; `GridBinner.from_values(values, n_bins)` builds
  edges spanning the path CV range.
- `resample(walkers, target_count, rng, clone_fn, release_fn)` — Huber–Kim
  split/merge within one bin, conserving total weight exactly.
- `WeightedEnsembleStage` — the `SamplingStage` implementation.

## Using it from Python

```python
from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.toy import ToyLangevinEngine
from pathgennie.core.progress import EscapeMetric
from pathgennie.core.parallel import SerialExecutor
from pathgennie.sampling import WeightedEnsembleStage, build_path_ensemble
import numpy as np

engine = ToyLangevinEngine(dt=0.005, kT=2.0)
initial = engine.create_state((-1.0, -1.4))

# 1) discover a path, retaining restartable seeds
progress = EscapeMetric(lambda c: np.array([c[0, 1]]), start_cv=np.array([-1.4]), escape_metric="cv0")
driver = PathGennieDriver(engine, progress, lambda c: False,
                          executor=SerialExecutor(), sigma=0.3, seed=0, verbosity=0)
traj, metrics, handles = driver.run(initial, tau1=5, tau2=10, max_trial=6,
                                    max_cycle=60, save_freq=1, collect_seeds=True)
ens = build_path_ensemble(traj, metrics, handles=handles, cv_fn=lambda c: c[0, 1])

# 2) run WE along the y coordinate
stage = WeightedEnsembleStage(cv_fn=lambda c: c[0, 1], tau_steps=10,
                              n_iterations=400, n_bins=16, target_count=6, seed=1,
                              kT=2.0, burn_in=0.3)
result = stage.run(ens, engine)

result.free_energy            # F over result.metadata["bin_centers"]
result.weights                # final walker weights (sum to 1)
result.metadata["weight_trace"]      # total weight per iteration (== 1)
result.metadata["bin_weight_trace"]  # (n_iterations, n_bins) occupancy
```

### Discard the transient — `burn_in`

WE starts from whatever seeded it. A common and sensible choice is one walker per
bin, but that is a *uniform* distribution: maximally unlike the Boltzmann
distribution being estimated. Averaging bin occupancy from iteration 0 therefore
mixes the relaxation away from that artificial start into the answer, **flattening
the profile and biasing barriers low**.

`burn_in` discards leading iterations — an `int` counts them, a `float` in
`(0, 1)` is a fraction of the run. It defaults to `0` for backwards
compatibility, but 0 is rarely the right choice.

Do not guess the value: `metadata["bin_weight_trace"]` records per-iteration,
per-bin occupancy, so you can re-estimate over later and later windows and take
the point beyond which the profile stops moving — without re-running anything.

```python
trace, kT = result.metadata["bin_weight_trace"], 2.0

def profile(window):
    p = window.sum(axis=0); p = p / p.sum()
    with np.errstate(divide="ignore"):
        f = -kT * np.log(p)
    return f - f[np.isfinite(f)].min()

second_half   = profile(trace[len(trace) // 2:])
final_quarter = profile(trace[3 * len(trace) // 4:])
# agreement between these is evidence of convergence; disagreement means
# the run is too short, whatever the nominal barrier says
```

`metadata["flux_trace"]` is the per-iteration flux, and rate constants use the
same burn-in — a rate is a steady-state quantity, so the transient contaminates it
in exactly the same way.

`examples/qmmm_reactive_sn2/amber/3_free_energy.py` implements this check and
refuses to endorse a barrier when the windows disagree.

### Rate constants (recycling)

Set `recycle=True` with a `source_cv` and `target_cv`; walkers crossing the
target add their weight to the flux and are re-injected at the source:

```python
stage = WeightedEnsembleStage(
    cv_fn=lambda c: c[0, 1], tau_steps=6, n_iterations=400, n_bins=16,
    recycle=True, source_cv=-1.4, target_cv=1.2, timestep_ps=0.002, kT=2.0,
)
result = stage.run(ens, engine)
result.rate_constants   # {"flux_per_iter": ..., "rate": ...}
```

`rate` uses `tau_steps * timestep_ps` when `timestep_ps` is given, else
per-iteration units.

## Multi-GPU

Pass a `ThreadDevicePool` as `executor=` to spread the per-iteration walker
segments across GPUs — exactly as for discovery.

## From `input.yaml` (backends)

When `pathgennie.downstream: weighted_ensemble` is set, the backend `run()` builds
the `PathEnsemble` (via `collect_seeds`) and runs WE automatically, writing
`free_energy.csv` (and `rate_constants.json` if recycling) into the output
directory:

```yaml
pathgennie:
  downstream: weighted_ensemble
  # ... discovery settings ...
weighted_ensemble:
  cv_component: 0        # which projection component to bin along
  tau_steps: 1000
  n_iterations: 200
  n_bins: 20
  target_count: 5
  recycle: false
```

> The real-MD backend path is wired through the same `run_downstream` helper that
> the toy tests cover, but cannot be executed in a CPU-only sandbox. See
> [roadmap.md](roadmap.md).

## Validation

`benchmarks/we_fes.py` compares the WE profile `F(y)` to the analytic Wolfe–Quapp
marginal:

```bash
python benchmarks/we_fes.py
```
