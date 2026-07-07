# Architecture

Before v0.2.0 the adaptive-sampling cycle was copy-pasted into each of the three
backends (`pg_omm.py`, `pg_amber.py`, `pg_gmx.py`). Multi-GPU support, a new CV,
or any algorithm change had to be written three times and could drift. v0.2.0
extracts the cycle into `pathgennie/core/` behind small protocols, so each
backend is now just an `Engine` implementation plus a `run()` config loader.

## The cycle (`core/driver.py`)

`PathGennieDriver.run()` implements exactly one loop:

```python
for cycle in range(max_cycle):
    trials  = executor.map(worker, range(max_trial))   # N samplers, τ1, fresh velocities
    metrics = [progress.metric(progress.project(t.coords)) for t in trials]
    chosen  = trials[softmax_select(metrics, sigma, rng)]
    runner  = engine.run_segment(chosen.handle, tau2, randomize_velocities=False, ...)
    anchor  = pick(runner, chosen, anchor)             # optional reject-worse rules
    if convergence_fn(anchor_coords): break
```

Key properties baked into the single implementation:

- **Reproducible.** One master `seed` seeds a `numpy.random.Generator` that
  produces both the per-segment seeds and the selection draw.
- **Bounded scratch.** Trial handles are released each cycle via
  `engine.release`, so disk/memory does not grow without bound.
- **Always saves the converged frame**, even when it does not fall on a
  `save_freq` boundary.
- **Adaptive-CV hook.** If the progress object defines `observe(coords, cycle)`,
  the driver calls it once per cycle (used by SPIB to retrain on the fly).
- **Restartable seeds.** `run(..., collect_seeds=True)` additionally returns a
  list of cloned anchor handles aligned with the saved frames — the inputs to a
  downstream sampling stage.

Constructor:

```python
PathGennieDriver(engine, progress, convergence_fn, *,
                 executor=SerialExecutor(), sigma=0.1, seed=None,
                 reject_worse_tau2=False, reject_worse_anchor=False, verbosity=1)
```

## The `Engine` protocol (`core/engine.py`)

A backend implements four methods; the driver never inspects a *handle* (an
opaque token — a restart-file path for the subprocess backends, a state-cache id
for the in-process ones):

```python
class Engine(Protocol):
    def clone_anchor(self, handle) -> handle: ...
    def run_segment(self, handle, n_steps, *, randomize_velocities, seed, device=None) -> handle: ...
    def get_coords(self, handle) -> np.ndarray:   # (n_atoms, 3) Angstrom
        ...
    def release(self, handle) -> None: ...
```

`randomize_velocities=True` marks a *sampler* (τ1, fresh Maxwell–Boltzmann
velocities); `False` marks a *runner* (τ2, continued velocities). `device` is the
GPU index assigned by the executor.

Implementations: `CoreAmberEngine`, `CoreGromacsEngine`, `OpenMMEngine`, and the
pure-NumPy `ToyLangevinEngine` (which additionally offers `create_state(coords)`).

## Progress variables (`core/progress.py`)

A `ProgressVariable` couples a CV projection with a scalar metric where **higher
is better**:

```python
class ProgressVariable(Protocol):
    def project(self, coords) -> np.ndarray: ...   # CV vector
    def metric(self, cv) -> float: ...             # progress score
```

Built-ins wrap the original backend behaviour verbatim:

- `EscapeMetric(projection_fn, start_cv, escape_metric="distance_from_start"|"cv0")`
  — maximise distance from the start (or just the first CV component).
- `TargetMetric(projection_fn, target_cv)` — minimise distance to a target
  (returned negated).

`CallableProjection` adapts a plain `projection_fn(coords, **kwargs)`.
`cv.spib.SPIBProgress` is a learned, adaptive implementation (see
[data-driven-cv.md](data-driven-cv.md)).

## Selection (`core/selection.py`)

`selection_probs(metrics, sigma)` min-max scales the metrics to `[0, 1]` and
forms weights `exp((scaled − 1)/sigma)` (shifting the max to 0 prevents
overflow). Small `sigma` → greedy (argmax); large `sigma` → uniform. An
all-equal batch returns a uniform distribution. `softmax_select(metrics, sigma,
rng)` draws one index using an explicit `Generator` for reproducibility.

## Parallel execution (`core/parallel.py`)

The driver evaluates the `N` samplers through a `ParallelExecutor`:

```python
class ParallelExecutor(Protocol):
    devices: list[int | None]
    def map(self, worker_fn, items) -> list: ...   # worker_fn(item, device)
```

- `SerialExecutor(device=None)` — the reference path; `map` runs items in order.
- `ThreadDevicePool(devices=[0,1,2,3], workers_per_device=1)` — assigns item `i`
  to `devices[i % len(devices)]` and runs them in a thread pool. Because the
  subprocess backends spend their time inside `subprocess.run` (GIL released) and
  each segment exports `CUDA_VISIBLE_DEVICES`, threads genuinely overlap across
  GPUs. Output order matches input order, and `G=1` reproduces `SerialExecutor`.

See [multi-gpu.md](multi-gpu.md) for usage and benchmarks.

## How a backend wires it together

Each backend `run(case_dir)`:

1. loads `input.yaml` and applies the run profile (`strategy.resolve_profile`);
2. builds its `Engine` (topology, executable, scratch dir, temperature, …);
3. builds a `ProgressVariable` from the case's `projection`/`convergence`
   modules and the `mode` (`escape`/`target`);
4. builds a `ThreadDevicePool` from `devices`/`workers_per_device`;
5. constructs a `PathGennieDriver` and calls `run()`;
6. writes the trajectory + `metrics.csv`.

The OpenMM backend wraps these steps in the `PathGennieMD` adapter; AMBER and
GROMACS call the driver directly.
