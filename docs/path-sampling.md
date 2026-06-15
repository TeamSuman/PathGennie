# Path sampling (TPS / TIS) via OpenPathSampling

Transition Path Sampling (TPS) and Transition Interface Sampling (TIS) are an
*alternative to Weighted Ensemble* for computing kinetics. Both need an initial
reactive A→B trajectory to bootstrap — exactly what a PathGennie run produces.
`pathgennie/sampling/path_sampling.py` bridges a discovered `PathEnsemble` to
[OpenPathSampling](https://openpathsampling.org/) (OPS).

Install the optional dependency: `pip install -e .[pathsampling]` (i.e.
`openpathsampling`).

## Two layers (same pattern as the OPES module)

**1. Dependency-free seed preparation** (always available, fully tested) — turn a
discovered path into an OPS-ready seed:

```python
import numpy as np
from pathgennie.sampling import build_path_ensemble
from pathgennie.sampling.path_sampling import (
    CVRangeState, label_frames, extract_transition_path, tis_interfaces, prepare_ops_seed,
)

cv = lambda c: c[0, 1]                      # scalar progress coordinate
A = CVRangeState("A", -2.0, -1.0)          # states as CV intervals
B = CVRangeState("B",  1.0,  2.0)

ens = build_path_ensemble(frames, metrics)  # from driver.run(...)
span = extract_transition_path(ens.frames, cv, A, B)   # (start, end) of the A->B sub-path
seed = prepare_ops_seed(ens, cv, A, B, interfaces=tis_interfaces(-1.0, 1.0, 6))
seed["reactive"], seed["seed_frames"].shape, seed["interfaces"]
```

- `CVRangeState` — a state as a closed CV interval.
- `label_frames` / `extract_transition_path` / `is_reactive` — label frames and
  slice the minimal reactive sub-path (from the last A frame before the first
  subsequent B frame).
- `tis_interfaces(lambda0, lambdaN, n, spacing="linear"|"exp")` — lay out TIS
  interfaces (exp packs them where crossing probabilities change fastest).
- `prepare_ops_seed` — bundles all of the above into the dict OPS needs.

**2. OPS-dependent stage** — `PathSamplingStage` (implements `SamplingStage`):

```python
import openpathsampling as paths
from openpathsampling.engines.openmm import Engine as OPSEngine   # OPS's own engine
from pathgennie.sampling import make_stage

ops_engine = OPSEngine(...)                 # build per OPS docs (topology, system, integrator)

# TPS:
stage = make_stage("tps", cv_fn=cv, state_a=(-2, -1), state_b=(1, 2),
                   ops_engine=ops_engine, n_steps=2000, storage_path="tps.nc")
result = stage.run(ens, engine=None)        # OPS propagates with ops_engine

# TIS (rate constants):
stage = make_stage("tis", cv_fn=cv, state_a=(-2, -1), state_b=(1, 2),
                   interfaces=list(tis_interfaces(-1.0, 1.0, 6)),
                   ops_engine=ops_engine, n_steps=5000)
result = stage.run(ens, engine=None)
result.rate_constants    # {"rate_matrix": ...} from OPS TISAnalysis
```

`PathSamplingStage` builds the OPS CV (`CoordinateFunctionCV`), state volumes
(`CVDefinedVolume`), the network (`TPSNetwork` / `MISTISNetwork`), a one-way
shooting move scheme, seeds it with the extracted reactive sub-path, runs
`PathSampling`, and returns kinetics in a `SamplingResult`.

## Important: OPS uses its own engine

OPS propagates shooting moves with **its own** engine (e.g.
`openpathsampling.engines.openmm`), not PathGennie's swarm `Engine`. So you pass
`ops_engine=`; the PathGennie engine argument to `run` is unused. Without
`openpathsampling` installed, or without an `ops_engine`, the stage raises an
informative error. The bridge targets the OpenPathSampling 1.x API and cannot be
executed in a CPU-only sandbox, so only the dependency-free preparation is
covered by CI.

## When to use TPS/TIS vs Weighted Ensemble

| | Weighted Ensemble | TPS / TIS (OPS) |
|---|---|---|
| What it gives | FES + steady-state rate | transition path ensemble; rate (TIS) |
| Bias | none (resampling weights) | none (shooting moves) |
| Best when | broad CV coverage / FES wanted | mechanism + rate of a known A→B event |
| PathGennie role | informed seeds + bins | informed initial reactive path |

Both consume the same `PathEnsemble`, so you can run either (or both) on one
discovery run.
