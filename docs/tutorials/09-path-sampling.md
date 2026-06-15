# Tutorial 09 — Kinetics with TPS/TIS (OpenPathSampling)

PathGennie's discovered path is an ideal *seed* for Transition Path Sampling
(TPS) and Transition Interface Sampling (TIS) — an alternative to Weighted
Ensemble for kinetics. This tutorial shows the dependency-free seed preparation
(runnable now) and the OpenPathSampling (OPS) run (needs `pip install -e
.[pathsampling]` and an OPS engine).

## 1. Discover a path and prepare the OPS seed (runnable now)

```python
import numpy as np
from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import SerialExecutor
from pathgennie.core.progress import EscapeMetric
from pathgennie.core.toy import ToyLangevinEngine
from pathgennie.sampling import build_path_ensemble
from pathgennie.sampling.path_sampling import (
    CVRangeState, extract_transition_path, tis_interfaces, prepare_ops_seed,
)

engine = ToyLangevinEngine(dt=0.005, kT=1.0)
initial = engine.create_state((-1.0, -1.4))
y = lambda c: np.array([c[0, 1]])
progress = EscapeMetric(y, start_cv=np.array([-1.4]), escape_metric="cv0")
driver = PathGennieDriver(engine, progress, lambda c: False,
                          executor=SerialExecutor(), sigma=0.3, seed=0, verbosity=0)
traj, metrics = driver.run(initial, tau1=5, tau2=10, max_trial=6, max_cycle=80, save_freq=1)
ens = build_path_ensemble(traj, metrics)

cv = lambda c: c[0, 1]
A = CVRangeState("A", -2.0, -1.0)
B = CVRangeState("B",  1.0,  2.0)

print("reactive A->B sub-path:", extract_transition_path(ens.frames, cv, A, B))
seed = prepare_ops_seed(ens, cv, A, B, interfaces=tis_interfaces(-1.0, 1.0, 6))
print("reactive:", seed["reactive"], " seed frames:", None if seed["seed_frames"] is None else seed["seed_frames"].shape)
print("TIS interfaces:", np.round(seed["interfaces"], 2))
```

(If the run didn't reach basin B, increase `max_cycle` or widen the state ranges.)

## 2. Run TPS / TIS with OpenPathSampling

```python
import openpathsampling as paths
from openpathsampling.engines.openmm import Engine as OPSEngine
from pathgennie.sampling import make_stage

ops_engine = OPSEngine(...)        # build per OPS docs (topology/system/integrator)

# TPS — sample the transition path ensemble
tps = make_stage("tps", cv_fn=cv, state_a=(-2, -1), state_b=(1, 2),
                 ops_engine=ops_engine, n_steps=2000, storage_path="tps.nc")
tps.run(ens, engine=None)

# TIS — rate constant
tis = make_stage("tis", cv_fn=cv, state_a=(-2, -1), state_b=(1, 2),
                 interfaces=list(tis_interfaces(-1.0, 1.0, 6)),
                 ops_engine=ops_engine, n_steps=5000)
result = tis.run(ens, engine=None)
print(result.rate_constants)
```

OPS propagates with **its own** engine (`ops_engine`), not PathGennie's swarm
engine. Without `openpathsampling` installed the stage raises an informative
`ImportError`.

See [path-sampling.md](../path-sampling.md) for the full reference and the
WE-vs-TPS/TIS comparison.
