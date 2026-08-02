# Tutorial 01 — Quickstart on the toy engine

> **Runnable version:** this tutorial's case ships as
> [`examples/toy_langevin/`](https://github.com/TeamSuman/PathGennie/tree/main/examples/toy_langevin).
> `cd examples/toy_langevin && python run_toy.py` — measured 6.9 s, converges at cycle 4,
> and needs no MD engine.


This runs the *entire* PathGennie driver — swarm → selection → runner →
convergence — with no MD binary or GPU, using the built-in Wolfe–Quapp toy
engine. It's the fastest way to see the method work and the foundation every
other tutorial builds on.

## Run it

```python
import numpy as np
from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import SerialExecutor
from pathgennie.core.progress import TargetMetric
from pathgennie.core.toy import ToyLangevinEngine

# 1) An engine (2-D Langevin on the Wolfe-Quapp surface).
engine = ToyLangevinEngine(dt=0.005, kT=1.0)
initial = engine.create_state((-1.174, 1.477))     # basin A

# 2) A progress variable: aim at basin B in (x, y) CV space.
xy = lambda coords: np.array([coords[0, 0], coords[0, 1]])
progress = TargetMetric(xy, target_cv=np.array([1.124, -1.485]))

# 3) Stop when we're close to the target basin.
def converged(coords):
    return np.linalg.norm(xy(coords) - np.array([1.124, -1.485])) < 0.4

# 4) Drive it.
driver = PathGennieDriver(engine, progress, converged,
                          executor=SerialExecutor(), sigma=0.2, seed=0)
traj, metrics = driver.run(initial, tau1=5, tau2=10, max_trial=8,
                           max_cycle=400, save_freq=5)

print("saved frames:", traj.shape[0])
print("final CV:", xy(traj[-1]))
```

You'll see the metric improve cycle by cycle and the run converge once it reaches
basin B.

## What just happened

- `max_trial=8` samplers each ran a `tau1=5`-step segment with fresh velocities.
- Each was scored by `progress`, and one was `softmax`-selected (`sigma=0.2`).
- The winner was extended for a `tau2=10`-step runner and became the new anchor.
- `seed=0` makes the whole run reproducible.

## Next

- Spread the swarm across GPUs → [02 — Multi-GPU](02-multi-gpu.md).
- Replace the hand-written CV with a learned one → [03 — SPIB](03-spib-cv.md).
- Turn the path into a free energy → [04 — Weighted Ensemble](04-weighted-ensemble.md).
