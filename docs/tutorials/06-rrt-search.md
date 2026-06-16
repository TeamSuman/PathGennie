# Tutorial 06 — Non-linear search with RRT-Connect

When the path must move *against* a simple progress metric (backtrack, turn, go
orthogonal), greedy selection gets stuck. RRT grows a tree in CV space toward
random targets; RRT-Connect grows two trees and links them. Runnable on the toy
engine.

## RRT toward a target basin

```python
import numpy as np
from pathgennie.core.parallel import SerialExecutor
from pathgennie.core.toy import ToyLangevinEngine
from pathgennie.search.rrt import RRT

engine = ToyLangevinEngine(dt=0.005, kT=1.0)
start = engine.create_state((-1.174, 1.477))           # basin A
xy = lambda c: np.array([c[0, 0], c[0, 1]])

rrt = RRT(engine, xy, lower=[-2, -2], upper=[2, 2],
          tau1=5, tau2=10, n_expand=8, sigma=0.05, goal_bias=0.2,
          executor=SerialExecutor(), seed=0)
result = rrt.build(start, target_cv=[1.124, -1.485], max_iter=300, goal_tol=0.5)

print("reached target basin:", result.success)
print("tree size:", result.tree_size)
print("path length:", len(result.path))
print("end CV:", result.path[-1].cv)
```

`result.path` is the list of `Node`s from root to goal; each has `.cv`, `.handle`,
and `.parent`.

## RRT-Connect (bidirectional)

```python
from pathgennie.search.rrt import rrt_connect

goal = engine.create_state((1.124, -1.485))            # basin B
result = rrt_connect(engine, xy, start, goal,
                     lower=[-2, -2], upper=[2, 2],
                     tau1=5, tau2=10, n_expand=8, sigma=0.05,
                     executor=SerialExecutor(), seed=2, max_iter=300, connect_tol=0.6)
print("linked:", result.success, " joined path length:", len(result.path))
```

## Tips

- `cv_fn` can be any CV — geometric or the learned SPIB latent. `lower`/`upper`
  bound the random-target box.
- Pass a `ThreadDevicePool` as `executor=` to run each expansion's swarm across
  GPUs.
- Increase `goal_bias` to pull harder toward the target; raise `n_expand` for a
  wider swarm per node.

See [non-linear-search.md](../non-linear-search.md) for the algorithm details.
