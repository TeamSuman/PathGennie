# Non-linear path search (RRT / RRT-Connect)

The greedy driver follows a monotone progress metric, so it cannot backtrack,
change direction, or move orthogonally to the CV — the failure mode the paper
shows on Wolfe–Quapp. `pathgennie/search/rrt.py` reframes each expansion as
growing a **Rapidly-exploring Random Tree** in CV space.

## Algorithm

Each iteration:

1. sample a random CV target `q_rand` (occasionally the goal — *goal biasing*);
2. find the nearest existing tree node `q_near` (by CV distance);
3. run a PathGennie swarm from `q_near` and select the sampler whose CV moves
   closest to `q_rand` — the existing `softmax_select` with metric
   `-||cv - q_rand||`;
4. extend the chosen sampler with a `tau2` runner and add it as a new node.

Because targets point in any CV direction and the tree remembers every node, the
search backtracks and explores naturally. **RRT-Connect** grows two trees — from
the start and from a goal configuration — and links them, crossing barriers much
faster. Everything reuses the shared `Engine` and `ParallelExecutor`, so RRT is
multi-GPU and backend-agnostic.

## RRT

```python
from pathgennie.core.toy import ToyLangevinEngine
from pathgennie.core.parallel import SerialExecutor
from pathgennie.search.rrt import RRT
import numpy as np

engine = ToyLangevinEngine(dt=0.005, kT=1.0)
start = engine.create_state((-1.174, 1.477))           # basin A
rrt = RRT(engine, lambda c: np.array([c[0, 0], c[0, 1]]),
          lower=[-2, -2], upper=[2, 2],
          tau1=5, tau2=10, n_expand=8, sigma=0.05, goal_bias=0.2,
          executor=SerialExecutor(), seed=0)
result = rrt.build(start, target_cv=[1.124, -1.485], max_iter=300, goal_tol=0.5)

result.success          # reached the target basin?
result.path             # list[Node] root -> goal, each with .cv, .handle, .parent
result.tree_size
```

With `target_cv=None`, `build` grows an exploratory tree for `max_iter`
expansions (blind exploration).

## RRT-Connect

```python
from pathgennie.search.rrt import rrt_connect
goal = engine.create_state((1.124, -1.485))            # basin B
result = rrt_connect(engine, lambda c: np.array([c[0, 0], c[0, 1]]),
                     start, goal, lower=[-2, -2], upper=[2, 2],
                     tau1=5, tau2=10, n_expand=8, executor=SerialExecutor(),
                     seed=2, max_iter=300, connect_tol=0.6)
result.path             # joined start -> goal path
```

## Notes

- CV space is whatever `cv_fn(coords) -> vector` you pass — a geometric CV or the
  learned SPIB latent. `lower`/`upper` bound the random-target sampling box.
- Tree nodes keep their engine handles alive (they *are* the backtracking memory);
  only discarded swarm trials are released each expansion.
- The `ExplorerPolicy` protocol in `pathgennie/core/policy.py` is the shared
  abstraction (`GreedyPolicy` reproduces the driver); RRT/RRT-Connect are the
  non-linear policies.
