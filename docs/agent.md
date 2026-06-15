# Agentic controller

The controller automates the "how hard to throw the swarm" decisions that are
otherwise hand-tuned: the swarm size `N` and segment lengths `tau1`/`tau2`, which
under-explored frontier to expand, when to refresh a learned CV, and when to stop.
`RuleBasedController` (`pathgennie/agent/controller.py`) is a deterministic,
testable policy — the plan's "rule-based first" — implementing a `Controller`
surface a future RL or LLM meta-controller could replace.

## Behaviour

Given the recent progress history (committed metrics, higher = better):

- **stalling** (progress rate below `stall_eps`) → *escalate*: grow `N` and
  lengthen `tau1`/`tau2` by `escalate`, clamped to bounds — push harder over the
  barrier;
- **progressing** → *relax*: shrink `N` toward its minimum by `relax` — save GPU
  time (the objective is progress-per-wall-second, not raw progress);
- **plateaued** for `stop_patience` updates → `should_stop` returns True.

It also offers count-based frontier selection (expand the least-visited region —
anti-trapping) and a CV-refresh schedule.

## Usage

```python
from pathgennie.agent import RuleBasedController, SwarmParams

ctrl = RuleBasedController(
    SwarmParams(n_trial=8, tau1=4, tau2=8),
    n_bounds=(4, 64), tau1_bounds=(2, 200), tau2_bounds=(4, 400),
    stall_window=5, stall_eps=1e-3, escalate=1.5, relax=0.75,
    stop_patience=20, refresh_every=50,
)

metric_history = []
for cycle in range(max_cycle):
    params = ctrl.update(metric_history)        # adapt N, tau1, tau2
    # ... run one driver cycle with params.n_trial / params.tau1 / params.tau2 ...
    metric_history.append(latest_metric)
    if ctrl.should_refresh_cv(cycle):
        ...                                     # trigger SPIB retraining
    if ctrl.should_stop(metric_history):
        break
```

`RuleBasedController.choose_frontier(visit_counts)` returns the least-visited node
index, for steering RRT expansion away from over-sampled regions.

## Extending it

`Controller` is a `Protocol` (`update`, `should_stop`). A contextual-bandit/RL
controller (state = local CV gradient / progress; action = `(N, tau1, tau2)`;
reward = progress per wall-second) or an LLM meta-controller can implement the
same surface and drop in without changing the driver.
