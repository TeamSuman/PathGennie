# Tutorial 07 — Metastable states, roadmap & the agent

This ties three pieces together: metastable-state labels (from SPIB or any
clustering) → a roadmap graph of all-pairs pathways → an agentic controller that
adapts the swarm. All runnable without MD.

## Build a roadmap from a state-label trajectory

`SPIBProgress.state_labels` gives a per-frame metastable-state id; feed that
sequence to a `Roadmap`. Here we use a synthetic label trajectory:

```python
from pathgennie.search.roadmap import Roadmap

rm = Roadmap()
# e.g. progress.state_labels from a SPIB run; here a synthetic 3-state walk:
labels = [0, 0, 1, 1, 2, 2, 1, 0, 1, 2, 0, 2]
rm.observe_sequence(labels)

print("states:", rm.nodes)
cost, path = rm.min_free_energy_path(0, 2)          # Dijkstra
print("min-free-energy path 0->2:", path, "cost", round(cost, 3))

for cost, route in rm.k_shortest_paths(0, 2, k=2):   # competing routes (Yen)
    print("route", route, "cost", round(cost, 3))

report = rm.all_pairs_paths(k=1)                      # every ordered pair
print("pairs found:", sorted(report))
```

Edges are weighted by `-log(transition fraction)`, so the cheapest path is the
maximum-likelihood / minimum-free-energy route.

## Drive the swarm with the agentic controller

```python
from pathgennie.agent import RuleBasedController, SwarmParams

ctrl = RuleBasedController(SwarmParams(n_trial=8, tau1=4, tau2=8),
                           stall_window=5, stall_eps=1e-3,
                           escalate=1.5, relax=0.75, stop_patience=20, refresh_every=50)

metric_history = []
for cycle in range(200):
    p = ctrl.update(metric_history)        # adapt N / tau1 / tau2 from progress
    # ... run one driver cycle with p.n_trial, p.tau1, p.tau2 ...
    metric_history.append(get_latest_metric())
    if ctrl.should_refresh_cv(cycle):
        ...                                # retrain SPIB
    if ctrl.should_stop(metric_history):
        break
```

When progress stalls the controller enlarges the swarm and lengthens the
segments; when it flows it relaxes the swarm to save compute; on a long plateau it
recommends stopping. `RuleBasedController.choose_frontier(visit_counts)` picks the
least-visited region to expand next (anti-trapping for RRT).

See [roadmap-graph.md](../roadmap-graph.md) and [agent.md](../agent.md).
