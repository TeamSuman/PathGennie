# Roadmap graph & all-pairs pathways

RRT and the swarm discover *transitions*; the roadmap turns the accumulated
transitions into a global weighted graph and extracts pathways between metastable
states. Nodes are states (e.g. SPIB metastable-state labels), and a directed edge
`u → v` is weighted by `-log(transition fraction)` — a free-energy-like cost, so
the *minimum-cost* path is the *maximum-likelihood / minimum-free-energy* route.

`pathgennie/search/roadmap.py` is pure standard library (no graph dependency).

## Building a roadmap

```python
from pathgennie.search.roadmap import Roadmap

rm = Roadmap()
# Feed a state-label trajectory (e.g. SPIBProgress.state_labels), or add edges:
rm.observe_sequence([0, 0, 1, 1, 2, 1, 0, 1, 2])
rm.add_transition(0, 2, count=1.0)

rm.nodes                       # discovered states
adj = rm.adjacency()           # {u: {v: -log p(u->v)}}
```

## Extracting pathways

```python
cost, path = rm.min_free_energy_path(0, 2)     # Dijkstra: single best route
routes = rm.k_shortest_paths(0, 2, k=3)        # Yen: competing parallel routes
report = rm.all_pairs_paths(k=1)               # {(s, t): [(cost, path), ...]}
```

- `dijkstra_path(adj, s, t)` — minimum-cost path; `(inf, [])` if unreachable.
- `k_shortest_paths(adj, s, t, k)` — Yen's algorithm: up to `k` loopless paths in
  increasing cost, formalising the paper's competing egress-route clustering.

## Where the states come from

Pair this with [SPIB](data-driven-cv.md): after a refresh, `SPIBProgress`
provides `state_labels` (per-frame metastable-state ids). Feeding that label
trajectory to `observe_sequence` builds a roadmap whose nodes are *learned*
metastable states — no separate clustering step. The per-edge `-log(fraction)`
weights are also ready to seed Weighted Ensemble or an MSM.
