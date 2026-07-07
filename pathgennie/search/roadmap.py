"""Conformational roadmap graph and all-pairs pathway extraction.

PathGennie / RRT discover *transitions*; the roadmap turns the accumulated
transitions into a global weighted graph and extracts pathways between metastable
states.  Nodes are states (e.g. SPIB metastable-state labels, Phase 3); a
directed edge ``u -> v`` is weighted by ``-log(transition fraction)``, a
free-energy-like cost so that the *minimum-cost* path is the
*maximum-likelihood / minimum-free-energy* route.

- :func:`dijkstra_path` — the single minimum-cost path between two states.
- :func:`k_shortest_paths` — Yen's algorithm for the *competing parallel*
  pathways (formalising the paper's post-hoc egress-route clustering).
- :class:`Roadmap` — accumulates transition counts (e.g. from a state-label
  trajectory) and exposes both, plus an all-pairs report.

Pure NumPy/standard-library; no external graph dependency.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Dict, Hashable, List, Optional, Tuple

__all__ = ["dijkstra_path", "k_shortest_paths", "Roadmap"]

Adjacency = Dict[Hashable, Dict[Hashable, float]]


def dijkstra_path(adj: Adjacency, source: Hashable, target: Hashable) -> Tuple[float, List[Hashable]]:
    """Minimum-cost path ``source -> target``; returns ``(cost, path)``.

    ``(inf, [])`` if unreachable. Edge weights must be non-negative.
    """

    if source == target:
        return 0.0, [source]
    dist: Dict[Hashable, float] = {source: 0.0}
    prev: Dict[Hashable, Hashable] = {}
    pq: List[Tuple[float, Hashable]] = [(0.0, source)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        for v, w in adj.get(u, {}).items():
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if target not in dist:
        return math.inf, []
    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    path.reverse()
    return dist[target], path


def k_shortest_paths(adj: Adjacency, source: Hashable, target: Hashable, k: int) -> List[Tuple[float, List[Hashable]]]:
    """Yen's algorithm: up to ``k`` shortest loopless paths, increasing cost.

    ``adj`` is treated as read-only (a working copy is used for spur searches).
    """

    cost0, path0 = dijkstra_path(adj, source, target)
    if not path0:
        return []
    accepted: List[Tuple[float, List[Hashable]]] = [(cost0, path0)]
    candidates: List[Tuple[float, List[Hashable]]] = []

    while len(accepted) < k:
        _, prev_path = accepted[-1]
        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[: i + 1]

            work: Adjacency = {u: dict(nbrs) for u, nbrs in adj.items()}
            # Remove the edge each known path takes after this root, and the
            # interior root nodes, so the spur search yields a genuinely new path.
            for _, p in accepted:
                if len(p) > i and p[: i + 1] == root_path and p[i] in work and p[i + 1] in work[p[i]]:
                    del work[p[i]][p[i + 1]]
            for node in root_path[:-1]:
                work.pop(node, None)
                for nbrs in work.values():
                    nbrs.pop(node, None)

            _, spur_path = dijkstra_path(work, spur_node, target)
            if spur_path:
                total_path = root_path[:-1] + spur_path
                total_cost = _path_cost(adj, total_path)
                pair = (total_cost, total_path)
                if pair not in candidates and pair not in accepted:
                    candidates.append(pair)

        if not candidates:
            break
        candidates.sort(key=lambda c: (c[0], [str(n) for n in c[1]]))
        accepted.append(candidates.pop(0))

    return accepted


def _path_cost(adj: Adjacency, path: List[Hashable]) -> float:
    total = 0.0
    for u, v in zip(path, path[1:]):
        if u not in adj or v not in adj[u]:
            return math.inf
        total += adj[u][v]
    return total


class Roadmap:
    """Accumulate transitions and extract pathways between states."""

    def __init__(self) -> None:
        self._counts: Dict[Tuple[Hashable, Hashable], float] = defaultdict(float)
        self._nodes: set = set()

    @property
    def nodes(self) -> List[Hashable]:
        return sorted(self._nodes, key=lambda x: (str(type(x)), x))

    def add_transition(self, u: Hashable, v: Hashable, count: float = 1.0) -> None:
        self._counts[(u, v)] += float(count)
        self._nodes.add(u)
        self._nodes.add(v)

    def observe_sequence(self, labels) -> None:
        """Accumulate transitions from a state-label trajectory (e.g. SPIB labels)."""
        labels = list(labels)
        for u, v in zip(labels, labels[1:]):
            if u != v:
                self.add_transition(u, v)

    def adjacency(self, *, self_loops: bool = False, eps: float = 1e-12) -> Adjacency:
        """Edge weights ``-log(p(u->v))`` from accumulated transition counts."""
        out_total: Dict[Hashable, float] = defaultdict(float)
        for (u, v), c in self._counts.items():
            if not self_loops and u == v:
                continue
            out_total[u] += c
        adj: Adjacency = {}
        for (u, v), c in self._counts.items():
            if not self_loops and u == v:
                continue
            p = c / out_total[u] if out_total[u] > 0 else eps
            adj.setdefault(u, {})[v] = -math.log(max(p, eps))
        return adj

    def min_free_energy_path(self, source: Hashable, target: Hashable) -> Tuple[float, List[Hashable]]:
        return dijkstra_path(self.adjacency(), source, target)

    def k_shortest_paths(self, source: Hashable, target: Hashable, k: int = 3) -> List[Tuple[float, List[Hashable]]]:
        return k_shortest_paths(self.adjacency(), source, target, k)

    def all_pairs_paths(self, k: int = 1) -> Dict[Tuple[Hashable, Hashable], List[Tuple[float, List[Hashable]]]]:
        """All-pairs pathway report; ``k`` competing routes per ordered pair."""
        adj = self.adjacency()
        report: Dict[Tuple[Hashable, Hashable], List[Tuple[float, List[Hashable]]]] = {}
        for s in self.nodes:
            for t in self.nodes:
                if s == t:
                    continue
                paths = k_shortest_paths(adj, s, t, k)
                if paths:
                    report[(s, t)] = paths
        return report
