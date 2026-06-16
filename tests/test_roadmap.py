"""Roadmap graph + pathway extraction tests (pure standard library / NumPy)."""

import math

from pathgennie.search.roadmap import Roadmap, dijkstra_path, k_shortest_paths


def test_dijkstra_simple():
    adj = {
        "A": {"B": 1.0, "C": 4.0},
        "B": {"C": 1.0, "D": 5.0},
        "C": {"D": 1.0},
        "D": {},
    }
    cost, path = dijkstra_path(adj, "A", "D")
    assert path == ["A", "B", "C", "D"]
    assert math.isclose(cost, 3.0)


def test_dijkstra_unreachable():
    adj = {"A": {"B": 1.0}, "B": {}, "C": {}}
    cost, path = dijkstra_path(adj, "A", "C")
    assert cost == math.inf and path == []


def test_k_shortest_paths_distinct_increasing():
    adj = {
        "A": {"B": 1.0, "C": 1.0},
        "B": {"D": 1.0},
        "C": {"D": 1.0},
        "D": {},
    }
    paths = k_shortest_paths(adj, "A", "D", k=2)
    assert len(paths) == 2
    costs = [c for c, _ in paths]
    assert costs[0] <= costs[1]
    # Two genuinely different parallel routes A->B->D and A->C->D.
    route_sets = {tuple(p) for _, p in paths}
    assert route_sets == {("A", "B", "D"), ("A", "C", "D")}


def test_roadmap_from_sequence_and_min_path():
    rm = Roadmap()
    # State trajectory visiting 0 -> 1 -> 2 mostly, with a rare 0 -> 2 jump.
    seq = [0, 0, 1, 1, 2, 2, 1, 0, 1, 2, 0, 2]
    rm.observe_sequence(seq)
    assert set(rm.nodes) == {0, 1, 2}

    adj = rm.adjacency()
    # The frequent 0->1 edge must be cheaper than the rare 0->2 edge.
    assert adj[0][1] < adj[0][2]

    cost, path = rm.min_free_energy_path(0, 2)
    assert path[0] == 0 and path[-1] == 2
    assert cost < math.inf


def test_roadmap_all_pairs_report():
    rm = Roadmap()
    rm.observe_sequence([0, 1, 2, 1, 0, 1, 2])
    report = rm.all_pairs_paths(k=1)
    assert (0, 2) in report
    assert report[(0, 2)][0][1][0] == 0
