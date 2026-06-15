"""Rapidly-exploring Random Trees (RRT / RRT-Connect) over a PathGennie swarm.

The greedy driver follows a monotone progress metric, so it cannot backtrack,
change direction, or move orthogonally to the CV — the failure mode the paper
shows on Wolfe-Quapp.  RRT reframes each expansion as growing a tree in CV space:

1. sample a random CV target ``q_rand`` (occasionally the goal — *goal biasing*);
2. find the nearest existing tree node ``q_near`` (by CV distance);
3. run a PathGennie swarm from ``q_near`` and select the sampler whose CV moves
   closest to ``q_rand`` — i.e. the existing
   :func:`~pathgennie.core.selection.softmax_select` with metric ``-||cv-q_rand||``;
4. extend the chosen sampler with a ``tau2`` runner and add it as a new node.

Because targets can point in *any* CV direction and the tree remembers every
node, the search backtracks and explores naturally.  **RRT-Connect** grows two
trees — from the start and from a goal configuration — and links them, which
crosses barriers far faster.

This reuses the shared :class:`~pathgennie.core.engine.Engine` and
:class:`~pathgennie.core.parallel.ParallelExecutor`, so RRT is multi-GPU for free
and works with any backend (including the toy Langevin engine used in tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from pathgennie.core.engine import Engine, Handle
from pathgennie.core.parallel import ParallelExecutor, SerialExecutor
from pathgennie.core.selection import softmax_select

__all__ = ["Node", "RRTResult", "RRT", "rrt_connect"]


@dataclass
class Node:
    id: int
    handle: Handle
    cv: np.ndarray
    parent: Optional[int] = None


@dataclass
class RRTResult:
    success: bool
    path: List[Node]          # root -> goal (empty if not successful)
    tree_size: int
    goal_node: Optional[Node] = None


class RRT:
    """A single rapidly-exploring random tree in CV space."""

    def __init__(
        self,
        engine: Engine,
        cv_fn: Callable[[np.ndarray], np.ndarray],
        *,
        lower: Sequence[float],
        upper: Sequence[float],
        tau1: int = 5,
        tau2: int = 10,
        n_expand: int = 8,
        sigma: float = 0.05,
        goal_bias: float = 0.1,
        executor: Optional[ParallelExecutor] = None,
        seed: int = 0,
    ):
        self.engine = engine
        self.cv_fn = cv_fn
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.tau1 = int(tau1)
        self.tau2 = int(tau2)
        self.n_expand = int(n_expand)
        self.sigma = float(sigma)
        self.goal_bias = float(goal_bias)
        self.executor = executor or SerialExecutor()
        self.rng = np.random.default_rng(seed)
        self.nodes: List[Node] = []

    # -- helpers -------------------------------------------------------------
    def _cv(self, handle: Handle) -> np.ndarray:
        return np.atleast_1d(np.asarray(self.cv_fn(self.engine.get_coords(handle)), dtype=float))

    def add_node(self, handle: Handle, parent: Optional[int]) -> Node:
        node = Node(id=len(self.nodes), handle=handle, cv=self._cv(handle), parent=parent)
        self.nodes.append(node)
        return node

    def _seed(self) -> int:
        return int(self.rng.integers(1, 2_147_483_647))

    def sample_target(self, goal_cv: Optional[np.ndarray]) -> np.ndarray:
        if goal_cv is not None and self.rng.random() < self.goal_bias:
            return np.asarray(goal_cv, dtype=float)
        return self.rng.uniform(self.lower, self.upper)

    def nearest(self, q: np.ndarray) -> Node:
        cvs = np.stack([n.cv for n in self.nodes])
        d = np.linalg.norm(cvs - np.asarray(q, dtype=float), axis=1)
        return self.nodes[int(d.argmin())]

    # -- expansion -----------------------------------------------------------
    def extend(self, node: Node, q_rand: np.ndarray) -> Node:
        """Grow one new node from ``node`` toward ``q_rand`` via a swarm + runner."""
        q_rand = np.asarray(q_rand, dtype=float)
        seg_seeds = [self._seed() for _ in range(self.n_expand)]

        def worker(seg_seed, device):
            handle = self.engine.clone_anchor(node.handle)
            return self.engine.run_segment(
                handle, self.tau1, randomize_velocities=True, seed=seg_seed, device=device
            )

        trials = self.executor.map(worker, seg_seeds)
        cvs = [self._cv(t) for t in trials]
        metrics = np.array([-np.linalg.norm(cv - q_rand) for cv in cvs], dtype=float)
        chosen_idx = softmax_select(metrics, self.sigma, self.rng)
        chosen = trials[chosen_idx]
        for j, t in enumerate(trials):
            if j != chosen_idx:
                self.engine.release(t)

        runner = self.engine.run_segment(
            chosen, self.tau2, randomize_velocities=False,
            seed=self._seed(), device=self.executor.devices[0],
        )
        if runner is not chosen:
            self.engine.release(chosen)
        return self.add_node(runner, node.id)

    def path_to(self, node: Node) -> List[Node]:
        chain: List[Node] = []
        current: Optional[Node] = node
        while current is not None:
            chain.append(current)
            current = self.nodes[current.parent] if current.parent is not None else None
        chain.reverse()
        return chain

    # -- top-level driver ----------------------------------------------------
    def build(
        self,
        initial_handle: Handle,
        *,
        target_cv: Optional[Sequence[float]] = None,
        max_iter: int = 200,
        goal_tol: float = 0.3,
    ) -> RRTResult:
        goal = None if target_cv is None else np.asarray(target_cv, dtype=float)
        self.add_node(self.engine.clone_anchor(initial_handle), None)

        for _ in range(max_iter):
            q = self.sample_target(goal)
            new = self.extend(self.nearest(q), q)
            if goal is not None and np.linalg.norm(new.cv - goal) <= goal_tol:
                return RRTResult(True, self.path_to(new), len(self.nodes), new)

        if goal is not None:
            cvs = np.stack([n.cv for n in self.nodes])
            best = self.nodes[int(np.linalg.norm(cvs - goal, axis=1).argmin())]
            return RRTResult(False, self.path_to(best), len(self.nodes), best)
        return RRTResult(False, [], len(self.nodes), None)


def rrt_connect(
    engine: Engine,
    cv_fn: Callable[[np.ndarray], np.ndarray],
    start_handle: Handle,
    goal_handle: Handle,
    *,
    lower: Sequence[float],
    upper: Sequence[float],
    tau1: int = 5,
    tau2: int = 10,
    n_expand: int = 8,
    sigma: float = 0.05,
    executor: Optional[ParallelExecutor] = None,
    seed: int = 0,
    max_iter: int = 200,
    connect_tol: float = 0.3,
) -> RRTResult:
    """Bidirectional RRT-Connect between two configurations.

    Grows a tree from ``start_handle`` and another from ``goal_handle``; each
    iteration one tree extends toward a random target and the other greedily
    extends toward the new node until they are within ``connect_tol`` in CV space.
    Returns the joined start->goal path.
    """

    common = dict(lower=lower, upper=upper, tau1=tau1, tau2=tau2, n_expand=n_expand,
                  sigma=sigma, goal_bias=0.0, executor=executor)
    tree_a = RRT(engine, cv_fn, seed=seed, **common)
    tree_b = RRT(engine, cv_fn, seed=seed + 1, **common)
    tree_a.add_node(engine.clone_anchor(start_handle), None)
    tree_b.add_node(engine.clone_anchor(goal_handle), None)

    a, b = tree_a, tree_b
    a_is_start = True
    for _ in range(max_iter):
        q = a.sample_target(None)
        a_new = a.extend(a.nearest(q), q)

        # Greedily grow b toward a_new.
        b_node = b.extend(b.nearest(a_new.cv), a_new.cv)
        if np.linalg.norm(b_node.cv - a_new.cv) <= connect_tol:
            path_a = a.path_to(a_new)
            path_b = b.path_to(b_node)
            # Order the joined path start -> goal.
            if a_is_start:
                joined = path_a + list(reversed(path_b))
            else:
                joined = path_b + list(reversed(path_a))
            return RRTResult(True, joined, len(a.nodes) + len(b.nodes), a_new)

        a, b = b, a
        a_is_start = not a_is_start

    return RRTResult(False, [], len(tree_a.nodes) + len(tree_b.nodes), None)
