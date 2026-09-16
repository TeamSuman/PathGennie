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

# scipy is an optional accelerator for `RRT.nearest`: below the rebuild
# threshold (or when it is not installed) nearest-neighbour search falls back
# to a plain linear scan, which is exact and fast enough for small trees.
try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - exercised via the scipy-missing fallback
    cKDTree = None  # type: ignore[assignment]

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
    reason: Optional[str] = None  # e.g. "max_nodes" when build() stopped on the node cap


class RRT:
    """A single rapidly-exploring random tree in CV space.

    ``scale`` (optional): a per-dimension divisor for anisotropic CV spaces,
    where components have very different natural ranges (e.g. an Angstrom
    distance next to a radian dihedral) so unweighted Euclidean distance would
    be dominated by whichever component happens to have the larger range.

    The public API -- ``lower``/``upper`` here, and ``target_cv``/``goal_tol``
    on :meth:`build` -- is always expressed in *raw* CV units, i.e. whatever
    ``cv_fn`` returns. ``scale`` is applied once, at the single choke point
    :meth:`_cv` (for CVs computed from the engine) and when raw bounds/targets
    enter the object (``__init__`` for ``lower``/``upper``, :meth:`build` for
    ``target_cv``); every internal computation from then on -- ``Node.cv``,
    :meth:`nearest`, the swarm-selection metric in :meth:`extend`, and the
    ``goal_tol`` distance check -- operates in that single, consistent scaled
    space, with no further division needed. With ``scale=None`` (the default)
    this is the identity and behaviour is unchanged.
    """

    #: `nearest` rebuilds its scipy cKDTree only every this many new nodes
    #: (see `nearest` for why an exact-but-lazily-rebuilt tree is safe here).
    _KD_REBUILD_INTERVAL = 64

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
        scale: Optional[Sequence[float]] = None,
    ):
        self.engine = engine
        self.cv_fn = cv_fn
        self.scale = None if scale is None else np.asarray(scale, dtype=float)
        self.lower = self._to_scaled(np.asarray(lower, dtype=float))
        self.upper = self._to_scaled(np.asarray(upper, dtype=float))
        self.tau1 = int(tau1)
        self.tau2 = int(tau2)
        self.n_expand = int(n_expand)
        self.sigma = float(sigma)
        self.goal_bias = float(goal_bias)
        self.executor = executor or SerialExecutor()
        self.rng = np.random.default_rng(seed)
        self.nodes: List[Node] = []
        self._tree = None            # lazily-built scipy cKDTree (scaled-space CVs)
        self._tree_node_count = 0    # node count as of the last tree rebuild

    # -- helpers -------------------------------------------------------------
    def _to_scaled(self, cv: np.ndarray) -> np.ndarray:
        """Convert a raw CV (cv_fn's units) into the internal scaled space."""
        cv = np.asarray(cv, dtype=float)
        if self.scale is None:
            return cv
        return cv / self.scale

    def _cv(self, handle: Handle) -> np.ndarray:
        raw = np.atleast_1d(np.asarray(self.cv_fn(self.engine.get_coords(handle)), dtype=float))
        return self._to_scaled(raw)

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
        """Return the tree node closest to ``q`` (both already in scaled space).

        Rebuilding a scipy ``cKDTree`` after every single insertion would cost
        O(n log n) per expansion; instead the tree is rebuilt only every
        ``_KD_REBUILD_INTERVAL`` (64) new nodes and the handful of nodes added
        since that rebuild are checked with a plain linear scan and folded into
        the same argmin. That combination is still the *exact* nearest node
        (not an approximation), just amortised to O(log n) on average. Below
        the rebuild threshold, or if scipy is not installed, this is a plain
        linear scan throughout -- exact and fast enough for small trees.
        """
        q = np.asarray(q, dtype=float)
        n = len(self.nodes)

        if cKDTree is not None and n >= self._KD_REBUILD_INTERVAL:
            if self._tree is None or n - self._tree_node_count >= self._KD_REBUILD_INTERVAL:
                cvs = np.stack([nd.cv for nd in self.nodes])
                self._tree = cKDTree(cvs)
                self._tree_node_count = n

            _, idx = self._tree.query(q)
            best_node = self.nodes[int(idx)]
            best_dist = float(np.linalg.norm(best_node.cv - q))
            for nd in self.nodes[self._tree_node_count:]:
                d = float(np.linalg.norm(nd.cv - q))
                if d < best_dist:
                    best_dist = d
                    best_node = nd
            return best_node

        cvs = np.stack([nd.cv for nd in self.nodes])
        d = np.linalg.norm(cvs - q, axis=1)
        return self.nodes[int(d.argmin())]

    # -- expansion -----------------------------------------------------------
    def extend(self, node: Node, q_rand: np.ndarray) -> Node:
        """Grow one new node from ``node`` toward ``q_rand`` via a swarm + runner."""
        q_rand = np.asarray(q_rand, dtype=float)
        seg_seeds = [self._seed() for _ in range(self.n_expand)]

        def worker(seg_seed, device):
            handle = self.engine.clone_anchor(node.handle)
            seg = self.engine.run_segment(
                handle, self.tau1, randomize_velocities=True, seed=seg_seed, device=device
            )
            # Release the cloned-anchor input; the segment output (seg) is a new
            # handle. Without this every expansion leaks one clone (a scratch
            # restart file / engine-cache entry), filling scratch on long searches.
            if seg != handle:
                self.engine.release(handle)
            return seg

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
        goal_test: Optional[Callable[[Handle], bool]] = None,
        max_nodes: Optional[int] = None,
    ) -> RRTResult:
        """Grow the tree for up to ``max_iter`` expansions.

        ``target_cv`` (raw CV units, optional) drives goal-biased sampling
        (:meth:`sample_target`) regardless of how success is judged.

        Success is judged one of two ways:

        * ``goal_test`` given: a new node succeeds when ``goal_test(new.handle)``
          is true (e.g. true basin membership) -- ``goal_tol`` is then ignored.
        * ``goal_test`` is None and ``target_cv`` given: the legacy behaviour,
          success when the new node's CV is within ``goal_tol`` of ``target_cv``.

        ``max_nodes`` (optional): stop once the tree would exceed this many
        nodes (root included) without having succeeded, returning
        ``RRTResult(success=False, ..., reason="max_nodes")``. Calling
        :meth:`build` again on a tree that already has nodes (e.g. one
        restored by :meth:`resume`) continues growing that same tree instead
        of adding a second root; ``initial_handle`` is then unused.
        """
        goal = None if target_cv is None else self._to_scaled(np.asarray(target_cv, dtype=float))
        if not self.nodes:
            self.add_node(self.engine.clone_anchor(initial_handle), None)

        for _ in range(max_iter):
            if max_nodes is not None and len(self.nodes) >= max_nodes:
                return self._exhausted(goal, reason="max_nodes")

            q = self.sample_target(goal)
            new = self.extend(self.nearest(q), q)

            if goal_test is not None:
                success = bool(goal_test(new.handle))
            elif goal is not None:
                success = np.linalg.norm(new.cv - goal) <= goal_tol
            else:
                success = False
            if success:
                return RRTResult(True, self.path_to(new), len(self.nodes), new)

        return self._exhausted(goal, reason=None)

    def _exhausted(self, goal: Optional[np.ndarray], *, reason: Optional[str]) -> RRTResult:
        """Build the failure result once ``build`` cannot succeed further."""
        if goal is not None and self.nodes:
            cvs = np.stack([n.cv for n in self.nodes])
            best = self.nodes[int(np.linalg.norm(cvs - goal, axis=1).argmin())]
            return RRTResult(False, self.path_to(best), len(self.nodes), best, reason=reason)
        return RRTResult(False, [], len(self.nodes), None, reason=reason)

    # -- checkpoint / resume ---------------------------------------------------
    def checkpoint(self, path: str) -> None:
        """Save the tree to ``path`` (an ``.npz``): each node's (scaled) CV,
        its parent index (-1 for the root), and its engine coordinates (via
        ``engine.get_coords``, Angstrom) -- everything :meth:`resume` needs to
        recreate the tree via ``engine.create_handle``. The RNG state and
        ``scale``/``lower``/``upper``/... constructor arguments are *not*
        saved; pass the same ones again to :meth:`resume`.
        """
        if self.nodes:
            cvs = np.stack([n.cv for n in self.nodes])
            parents = np.array([-1 if n.parent is None else n.parent for n in self.nodes], dtype=np.int64)
            coords = np.stack([np.asarray(self.engine.get_coords(n.handle), dtype=float) for n in self.nodes])
        else:
            cvs = np.empty((0,), dtype=float)
            parents = np.empty((0,), dtype=np.int64)
            coords = np.empty((0, 0, 3), dtype=float)
        np.savez(path, cvs=cvs, parents=parents, coords=coords)

    @classmethod
    def resume(
        cls,
        path: str,
        engine: Engine,
        cv_fn: Callable[[np.ndarray], np.ndarray],
        **kwargs,
    ) -> "RRT":
        """Reconstruct an RRT from a checkpoint written by :meth:`checkpoint`.

        ``**kwargs`` are forwarded to ``__init__`` (``lower``/``upper`` are
        still required, exactly like building a fresh RRT; pass the same
        ``scale`` used originally too, since it is not itself checkpointed).
        Node handles are recreated via ``engine.create_handle(coords)`` --
        velocities are not restored, which is fine because every future
        expansion starts samplers with ``randomize_velocities=True`` anyway.
        Call :meth:`build` on the result to keep growing the same tree.
        """
        data = np.load(path)
        cvs, parents, coords = data["cvs"], data["parents"], data["coords"]

        rrt = cls(engine, cv_fn, **kwargs)
        nodes: List[Node] = []
        for i in range(len(cvs)):
            handle = engine.create_handle(coords[i])
            parent = None if int(parents[i]) < 0 else int(parents[i])
            nodes.append(Node(id=i, handle=handle, cv=np.asarray(cvs[i], dtype=float), parent=parent))
        rrt.nodes = nodes
        return rrt


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

    tree_a = RRT(engine, cv_fn, seed=seed, lower=lower, upper=upper, tau1=tau1, tau2=tau2, n_expand=n_expand, sigma=sigma, goal_bias=0.0, executor=executor)
    tree_b = RRT(engine, cv_fn, seed=seed + 1, lower=lower, upper=upper, tau1=tau1, tau2=tau2, n_expand=n_expand, sigma=sigma, goal_bias=0.0, executor=executor)
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
