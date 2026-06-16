"""Exploration policies for PathGennie.

The greedy/anchor cycle in :class:`pathgennie.core.driver.PathGennieDriver` is one
*exploration policy*: at every step it expands a single anchor and aims at a fixed
escape/target metric.  Richer non-linear search — Rapidly-exploring Random Trees
(RRT) and RRT-Connect (:mod:`pathgennie.search.rrt`) — expand *different* tree
nodes toward *random* CV targets, which lets the search backtrack, change
direction, and move orthogonally to a hand-crafted CV.

This module defines the thin :class:`ExplorerPolicy` protocol those searchers
share so they remain interchangeable, plus :class:`GreedyPolicy`, the trivial
policy that reproduces the driver's behaviour (always expand the current node,
always aim at the global metric).

A policy only decides *which node to expand and what CV target to aim at*; running
the swarm and selecting a sampler is still done with the shared
:class:`~pathgennie.core.engine.Engine` and
:func:`~pathgennie.core.selection.softmax_select`, so policies never duplicate MD
or selection logic.
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

__all__ = ["ExplorerPolicy", "GreedyPolicy"]


class ExplorerPolicy(Protocol):
    """Decide the next (node, CV-target) to expand from a collection of nodes."""

    def propose(self, nodes, rng: np.random.Generator):
        """Return ``(node, target_cv)``: the node to expand and the CV to aim at."""
        ...


class GreedyPolicy:
    """Always expand the most recently committed node toward a fixed target.

    Reproduces :class:`PathGennieDriver`'s behaviour: ``target_cv`` is a constant
    (or ``None`` for a pure escape metric handled by the progress variable).
    """

    def __init__(self, target_cv: Optional[np.ndarray] = None):
        self.target_cv = None if target_cv is None else np.asarray(target_cv, dtype=float)

    def propose(self, nodes, rng: np.random.Generator):
        return nodes[-1], self.target_cv
