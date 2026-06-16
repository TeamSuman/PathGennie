"""Agentic orchestration for PathGennie.

The controller automates the "how hard to throw the swarm" decisions that are
otherwise hand-tuned: the swarm size ``N`` and segment lengths ``tau1``/``tau2``,
when to expand an under-explored frontier, when to refresh a learned CV, and when
to stop.  :class:`RuleBasedController` is a deterministic, testable policy (the
plan's "rule-based first"); it implements the same ``Controller`` surface a future
RL or LLM meta-controller would, so it can be swapped without touching the driver.
"""

from .controller import Controller, RuleBasedController, SwarmParams

__all__ = ["Controller", "RuleBasedController", "SwarmParams"]
