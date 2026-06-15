"""Rule-based adaptive controller for the PathGennie swarm.

Each round the controller looks at the recent progress history and adjusts the
sampling effort:

* **stalling** (progress rate below a threshold) → *escalate*: enlarge the swarm
  ``N`` and lengthen the sampler segment ``tau1`` (and ``tau2``) so the search
  pushes harder over the barrier;
* **progressing well** → *relax*: shrink ``N`` back toward its minimum to save GPU
  time (progress-per-wall-second, not raw progress, is the objective);
* **plateaued** for a long time → recommend stopping.

It also offers count-based frontier selection (expand the least-visited region —
anti-trapping) and a CV-refresh schedule (for SPIB).  Everything is deterministic
given its inputs, so it is unit-testable; the same surface (``Controller``) could
later be backed by a contextual-bandit/RL agent or an LLM meta-controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

import numpy as np

__all__ = ["SwarmParams", "Controller", "RuleBasedController"]


@dataclass
class SwarmParams:
    n_trial: int
    tau1: int
    tau2: int


class Controller(Protocol):
    def update(self, metric_history: Sequence[float]) -> SwarmParams: ...
    def should_stop(self, metric_history: Sequence[float]) -> bool: ...


class RuleBasedController:
    def __init__(
        self,
        params: SwarmParams,
        *,
        n_bounds: tuple = (4, 64),
        tau1_bounds: tuple = (2, 200),
        tau2_bounds: tuple = (4, 400),
        stall_window: int = 5,
        stall_eps: float = 1e-3,
        escalate: float = 1.5,
        relax: float = 0.75,
        stop_patience: int = 20,
        refresh_every: int = 50,
    ):
        self.params = SwarmParams(int(params.n_trial), int(params.tau1), int(params.tau2))
        self.n_bounds = n_bounds
        self.tau1_bounds = tau1_bounds
        self.tau2_bounds = tau2_bounds
        self.stall_window = int(stall_window)
        self.stall_eps = float(stall_eps)
        self.escalate = float(escalate)
        self.relax = float(relax)
        self.stop_patience = int(stop_patience)
        self.refresh_every = int(refresh_every)
        self._best = -np.inf
        self._since_improve = 0
        self._last_refresh = -self.refresh_every

    # -- progress bookkeeping ------------------------------------------------
    def _progress_rate(self, metric_history: Sequence[float]) -> Optional[float]:
        if len(metric_history) < 2:
            return None
        window = list(metric_history)[-self.stall_window:]
        if len(window) < 2:
            return None
        return (window[-1] - window[0]) / (len(window) - 1)

    def _note_improvement(self, metric_history: Sequence[float]) -> None:
        if not metric_history:
            return
        latest = float(metric_history[-1])
        if latest > self._best + self.stall_eps:
            self._best = latest
            self._since_improve = 0
        else:
            self._since_improve += 1

    # -- public API ----------------------------------------------------------
    def update(self, metric_history: Sequence[float]) -> SwarmParams:
        """Return adjusted swarm parameters given the progress history."""
        self._note_improvement(metric_history)
        rate = self._progress_rate(metric_history)
        if rate is None:
            return self.params

        n_lo, n_hi = self.n_bounds
        t1_lo, t1_hi = self.tau1_bounds
        t2_lo, t2_hi = self.tau2_bounds
        if rate < self.stall_eps:
            # Stalling: push harder.
            self.params.n_trial = int(min(n_hi, max(n_lo, round(self.params.n_trial * self.escalate))))
            self.params.tau1 = int(min(t1_hi, max(t1_lo, round(self.params.tau1 * self.escalate))))
            self.params.tau2 = int(min(t2_hi, max(t2_lo, round(self.params.tau2 * self.escalate))))
        else:
            # Progressing: relax the swarm size to save compute.
            self.params.n_trial = int(min(n_hi, max(n_lo, round(self.params.n_trial * self.relax))))
        return self.params

    def should_stop(self, metric_history: Sequence[float]) -> bool:
        """Recommend stopping once progress has plateaued for ``stop_patience``."""
        return self._since_improve >= self.stop_patience

    def should_refresh_cv(self, cycle: int) -> bool:
        """True on a refresh-schedule boundary (drives SPIB retraining)."""
        if cycle - self._last_refresh >= self.refresh_every:
            self._last_refresh = cycle
            return True
        return False

    @staticmethod
    def choose_frontier(visit_counts: Sequence[int]) -> int:
        """Pick the least-visited node (count-based novelty / anti-trapping)."""
        counts = np.asarray(visit_counts, dtype=float)
        if counts.size == 0:
            raise ValueError("visit_counts must be non-empty")
        return int(counts.argmin())
