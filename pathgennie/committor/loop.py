"""Closed loop: the committor learned from the swarms steers the next swarms.

One *round* is

1. pick restart configurations among everything visited so far, spread evenly
   over the current committor estimate (so every stage of the transition is
   revisited, not only the neighbourhood of the basins);
2. from each, run a short PathGennie *episode* until a basin is entered: the
   target is B with probability ``q`` and A otherwise, the progress metric is
   the log-probability of reaching that target, and trials are selected with
   unscaled Boltzmann weights -- i.e. with probability proportional to the
   committor at their endpoints;
3. retrain the committor (warm start) on all swarms recorded so far.

Before the first model exists, ``bootstrap`` runs episodes from A and from B
with a user-supplied progress variable. That CV may be poor: it only has to
move the system away from each basin. Each bootstrap run labels its own side
(runs from A see endpoints in A, runs from B see endpoints in B), which is
enough for a first, crude committor that the rounds then correct.

Why the target is drawn with probability ``q``: episodes then end in B about as
often as unbiased trajectories from the same point would, so both basins keep
feeding boundary information and the model cannot drift into predicting "always
B". Steering every episode to B would starve the A side.

The engine only ever runs plain, unbiased segments. All steering is selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from ..core.box import create_handle, frame_coords
from ..core.driver import PathGennieDriver
from ..core.parallel import SerialExecutor
from .basins import LABEL_NONE, BasinPair
from .progress import CommittorProgress
from .records import SwarmRecorder

__all__ = ["LoopConfig", "CommittorLoop"]


@dataclass
class LoopConfig:
    tau1: int = 20                  # swarm (sampler) length, steps
    tau2: int = 40                  # runner length, steps
    n_trials: int = 16              # trials per swarm
    episodes_per_round: int = 16
    max_cycles: int = 60            # per episode; an episode also ends on basin entry
    bootstrap_episodes: int = 1     # per basin
    bootstrap_max_cycles: int = 200
    bootstrap_retries: int = 10     # extra episodes per basin while it lacks boundary data
    bootstrap_sigma: float = 0.5    # softmax temperature of the bootstrap runs
    sigma: float = 1.0              # Boltzmann temperature of committor steering (1 = Doob weights)
    floor: float = 1e-3             # committor regularisation, see CommittorProgress
    n_models: int = 4               # ensemble size
    hidden: tuple = (64, 64)
    train_steps: int = 2500         # first fit
    refresh_steps: int = 800        # warm-started refits
    explore_fraction: float = 0.25  # share of restarts drawn where the ensemble disagrees most
    q_bins: int = 10
    seed: int = 0


@dataclass
class RoundReport:
    round: int                      # 0 = bootstrap
    swarms: int
    md_steps: int                   # all MD so far: trials and runners
    end_in_a: int                   # episodes of this round that ended in A
    end_in_b: int
    unfinished: int
    extra: dict = field(default_factory=dict)


class CommittorLoop:
    """Run the closed loop on any PathGennie engine.

    Parameters
    ----------
    engine:
        A :class:`~pathgennie.core.engine.Engine`.
    basins, cv_fn:
        The two states and the CV they are defined in.
    featurize:
        ``f(coords) -> 1-D array``, the network input.
    start_a, start_b:
        Handles of one configuration in each basin.
    bootstrap_progress:
        ``f(target) -> ProgressVariable`` for ``target`` in ``("A", "B")``: the
        geometric progress variable used before a model exists.
    on_round:
        Optional ``f(loop, report)`` called after each (re)training, e.g. to
        score the model against a reference; whatever it returns is stored in
        ``report.extra``.
    """

    def __init__(self, engine, basins: BasinPair, cv_fn: Callable, featurize: Callable, *,
                 start_a, start_b, bootstrap_progress: Callable[[str], object],
                 config: Optional[LoopConfig] = None, executor=None,
                 on_round: Optional[Callable] = None):
        self.engine = engine
        self.basins = basins
        self.cv_fn = cv_fn
        self.featurize = featurize
        self.start_a = start_a
        self.start_b = start_b
        self.bootstrap_progress = bootstrap_progress
        self.config = config or LoopConfig()
        self.executor = executor or SerialExecutor()
        self.on_round = on_round
        self.rng = np.random.default_rng(self.config.seed)
        self.recorder = SwarmRecorder(featurize=featurize, basins=basins, basin_cv=cv_fn,
                                      keep_coords=True)
        self.model = None
        self.reports: List[RoundReport] = []
        self._round = 0

    # -- bookkeeping -------------------------------------------------------------
    @property
    def md_steps(self) -> int:
        """Integrator steps spent so far, trials *and* runners."""
        trials = sum(r["ends"].shape[0] * r["lag_steps"] for r in self.recorder._rows)
        return int(trials + len(self.recorder.runner_starts) * self.config.tau2)

    def _basin_entry(self) -> Callable:
        """Episode stop criterion: a basin is *entered*.

        An episode that starts inside a basin (the bootstrap runs do) must first
        leave it; stopping on mere membership would end it after one cycle.
        """
        state = {"outside": False}

        def entered(coords) -> bool:
            if self.basins.label(self.cv_fn(coords)) == LABEL_NONE:
                state["outside"] = True
                return False
            return state["outside"]

        return entered

    def _seed(self) -> int:
        return int(self.rng.integers(1, 2_147_483_647))

    def _episode(self, handle, progress, *, selection: str, sigma: float, max_cycles: int) -> int:
        """Run one episode; return the basin label it ended in (-1 if neither)."""
        driver = PathGennieDriver(
            self.engine, progress, self._basin_entry(), executor=self.executor,
            sigma=sigma, seed=self._seed(), verbosity=0,
            swarm_recorder=self.recorder, selection=selection,
        )
        cfg = self.config
        traj, _ = driver.run(handle, tau1=cfg.tau1, tau2=cfg.tau2, max_trial=cfg.n_trials,
                             max_cycle=max_cycles, save_freq=max_cycles + 1)
        # The driver releases neither the handle it was given nor the anchor it
        # ends on; over thousands of episodes that would fill scratch.
        for stale in {handle, driver.last_anchor}:
            self.engine.release(stale)
        if len(traj) == 0:
            return LABEL_NONE
        return self.basins.label(self.cv_fn(traj[-1]))

    def _train(self) -> None:
        from .train import train_ensemble

        cfg = self.config
        first = self.model is None
        self.model = train_ensemble(
            self.recorder.dataset(), cfg.n_models, seed=cfg.seed,
            init=self.model, hidden=cfg.hidden,
            steps=cfg.train_steps if first else cfg.refresh_steps,
        )

    def _report(self, ends: List[int]) -> RoundReport:
        ends = np.asarray(ends)
        report = RoundReport(
            round=self._round, swarms=len(self.recorder), md_steps=self.md_steps,
            end_in_a=int((ends == 0).sum()), end_in_b=int((ends == 1).sum()),
            unfinished=int((ends == LABEL_NONE).sum()),
        )
        if self.on_round is not None:
            report.extra = self.on_round(self, report) or {}
        self.reports.append(report)
        self._round += 1
        return report

    # -- stages ------------------------------------------------------------------
    def _boundary_counts(self) -> tuple:
        """Trial endpoints in A and in B among swarms that can be trained on."""
        data = self.recorder.dataset()
        usable = (data.start_label == LABEL_NONE)[:, None] & data.mask
        return (int(((data.end_label == 0) & usable).sum()),
                int(((data.end_label == 1) & usable).sum()))

    def bootstrap(self) -> RoundReport:
        """Episodes from each basin with the geometric progress variable, then a first fit.

        A committor needs boundary data from both basins. One episode per basin
        usually provides it (the trials of the first swarms outside a basin fall
        back into it), but not always: the runner can carry the anchor straight
        out, and the episode can end without a single trial endpoint inside its
        own basin. So after the requested episodes, further ones are launched
        from whichever basin is still missing, up to ``bootstrap_retries`` times.
        """
        cfg = self.config
        ends = []

        def episode(start, target):
            ends.append(self._episode(
                self.engine.clone_anchor(start), self.bootstrap_progress(target),
                selection="softmax", sigma=cfg.bootstrap_sigma,
                max_cycles=cfg.bootstrap_max_cycles))

        for _ in range(cfg.bootstrap_episodes):
            episode(self.start_a, "B")
            episode(self.start_b, "A")
        for _ in range(cfg.bootstrap_retries):
            n_a, n_b = self._boundary_counts()
            if n_a and n_b:
                break
            if not n_a:
                episode(self.start_a, "B")
            if not n_b:
                episode(self.start_b, "A")
        else:
            n_a, n_b = self._boundary_counts()
            if not (n_a and n_b):
                raise RuntimeError(
                    f"bootstrap found {n_a} trial endpoints in A and {n_b} in B after "
                    f"{cfg.bootstrap_retries} extra episodes per basin. Check the basin "
                    "definitions (too small to be re-entered?) and the swarm length.")
        self._train()
        return self._report(ends)

    def restart_points(self, n: int) -> List[int]:
        """Indices (into the recorded swarms) of ``n`` restart configurations.

        Most are spread evenly over bins of the predicted committor. A share
        ``explore_fraction`` is drawn where the ensemble members disagree most,
        which is where another swarm is most informative.
        """
        cfg = self.config
        data = self.recorder.dataset()
        candidates = np.flatnonzero(data.start_label == LABEL_NONE)
        if candidates.size == 0:
            raise RuntimeError("no visited configuration lies outside the basins")
        feats = data.starts[candidates]
        q = self.model.predict(feats)

        n_explore = int(round(cfg.explore_fraction * n)) if len(self.model) > 1 else 0
        picks: List[int] = []
        if n_explore:
            spread = self.model.std(feats)
            top = np.argsort(spread)[-max(4 * n_explore, n_explore):]
            picks += list(candidates[self.rng.choice(top, size=n_explore, replace=top.size < n_explore)])

        bins = np.minimum((q * cfg.q_bins).astype(int), cfg.q_bins - 1)
        occupied = [np.flatnonzero(bins == b) for b in range(cfg.q_bins)]
        occupied = [members for members in occupied if members.size]
        for k in range(n - n_explore):
            members = occupied[k % len(occupied)]
            picks.append(int(candidates[self.rng.choice(members)]))
        return picks

    def round(self) -> RoundReport:
        """One round of committor-steered episodes followed by a warm-started refit."""
        if self.model is None:
            raise RuntimeError("call bootstrap() first")
        cfg = self.config
        ends = []
        for index in self.restart_points(cfg.episodes_per_round):
            coords = self.recorder.coords_starts[index]
            q = float(self.model.predict(self.recorder._rows[index]["start"]))
            q = cfg.floor + (1.0 - 2.0 * cfg.floor) * q
            target = "B" if self.rng.random() < q else "A"
            progress = CommittorProgress(self.model, self.featurize, target=target, floor=cfg.floor)
            ends.append(self._episode(
                create_handle(self.engine, coords), progress,
                selection="boltzmann", sigma=cfg.sigma, max_cycles=cfg.max_cycles))
        self._train()
        return self._report(ends)

    def run(self, n_rounds: int) -> "CommittorLoop":
        if self.model is None:
            self.bootstrap()
        for _ in range(int(n_rounds)):
            self.round()
        return self
