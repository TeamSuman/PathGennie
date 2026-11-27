"""Committor milestone M3 on the Wolfe-Quapp toy potential: the closed loop.

    python benchmarks/committor_loop_toy.py [--seeds 10] [--rounds 8] [--json out.json]

The loop is bootstrapped with a deliberately poor progress variable (the x
coordinate alone, which cannot tell the product from an off-path minimum) and
must then learn the committor on its own. The model is scored against the exact
committor on a grid covering the *whole* thermally relevant landscape -- both
reaction channels -- not only the region a particular run happened to visit.

Gates
  (a) from the poor CV: r(TS) > 0.95 and RMSE(TS) < 0.10
  (b) fewer MD steps to RMSE(TS) < 0.10 than fixed-CV discovery + offline training
  (c) both channels resolved (RMSE(TS) < 0.10 in each) in at least 9 of 10 seeds

Ablations (fewer seeds) show which ingredients matter.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from pathgennie.committor import BasinPair, SphereBasin, SwarmDataset, SwarmRecorder
from pathgennie.committor.loop import CommittorLoop, LoopConfig
from pathgennie.committor.reference import grid_committor
from pathgennie.committor.train import committor_scores, train_ensemble
from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.progress import TargetMetric
from pathgennie.core.toy import ToyLangevinEngine, wolfe_quapp_potential

A = np.array([-1.174, 1.477])
B = np.array([1.124, -1.485])
R = 0.4
BASINS = BasinPair(SphereBasin(A, R), SphereBasin(B, R))
TARGET_RMSE = 0.10


def xy(coords):
    return np.array([coords[0, 0], coords[0, 1]])


def x_only(target):
    """The poor CV: progress is measured along x alone."""
    return TargetMetric(lambda c: np.array([c[0, 0]]),
                        target_cv=np.array([B[0] if target == "B" else A[0]]))


def good_cv(target):
    return TargetMetric(xy, target_cv=B if target == "B" else A)


class Scorer:
    """Scores a model on a grid over the landscape, per reaction channel."""

    def __init__(self):
        ref = grid_committor(
            wolfe_quapp_potential,
            lambda x, y: (x - A[0]) ** 2 + (y - A[1]) ** 2 < R * R,
            lambda x, y: (x - B[0]) ** 2 + (y - B[1]) ** 2 < R * R,
            xlim=(-2.6, 2.6), ylim=(-2.6, 2.6), n=201,
        )
        grid = np.array([[x, y] for x in np.linspace(-2, 2, 41) for y in np.linspace(-2.2, 2.2, 45)])
        self.points = grid[wolfe_quapp_potential(grid[:, 0], grid[:, 1]) < -1.0]
        self.exact = ref(self.points)
        ts = (self.exact > 0.05) & (self.exact < 0.95)
        # The two channels pass through the minima near (-0.9, -1.3) and (1.0, 1.3).
        self.lower = ts & (self.points.sum(axis=1) < 0)
        self.upper = ts & (self.points.sum(axis=1) >= 0)

    def __call__(self, model) -> dict:
        pred = model.predict(self.points)
        s = committor_scores(pred, self.exact)
        rmse = lambda sel: float(np.sqrt(np.mean((pred[sel] - self.exact[sel]) ** 2)))  # noqa: E731
        return dict(r_ts=s["pearson_transition"], rmse_ts=s["rmse_transition"],
                    rmse_lower=rmse(self.lower), rmse_upper=rmse(self.upper))


def run_loop(scorer, seed, rounds, bootstrap, **overrides):
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    loop = CommittorLoop(
        engine, BASINS, xy, xy,
        start_a=engine.create_state(A), start_b=engine.create_state(B),
        bootstrap_progress=bootstrap, config=LoopConfig(seed=seed, **overrides),
        on_round=lambda lp, report: scorer(lp.model),
    )
    loop.run(rounds)
    return [dict(round=r.round, md_steps=r.md_steps, **r.extra) for r in loop.reports]


def steps_to_target(curve):
    """MD steps at the first round from which RMSE(TS) stays below the target."""
    for i, row in enumerate(curve):
        if all(later["rmse_ts"] < TARGET_RMSE for later in curve[i:]):
            return row["md_steps"]
    return None


def summarise(label, curves):
    final = [c[-1] for c in curves]
    reached = [steps_to_target(c) for c in curves]
    both = sum(f["rmse_lower"] < TARGET_RMSE and f["rmse_upper"] < TARGET_RMSE for f in final)
    hit = [r for r in reached if r is not None]
    row = dict(
        seeds=len(curves),
        r_ts_median=float(np.median([f["r_ts"] for f in final])),
        r_ts_min=float(np.min([f["r_ts"] for f in final])),
        rmse_ts_median=float(np.median([f["rmse_ts"] for f in final])),
        rmse_ts_max=float(np.max([f["rmse_ts"] for f in final])),
        both_channels=int(both),
        reached_target=len(hit),
        steps_to_target_median=None if not hit else float(np.median(hit)),
        final_md_steps=float(np.median([f["md_steps"] for f in final])),
    )
    steps = "      -" if not hit else f"{row['steps_to_target_median']:7.0f}"
    print(f"    {label:28s} {row['r_ts_median']:.3f} ({row['r_ts_min']:.3f})  "
          f"{row['rmse_ts_median']:.3f} ({row['rmse_ts_max']:.3f})   {both:2d}/{len(curves):<2d}     "
          f"{len(hit):2d}/{len(curves):<2d}  {steps}")
    return row


def offline_baseline(scorer, n_runs, seed):
    """Fixed, good geometric CV; discovery runs both ways; train offline."""
    runs, md_steps = [], 0
    for k in range(n_runs):
        for src, dst, offset in ((A, B, 0), (B, A, 1000)):
            engine = ToyLangevinEngine(dt=0.005, kT=1.0)
            rec = SwarmRecorder(featurize=xy, basins=BASINS)
            driver = PathGennieDriver(
                engine, TargetMetric(xy, target_cv=dst),
                lambda c, dst=dst: bool(np.linalg.norm(xy(c) - dst) < 0.3),
                sigma=0.5, seed=100 * seed + offset + k, verbosity=0, swarm_recorder=rec,
            )
            driver.run(engine.create_state(src), tau1=20, tau2=40, max_trial=16,
                       max_cycle=2000, save_freq=5)
            runs.append(rec.dataset())
            md_steps += rec.dataset().total_steps + len(rec.runner_starts) * 40
    model = train_ensemble(SwarmDataset.concatenate(runs), 4, seed=seed, steps=2500)
    return dict(md_steps=md_steps, **scorer(model))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--ablation-seeds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    scorer = Scorer()
    out = {}

    print(f"M3  closed loop, scored on {int(scorer.lower.sum() + scorer.upper.sum())} transition-region grid "
          f"points ({int(scorer.lower.sum())} lower channel, {int(scorer.upper.sum())} upper channel)")
    print(f"    after {args.rounds} rounds             r(TS) med (min)  RMSE(TS) med (max)  both ch.  "
          f"reached  steps to RMSE<{TARGET_RMSE}")
    print("    " + "-" * 100)
    main_curves = [run_loop(scorer, s, args.rounds, x_only) for s in range(args.seeds)]
    out["curves"] = main_curves
    out["loop_poor_cv"] = main_row = summarise("closed loop, x-only bootstrap", main_curves)

    ablations = {
        "good-CV bootstrap": dict(bootstrap=good_cv),
        "no ensemble-guided restarts": dict(bootstrap=x_only, explore_fraction=0.0),
        "single network": dict(bootstrap=x_only, n_models=1),
    }
    for label, kw in ablations.items():
        kw = dict(kw)
        bootstrap = kw.pop("bootstrap")
        out[label] = summarise(label, [run_loop(scorer, s, args.rounds, bootstrap, **kw)
                                       for s in range(args.ablation_seeds)])

    print("\n    fixed good CV + offline training (ensemble of 4), same grid")
    print("    runs each way   md steps   r(TS)   RMSE(TS)   lower   upper")
    out["offline"] = {}
    for n_runs in (2, 4, 8, 16):
        rows = [offline_baseline(scorer, n_runs, seed) for seed in range(3)]
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        out["offline"][n_runs] = mean
        print(f"    {n_runs:6d}        {mean['md_steps']:9.0f}   {mean['r_ts']:.3f}   {mean['rmse_ts']:.3f}"
              f"      {mean['rmse_lower']:.3f}   {mean['rmse_upper']:.3f}")

    offline_hit = [v["md_steps"] for v in out["offline"].values() if v["rmse_ts"] < TARGET_RMSE]
    gate_a = main_row["r_ts_min"] > 0.95 and main_row["rmse_ts_max"] < TARGET_RMSE
    gate_c = main_row["both_channels"] >= int(np.ceil(0.9 * args.seeds))
    loop_steps = main_row["steps_to_target_median"]
    gate_b = loop_steps is not None and (not offline_hit or loop_steps < min(offline_hit))
    out["gates"] = dict(a=bool(gate_a), b=bool(gate_b), c=bool(gate_c),
                        offline_steps_to_target=min(offline_hit) if offline_hit else None)
    print(f"\n    gate (a) accuracy from a poor CV, every seed      -> {'PASS' if gate_a else 'FAIL'}")
    print(f"    gate (b) cheaper than fixed CV + offline training  -> {'PASS' if gate_b else 'FAIL'}"
          f"  (loop {loop_steps}, offline {min(offline_hit) if offline_hit else 'never reached'})")
    print(f"    gate (c) both channels in >= 90% of seeds          -> {'PASS' if gate_c else 'FAIL'}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
    return 0 if (gate_a and gate_b and gate_c) else 1


if __name__ == "__main__":
    t0 = time.time()
    code = main()
    print(f"\ndone in {time.time() - t0:.0f} s")
    raise SystemExit(code)
