"""Committor milestones M0 and M1 on the Wolfe-Quapp toy potential.

    python benchmarks/committor_toy.py [--json out.json]

M0  the shooting committor agrees with the numerically exact committor
M1  a network trained on the swarms of ordinary PathGennie discovery runs
    reproduces the exact committor along the discovered transition tube

It also reports how accuracy depends on the amount of data and on the swarm
length tau1, and what the same accuracy would cost by shooting. Everything runs
on CPU in a few minutes; no MD engine is needed.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from pathgennie.committor import BasinPair, SphereBasin, SwarmDataset, SwarmRecorder, shooting_committor
from pathgennie.committor.basins import LABEL_NONE
from pathgennie.committor.reference import grid_committor
from pathgennie.committor.train import committor_scores, train_committor
from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.progress import TargetMetric
from pathgennie.core.toy import ToyLangevinEngine, wolfe_quapp_potential

A = np.array([-1.174, 1.477])
B = np.array([1.124, -1.485])
R = 0.4
KT = 1.0
DT = 0.005
BASINS = BasinPair(SphereBasin(A, R), SphereBasin(B, R))


def xy(coords):
    return np.array([coords[0, 0], coords[0, 1]])


def discover(seed, src, dst, tau1, tau2=40, n_trials=16, sigma=0.5):
    engine = ToyLangevinEngine(dt=DT, kT=KT)
    rec = SwarmRecorder(featurize=xy, basins=BASINS)
    driver = PathGennieDriver(
        engine, TargetMetric(xy, target_cv=dst),
        lambda c: bool(np.linalg.norm(xy(c) - dst) < 0.3),
        sigma=sigma, seed=seed, verbosity=0, swarm_recorder=rec,
    )
    driver.run(engine.create_state(src), tau1=tau1, tau2=tau2, max_trial=n_trials,
               max_cycle=2000, save_freq=5)
    return rec.dataset()


def collect(seeds, tau1):
    runs = []
    for seed in seeds:
        runs.append(discover(seed, A, B, tau1))
        runs.append(discover(1000 + seed, B, A, tau1))
    return SwarmDataset.concatenate(runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", default=None, help="write the results to this file")
    parser.add_argument("--train-seeds", type=int, default=3, help="network initialisations per setting")
    args = parser.parse_args()
    out = {}

    reference = grid_committor(
        wolfe_quapp_potential,
        lambda x, y: (x - A[0]) ** 2 + (y - A[1]) ** 2 < R * R,
        lambda x, y: (x - B[0]) ** 2 + (y - B[1]) ** 2 < R * R,
        xlim=(-2.6, 2.6), ylim=(-2.6, 2.6), n=201, kT=KT,
    )

    # ---- M0: shooting vs exact ------------------------------------------------
    print("M0  shooting committor vs exact (400 shots per point)")
    print("    point            exact   shooting  +-      z      steps")
    engine = ToyLangevinEngine(dt=DT, kT=KT)
    points = [[-1.0, 0.3], [0.9, 0.2], [-0.9, -1.3], [0.0, 0.0], [-0.3, 1.0], [0.5, -0.9]]
    zs, shoot_steps = [], []
    for point in points:
        res = shooting_committor(engine, engine.create_state(point), BASINS, xy,
                                 n_shots=400, max_steps=200_000, check_every=5, seed=1)
        exact = float(reference(np.array([point]))[0])
        z = (res.committor - exact) / res.stderr
        zs.append(z)
        shoot_steps.append(res.total_steps / 400)
        print(f"    {str(point):16s} {exact:.3f}   {res.committor:.3f}     {res.stderr:.3f}  {z:+.2f}  {res.total_steps}")
    out["M0"] = dict(max_abs_z=float(np.max(np.abs(zs))), passed=bool(np.max(np.abs(zs)) < 4.0),
                     mean_steps_per_shot=float(np.mean(shoot_steps)))
    print(f"    max |z| = {out['M0']['max_abs_z']:.2f}  ->  {'PASS' if out['M0']['passed'] else 'FAIL'} (gate: < 4)")
    steps_per_shot = out["M0"]["mean_steps_per_shot"]

    # ---- held-out test set: starts of independent discovery runs ---------------
    test = collect(range(500, 508), tau1=20)
    test_points = test.starts[test.start_label == LABEL_NONE]
    exact = reference(test_points)
    da = np.linalg.norm(test_points - A, axis=1)
    db = np.linalg.norm(test_points - B, axis=1)
    geometric = committor_scores(da / (da + db), exact)
    out["geometric_cv"] = geometric

    def evaluate(data, label):
        scores = [committor_scores(train_committor(data, steps=3000, seed=s).predict(test_points), exact)
                  for s in range(args.train_seeds)]
        row = dict(
            swarms=len(data), md_steps=data.total_steps,
            pearson=float(np.mean([s["pearson"] for s in scores])),
            rmse=float(np.mean([s["rmse"] for s in scores])),
            pearson_transition=float(np.mean([s["pearson_transition"] for s in scores])),
            rmse_transition=float(np.mean([s["rmse_transition"] for s in scores])),
            rmse_transition_worst=float(np.max([s["rmse_transition"] for s in scores])),
        )
        print(f"    {label:10s} {row['swarms']:6d} {row['md_steps']:9d}   {row['pearson']:.3f}  {row['rmse']:.3f}"
              f"    {row['pearson_transition']:.3f}  {row['rmse_transition']:.3f} ({row['rmse_transition_worst']:.3f})")
        return row

    header = ("    setting    swarms  md steps   r      RMSE     r(TS)  RMSE(TS) (worst)\n"
              "    " + "-" * 74)

    # ---- M1: data efficiency ---------------------------------------------------
    print(f"\nM1  committor from discovery swarms, tau1 = 20 steps "
          f"({len(test_points)} held-out points, {geometric['n_transition']} in the transition region)")
    print(header)
    out["M1_data"] = {}
    for n_runs in (1, 2, 4, 8, 16):
        out["M1_data"][n_runs] = evaluate(collect(range(n_runs), tau1=20), f"{n_runs} run(s)x2")
    print(f"    geometric CV {'':20s} {geometric['pearson']:.3f}  {geometric['rmse']:.3f}"
          f"    {geometric['pearson_transition']:.3f}  {geometric['rmse_transition']:.3f}")
    gate = out["M1_data"][8]
    passed = gate["pearson_transition"] > 0.95 and gate["rmse_transition"] < 0.10
    out["M1"] = dict(passed=bool(passed), **gate)
    print(f"    gate (8 runs each way): r(TS) > 0.95 and RMSE(TS) < 0.10  ->  {'PASS' if passed else 'FAIL'}")

    # What the same accuracy costs by shooting: a binomial standard error equal to
    # the model's transition-region RMSE needs 0.25 / rmse^2 shots per point (q = 1/2).
    shots = 0.25 / gate["rmse_transition"] ** 2
    cost = shots * steps_per_shot * geometric["n_transition"]
    out["M1"]["shooting_equivalent_steps"] = float(cost)
    print(f"    shooting to the same error on the {geometric['n_transition']} transition points: "
          f"~{cost:,.0f} steps ({cost / gate['md_steps']:.0f}x the swarm data)")

    # ---- lag dependence --------------------------------------------------------
    print("\n    dependence on the swarm length tau1 (8 runs each way)")
    print(header)
    out["M1_lag"] = {}
    for tau1 in (2, 5, 20, 50):
        out["M1_lag"][tau1] = evaluate(collect(range(8), tau1=tau1), f"tau1={tau1}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
    return 0 if (out["M0"]["passed"] and out["M1"]["passed"]) else 1


if __name__ == "__main__":
    t0 = time.time()
    code = main()
    print(f"\ndone in {time.time() - t0:.0f} s")
    raise SystemExit(code)
