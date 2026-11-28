"""Committor milestone M3b: the closed loop on alanine dipeptide (OpenMM).

    python m3b_loop.py --reference M2_OUT --out OUT --seed 0

The loop is bootstrapped with a **single dihedral** (phi) as its progress
variable and scored each round against the shooting reference of the M2 study
(``m2_study.py`` stages ``discover``, ``reference``, ``enrich``). The reference is
read for reporting only; nothing in the loop sees it.

Protocol and gate were fixed before the first run and are not to be tuned
against the reference:

  steering   tau1 = 10 fs, tau2 = 40 fs, 8 trials; 16 episodes per round of at
             most 80 cycles; Boltzmann selection at sigma = 0.3
  learning   150 configurations per round, spread over the committor estimate;
             8 trials each of at most 300 fs, stopped at basin entry (20 fs checks)
  model      ensemble of 4 networks (128, 128); 6000 steps, then 2000 per round;
             15 % of the swarms held out for early stopping
  rounds     6

  gate       after the last round, in every seed: r > 0.9 and RMSE(TS) < 0.15
             against shooting, with less MD than the best M2 protocol
             (6,058,420 steps)

The two starting structures (one per basin) are taken from the M2 discovery
stage; only their coordinates are used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import adp  # noqa: E402
import m2_study  # noqa: E402

PROTOCOL = dict(
    tau1=10, tau2=40, n_trials=8, episodes_per_round=16, max_cycles=80, sigma=0.3,
    bootstrap_max_cycles=300, bootstrap_sigma=0.2,
    learn_starts=150, learn_trials=8, learn_steps=300, learn_check_every=20,
    n_models=4, hidden=(128, 128), train_steps=6000, refresh_steps=2000, val_fraction=0.15,
)
ROUNDS = 6
M2_BEST_MD_STEPS = 6_058_420


def phi_progress(target: str):
    """The deliberately poor bootstrap CV: phi alone."""
    from pathgennie.core.progress import TargetMetric

    centre = adp.C7AX if target == "B" else adp.C7EQ
    return TargetMetric(lambda c: np.array([adp.phi_psi(c)[0]]), target_cv=np.array([centre[0]]),
                        periodic=[360.0])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True, help="output directory of m2_study.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--smoke", action="store_true", help="tiny settings, to check the plumbing only")
    parser.add_argument("--psf", default=None)
    parser.add_argument("--params", nargs="*", default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    from pathgennie.backends.openmm.engine import OpenMMEngine
    from pathgennie.committor.loop import CommittorLoop, LoopConfig
    from pathgennie.committor.train import committor_scores

    torch.set_num_threads(1)
    protocol = dict(PROTOCOL)
    if args.smoke:
        protocol.update(episodes_per_round=2, max_cycles=10, bootstrap_max_cycles=40, learn_starts=6,
                        learn_steps=60, train_steps=300, refresh_steps=100, n_models=2)

    ref = m2_study._load_reference(args.reference)
    committed = ref["n_a"] + ref["n_b"]
    q_ref = ref["n_b"] / committed
    features = np.stack([adp.heavy_atom_distances(c) for c in ref["coords"]])
    disc = np.load(args.reference / "discover.npz")

    sim, _, label = adp.build_simulation(psf=args.psf, params=args.params)
    engine = OpenMMEngine(sim, 300.0)
    curve = []
    t0 = time.time()

    def score(loop, report):
        s = committor_scores(loop.model.predict(features), q_ref)
        data = loop.recorder.dataset()
        row = dict(round=report.round, md_steps=report.md_steps, swarms=report.swarms,
                   learning_swarms=len(loop._learn_rows),
                   episodes_end_a=report.end_in_a, episodes_end_b=report.end_in_b,
                   episodes_unfinished=report.unfinished,
                   pearson=s["pearson"], rmse=s["rmse"],
                   pearson_ts=s["pearson_transition"], rmse_ts=s["rmse_transition"],
                   n_ts=s["n_transition"], wall_s=round(time.time() - t0),
                   train_evals=[len(m.val_history) for m in loop.model.models])
        curve.append(row)
        print(f"seed {args.seed} round {row['round']}: {row['md_steps']:>9,d} MD steps, "
              f"{row['swarms']:5d} swarms ({row['learning_swarms']} learning), episodes A/B/open "
              f"{row['episodes_end_a']}/{row['episodes_end_b']}/{row['episodes_unfinished']}, "
              f"r {row['pearson']:.3f} RMSE {row['rmse']:.3f} | r(TS) {row['pearson_ts']:.3f} "
              f"RMSE(TS) {row['rmse_ts']:.3f} | {row['wall_s']} s", flush=True)
        with open(args.out / f"m3b_seed{args.seed}.json", "w") as fh:
            json.dump(dict(force_field=label, seed=args.seed, protocol=protocol, smoke=args.smoke,
                           curve=curve), fh, indent=2)
        return row

    loop = CommittorLoop(
        engine, adp.basins(), adp.phi_psi, adp.heavy_atom_distances,
        start_a=engine.create_handle(disc["c7eq"]), start_b=engine.create_handle(disc["c7ax"]),
        bootstrap_progress=phi_progress, config=LoopConfig(seed=args.seed, **protocol),
        on_round=score,
    )
    loop.run(args.rounds)
    if not args.smoke:
        loop.recorder.save(args.out / f"m3b_seed{args.seed}_swarms.h5")
        final = curve[-1]
        passed = final["pearson"] > 0.9 and final["rmse_ts"] < 0.15 and final["md_steps"] < M2_BEST_MD_STEPS
        print(f"seed {args.seed}: gate {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
