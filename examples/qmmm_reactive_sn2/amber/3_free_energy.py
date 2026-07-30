#!/usr/bin/env python
"""Stage 3 -- free energy along the refined PathCV.

A refined path says *where* the reaction goes; it says nothing about the barrier.
This stage runs Weighted Ensemble binned on the path progress coordinate ``s``, so
the profile is expressed in the coordinate the path itself defines rather than in
a hand-picked distance.

WE suits this because it is unbiased: walkers propagate under the plain QM/MM
Hamiltonian and only their statistical *weights* are split and merged, so the
profile needs no reweighting and no biasing-potential book-keeping.

    python 3_free_energy.py --refined results/refinement/refined_path.npy

The default budget is a demonstration, not a converged barrier -- see the
walltime note in README.md before quoting numbers from it.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Same feature space as stages 2 and 4 -- the PathCV loaded below was built in it.
from sn2_cv import path_features as feature_fn  # noqa: E402

KB_KCAL = 0.0019872041  # kcal/mol/K


def seed_frames(ensemble: Path, n_bins: int, s_of) -> np.ndarray:
    """Pick one frame per ``s`` bin from the stage-1 trajectories.

    Seeding every bin matters: WE only splits walkers that reach a bin, so a run
    started entirely in the reactant well spends its whole budget crawling out.
    """
    from pathgennie.backends.amber.utils import read_native_trajectory

    pool = []
    for case in sorted(ensemble.glob("seed_*")):
        nc = glob.glob(str(case / "pathgennie_sn2" / "output" / "*.nc"))
        top = case / "sn2.prmtop"
        if nc and top.exists():
            traj = np.asarray(read_native_trajectory(nc[0], str(top)))
            if traj.ndim == 3:
                pool.extend(traj)
    if not pool:
        raise SystemExit(f"no stage-1 trajectories found under {ensemble}")

    s_vals = np.array([s_of(f) for f in pool])
    picked = []
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        idx = np.flatnonzero((s_vals >= lo) & (s_vals < hi))
        if len(idx):
            picked.append(pool[idx[len(idx) // 2]])
    return np.asarray(picked)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refined", type=Path,
                   default=HERE / "results" / "refinement" / "refined_path.npy")
    p.add_argument("--ensemble", type=Path, default=HERE / "ensemble")
    p.add_argument("--iterations", type=int, default=24)
    p.add_argument("--walkers-per-bin", type=int, default=8)
    p.add_argument("--bins", type=int, default=16)
    p.add_argument("--tau-steps", type=int, default=100)   # 50 fs at dt = 0.5 fs
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--outdir", type=Path, default=HERE / "results" / "free_energy")
    p.add_argument("--sander", default="sander")
    args = p.parse_args()

    from pathgennie.backends.amber.engine import CoreAmberEngine
    from pathgennie.sampling import WeightedEnsembleStage, build_path_ensemble
    from pathrefinement.pathcv import PathCV

    seed_dir = next((c for c in sorted(args.ensemble.glob("seed_*"))
                     if (c / "sn2.prmtop").exists()), None)
    if seed_dir is None:
        raise SystemExit(f"no seed directory with sn2.prmtop under {args.ensemble}")

    refined = np.load(args.refined)                       # (N, 2)
    path_cv = PathCV(refined[:, np.newaxis, :], enforce_equidistance=False,
                     normalize_output=True)

    def s_of(coords) -> float:
        s, _z = path_cv.compute(np.atleast_2d(feature_fn(coords)))
        return float(s)

    # The same QM/MM Hamiltonian as stages 1 and 2 -- a free energy computed at a
    # different level of theory than the path is not a profile *of* that path.
    engine = CoreAmberEngine(
        topology=seed_dir / "sn2.prmtop",
        executable=args.sander,
        scratch_dir=args.outdir / "scratch",
        temperature=args.temperature,
        mdin_controls=dict(dt=0.0005, ntc=1, ntf=1, ntb=0, cut=999.0,
                           ntpr=100000, ntwx=0, ntwr=100000, ntxo=1, ifqnt=1),
        extra_mdin_text="&qmmm\n  qmmask=':1-2',\n  qmcharge=-1,\n  qm_theory='DFTB3',\n/\n",
    )

    frames = seed_frames(args.ensemble, args.bins, s_of)
    handles = [engine.create_handle(f) for f in frames]
    print(f"seeded {len(handles)} walkers spanning s = "
          f"{min(s_of(f) for f in frames):.2f} to {max(s_of(f) for f in frames):.2f}")

    ensemble = build_path_ensemble(
        frames=frames,
        metrics=np.array([s_of(f) for f in frames]),
        handles=handles,
    )

    stage = WeightedEnsembleStage(
        cv_fn=s_of,
        tau_steps=args.tau_steps,
        n_iterations=args.iterations,
        n_bins=args.bins,
        bin_edges=np.linspace(0.0, 1.0, args.bins + 1),
        target_count=args.walkers_per_bin,
        seed=7,
        kT=KB_KCAL * args.temperature,
    )
    result = stage.run(ensemble, engine)

    args.outdir.mkdir(parents=True, exist_ok=True)
    fe = np.asarray(result.free_energy, dtype=float)
    centers = np.asarray(result.metadata["bin_centers"], dtype=float)
    np.savez(args.outdir / "fes_along_s.npz", free_energy=fe, s=centers)

    ok = np.isfinite(fe)
    print(f"\n{'s':>7} {'F (kcal/mol)':>14}")
    for s_val, f_val in zip(centers[ok], fe[ok]):
        print(f"{s_val:>7.3f} {f_val:>14.2f}")
    if ok.sum() < len(fe):
        print(f"\n{len(fe) - int(ok.sum())} of {len(fe)} bins never visited -- "
              "increase --iterations or --walkers-per-bin.")
    print(f"\napparent barrier: {fe[ok].max():.2f} kcal/mol "
          f"(WE free energies are already shifted to min = 0)")
    print("Demonstration budget -- verify convergence before quoting this.")
    print(f"written to {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
