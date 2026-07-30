#!/usr/bin/env python
"""Stage 2 -- refine the raw QM/MM paths into a single smooth PathCV.

The seeds from stage 1 are noisy, thermally scattered realisations of the same
mechanism. This stage alternates short QM/MM exploration around the current path
with a principal-curve/neural consensus fit until the path stops moving.

The exploration is driven through :class:`pathrefinement.samplers.EngineSampler`,
which speaks the core ``Engine`` protocol rather than OpenMM. That is what makes
QM/MM refinement possible at all -- AMBER is the only backend that can run a QM
Hamiltonian, and the refiner's built-in walker is OpenMM-only.

Feature space is the 2-D pair (d(C-Cl_attacking), d(C-Cl_leaving)) -- the same
plane the mechanism is usually drawn in, so the refined path can be plotted
directly (stage 4). It is defined once in ``sn2_cv.py`` and imported here, so the
stages cannot disagree on component order.

    python 2_refine_pathcv.py --ensemble ensemble --iterations 6
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# One definition of the feature space, shared with stages 3 and 4.
from sn2_cv import path_features as features  # noqa: E402
from sn2_cv import path_features_traj  # noqa: E402

N_NODES = 20


def arclength_resample(path: np.ndarray, n: int) -> np.ndarray:
    """Re-space a polyline to ``n`` equidistant nodes along its own arc length."""
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if d[-1] <= 0:
        return np.repeat(path[:1], n, axis=0)
    t = np.linspace(0.0, d[-1], n)
    return np.column_stack([np.interp(t, d, path[:, k]) for k in range(path.shape[1])])


def first_seed(ensemble: Path) -> Path:
    """Any seed directory will do -- they share a topology and starting structure."""
    for case in sorted(ensemble.glob("seed_*")):
        if (case / "sn2.prmtop").exists() and (case / "sn2.rst7").exists():
            return case
    raise SystemExit(f"no seed directory with sn2.prmtop + sn2.rst7 under {ensemble}")


def load_ensemble(ensemble: Path) -> list[np.ndarray]:
    """Read every converged seed trajectory and project it into feature space."""
    from pathgennie.backends.amber.utils import read_native_trajectory

    paths = []
    for case in sorted(ensemble.glob("seed_*")):
        nc = glob.glob(str(case / "pathgennie_sn2" / "output" / "*.nc"))
        top = case / "sn2.prmtop"
        if not nc or not top.exists():
            continue
        traj = np.asarray(read_native_trajectory(nc[0], str(top)))
        if traj.ndim != 3 or len(traj) < 2:
            continue
        paths.append(path_features_traj(traj))
    return paths


def consensus(paths: list[np.ndarray], n: int = N_NODES) -> np.ndarray:
    """Average the seeds after putting them on a common arc-length parameter.

    Averaging raw frames would be meaningless -- the seeds have different lengths
    and different dwell times, so frame *k* of one is not frame *k* of another.
    """
    return np.mean([arclength_resample(p, n) for p in paths], axis=0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ensemble", type=Path, default=HERE / "ensemble")
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--walkers", type=int, default=4)
    p.add_argument("--outdir", type=Path, default=HERE / "results" / "refinement")
    p.add_argument("--sander", default="sander")
    args = p.parse_args()

    paths = load_ensemble(args.ensemble)
    if len(paths) < 2:
        print(f"need at least 2 converged seeds in {args.ensemble}; found {len(paths)}")
        return 1
    initial = consensus(paths)
    print(f"seeds: {len(paths)} -> initial path {initial.shape}")

    from pathgennie.backends.amber.engine import CoreAmberEngine
    from pathrefinement.refiner import PathRefiner, PathRefinementConfig
    from pathrefinement.samplers import EngineSampler

    seed_dir = first_seed(args.ensemble)

    # Same QM/MM Hamiltonian as stage 1 -- refinement must not silently change
    # the level of theory the path was discovered at.
    engine = CoreAmberEngine(
        topology=seed_dir / "sn2.prmtop",
        executable=args.sander,
        scratch_dir=args.outdir / "scratch",
        temperature=300.0,
        mdin_controls=dict(dt=0.0005, ntc=1, ntf=1, ntb=0, cut=999.0,
                           ntpr=100000, ntwx=0, ntwr=100000, ntxo=1, ifqnt=1),
        extra_mdin_text="&qmmm\n  qmmask=':1-2',\n  qmcharge=-1,\n  qm_theory='DFTB3',\n/\n",
    )

    sampler = EngineSampler(
        engine,
        initial_handle=str((seed_dir / "sn2.rst7").resolve()),
        feature_fn=features,
        tau1=10, tau2=10,        # short segments: the barrier is enthalpic
        max_trial=30, max_cycle=300,
        sigma=0.05, tol=0.05,
    )

    cfg = PathRefinementConfig(
        n_iterations=args.iterations,
        n_trajectories=args.walkers,
        nn_epochs=1500,
        device="cpu",
        seed=42,
        verbosity=1,
    )
    result = PathRefiner(potential=None, config=cfg, sampler=sampler).refine(initial)

    args.outdir.mkdir(parents=True, exist_ok=True)
    result.save(str(args.outdir))
    np.save(args.outdir / "seed_paths.npy", np.array([arclength_resample(p, N_NODES)
                                                      for p in paths]))
    print(f"\nconverged={result.converged} after {result.n_iterations_run} iterations")
    print(f"written to {args.outdir}")
    print("next: python 3_free_energy.py --refined", args.outdir / "refined_path.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
