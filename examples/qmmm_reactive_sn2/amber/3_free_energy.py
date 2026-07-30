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
    p.add_argument("--burn-in", type=float, default=0.3,
                   help="fraction (<1) or count of leading iterations to discard")
    p.add_argument("--workers", type=int, default=8,
                   help="concurrent sander processes; one CPU core each")
    p.add_argument("--outdir", type=Path, default=HERE / "results" / "free_energy")
    p.add_argument("--sander", default="sander")
    args = p.parse_args()

    from pathgennie.backends.amber.engine import CoreAmberEngine
    from pathgennie.core.parallel import ThreadDevicePool
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
        # Seeding puts one walker per bin -- a uniform distribution, which is the
        # opposite of the Boltzmann one being estimated. Averaging that transient
        # in flattens the profile and biases the barrier low.
        burn_in=args.burn_in,
        executor=ThreadDevicePool(devices=None, workers_per_device=args.workers),
    )
    result = stage.run(ensemble, engine)

    args.outdir.mkdir(parents=True, exist_ok=True)
    fe = np.asarray(result.free_energy, dtype=float)
    centers = np.asarray(result.metadata["bin_centers"], dtype=float)
    trace = np.asarray(result.metadata["bin_weight_trace"], dtype=float)
    n_burn = int(result.metadata["burn_in"])
    np.savez(args.outdir / "fes_along_s.npz", free_energy=fe, s=centers,
             bin_weight_trace=trace, burn_in=n_burn, kT=KB_KCAL * args.temperature)

    kT = KB_KCAL * args.temperature

    def profile(window: np.ndarray) -> np.ndarray:
        """Free energy from an arbitrary slice of iterations."""
        w = window.sum(axis=0)
        tot = w.sum()
        with np.errstate(divide="ignore"):
            f = -kT * np.log(w / tot if tot > 0 else w)
        good = f[np.isfinite(f)]
        return f - good.min() if good.size else f

    ok = np.isfinite(fe)
    print(f"\n{'s':>7} {'F (kcal/mol)':>14}")
    for s_val, f_val in zip(centers[ok], fe[ok]):
        print(f"{s_val:>7.3f} {f_val:>14.2f}")

    # WE climbs a barrier by a ratchet: a walker that strays into the next bin is
    # split there, so its descendants are more numerous and probability creeps
    # upward. Emptying a bin is normal -- the ratchet refills it. What matters is
    # whether the two sides ever MEET. If a run of bins is never entered from
    # either side after burn-in, the ratchet stalled and everything above the
    # highest reached bin is unmeasured.
    occupied = trace > 0
    live = np.flatnonzero(occupied[n_burn:].any(axis=0))   # reached post-burn-in
    unreached = [b for b in range(trace.shape[1]) if not occupied[:, b].any()]
    gap: list[int] = []
    if live.size:
        breaks = np.flatnonzero(np.diff(live) > 1)
        if breaks.size:                     # widest unbridged stretch
            k = int(breaks[np.argmax(np.diff(live)[breaks])])
            gap = list(range(int(live[k]) + 1, int(live[k + 1])))

    if gap:
        lo, hi = gap[0] - 1, gap[-1] + 1
        refills = {b: int(np.sum(np.diff(np.flatnonzero(occupied[:, b])) > 1))
                   for b in (lo, hi) if occupied[:, b].any()}
        print(f"\nThe WE ratchet stalled: bins s = {centers[gap[0]]:.3f}–{centers[gap[-1]]:.3f} "
              f"({len(gap)} bins) were never\nentered from either side after burn-in.")
        print(f"  reached from below: up to s = {centers[lo]:.3f}"
              f"   (that bin emptied and refilled {refills.get(lo, 0)}× — the ratchet works)")
        print(f"  reached from above: down to s = {centers[hi]:.3f}"
              f"   (refilled {refills.get(hi, 0)}×)")
        print("  So this is a STALL, not an inability to repopulate: the climb rate falls\n"
              "  off exponentially with height and ran out of budget. The reported maximum\n"
              "  is a LOWER BOUND on the barrier.\n"
              "  Remedies: more walkers per bin, finer bins across the stalled region, a\n"
              "  longer tau, or a bias along s (umbrella/OPES).")
    if unreached:
        print(f"\n{len(unreached)} bin(s) never reached at all -- widen the seeding or "
              "lengthen the run.")

    # Adjacent-bin occupancy ratios should follow exp(-dF/kT) wherever sampling is
    # adequate. Printing them shows whether WE is behaving correctly (it usually
    # is) as distinct from having enough budget (often it does not).
    w = trace[n_burn:].sum(axis=0)
    live_pairs = [(b, b + 1) for b in range(len(w) - 1) if w[b] > 0 and w[b + 1] > 0]
    if live_pairs:
        print("\nper-bin occupancy ratios (should track exp(-dF/kT) where sampled)")
        for a, b in live_pairs:
            print(f"  s {centers[a]:.3f} -> {centers[b]:.3f}   ratio {w[b] / w[a]:8.4f}"
                  f"   implied dF {-kT * np.log(w[b] / w[a]):+6.2f} kcal/mol")

    # Convergence evidence, not a convergence claim: re-estimate over the second
    # half, third quarter, and final quarter. If those agree the profile has
    # stopped moving; if they do not, the run is too short and says so.
    n_it = trace.shape[0]
    windows = {
        "2nd half   ": trace[n_it // 2:],
        "3rd quarter": trace[n_it // 2: 3 * n_it // 4],
        "4th quarter": trace[3 * n_it // 4:],
    }
    print(f"\nconvergence check (burn-in used: {n_burn}/{n_it} iterations)")
    print(f"  {'window':<12} {'barrier':>9}  max |dF| vs full estimate")
    drifts = []
    for name, win in windows.items():
        if win.size == 0:
            continue
        f = profile(win)
        both = np.isfinite(f) & ok
        drift = float(np.abs(f[both] - fe[both]).max()) if both.any() else float("nan")
        drifts.append(drift)
        bar = float(f[np.isfinite(f)].max()) if np.isfinite(f).any() else float("nan")
        print(f"  {name:<12} {bar:>9.2f}  {drift:>9.2f} kcal/mol")

    # This reaction is an *identity* substitution, so F(s) must be exactly
    # symmetric about s = 0.5. Any asymmetry is pure sampling error -- a free,
    # assumption-light error bar that most systems do not give you.
    n_bin = len(centers)
    print("\nsymmetry check (identity reaction: F(s) must mirror about s = 0.5)")
    asym = []
    for lo in range(n_bin // 2):
        hi = n_bin - 1 - lo
        if ok[lo] and ok[hi]:
            d = abs(fe[lo] - fe[hi])
            asym.append(d)
            print(f"  s={centers[lo]:.3f} {fe[lo]:6.2f}  <->  "
                  f"s={centers[hi]:.3f} {fe[hi]:6.2f}   |diff| = {d:.2f}")
    if asym:
        print(f"  worst asymmetry {max(asym):.2f} kcal/mol -- a lower bound on the "
              "error in this profile")

    worst = max([d for d in drifts if np.isfinite(d)], default=float("nan"))
    label = "highest sampled point (LOWER BOUND)" if gap else "apparent barrier"
    print(f"\n{label}: {fe[ok].max():.2f} kcal/mol "
          f"(WE free energies are already shifted to min = 0)")
    if np.isfinite(worst) and worst < 0.5 and ok.all() and not gap:
        print(f"Windows agree to {worst:.2f} kcal/mol and every bin stayed populated: "
              "the profile has stopped moving.")
    else:
        reasons = []
        if not (np.isfinite(worst) and worst < 0.5):
            reasons.append(f"windows disagree by up to {worst:.2f} kcal/mol")
        if gap:
            reasons.append(f"the ratchet stalled across {len(gap)} bin(s); barrier unsampled")
        if unreached:
            reasons.append(f"{len(unreached)} bin(s) never reached")
        print("NOT CONVERGED: " + "; ".join(reasons)
              + ".\nDo not quote this number.")
    print(f"written to {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
