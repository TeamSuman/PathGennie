"""Committor milestone M2: alanine dipeptide in vacuum (OpenMM).

    python m2_study.py discover  --out OUT     # PathGennie runs both ways (train + test pools)
    python m2_study.py reference --out OUT     # shooting committors at held-out configurations
    python m2_study.py swarms    --out OUT     # selection-free swarms at fixed starts, several lags
    python m2_study.py analyze   --out OUT     # train, score, write OUT/m2_summary.json

Every stage is resumable (it skips work whose output exists) and parallel over
``--workers`` processes, each with its own single-threaded OpenMM Context, so the
same script runs on a laptop or as a cluster job.

Design
------
*Reference.* Held-out configurations come from discovery runs that share no
seed with the training runs. Each gets ``--shots1`` shots; those with a mixed
outcome (the transition region) are topped up to ``--shots2``, so the precision
is spent where the committor is not trivially 0 or 1.

*Lag scan.* Accuracy against swarm length is measured with the **same** start
configurations and the same number of trials at every lag, so coverage is held
fixed and only the lag changes (the toy-model scan in benchmarks/ changed both).

*Protocols scored.*
  ``discovery``  the swarms recorded during the training discovery runs
                 themselves (tau1 = 20 fs, selection on): the M1 protocol.
  ``lag=N``      selection-free swarms of N steps at the fixed starts.

Langevin noise is not re-seeded per segment (that needs a Context reinitialise),
so repeated runs agree statistically, not bit for bit.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import adp  # noqa: E402

LAGS = (2, 5, 20, 50, 200, 500)          # fs (1 fs time step)
TAU1, TAU2, N_TRIALS = 20, 40, 16         # discovery runs
SIGMA = 0.2                               # softmax temperature of the discovery runs
_ENGINE = None
_FF = None


# ---- per-process engine ---------------------------------------------------------

def _init_worker(psf, params):
    global _ENGINE, _FF
    from pathgennie.backends.openmm.engine import OpenMMEngine

    sim, positions, _FF = adp.build_simulation(psf=psf, params=params)
    _ENGINE = OpenMMEngine(sim, 300.0)
    _ENGINE._start = _ENGINE.create_state(positions)


def _pool(args):
    return mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                        initargs=(args.psf, args.params))


# ---- stage: discover ----------------------------------------------------------------

def _discover_one(job):
    """One PathGennie run ``src -> dst`` from given coordinates; returns its records."""
    from pathgennie.committor import SwarmRecorder
    from pathgennie.core.driver import PathGennieDriver

    seed, target, start_coords = job
    basins = adp.basins()
    label = 1 if target == "B" else 0
    rec = SwarmRecorder(featurize=adp.heavy_atom_distances, basins=basins,
                        basin_cv=adp.phi_psi, keep_coords=True)
    handle = _ENGINE.clone_anchor(_ENGINE._start) if start_coords is None \
        else _ENGINE.create_handle(start_coords)
    driver = PathGennieDriver(
        _ENGINE, adp.dihedral_progress(target),
        lambda c: basins.label(adp.phi_psi(c)) == label,
        sigma=SIGMA, seed=seed, verbosity=0, swarm_recorder=rec,
    )
    traj, metrics = driver.run(handle, tau1=TAU1, tau2=TAU2, max_trial=N_TRIALS,
                               max_cycle=1500, save_freq=1)
    reached = bool(basins.label(adp.phi_psi(traj[-1])) == label)
    data = rec.dataset()
    for stale in {handle, driver.last_anchor}:
        _ENGINE.release(stale)
    return dict(seed=seed, target=target, reached=reached, cycles=len(metrics),
                final=np.asarray(traj[-1]), coords=np.stack(rec.coords_starts),
                runner_steps=len(rec.runner_starts) * TAU2,
                data={name: getattr(data, name) for name in data._FIELDS})


def stage_discover(args):
    from pathgennie.committor import SwarmDataset

    out = args.out
    if (out / "discover.npz").exists():
        print("discover: done already")
        return
    t0 = time.time()
    with _pool(args) as pool:
        # A structure in each basin: relax into C7eq, then one run to C7ax.
        to_a = pool.map(_discover_one, [(900, "A", None)])[0]
        to_b = pool.map(_discover_one, [(901, "B", to_a["final"])])[0]
        if not (to_a["reached"] and to_b["reached"]):
            raise RuntimeError(
                "could not prepare structures in both basins: "
                f"to C7eq reached={to_a['reached']} after {to_a['cycles']} cycles, "
                f"to C7ax reached={to_b['reached']} after {to_b['cycles']} cycles "
                f"(ended at phi, psi = {adp.phi_psi(to_b['final']).round(1)})")
        c7eq, c7ax = to_a["final"], to_b["final"]
        jobs = []
        for pool_name, seeds in (("train", range(args.train_runs)),
                                 ("test", range(1000, 1000 + args.test_runs))):
            for s in seeds:
                jobs.append((pool_name, (2 * s, "B", c7eq)))
                jobs.append((pool_name, (2 * s + 1, "A", c7ax)))
        results = pool.map(_discover_one, [j for _, j in jobs])
    save = dict(c7eq=c7eq, c7ax=c7ax)
    for pool_name in ("train", "test"):
        runs = [r for (name, _), r in zip(jobs, results) if name == pool_name]
        data = SwarmDataset.concatenate([SwarmDataset(**r["data"]) for r in runs])
        coords = np.concatenate([r["coords"] for r in runs])
        save[f"{pool_name}_coords"] = coords
        save[f"{pool_name}_start_label"] = data.start_label
        save[f"{pool_name}_md_steps"] = data.total_steps + sum(r["runner_steps"] for r in runs)
        save[f"{pool_name}_reached"] = np.array([r["reached"] for r in runs])
        save[f"{pool_name}_cycles"] = np.array([r["cycles"] for r in runs])
        if pool_name == "train":
            data.save(out / "swarms_discovery.h5")
        print(f"discover/{pool_name}: {len(runs)} runs, {int(save[f'{pool_name}_reached'].sum())} reached "
              f"the far basin, median {int(np.median(save[f'{pool_name}_cycles']))} cycles, "
              f"{len(data)} swarms, {int(save[f'{pool_name}_md_steps'])} MD steps")
    np.savez_compressed(out / "discover.npz", **save)
    print(f"discover: {time.time() - t0:.0f} s")


# ---- stage: reference ------------------------------------------------------------------

def _shoot_one(job):
    from pathgennie.committor import shooting_committor

    index, coords, n_shots, seed, max_steps = job
    handle = _ENGINE.create_handle(coords)
    res = shooting_committor(_ENGINE, handle, adp.basins(), adp.phi_psi, n_shots=n_shots,
                             max_steps=max_steps, check_every=20, seed=seed)
    _ENGINE.release(handle)
    return index, res.n_a, res.n_b, res.n_open, res.total_steps


def stage_reference(args):
    out = args.out
    disc = np.load(out / "discover.npz")
    path = out / "reference.npz"
    rng = np.random.default_rng(12345)
    outside = np.flatnonzero(disc["test_start_label"] == -1)
    chosen = np.sort(rng.choice(outside, size=min(args.configs, outside.size), replace=False))
    coords = disc["test_coords"][chosen]
    n = len(coords)

    if path.exists():
        state = dict(np.load(path))
    else:
        state = dict(coords=coords, n_a=np.zeros(n, int), n_b=np.zeros(n, int),
                     n_open=np.zeros(n, int), steps=np.zeros(n, int), stage=np.zeros(n, int))

    def run(stage, indices, n_shots):
        todo = [i for i in indices if state["stage"][i] < stage]
        if not todo:
            return
        t0 = time.time()
        jobs = [(i, coords[i], n_shots, 7_000_000 * stage + i, args.max_steps) for i in todo]
        with _pool(args) as pool:
            for k, (i, n_a, n_b, n_open, steps) in enumerate(pool.imap_unordered(_shoot_one, jobs, chunksize=1)):
                state["n_a"][i] += n_a
                state["n_b"][i] += n_b
                state["n_open"][i] += n_open
                state["steps"][i] += steps
                state["stage"][i] = stage
                if (k + 1) % 20 == 0 or k + 1 == len(jobs):
                    np.savez_compressed(path, **state)
                    print(f"reference stage {stage}: {k + 1}/{len(jobs)} configurations, "
                          f"{time.time() - t0:.0f} s", flush=True)

    run(1, range(n), args.shots1)
    mixed = [i for i in range(n) if state["n_a"][i] > 0 and state["n_b"][i] > 0]
    run(2, mixed, args.shots2 - args.shots1)
    np.savez_compressed(path, **state)
    committed = state["n_a"] + state["n_b"]
    q = state["n_b"] / np.maximum(committed, 1)
    print(f"reference: {n} configurations, {len(mixed)} with mixed outcomes, "
          f"{int((state['n_open'] > 0).sum())} with uncommitted shots, "
          f"{int(state['steps'].sum())} MD steps; q in (0.05, 0.95): {int(((q > 0.05) & (q < 0.95)).sum())}")


# ---- stage: swarms -------------------------------------------------------------------------

def _swarm_chunk(job):
    from pathgennie.committor import SwarmRecorder
    from pathgennie.committor.records import run_swarms

    coords_chunk, lag, seed = job
    rec = SwarmRecorder(featurize=adp.heavy_atom_distances, basins=adp.basins(), basin_cv=adp.phi_psi)
    handles = [_ENGINE.create_handle(c) for c in coords_chunk]
    run_swarms(_ENGINE, handles, rec, n_trials=N_TRIALS, n_steps=lag, seed=seed)
    for h in handles:
        _ENGINE.release(h)
    data = rec.dataset()
    return {name: getattr(data, name) for name in data._FIELDS}


def stage_swarms(args):
    from pathgennie.committor import SwarmDataset

    out = args.out
    disc = np.load(out / "discover.npz")
    rng = np.random.default_rng(777)
    outside = np.flatnonzero(disc["train_start_label"] == -1)
    chosen = np.sort(rng.choice(outside, size=min(args.starts, outside.size), replace=False))
    coords = disc["train_coords"][chosen]
    np.save(out / "swarm_starts.npy", coords)
    chunks = np.array_split(coords, max(args.workers * 4, 1))
    with _pool(args) as pool:
        for lag in LAGS:
            target = out / f"swarms_lag{lag}.h5"
            if target.exists():
                continue
            t0 = time.time()
            parts = pool.map(_swarm_chunk, [(c, lag, 31 * lag + k) for k, c in enumerate(chunks) if len(c)])
            data = SwarmDataset.concatenate([SwarmDataset(**p) for p in parts])
            data.save(target)
            print(f"swarms lag {lag:4d} fs: {len(data)} swarms, {data.total_steps} MD steps, "
                  f"endpoints in A/B {int((data.end_label == 0).sum())}/{int((data.end_label == 1).sum())}, "
                  f"{time.time() - t0:.0f} s", flush=True)


# ---- stage: analyze -------------------------------------------------------------------------

def stage_analyze(args):
    from pathgennie.committor import SwarmDataset
    from pathgennie.committor.train import committor_scores, train_ensemble

    out = args.out
    disc = np.load(out / "discover.npz")
    ref = np.load(out / "reference.npz")
    committed = ref["n_a"] + ref["n_b"]
    ok = committed >= 10
    q_ref = (ref["n_b"] / np.maximum(committed, 1))[ok]
    features = np.stack([adp.heavy_atom_distances(c) for c in ref["coords"][ok]])
    ts = (q_ref > 0.05) & (q_ref < 0.95)
    # What shooting noise alone contributes to an RMSE against this reference.
    floor = np.sqrt(np.mean((q_ref * (1 - q_ref) / committed[ok])[ts])) if ts.any() else float("nan")
    phi_psi = np.array([adp.phi_psi(c) for c in ref["coords"][ok]])
    da = np.linalg.norm(adp.angular_delta(phi_psi, adp.C7EQ), axis=1)
    db = np.linalg.norm(adp.angular_delta(phi_psi, adp.C7AX), axis=1)
    geometric = committor_scores(da / (da + db), q_ref)

    summary = dict(
        force_field="charmm36_2024.xml (stand-in, not CHARMM27)" if args.psf is None else args.psf,
        reference=dict(configurations=int(ok.sum()), transition=int(ts.sum()),
                       md_steps=int(ref["steps"].sum()), shooting_noise_rmse_ts=float(floor),
                       uncommitted_shots=int(ref["n_open"].sum())),
        geometric_cv=geometric, protocols={},
    )
    print(f"reference: {int(ok.sum())} configurations, {int(ts.sum())} in the transition region, "
          f"{int(ref['steps'].sum()):,} MD steps; shooting noise alone gives RMSE(TS) = {floor:.3f}")
    print(f"{'protocol':14s} {'MD steps':>10s}  {'r':>6s} {'RMSE':>6s}  {'r(TS)':>6s} {'RMSE(TS)':>8s}  "
          f"{'A/B endpoints':>14s}")
    print(f"{'geometric CV':14s} {'-':>10s}  {geometric['pearson']:6.3f} {geometric['rmse']:6.3f}  "
          f"{geometric['pearson_transition']:6.3f} {geometric['rmse_transition']:8.3f}")

    models = {}

    def evaluate(name, data, md_steps, steps):
        n_a = int(((data.end_label == 0) & data.mask)[data.start_label == -1].sum())
        n_b = int(((data.end_label == 1) & data.mask)[data.start_label == -1].sum())
        row = dict(md_steps=int(md_steps), swarms=len(data), endpoints_in_a=n_a, endpoints_in_b=n_b,
                   train_steps=steps)
        if n_a == 0 or n_b == 0:
            row["status"] = "no boundary data from both basins"
            print(f"{name:14s} {md_steps:10d}  -- cannot train: endpoints in A/B = {n_a}/{n_b}")
        else:
            model = train_ensemble(data, args.models, seed=0, steps=steps, hidden=(128, 128))
            models[name] = model
            pred = model.predict(features)
            row.update(committor_scores(pred, q_ref))
            row["member_rmse_ts"] = [committor_scores(m.predict(features), q_ref)["rmse_transition"]
                                     for m in model.models]
            row["final_loss"] = float(np.mean([m.history[-1] for m in model.models]))
            print(f"{name:14s} {md_steps:10d}  {row['pearson']:6.3f} {row['rmse']:6.3f}  "
                  f"{row['pearson_transition']:6.3f} {row['rmse_transition']:8.3f}  {n_a:7d}/{n_b:<6d}",
                  flush=True)
        summary["protocols"][name] = row

    disc_data = SwarmDataset.load(out / "swarms_discovery.h5")
    evaluate("discovery", disc_data, int(disc["train_md_steps"]), args.train_steps)
    lag_data = {}
    for lag in LAGS:
        path = out / f"swarms_lag{lag}.h5"
        if not path.exists():
            continue
        lag_data[lag] = SwarmDataset.load(path)
        # Boundary information travels one lag per sweep, so short lags need more sweeps.
        steps = args.train_steps * (4 if lag <= 5 else 1)
        evaluate(f"lag={lag}", lag_data[lag], lag_data[lag].total_steps, steps)

    # Self-consistency of one fixed function at every lag: for the model trained at
    # the longest lag, how far is the swarm average of q at the endpoints from q at
    # the start? Zero (up to sampling noise) wherever a position-only committor
    # satisfies the relation; a systematic excess marks lags where it does not.
    longest = f"lag={max(lag_data)}" if lag_data else None
    if longest in models:
        best = models[longest]
        print(f"\nself-consistency residual of the {longest} model, by lag "
              f"(swarm mean of q at endpoints minus q at start)")
        print(f"{'lag (fs)':>8s} {'RMS residual':>13s} {'expected from noise':>20s} {'ratio':>6s} {'mean |shift in q|':>18s}")
        summary["residual"] = {}
        for lag, data in lag_data.items():
            use = data.start_label == -1
            q0 = best.predict(data.starts[use])
            ends = data.ends[use]
            q1 = best.predict(ends.reshape(-1, ends.shape[-1])).reshape(ends.shape[:2])
            q1 = np.where(data.end_label[use] >= 0, data.end_label[use], q1)
            resid = q1.mean(axis=1) - q0
            noise = q1.std(axis=1, ddof=1) / np.sqrt(q1.shape[1])
            row = dict(rms_residual=float(np.sqrt(np.mean(resid**2))),
                       rms_noise=float(np.sqrt(np.mean(noise**2))),
                       mean_abs_shift=float(np.mean(np.abs(q1 - q0[:, None]))))
            row["ratio"] = row["rms_residual"] / max(row["rms_noise"], 1e-12)
            summary["residual"][lag] = row
            print(f"{lag:8d} {row['rms_residual']:13.4f} {row['rms_noise']:20.4f} {row['ratio']:6.2f} "
                  f"{row['mean_abs_shift']:18.4f}")

    # Gate: r > 0.9 and RMSE(TS) < 0.15 against shooting with less MD than the shooting itself.
    trained = {k: v for k, v in summary["protocols"].items() if "pearson" in v}
    if trained:
        name, best_row = min(trained.items(), key=lambda kv: kv[1]["rmse_transition"])
        passed = (best_row["pearson"] > 0.9 and best_row["rmse_transition"] < 0.15
                  and best_row["md_steps"] < summary["reference"]["md_steps"])
        summary["gate"] = dict(protocol=name, passed=bool(passed))
        print(f"\nM2 gate (r > 0.9, RMSE(TS) < 0.15, less MD than the shooting reference): "
              f"best protocol {name} -> {'PASS' if passed else 'FAIL'}")
    with open(out / "m2_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["discover", "reference", "swarms", "analyze"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--psf", default=None, help="CHARMM PSF (default: bundled CHARMM36 stand-in)")
    parser.add_argument("--params", nargs="*", default=None, help="CHARMM rtf/prm/str files")
    parser.add_argument("--train-runs", type=int, default=12, help="discovery runs each way (training pool)")
    parser.add_argument("--test-runs", type=int, default=8, help="discovery runs each way (held-out pool)")
    parser.add_argument("--configs", type=int, default=300, help="held-out configurations to shoot from")
    parser.add_argument("--shots1", type=int, default=20)
    parser.add_argument("--shots2", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=30_000, help="per shot")
    parser.add_argument("--starts", type=int, default=400, help="fixed starts for the lag scan")
    parser.add_argument("--models", type=int, default=4, help="ensemble size")
    parser.add_argument("--train-steps", type=int, default=10_000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {"discover": stage_discover, "reference": stage_reference,
     "swarms": stage_swarms, "analyze": stage_analyze}[args.stage](args)


if __name__ == "__main__":
    main()
