#!/usr/bin/env python
"""Stage 4 -- plot the paths on the 2-D distance plane.

d(C-Cl_attacking) against d(C-Cl_leaving) is the plane the mechanism is normally
drawn in, and different mechanisms occupy visibly different regions of it, so one
figure shows both how refinement moved the path and which channel it took:

  * S_N2 -- a diagonal crossing near the symmetry line (concerted)
  * S_N1 -- an L-shaped detour through the top-right corner (both bonds long: a
    free carbocation) before the new bond forms
  * E1 / E2 -- never reach short d(C-X); the path leaves the plane entirely,
    which is why an elimination channel shows up here as a *truncated* trace

    python 4_plot_2d_cv.py --refinement results/refinement

Passing extra ``--extra label=path.npy`` families lets you overlay a second
substrate (e.g. the tert-butyl E2 run) in the same axes for comparison.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# The same feature space stages 2 and 3 use, so the refined path and the raw seeds
# are plotted in the same coordinates. Defining it twice once put them on
# transposed axes -- undetectably, because this reaction is symmetric.
from sn2_cv import path_features_traj  # noqa: E402

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")          # blue, orange, green
INK, MUTED, GRID, SURFACE = "#1a1a19", "#5c5c58", "#e5e5e1", "#fcfcfb"


def load_seed_paths(ensemble: Path) -> list[np.ndarray]:
    from pathgennie.backends.amber.utils import read_native_trajectory

    out = []
    for case in sorted(ensemble.glob("seed_*")):
        nc = glob.glob(str(case / "pathgennie_sn2" / "output" / "*.nc"))
        top = case / "sn2.prmtop"
        if nc and top.exists():
            traj = np.asarray(read_native_trajectory(nc[0], str(top)))
            if traj.ndim == 3 and len(traj) > 1:
                out.append(path_features_traj(traj))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refinement", type=Path, default=HERE / "results" / "refinement")
    p.add_argument("--ensemble", type=Path, default=HERE / "ensemble")
    p.add_argument("--ts", type=float, default=None,
                   help="symmetric-TS distance in A; drawn on the d1 = d2 line")
    p.add_argument("--extra", action="append", default=[], metavar="LABEL=FILE.npy",
                   help="overlay another (T,2) path, e.g. a tert-butyl E2 run")
    p.add_argument("--out", type=Path, default=HERE / "results" / "paths_2d_cv")
    args = p.parse_args()

    seeds = load_seed_paths(args.ensemble)
    initial = np.load(args.refinement / "initial_path.npy")
    refined = np.load(args.refinement / "refined_path.npy")
    history = np.load(args.refinement / "path_history.npz")
    mids = [history[k] for k in sorted(history.files, key=lambda s: int(s.split("_")[1]))][1:-1]

    fig, ax = plt.subplots(figsize=(7.2, 6.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    pts = np.vstack([initial, refined] + seeds)
    lo, hi = float(pts.min()) - 0.15, float(pts.max()) + 0.15
    # Any symmetric TS must lie on d1 = d2; a recessive guide, not a data series.
    ax.plot([lo, hi], [lo, hi], color=GRID, lw=1.2, ls="--", zorder=1)

    for i, s in enumerate(seeds):
        ax.plot(s[:, 0], s[:, 1], color=SERIES[0], lw=1.0, alpha=0.35, zorder=2,
                label=f"raw QM/MM seeds (n={len(seeds)})" if i == 0 else None)
    for i, m in enumerate(mids):
        ax.plot(m[:, 0], m[:, 1], color=SERIES[2], lw=1.2, alpha=0.55, zorder=3,
                label=f"refinement iterations (n={len(mids)})" if i == 0 else None)
    ax.plot(initial[:, 0], initial[:, 1], color=MUTED, lw=2.0, ls=":", zorder=4,
            label="initial (seed consensus)")
    ax.plot(refined[:, 0], refined[:, 1], color=SERIES[1], lw=2.6, zorder=5,
            label="refined PathCV")

    for spec in args.extra:
        label, _, fname = spec.partition("=")
        extra = np.load(fname)
        ax.plot(extra[:, 0], extra[:, 1], color=INK, lw=2.0, ls="--", zorder=5,
                label=label)

    if args.ts is not None:
        ax.plot(args.ts, args.ts, marker="*", ms=20, color=INK, mec=SURFACE, mew=1.5,
                ls="none", zorder=6, label=f"reference TS ({args.ts:.3f} Å)")

    ax.set_xlabel("d(C–Cl$_{attacking}$)  /  Å", fontsize=11, color=INK)
    ax.set_ylabel("d(C–Cl$_{leaving}$)  /  Å", fontsize=11, color=INK)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title("QM/MM path refinement on the 2-D distance plane",
                 fontsize=12.5, color=INK, pad=32, loc="left")
    ax.text(0.0, 1.02, "Cl⁻ + CH₃Cl → ClCH₃ + Cl⁻ · DFTB3/3ob-3-1, gas phase",
            transform=ax.transAxes, fontsize=9.5, color=MUTED, ha="left", va="bottom")

    leg = ax.legend(loc="upper right", frameon=True, fontsize=9,
                    facecolor=SURFACE, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(INK)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}.png / .pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
