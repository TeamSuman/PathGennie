#!/usr/bin/env python
"""Validate the Weighted Ensemble stage against an analytic free energy.

Runs path-informed WE on the toy Wolfe-Quapp engine along the y coordinate and
compares the recovered free-energy profile F(y) to the analytic marginal

    F(y) = -kT ln integral e^{-V(x,y)/kT} dx

computed numerically.  Prints both profiles (shifted to min 0) and their
correlation; this is a qualitative physics check, not a CI test.
"""

from __future__ import annotations

import numpy as np

from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import SerialExecutor
from pathgennie.core.progress import EscapeMetric
from pathgennie.core.toy import ToyLangevinEngine, wolfe_quapp_potential
from pathgennie.sampling import WeightedEnsembleStage, build_path_ensemble


def analytic_marginal(y_grid, kT):
    x = np.linspace(-2.5, 2.5, 4001)
    fe = []
    for y in y_grid:
        v = np.array([wolfe_quapp_potential(xi, y) for xi in x])
        trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        z = trapz(np.exp(-v / kT), x)
        fe.append(-kT * np.log(z))
    fe = np.array(fe)
    return fe - fe.min()


def main():
    kT = 2.0
    engine = ToyLangevinEngine(dt=0.005, kT=kT)
    initial = engine.create_state((-1.0, -1.4))
    progress = EscapeMetric(lambda c: np.array([c[0, 1]]), start_cv=np.array([-1.4]), escape_metric="cv0")
    driver = PathGennieDriver(
        engine, progress, convergence_fn=lambda c: False,
        executor=SerialExecutor(), sigma=0.3, seed=0, verbosity=0,
    )
    traj, metrics, handles = driver.run(
        initial, tau1=5, tau2=10, max_trial=6, max_cycle=60, save_freq=1, collect_seeds=True,
    )
    ens = build_path_ensemble(traj, metrics, handles=handles, cv_fn=lambda c: c[0, 1])

    stage = WeightedEnsembleStage(
        cv_fn=lambda c: c[0, 1], tau_steps=10, n_iterations=400,
        n_bins=16, target_count=6, seed=1, kT=kT,
    )
    result = stage.run(ens, engine)

    centers = result.metadata["bin_centers"]
    fe_we = result.free_energy
    finite = np.isfinite(fe_we)
    fe_an = analytic_marginal(centers, kT)

    print(f"{'y':>8} {'F_WE':>10} {'F_analytic':>12}")
    for c, a, w, ok in zip(centers, fe_an, fe_we, finite):
        print(f"{c:>8.3f} {('%.3f' % w) if ok else '   inf':>10} {a:>12.3f}")

    # Correlation over occupied bins (shifted).
    we = fe_we[finite] - fe_we[finite].min()
    an = fe_an[finite] - fe_an[finite].min()
    if we.size > 2:
        corr = np.corrcoef(we, an)[0, 1]
        print(f"\nPearson correlation (occupied bins): {corr:.3f}")


if __name__ == "__main__":
    main()
