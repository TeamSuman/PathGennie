"""Glue that runs a configured downstream stage after path discovery.

A backend ``run()`` calls :func:`run_downstream` when ``pathgennie.downstream`` is
set in ``input.yaml``.  It assembles a :class:`PathEnsemble` from the driver's
trajectory + restartable seed handles, constructs the named stage via
:func:`pathgennie.sampling.make_stage`, runs it on the same engine, and writes the
free-energy profile / rate constants next to the trajectory.

This helper is engine-agnostic and unit-tested with the toy Langevin engine; the
per-backend wiring (which supplies a real MD engine and seed handles) reuses it
unchanged.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .base import SamplingResult, build_path_ensemble

__all__ = ["make_scalar_cv", "run_downstream", "write_result"]


def make_scalar_cv(proj_fn: Callable[..., np.ndarray], projection_args: dict, component: int = 0) -> Callable[[np.ndarray], float]:
    """Reduce a (possibly vector) projection to the scalar CV a stage bins along."""
    args = projection_args or {}

    def scalar(coords: np.ndarray) -> float:
        v = np.atleast_1d(np.asarray(proj_fn(coords, **args), dtype=float))
        return float(v[component])

    return scalar


def write_result(result: SamplingResult, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if result.free_energy is not None:
        x = result.metadata.get("bin_centers")
        if x is None:
            x = result.metadata.get("grid")
        with (output_dir / "free_energy.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["cv", "free_energy"])
            fe = np.asarray(result.free_energy).ravel()
            xs = np.asarray(x).ravel() if x is not None else [None] * fe.size
            for xi, fi in zip(xs, fe):
                writer.writerow(["" if xi is None else xi, fi])
    if result.rate_constants is not None:
        (output_dir / "rate_constants.json").write_text(
            json.dumps(result.rate_constants, indent=2), encoding="utf-8"
        )


def run_downstream(
    downstream: str,
    stage_cfg: dict,
    *,
    engine,
    traj: np.ndarray,
    metrics: np.ndarray,
    seed_handles: Optional[Sequence] = None,
    scalar_cv_fn: Callable[[np.ndarray], float],
    output_dir: Path,
    state_labels: Optional[np.ndarray] = None,
) -> SamplingResult:
    """Build a PathEnsemble, run the named stage, and persist its result."""
    from pathgennie.sampling import make_stage  # late import avoids any import cycle

    ensemble = build_path_ensemble(
        traj, metrics, handles=list(seed_handles) if seed_handles else None,
        cv_fn=scalar_cv_fn, state_labels=state_labels,
    )
    stage = make_stage(downstream, cv_fn=scalar_cv_fn, **dict(stage_cfg))
    result = stage.run(ensemble, engine)
    write_result(result, output_dir)
    return result
