#!/usr/bin/env python
"""Run the Toy Langevin Wolfe-Quapp adaptive sampling case.

Demonstrates:
- Target-mode adaptive sampling on the 2D Wolfe-Quapp surface
- Intra-segment frame capture (save_subframes: true, subframe_stride: 5)
- Periodic HDF5 checkpointing (checkpoint_freq: 10, checkpoint_path: checkpoint.h5)
- Automatic restart/resume from checkpoint
- Output overwrite protection (overwrite: false)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

# Ensure pathgennie package is on python path
CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import SerialExecutor
from pathgennie.core.progress import TargetMetric
from pathgennie.core.toy import ToyLangevinEngine, wolfe_quapp_gradient
from pathgennie.utils.config import load_config
from pathgennie.utils.scratch import resolve_scratch_dir

SOURCE = (-1.174, 1.477)
TARGET = np.array([1.124, -1.485])


def _xy(coords: np.ndarray) -> np.ndarray:
    return np.array([coords[0, 0], coords[0, 1]])


def run(case_dir: Path, config_name: str = "input.yaml") -> None:
    case_dir = case_dir.resolve()
    cfg_model = load_config(case_dir / config_name)
    cfg = cfg_model.model_dump(exclude_none=True)

    pg_cfg = cfg["pathgennie"]
    md_cfg = cfg.get("md", {})

    workdir = case_dir / cfg.get("workdir", "pathgennie_toy_run")
    output_dir = workdir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_path = output_dir / "reactive_path.npy"
    metrics_path = output_dir / "metrics.csv"

    checkpoint_rel = pg_cfg.get("checkpoint_path", "checkpoint.h5")
    checkpoint_path = workdir / Path(checkpoint_rel).name
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    overwrite = pg_cfg.get("overwrite", False)

    # Overwrite protection check
    is_resuming = False
    if checkpoint_path.exists():
        from pathgennie.core.storage import HDF5Storage
        if HDF5Storage.load_checkpoint(checkpoint_path) is not None:
            is_resuming = True

    if not overwrite and not is_resuming:
        existing = [p for p in [trajectory_path, metrics_path] if p.exists()]
        if existing:
            names = ", ".join(str(p) for p in existing)
            raise FileExistsError(
                f"Output file(s) already exist: {names}. "
                f"Set 'overwrite: true' in pathgennie config to overwrite."
            )

    engine = ToyLangevinEngine(
        dt=float(md_cfg.get("dt", 0.005)),
        kT=float(md_cfg.get("kT", 1.0)),
        gamma=float(md_cfg.get("gamma", 1.0)),
    )
    initial_handle = engine.create_state(SOURCE)
    target_pos = np.asarray(pg_cfg.get("target_projection", TARGET), dtype=float)
    progress = TargetMetric(_xy, target_cv=target_pos)

    def converged(coords: np.ndarray) -> bool:
        return bool(np.linalg.norm(_xy(coords) - target_pos) < 0.5)

    driver = PathGennieDriver(
        engine,
        progress,
        converged,
        executor=SerialExecutor(),
        sigma=float(pg_cfg.get("sigma", 0.1)),
        seed=pg_cfg.get("seed"),
        verbosity=int(pg_cfg.get("verbosity", 1)),
        save_subframes=bool(pg_cfg.get("save_subframes", False)),
        subframe_stride=int(pg_cfg.get("subframe_stride", 1)),
        checkpoint_freq=int(pg_cfg.get("checkpoint_freq", 0)),
    )

    traj, metrics = driver.run(
        initial_handle,
        tau1=int(pg_cfg["tau1_steps"]),
        tau2=int(pg_cfg["tau2_steps"]),
        max_trial=int(pg_cfg["max_trial"]),
        max_cycle=int(pg_cfg["max_cycle"]),
        save_freq=int(pg_cfg.get("save_freq", 5)),
        checkpoint_path=str(checkpoint_path),
        checkpoint_freq=int(pg_cfg.get("checkpoint_freq", 0)),
    )

    np.save(trajectory_path, traj)
    np.savetxt(metrics_path, metrics, delimiter=",", header="metric", comments="")

    print(f"Toy Langevin run complete.")
    print(f"Trajectory shape: {traj.shape}")
    print(f"Metrics count: {len(metrics)}")
    print(f"Saved trajectory: {trajectory_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Checkpoint file: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Run PathGennie Toy Langevin case")
    parser.add_argument("--case", type=Path, default=CASE_DIR, help="Case directory")
    parser.add_argument("--config", default="input.yaml", help="Config file name")
    args = parser.parse_args()
    run(args.case, args.config)


if __name__ == "__main__":
    main()
