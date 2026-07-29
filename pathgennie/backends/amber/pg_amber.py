#!/usr/bin/env python
"""Generic PathGennie Amber runner driven by a case-local input.yaml.

The adaptive cycle now lives in :mod:`pathgennie.core`; this module only loads
the case configuration, builds a device-aware :class:`CoreAmberEngine`, and runs
the shared :class:`~pathgennie.core.driver.PathGennieDriver` over a device pool so
the swarm spreads across all configured GPUs.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np

from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import ThreadDevicePool
from pathgennie.core.progress import DEFAULT_ESCAPE_METRIC, EscapeMetric, TargetMetric
from pathgennie.core.strategy import resolve_profile
from pathgennie.utils.config import load_config
from pathgennie.utils.scratch import resolve_scratch_dir

from .engine import CoreAmberEngine
from .utils import (
    default_mdin_controls,
    enrich_args,
    load_function,
    parse_prmtop,
    read_rst7_coords,
    resolve_case_path,
    wrap_frames_pbc,
    write_metrics_csv,
    write_trajectory,
)


def run(case_dir: Path, config_name: str = "input.yaml") -> None:
    case_dir = case_dir.resolve()
    os.chdir(case_dir)
    cfg_model = load_config(case_dir / config_name)
    cfg = cfg_model.model_dump(exclude_none=True)

    workdir = resolve_case_path(case_dir, cfg.get("workdir", "pathgennie_run"))
    scratch_dir = resolve_scratch_dir(workdir, cfg.get("scratch_root"))
    output_dir = workdir / "output"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    amber_cfg = cfg["amber"]
    pg_cfg = resolve_profile(cfg["pathgennie"])
    topology = resolve_case_path(case_dir, amber_cfg["topology"])
    initial_restart = resolve_case_path(case_dir, amber_cfg["initial_restart"])
    executable_str = amber_cfg["executable"]
    expanded_path = str(Path(executable_str).expanduser())
    resolved_exe = shutil.which(expanded_path) or shutil.which(executable_str)
    if resolved_exe is None:
        raise FileNotFoundError(f"Amber executable not found: {executable_str}")
    executable = Path(resolved_exe)

    missing = [str(path) for path in (topology, initial_restart) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s): " + ", ".join(missing))

    topology_info = parse_prmtop(topology)
    temperature = float(pg_cfg.get("temperature", 300.0))
    md_cfg = cfg.get("md", {})
    system_kind = md_cfg.get("system", amber_cfg.get("system", "explicit"))
    mdin_controls = default_mdin_controls(system_kind)
    mdin_controls.update(md_cfg.get("controls", {}))
    extra_mdin_text = md_cfg.get("extra_text", "")

    command_prefix = []
    if amber_cfg.get("mpi_launcher"):
        command_prefix = [
            str(amber_cfg["mpi_launcher"]),
            "-np",
            str(int(amber_cfg.get("mpi_ranks", 1))),
            *[str(arg) for arg in amber_cfg.get("mpi_launcher_args", [])],
        ]

    proj_fn = load_function(case_dir, cfg["projection"]["module"], cfg["projection"]["function"])
    conv_fn = load_function(case_dir, cfg["convergence"]["module"], cfg["convergence"]["function"])

    projection_args = {
        key: value for key, value in cfg.get("projection", {}).items() if key not in {"module", "function", "reference", "periodic"}
    }
    convergence_args = {
        key: value for key, value in cfg.get("convergence", {}).items() if key not in {"module", "function"}
    }
    projection_args = enrich_args(projection_args, topology_info)
    convergence_args = enrich_args(convergence_args, topology_info)

    start_coords = read_rst7_coords(initial_restart)
    start_cv = np.asarray(proj_fn(start_coords, **projection_args), dtype=float)
    print(f"Initial CV: {start_cv}")

    # Optional per-component CV periods (e.g. [360.0, 360.0] for dihedrals in
    # degrees). Omit, or use null per component, for non-periodic CVs.
    cv_periodic = cfg.get("projection", {}).get("periodic")

    mode = pg_cfg.get("mode", "escape")
    if mode == "target":
        if "target_projection" not in pg_cfg:
            raise ValueError("pathgennie.mode is 'target', but pathgennie.target_projection is missing")
        target_projection = np.asarray(pg_cfg["target_projection"], dtype=float).reshape(-1)
        progress = TargetMetric(proj_fn, target_projection, projection_args=projection_args,
                                periodic=cv_periodic)
        assert target_projection is not None
        progress: ProgressVariable
    else:
        progress = EscapeMetric(
            proj_fn, start_cv, projection_args=projection_args,
            escape_metric=pg_cfg.get("escape_metric", DEFAULT_ESCAPE_METRIC),
            periodic=cv_periodic,
        )

    def convergence(coords: np.ndarray) -> bool:
        return bool(conv_fn(coords, **convergence_args))

    # Guard against CPU oversubscription: when several segments run concurrently
    # (workers_per_device > 1 on CPU, or an MPI/threaded pmemd), each subprocess
    # otherwise grabs every core. cpu_threads_per_worker pins per-worker threads.
    env_overrides = {}
    cpu_threads = pg_cfg.get("cpu_threads_per_worker")
    if cpu_threads:
        n_threads = str(int(cpu_threads))
        env_overrides = {
            "OMP_NUM_THREADS": n_threads,
            "MKL_NUM_THREADS": n_threads,
            "OPENBLAS_NUM_THREADS": n_threads,
        }

    engine = CoreAmberEngine(
        topology=topology,
        executable=executable,
        scratch_dir=scratch_dir,
        temperature=temperature,
        mdin_controls=mdin_controls,
        extra_mdin_text=extra_mdin_text,
        command_prefix=command_prefix,
        env_overrides=env_overrides,
    )

    # Device pool: `devices` lists GPU indices; falls back to legacy tau1_workers.
    devices = pg_cfg.get("devices", amber_cfg.get("devices"))
    workers_per_device = int(pg_cfg.get("workers_per_device", pg_cfg.get("tau1_workers", 1)))
    executor = ThreadDevicePool(devices=devices, workers_per_device=workers_per_device)

    trajectory_path = output_dir / cfg.get("output", {}).get("trajectory", "reactive_path.pdb")
    metrics_path = output_dir / cfg.get("output", {}).get("metrics", "metrics.csv")
    overwrite = pg_cfg.get("overwrite", False)
    checkpoint_path = pg_cfg.get("checkpoint_path")

    is_resuming = False
    if checkpoint_path:
        from pathgennie.core.storage import HDF5Storage
        if HDF5Storage.load_checkpoint(checkpoint_path) is not None:
            is_resuming = True

    if not overwrite and not is_resuming:
        existing = [p for p in [trajectory_path, metrics_path] if p.exists()]
        if checkpoint_path and Path(checkpoint_path).exists():
            existing.append(Path(checkpoint_path))
        if existing:
            names = ", ".join(str(p) for p in existing)
            raise FileExistsError(
                f"Output file(s) already exist: {names}. "
                f"Set 'overwrite: true' in pathgennie config to overwrite, or remove existing files."
            )

    driver = PathGennieDriver(
        engine, progress, convergence,
        executor=executor,
        sigma=pg_cfg["sigma"],
        seed=pg_cfg.get("seed"),
        reject_worse_tau2=pg_cfg.get("reject_worse_tau2", False),
        reject_worse_anchor=pg_cfg.get("reject_worse_anchor", False),
        verbosity=pg_cfg.get("verbosity", 1),
        save_subframes=pg_cfg.get("save_subframes", False),
        subframe_stride=pg_cfg.get("subframe_stride", 1),
        checkpoint_freq=pg_cfg.get("checkpoint_freq", 0),
    )

    downstream = pg_cfg.get("downstream")
    result = driver.run(
        str(initial_restart),
        tau1=pg_cfg["tau1_steps"],
        tau2=pg_cfg["tau2_steps"],
        max_trial=pg_cfg["max_trial"],
        max_cycle=pg_cfg["max_cycle"],
        save_freq=pg_cfg.get("save_freq", 10),
        collect_seeds=bool(downstream),
        checkpoint_path=checkpoint_path,
        checkpoint_freq=pg_cfg.get("checkpoint_freq", 0),
    )
    seed_handles = None
    if downstream:
        traj, metrics, seed_handles = result
    else:
        traj, metrics = result

    if cfg.get("output", {}).get("wrap_pbc", False):
        traj = wrap_frames_pbc(traj, topology_info)
    # Physical time between saved frames: each saved cycle spans tau1 + tau2
    # integrator steps, and frames are kept every save_freq cycles.
    timestep_ps = float(mdin_controls.get("dt", 0.002))
    save_freq = int(pg_cfg.get("save_freq", 10))
    if pg_cfg.get("save_subframes", False):
        trajectory_dt = pg_cfg.get("subframe_stride", 1) * timestep_ps
    else:
        trajectory_dt = save_freq * (pg_cfg["tau1_steps"] + pg_cfg["tau2_steps"]) * timestep_ps
    write_trajectory(trajectory_path, topology_info, traj, dt=trajectory_dt)
    write_metrics_csv(metrics_path, metrics)

    # Optional downstream enhanced-sampling stage (uses scratch, so before cleanup).
    if downstream:
        from pathgennie.sampling.runner import make_scalar_cv, run_downstream
        stage_cfg = dict(cfg.get(downstream, {}))
        component = stage_cfg.pop("cv_component", 0)
        scalar_cv = make_scalar_cv(proj_fn, projection_args, component)
        run_downstream(
            downstream, stage_cfg, engine=engine, traj=traj, metrics=metrics,
            seed_handles=seed_handles, scalar_cv_fn=scalar_cv, output_dir=output_dir,
            executor=executor,
        )

    shutil.rmtree(scratch_dir)

    print(f"Saved frames: {len(traj)}")
    print(f"Metric samples: {len(metrics)}")
    print(f"Reactive path: {trajectory_path}")
    print(f"Metrics: {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=Path.cwd(), help=" Directory containing input.yaml")
    parser.add_argument("--config", default="input.yaml", help="YAML config name inside the case directory")
    args = parser.parse_args()
    run(args.case, args.config)


if __name__ == "__main__":
    main()
