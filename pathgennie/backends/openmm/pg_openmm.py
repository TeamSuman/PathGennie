#!/usr/bin/env python
"""Generic PathGennie OpenMM runner driven by a case-local input.yaml."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import yaml
from openmm import LangevinMiddleIntegrator, Platform, unit
from openmm.app import PME, AmberInpcrdFile, AmberPrmtopFile, HBonds, Simulation

from pathgennie.backends.amber.utils import (
    enrich_args,
    load_function,
    parse_prmtop,
    read_rst7_coords,
    resolve_case_path,
    wrap_frames_pbc,
    write_metrics_csv,
    write_trajectory,
)

from pathgennie.core.strategy import resolve_profile

from .pg_omm import PathGennieMD


def build_simulation(
    prmtop_path: Path,
    inpcrd_path: Path,
    *,
    temperature: float,
    timestep_ps: float,
    friction_per_ps: float,
    platform_name: str = "CPU",
) -> Simulation:
    top = AmberPrmtopFile(str(prmtop_path))
    crd = AmberInpcrdFile(str(inpcrd_path))
    system = top.createSystem(
        nonbondedMethod=PME,  # type: ignore
        nonbondedCutoff=1.0 * unit.nanometer,  # type: ignore
        constraints=HBonds,
    )
    integrator = LangevinMiddleIntegrator(
        temperature * unit.kelvin,  # type: ignore
        friction_per_ps / unit.picosecond,  # type: ignore
        timestep_ps * unit.picoseconds,  # type: ignore
    )
    platform = Platform.getPlatformByName(platform_name)
    simulation = Simulation(top.topology, system, integrator, platform)
    simulation.context.setPositions(crd.positions)
    if crd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*crd.boxVectors)  # type: ignore
    return simulation


def run(case_dir: Path, config_name: str = "input.yaml") -> None:
    case_dir = case_dir.resolve()
    os.chdir(case_dir)
    cfg = yaml.safe_load((case_dir / config_name).read_text(encoding="utf-8"))

    workdir = resolve_case_path(case_dir, cfg.get("workdir", "pathgennie_openmm_run"))
    output_dir = workdir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    openmm_cfg = cfg["openmm"]
    pg_cfg = resolve_profile(cfg["pathgennie"])
    topology = resolve_case_path(case_dir, openmm_cfg["topology"])
    initial_restart = resolve_case_path(case_dir, openmm_cfg["initial_restart"])

    missing = [str(path) for path in (topology, initial_restart) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s): " + ", ".join(missing))

    topology_info = parse_prmtop(topology)
    temperature = float(pg_cfg.get("temperature", 300.0))
    md_cfg = cfg.get("md", {})

    proj_fn = load_function(case_dir, cfg["projection"]["module"], cfg["projection"]["function"])
    conv_fn = load_function(case_dir, cfg["convergence"]["module"], cfg["convergence"]["function"])

    projection_args = {
        key: value for key, value in cfg.get("projection", {}).items() if key not in {"module", "function", "reference"}
    }
    convergence_args = {
        key: value for key, value in cfg.get("convergence", {}).items() if key not in {"module", "function"}
    }
    projection_args = enrich_args(projection_args, topology_info)
    convergence_args = enrich_args(convergence_args, topology_info)

    initial_cv = proj_fn(read_rst7_coords(initial_restart), **projection_args)
    print(f"Initial CV: {initial_cv}")

    mode = pg_cfg.get("mode", "target")
    target_projection = None
    if mode == "target":
        if "target_projection" not in pg_cfg:
            raise ValueError("pathgennie.mode is 'target', but pathgennie.target_projection is missing")
        target_projection = np.asarray(pg_cfg["target_projection"], dtype=float)
        if target_projection.ndim == 0:
            target_projection = target_projection.reshape(1)

    simulation = build_simulation(
        topology,
        initial_restart,
        temperature=temperature,
        timestep_ps=float(md_cfg.get("timestep_ps", 0.001)),
        friction_per_ps=float(md_cfg.get("friction_per_ps", 1.0)),
        platform_name=openmm_cfg.get("platform", "CPU"),
    )

    runner = PathGennieMD(
        simulation=simulation,
        projection_fn=proj_fn,
        projection_args=projection_args,
        mode=mode,
        target_projection=target_projection,
        convergence_fn=conv_fn,
        convergence_args=convergence_args,
        temperature=temperature,
        sigma=pg_cfg.get("sigma", 0.05),
        seed=pg_cfg.get("seed"),
    )
    downstream = pg_cfg.get("downstream")
    result = runner.run(
        initial_pos=AmberInpcrdFile(str(initial_restart)).positions,  # type: ignore
        tau1=pg_cfg["tau1_steps"],
        tau2=pg_cfg["tau2_steps"],
        max_trial=pg_cfg["max_trial"],
        max_cycle=pg_cfg["max_cycle"],
        save_freq=pg_cfg.get("save_freq", 1),
        verbosity=pg_cfg.get("verbosity", 1),
        collect_seeds=bool(downstream),
    )
    seed_handles = None
    if downstream:
        trajectory, metrics, seed_handles = result
    else:
        trajectory, metrics = result

    trajectory_path = output_dir / cfg.get("output", {}).get("trajectory", "reactive_path.dcd")
    metrics_path = output_dir / cfg.get("output", {}).get("metrics", "metrics.csv")
    if cfg.get("output", {}).get("wrap_pbc", False):
        trajectory = wrap_frames_pbc(trajectory, topology_info)
    write_trajectory(trajectory_path, topology_info, trajectory)
    write_metrics_csv(metrics_path, metrics)

    if downstream:
        from pathgennie.sampling.runner import make_scalar_cv, run_downstream
        stage_cfg = dict(cfg.get(downstream, {}))
        component = stage_cfg.pop("cv_component", 0)
        scalar_cv = make_scalar_cv(proj_fn, projection_args, component)
        run_downstream(
            downstream, stage_cfg, engine=runner.engine, traj=trajectory, metrics=metrics,
            seed_handles=seed_handles, scalar_cv_fn=scalar_cv, output_dir=output_dir,
        )
    print(f"Saved frames: {len(trajectory)}")
    print(f"Metric samples: {len(metrics)}")
    print(f"Reactive path: {trajectory_path}")
    print(f"Metrics: {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=Path.cwd(), help="Directory containing input.yaml")
    parser.add_argument("--config", default="input.yaml", help="YAML config name inside the case directory")
    args = parser.parse_args()
    run(args.case, args.config)


if __name__ == "__main__":
    main()
