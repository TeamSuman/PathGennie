#!/usr/bin/env python
"""Generic PathGennie OpenMM runner driven by a case-local input.yaml."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
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
from pathgennie.backends.gromacs.utils import (
    read_gro_coords,
    read_masses_from_topology,
    read_topology_info,
)
from openmm.app import GromacsGroFile, GromacsTopFile

from pathgennie.core.progress import DEFAULT_ESCAPE_METRIC
from pathgennie.core.strategy import resolve_profile
from pathgennie.utils.config import load_config

from .pg_omm import PathGennieMD


def build_simulation(
    prmtop_path: Path,
    inpcrd_path: Path,
    *,
    temperature: float,
    timestep_ps: float,
    friction_per_ps: float,
    platform_name: str = "CPU",
    plumed_file: str | None = None,
) -> Simulation:
    if prmtop_path.suffix in {".top", ".itp"} or inpcrd_path.suffix == ".gro":
        gro = GromacsGroFile(str(inpcrd_path))
        top = GromacsTopFile(
            str(prmtop_path),
            periodicBoxVectors=gro.getPeriodicBoxVectors()
        )
        system = top.createSystem(
            nonbondedMethod=PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=HBonds,
        )
        positions = gro.positions
        box_vectors = gro.getPeriodicBoxVectors()
    else:
        top = AmberPrmtopFile(str(prmtop_path))
        crd = AmberInpcrdFile(str(inpcrd_path))
        system = top.createSystem(
            nonbondedMethod=PME,  # type: ignore
            nonbondedCutoff=1.0 * unit.nanometer,  # type: ignore
            constraints=HBonds,
        )
        positions = crd.positions
        box_vectors = crd.boxVectors

    if plumed_file is not None:
        import openmmplumed
        with open(plumed_file, "r") as f:
            script = f.read()
        system.addForce(openmmplumed.PlumedForce(script))
    integrator = LangevinMiddleIntegrator(
        temperature * unit.kelvin,  # type: ignore
        friction_per_ps / unit.picosecond,  # type: ignore
        timestep_ps * unit.picoseconds,  # type: ignore
    )
    platform = Platform.getPlatformByName(platform_name)
    simulation = Simulation(top.topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(*box_vectors)  # type: ignore
    return simulation


def run(case_dir: Path, config_name: str = "input.yaml") -> None:
    case_dir = case_dir.resolve()
    os.chdir(case_dir)
    cfg_model = load_config(case_dir / config_name)
    cfg = cfg_model.model_dump(exclude_none=True)

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

    # Gromacs or Amber topology info parsing
    if topology.suffix in {".top", ".itp"} or initial_restart.suffix == ".gro":
        topology_info = read_topology_info(initial_restart)
    else:
        topology_info = parse_prmtop(topology)

    # A .gro coordinate file carries no masses, so read_topology_info returns
    # placeholders. Recover the real ones from the topology, or a mass-weighted CV
    # would silently degrade to an unweighted centroid.
    if topology_info.get("masses_are_placeholder"):
        real_masses = read_masses_from_topology(topology)
        expected = len(topology_info.get("atom_names", []))
        if real_masses is not None and (expected == 0 or real_masses.size == expected):
            topology_info["masses"] = real_masses
            topology_info["masses_are_placeholder"] = False

    temperature = float(pg_cfg.get("temperature", 300.0))
    md_cfg = cfg.get("md", {})

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

    if topology.suffix in {".top", ".itp"} or initial_restart.suffix == ".gro":
        initial_coords = read_gro_coords(initial_restart)
    else:
        initial_coords = read_rst7_coords(initial_restart)

    initial_cv = proj_fn(initial_coords, **projection_args)
    print(f"Initial CV: {initial_cv}")

    mode = pg_cfg.get("mode", "target")
    target_projection = None
    if mode == "target":
        if "target_projection" not in pg_cfg:
            raise ValueError("pathgennie.mode is 'target', but pathgennie.target_projection is missing")
        target_projection = np.asarray(pg_cfg["target_projection"], dtype=float)
        if target_projection.ndim == 0:
            target_projection = target_projection.reshape(1)

    system_file = cfg.get("system", {}).get("system_file")
    if system_file is not None:
        system_file_path = resolve_case_path(case_dir, system_file)
        if system_file_path.exists():
            print(f"Loading custom system builder from {system_file_path}...")
            make_system_fn = load_function(case_dir, system_file_path.stem, "make_system")
            
            if topology.suffix in {".top", ".itp"} or initial_restart.suffix == ".gro":
                gro = GromacsGroFile(str(initial_restart))
                top = GromacsTopFile(
                    str(topology),
                    periodicBoxVectors=gro.getPeriodicBoxVectors()
                )
                positions = gro.positions
                box_vectors = gro.getPeriodicBoxVectors()
            else:
                top = AmberPrmtopFile(str(topology))
                crd = AmberInpcrdFile(str(initial_restart))
                positions = crd.positions
                box_vectors = crd.boxVectors

            timestep_ps = float(md_cfg.get("timestep_ps", 0.002))
            system, integrator = make_system_fn(
                top,
                temp=temperature,
                dt=timestep_ps,
                pressure=float(md_cfg.get("pressure", 1.0))
            )
            
            platform_name = openmm_cfg.get("platform", "CUDA")
            platform = Platform.getPlatformByName(platform_name)
            simulation = Simulation(top.topology, system, integrator, platform)
            simulation.context.setPositions(positions)
            if box_vectors is not None:
                simulation.context.setPeriodicBoxVectors(*box_vectors)  # type: ignore
        else:
            raise FileNotFoundError(f"Custom system file not found: {system_file_path}")
    else:
        timestep_ps = float(md_cfg.get("timestep_ps", 0.001))
        simulation = build_simulation(
            topology,
            initial_restart,
            temperature=temperature,
            timestep_ps=timestep_ps,
            friction_per_ps=float(md_cfg.get("friction_per_ps", 1.0)),
            platform_name=openmm_cfg.get("platform", "CUDA"),
            plumed_file=md_cfg.get("plumed_file", None),
        )

    equilibration_steps = int(md_cfg.get("equilibration_steps", 0))
    if equilibration_steps > 0:
        print(f"Running {equilibration_steps} equilibration steps...")
        from openmm.app import StateDataReporter
        import sys
        
        # Add a reporter to show progress during equilibration every 10% of the steps
        report_freq = max(1, equilibration_steps // 10)
        reporter = StateDataReporter(sys.stdout, report_freq, step=True, potentialEnergy=True, temperature=True, speed=True)
        simulation.reporters.append(reporter)
        
        simulation.step(equilibration_steps)
        
        # Remove the reporter so it doesn't clutter PathGennie's output
        simulation.reporters.remove(reporter)

    initial_pos = simulation.context.getState(getPositions=True).getPositions()
    
    if equilibration_steps > 0:
        # Re-evaluate the CV after equilibration
        pos_array = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        eq_cv = proj_fn(pos_array, **projection_args)
        print(f"Equilibrated CV: {eq_cv}")

    trajectory_path = output_dir / cfg.get("output", {}).get("trajectory", "reactive_path.dcd")
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

    runner = PathGennieMD(
        simulation=simulation,
        projection_fn=proj_fn,
        projection_args=projection_args,
        mode=mode,
        target_projection=target_projection,
        convergence_fn=conv_fn,
        convergence_args=convergence_args,
        escape_metric=pg_cfg.get("escape_metric", DEFAULT_ESCAPE_METRIC),
        periodic=cfg.get("projection", {}).get("periodic"),
        temperature=temperature,
        sigma=pg_cfg.get("sigma", 0.05),
        seed=pg_cfg.get("seed"),
        # Single-GPU saturation: run this many concurrent walkers on one card
        # ("auto" sizes from cores + free GPU memory); `devices` picks the card.
        workers_per_device=pg_cfg.get("workers_per_device", pg_cfg.get("tau1_workers", 1)),
        device=(pg_cfg.get("devices") or [None])[0],
        save_subframes=pg_cfg.get("save_subframes", False),
        subframe_stride=pg_cfg.get("subframe_stride", 1),
        checkpoint_freq=pg_cfg.get("checkpoint_freq", 0),
    )
    downstream = pg_cfg.get("downstream")
    result = runner.run(
        initial_pos=initial_pos,
        tau1=pg_cfg["tau1_steps"],
        tau2=pg_cfg["tau2_steps"],
        max_trial=pg_cfg["max_trial"],
        max_cycle=pg_cfg["max_cycle"],
        save_freq=pg_cfg.get("save_freq", 1),
        verbosity=pg_cfg.get("verbosity", 1),
        collect_seeds=bool(downstream),
        checkpoint_path=checkpoint_path,
        checkpoint_freq=pg_cfg.get("checkpoint_freq", 0),
    )
    seed_handles = None
    if downstream:
        trajectory, metrics, seed_handles = result
    else:
        trajectory, metrics = result
    if cfg.get("output", {}).get("wrap_pbc", False):
        trajectory = wrap_frames_pbc(trajectory, topology_info)
    # Physical time between saved frames: each saved cycle spans tau1 + tau2
    # integrator steps, and frames are kept every save_freq cycles.
    save_freq = int(pg_cfg.get("save_freq", 1))
    if pg_cfg.get("save_subframes", False):
        trajectory_dt = pg_cfg.get("subframe_stride", 1) * timestep_ps
    else:
        trajectory_dt = save_freq * (pg_cfg["tau1_steps"] + pg_cfg["tau2_steps"]) * timestep_ps
    write_trajectory(trajectory_path, topology_info, trajectory, dt=trajectory_dt)
    write_metrics_csv(metrics_path, metrics)

    if downstream:
        from pathgennie.sampling.runner import make_scalar_cv, run_downstream
        stage_cfg = dict(cfg.get(downstream, {}))
        component = stage_cfg.pop("cv_component", 0)
        scalar_cv = make_scalar_cv(proj_fn, projection_args, component)
        run_downstream(
            downstream, stage_cfg, engine=runner.engine, traj=trajectory, metrics=metrics,
            seed_handles=seed_handles, scalar_cv_fn=scalar_cv, output_dir=output_dir,
            executor=getattr(runner, "executor", None),
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
