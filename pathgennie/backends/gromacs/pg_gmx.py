#!/usr/bin/env python
"""Generic PathGennie GROMACS runner driven by a case-local input.yaml."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

try:
    from tqdm.auto import trange  # type: ignore
except ModuleNotFoundError:

    def trange(*args, **kwargs):
        return range(*args)


from .utils import (
    enrich_args,
    load_function,
    read_gro_coords,
    read_topology_info,
    resolve_case_path,
    write_metrics_csv,
    write_trajectory,
)

RNG = random.SystemRandom()

IGNORED_AMBER_MDP_KEYS = {
    "cut",
    "gamma-ln",
    "igb",
    "imin",
    "ioutfm",
    "irest",
    "ntb",
    "ntc",
    "ntf",
    "ntp",
    "ntpr",
    "ntwx",
    "ntwr",
    "ntx",
    "ntxo",
    "temp0",
    "tempi",
}


def write_mdp(
    path: Path,
    nsteps: int,
    temperature: float,
    controls: dict[str, object],
    *,
    generate_velocities: bool,
    random_seed: int,
) -> None:
    values = dict(controls)
    values.update(
        {
            "nsteps": int(nsteps),
            "ref-t": format_ref_t(values, temperature),
            "gen-temp": float(temperature),
            "gen-vel": "yes" if generate_velocities else "no",
            "gen-seed": int(random_seed) if generate_velocities else -1,
            "continuation": "no" if generate_velocities else "yes",
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["; PathGennie GROMACS MD"]
    for key, value in values.items():
        if isinstance(value, bool):
            value = "yes" if value else "no"
        lines.append(f"{key} = {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_ref_t(controls: dict[str, object], temperature: float) -> str | float:
    tc_groups = str(controls.get("tc-grps", "")).split()
    if len(tc_groups) <= 1:
        return float(temperature)
    return " ".join(f"{float(temperature):.6g}" for _ in tc_groups)


def read_mdp(path: Path) -> dict[str, str]:
    controls: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        controls[normalize_mdp_key(key)] = value.strip()
    return controls


def normalize_mdp_key(key: object) -> str:
    return str(key).strip().replace("_", "-")


def normalize_mdp_controls(controls: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in controls.items():
        mdp_key = normalize_mdp_key(key)
        if mdp_key in IGNORED_AMBER_MDP_KEYS:
            continue
        normalized[mdp_key] = value
    return normalized


def split_grompp_only_args(
    grompp_args: list[str],
    mdrun_args: list[str],
) -> tuple[list[str], list[str]]:
    """Move legacy mdrun_args that actually belong to grompp."""

    remaining_mdrun_args: list[str] = []
    index = 0
    while index < len(mdrun_args):
        arg = mdrun_args[index]
        if arg == "-n":
            if index + 1 >= len(mdrun_args):
                raise ValueError("GROMACS argument '-n' requires an index file path")
            grompp_args.extend([arg, mdrun_args[index + 1]])
            index += 2
            continue
        remaining_mdrun_args.append(arg)
        index += 1
    return grompp_args, remaining_mdrun_args


class GromacsState:
    def __init__(self, gro: Path, cpt: Path | None = None):
        self.gro = gro
        self.cpt = cpt if cpt is not None and cpt.exists() else None


class GenericGromacsEngine:
    def __init__(
        self,
        *,
        topology: Path,
        executable: Path,
        scratch_dir: Path,
        tau1_steps: int,
        tau2_steps: int,
        temperature: float,
        mdp_controls: dict[str, object],
        maxwarn: int = 1,
        grompp_args: list[str] | None = None,
        mdrun_args: list[str] | None = None,
    ):
        self.topology = str(topology)
        self.exe = str(executable)
        self.scratch_dir = scratch_dir
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.tau1_steps = int(tau1_steps)
        self.tau2_steps = int(tau2_steps)
        self.temperature = float(temperature)
        self.mdp_controls = mdp_controls
        self.maxwarn = int(maxwarn)
        self.grompp_args = grompp_args or []
        self.mdrun_args = mdrun_args or []

    def state_path(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        scratch_path = self.scratch_dir / path
        if scratch_path.exists():
            return scratch_path
        if path.exists():
            return path
        return scratch_path

    def state(self, value: str | Path | GromacsState) -> GromacsState:
        if isinstance(value, GromacsState):
            return value
        gro = self.state_path(value)
        cpt = gro.with_suffix(".cpt")
        return GromacsState(gro=gro, cpt=cpt if cpt.exists() else None)

    def copy_state(self, src, dst):
        src_state = self.state(src)
        dst_path = Path(dst)
        if dst_path.suffix != ".gro":
            dst_path = dst_path.with_suffix(".gro")
        if not dst_path.is_absolute():
            dst_path = self.scratch_dir / dst_path
        shutil.copy(src_state.gro, dst_path)
        dst_cpt = dst_path.with_suffix(".cpt")
        if src_state.cpt is not None:
            shutil.copy(src_state.cpt, dst_cpt)
        elif dst_cpt.exists():
            dst_cpt.unlink()
        return GromacsState(dst_path, dst_cpt)

    def run_segment(self, input_state, output_prefix):
        input_state = self.state(input_state)
        output_name = Path(output_prefix).name
        is_tau2 = output_name.startswith("tau2_")
        output_prefix = self.scratch_dir / output_name
        mdp = self.scratch_dir / f"{output_name}.mdp"
        tpr = output_prefix.with_suffix(".tpr")
        out_gro = output_prefix.with_suffix(".gro")
        out_cpt = output_prefix.with_suffix(".cpt")

        write_mdp(
            mdp,
            self.tau2_steps if is_tau2 else self.tau1_steps,
            self.temperature,
            self.mdp_controls,
            generate_velocities=not is_tau2,
            random_seed=RNG.randint(1, 2_000_000_000),
        )

        grompp_cmd = [
            self.exe,
            "grompp",
            "-f",
            str(mdp),
            "-c",
            str(input_state.gro),
            "-p",
            self.topology,
            "-o",
            str(tpr),
            "-po",
            str(output_prefix.with_suffix(".mdout.mdp")),
            "-maxwarn",
            str(self.maxwarn),
            *self.grompp_args,
        ]
        if is_tau2 and input_state.cpt is not None:
            grompp_cmd.extend(["-t", str(input_state.cpt)])

        self._run(grompp_cmd, "grompp")

        mdrun_cmd = [
            self.exe,
            "mdrun",
            "-s",
            str(tpr),
            "-deffnm",
            str(output_prefix),
            "-c",
            str(out_gro),
            "-cpo",
            str(out_cpt),
            *self.mdrun_args,
        ]
        self._run(mdrun_cmd, "mdrun")
        return GromacsState(out_gro, out_cpt)

    def load_coords(self, state):
        return read_gro_coords(self.state(state).gro)

    def _run(self, cmd: list[str], stage: str) -> None:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            message = [
                f"GROMACS {stage} failed with exit code {exc.returncode}.",
                "Command: " + " ".join(cmd),
            ]
            if exc.stdout:
                message.append("stdout:\n" + exc.stdout[-4000:])
            if exc.stderr:
                message.append("stderr:\n" + exc.stderr[-4000:])
            raise RuntimeError("\n".join(message)) from exc


class GenericPathGennieGromacs:
    def __init__(
        self,
        *,
        engine,
        projection_fn,
        convergence_fn,
        sigma=0.1,
        mode="escape",
        target_projection=None,
        projection_args=None,
        convergence_args=None,
        tau1_workers=1,
        escape_metric="cv0",
        reject_worse_tau2=False,
        reject_worse_anchor=False,
    ):
        self.engine = engine
        self.proj_fn = projection_fn
        self.conv_fn = convergence_fn
        self.mode = mode
        self.sigma = sigma
        self.target = target_projection
        self.proj_args = projection_args or {}
        self.conv_args = convergence_args or {}
        self.tau1_workers = max(1, int(tau1_workers))
        self.escape_metric = escape_metric
        self.reject_worse_tau2 = bool(reject_worse_tau2)
        self.reject_worse_anchor = bool(reject_worse_anchor)

    def metric(self, cv, start_proj):
        if self.mode == "escape":
            if self.escape_metric == "distance_from_start":
                return np.linalg.norm(cv - start_proj)
            return float(cv[0])
        return -np.linalg.norm(cv - self.target)

    def run(self, initial_state, max_trial=20, max_cycle=1000, save_freq=10):
        anchor_state = self.engine.copy_state(initial_state, "anchor.gro")
        start_pos = self.engine.load_coords(anchor_state)
        start_proj = self.proj_fn(start_pos, **self.proj_args)
        anchor_pos = start_pos
        anchor_cv = start_proj
        anchor_metric = self.metric(anchor_cv, start_proj)

        trajectory = []
        metric_history = []

        def run_trial(cycle, trial):
            trial_input = self.engine.copy_state(anchor_state, f"trial_{trial}.gro")
            tau1_state = self.engine.run_segment(trial_input, f"tau1_{cycle}_{trial}")
            pos = self.engine.load_coords(tau1_state)
            cv = self.proj_fn(pos, **self.proj_args)
            return {"state": tau1_state, "metric": self.metric(cv, start_proj), "cv": cv}

        for cycle in trange(max_cycle):
            previous_anchor_state = anchor_state
            workers = min(self.tau1_workers, max_trial)
            if workers == 1:
                trials = [run_trial(cycle, trial) for trial in range(max_trial)]
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    trials = list(executor.map(lambda trial: run_trial(cycle, trial), range(max_trial)))

            metrics = np.array([trial["metric"] for trial in trials])
            mmin = metrics.min()
            mmax = metrics.max()
            if abs(mmax - mmin) < 1e-12:
                probs = np.ones(len(metrics)) / len(metrics)
            else:
                scaled = (metrics - mmin) / (mmax - mmin)
                weights = np.exp((scaled - 1.0) / self.sigma)
                probs = weights / weights.sum()

            chosen = trials[np.random.choice(len(trials), p=probs)]
            tau2_state = self.engine.run_segment(chosen["state"], f"tau2_{cycle}")
            tau2_pos = self.engine.load_coords(tau2_state)
            tau2_cv = self.proj_fn(tau2_pos, **self.proj_args)
            tau2_metric = self.metric(tau2_cv, start_proj)

            if self.reject_worse_tau2 and tau2_metric < chosen["metric"]:
                anchor_state = chosen["state"]
                pos = self.engine.load_coords(anchor_state)
                cv = chosen["cv"]
                metric = chosen["metric"]
            else:
                anchor_state = tau2_state
                pos = tau2_pos
                cv = tau2_cv
                metric = tau2_metric

            if self.reject_worse_anchor and metric < anchor_metric:
                anchor_state = previous_anchor_state
                pos = anchor_pos
                cv = anchor_cv
                metric = anchor_metric
            else:
                anchor_pos = pos
                anchor_cv = cv
                anchor_metric = metric
            metric_history.append(metric)

            if cycle % save_freq == 0:
                print(f"Cycle {cycle}: metric={metric:.4f}, CV={cv}")
                trajectory.append(pos.copy())

            if self.conv_fn(pos, **self.conv_args):
                print(f"Converged at cycle {cycle}")
                break

        return np.asarray(trajectory), np.asarray(metric_history)


def run(case_dir: Path, config_name: str = "input.yaml") -> None:
    case_dir = case_dir.resolve()
    os.chdir(case_dir)
    cfg = yaml.safe_load((case_dir / config_name).read_text(encoding="utf-8"))

    workdir = resolve_case_path(case_dir, cfg.get("workdir", "pathgennie_gmx_run"))
    scratch_dir = workdir / "scratch"
    output_dir = workdir / "output"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    gmx_cfg = cfg["gromacs"]
    pg_cfg = cfg["pathgennie"]
    topology = resolve_case_path(case_dir, gmx_cfg["topology"])
    initial_structure = resolve_case_path(case_dir, gmx_cfg["initial_structure"])
    executable = Path(gmx_cfg["executable"]).expanduser()

    missing = [str(path) for path in (topology, initial_structure) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s): " + ", ".join(missing))
    if not executable.exists():
        raise FileNotFoundError(f"GROMACS executable not found: {executable}")

    metadata_source = resolve_case_path(
        case_dir,
        gmx_cfg.get(
            "reference_template",
            gmx_cfg.get("reference_structure", gmx_cfg.get("metadata", initial_structure)),
        ),
    )
    if not metadata_source.exists():
        raise FileNotFoundError(f"GROMACS reference template file not found: {metadata_source}")
    topology_info = read_topology_info(metadata_source)
    temperature = float(pg_cfg.get("temperature", 300.0))
    mdp_path = resolve_case_path(case_dir, gmx_cfg.get("mdp", "md.mdp"))
    if not mdp_path.exists():
        raise FileNotFoundError(f"GROMACS MDP file not found: {mdp_path}")
    md_cfg = cfg.get("md", {})
    mdp_controls = read_mdp(mdp_path)
    mdp_controls.update(normalize_mdp_controls(md_cfg.get("controls", {})))
    grompp_args = [str(arg) for arg in gmx_cfg.get("grompp_args", [])]
    mdrun_args = [str(arg) for arg in gmx_cfg.get("mdrun_args", [])]
    grompp_args, mdrun_args = split_grompp_only_args(grompp_args, mdrun_args)

    write_mdp(
        workdir / "tau1.mdp",
        pg_cfg["tau1_steps"],
        temperature,
        mdp_controls,
        generate_velocities=True,
        random_seed=-1,
    )
    write_mdp(
        workdir / "tau2.mdp",
        pg_cfg["tau2_steps"],
        temperature,
        mdp_controls,
        generate_velocities=False,
        random_seed=-1,
    )
    print(
        f"Using generated GROMACS mdp files: {workdir / 'tau1.mdp'} "
        f"and {workdir / 'tau2.mdp'}"
    )

    proj_fn = load_function(case_dir, cfg["projection"]["module"], cfg["projection"]["function"])
    conv_fn = load_function(case_dir, cfg["convergence"]["module"], cfg["convergence"]["function"])

    projection_args = {
        key: value
        for key, value in cfg.get("projection", {}).items()
        if key not in {"module", "function", "reference"}
    }
    convergence_args = {
        key: value
        for key, value in cfg.get("convergence", {}).items()
        if key not in {"module", "function"}
    }
    projection_args = enrich_args(projection_args, topology_info)
    convergence_args = enrich_args(convergence_args, topology_info)

    initial_cv = proj_fn(read_gro_coords(initial_structure), **projection_args)
    print(f"Initial CV: {initial_cv}")

    mode = pg_cfg.get("mode", "escape")
    target_projection = None
    if mode == "target":
        if "target_projection" not in pg_cfg:
            raise ValueError("pathgennie.mode is 'target', but pathgennie.target_projection is missing")
        target_projection = np.asarray(pg_cfg["target_projection"], dtype=float)
        if target_projection.ndim == 0:
            target_projection = target_projection.reshape(1)

    engine = GenericGromacsEngine(
        topology=topology,
        executable=executable,
        scratch_dir=scratch_dir,
        tau1_steps=pg_cfg["tau1_steps"],
        tau2_steps=pg_cfg["tau2_steps"],
        temperature=temperature,
        mdp_controls=mdp_controls,
        maxwarn=gmx_cfg.get("maxwarn", 1),
        grompp_args=grompp_args,
        mdrun_args=mdrun_args,
    )
    runner = GenericPathGennieGromacs(
        engine=engine,
        projection_fn=proj_fn,
        convergence_fn=conv_fn,
        sigma=pg_cfg["sigma"],
        mode=mode,
        target_projection=target_projection,
        projection_args=projection_args,
        convergence_args=convergence_args,
        tau1_workers=pg_cfg.get("tau1_workers", 1),
        escape_metric=pg_cfg.get("escape_metric", "cv0"),
        reject_worse_tau2=pg_cfg.get("reject_worse_tau2", False),
        reject_worse_anchor=pg_cfg.get("reject_worse_anchor", False),
    )

    traj, metrics = runner.run(
        initial_state=str(initial_structure),
        max_trial=pg_cfg["max_trial"],
        max_cycle=pg_cfg["max_cycle"],
        save_freq=pg_cfg.get("save_freq", 10),
    )

    trajectory_path = output_dir / cfg.get("output", {}).get("trajectory", "reactive_path.xtc")
    metrics_path = output_dir / cfg.get("output", {}).get("metrics", "metrics.csv")
    write_trajectory(trajectory_path, topology_info, traj)
    write_metrics_csv(metrics_path, metrics)
    shutil.rmtree(scratch_dir)

    print(f"Saved frames: {len(traj)}")
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
