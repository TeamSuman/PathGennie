#!/usr/bin/env python
"""Generic PathGennie Amber runner driven by a case-local input.yaml."""

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
    default_mdin_controls,
    enrich_args,
    load_function,
    parse_prmtop,
    read_rst7_coords,
    resolve_case_path,
    wrap_frames_pbc,
    write_mdin,
    write_metrics_csv,
    write_trajectory,
)

RNG = random.SystemRandom()


class GenericAmberEngine:
    def __init__(
        self,
        *,
        topology: Path,
        executable: Path,
        scratch_dir: Path,
        tau1_steps: int,
        tau2_steps: int,
        temperature: float,
        mdin_controls: dict[str, object],
        extra_mdin_text: str = "",
        command_prefix: list[str] | None = None,
    ):
        self.topology = str(topology)
        self.exe = str(executable)
        self.scratch_dir = scratch_dir
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.tau1_steps = int(tau1_steps)
        self.tau2_steps = int(tau2_steps)
        self.temperature = float(temperature)
        self.mdin_controls = mdin_controls
        self.extra_mdin_text = extra_mdin_text
        self.command_prefix = command_prefix or []

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

    def copy_state(self, src, dst):
        dst_path = Path(dst)
        if not dst_path.is_absolute():
            dst_path = self.scratch_dir / dst_path
        shutil.copy(self.state_path(src), dst_path)

    def run_segment(self, input_rst, output_prefix):
        output_name = Path(output_prefix).name
        is_tau2 = output_name.startswith("tau2_")
        output_prefix = self.scratch_dir / output_name
        out_rst = output_prefix.with_suffix(".rst7")
        mdin = self.scratch_dir / f"{output_name}.mdin"
        write_mdin(
            mdin,
            self.tau2_steps if is_tau2 else self.tau1_steps,
            self.temperature,
            self.mdin_controls,
            continue_velocities=is_tau2,
            random_seed=RNG.randint(1, 2_000_000_000),
            extra_text=self.extra_mdin_text,
        )

        cmd = [
            *self.command_prefix,
            self.exe,
            "-O",
            "-i",
            str(mdin),
            "-p",
            self.topology,
            "-c",
            str(self.state_path(input_rst)),
            "-r",
            str(out_rst),
            "-o",
            str(output_prefix.with_suffix(".out")),
            "-inf",
            str(output_prefix.with_suffix(".mdinfo")),
        ]
        if self.mdin_controls.get("ntwx", 0):
            cmd.extend(["-x", str(output_prefix.with_suffix(".nc"))])
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return str(out_rst)

    def load_coords(self, rst):
        return read_rst7_coords(self.state_path(rst))


class GenericPathGennieAmber:
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

    def run(self, initial_restart, max_trial=20, max_cycle=1000, save_freq=10):
        anchor_rst = "anchor.rst7"
        self.engine.copy_state(initial_restart, anchor_rst)
        start_pos = self.engine.load_coords(anchor_rst)
        start_proj = self.proj_fn(start_pos, **self.proj_args)
        anchor_pos = start_pos
        anchor_cv = start_proj
        anchor_metric = self.metric(anchor_cv, start_proj)

        trajectory = []
        metric_history = []

        def run_trial(cycle, trial):
            trial_input = f"trial_{trial}.rst7"
            self.engine.copy_state(anchor_rst, trial_input)
            tau1_rst = self.engine.run_segment(trial_input, f"tau1_{cycle}_{trial}")
            pos = self.engine.load_coords(tau1_rst)
            cv = self.proj_fn(pos, **self.proj_args)
            return {"rst": tau1_rst, "metric": self.metric(cv, start_proj), "cv": cv}

        for cycle in trange(max_cycle):
            previous_anchor_rst = anchor_rst
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
            tau2_rst = self.engine.run_segment(chosen["rst"], f"tau2_{cycle}")
            tau2_pos = self.engine.load_coords(tau2_rst)
            tau2_cv = self.proj_fn(tau2_pos, **self.proj_args)
            tau2_metric = self.metric(tau2_cv, start_proj)

            if self.reject_worse_tau2 and tau2_metric < chosen["metric"]:
                anchor_rst = chosen["rst"]
                pos = self.engine.load_coords(anchor_rst)
                cv = chosen["cv"]
                metric = chosen["metric"]
                print(f"Cycle {cycle}: rejected tau2 metric={tau2_metric:.4f}; kept tau1 metric={metric:.4f}, CV={cv}")
            else:
                anchor_rst = tau2_rst
                pos = tau2_pos
                cv = tau2_cv
                metric = tau2_metric

            if self.reject_worse_anchor and metric < anchor_metric:
                print(
                    f"Cycle {cycle}: rejected candidate metric={metric:.4f}; "
                    f"kept anchor metric={anchor_metric:.4f}, CV={anchor_cv}"
                )
                anchor_rst = previous_anchor_rst
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

    workdir = resolve_case_path(case_dir, cfg.get("workdir", "pathgennie_run"))
    scratch_dir = workdir / "scratch"
    output_dir = workdir / "output"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    amber_cfg = cfg["amber"]
    pg_cfg = cfg["pathgennie"]
    topology = resolve_case_path(case_dir, amber_cfg["topology"])
    initial_restart = resolve_case_path(case_dir, amber_cfg["initial_restart"])
    executable = Path(amber_cfg["executable"]).expanduser()

    missing = [str(path) for path in (topology, initial_restart) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s): " + ", ".join(missing))
    if not executable.exists():
        raise FileNotFoundError(f"Amber executable not found: {executable}")

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

    write_mdin(
        workdir / "tau1.mdin",
        pg_cfg["tau1_steps"],
        temperature,
        mdin_controls,
        continue_velocities=False,
        random_seed=-1,
        extra_text=extra_mdin_text,
    )
    write_mdin(
        workdir / "tau2.mdin",
        pg_cfg["tau2_steps"],
        temperature,
        mdin_controls,
        continue_velocities=True,
        random_seed=-1,
        extra_text=extra_mdin_text,
    )
    print(f"Using generated Amber mdin files: {workdir / 'tau1.mdin'} and {workdir / 'tau2.mdin'}")

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

    mode = pg_cfg.get("mode", "escape")
    target_projection = None
    if mode == "target":
        if "target_projection" not in pg_cfg:
            raise ValueError("pathgennie.mode is 'target', but pathgennie.target_projection is missing")
        target_projection = np.asarray(pg_cfg["target_projection"], dtype=float)
        if target_projection.ndim == 0:
            target_projection = target_projection.reshape(1)

    engine = GenericAmberEngine(
        topology=topology,
        executable=executable,
        scratch_dir=scratch_dir,
        tau1_steps=pg_cfg["tau1_steps"],
        tau2_steps=pg_cfg["tau2_steps"],
        temperature=temperature,
        mdin_controls=mdin_controls,
        extra_mdin_text=extra_mdin_text,
        command_prefix=command_prefix,
    )
    runner = GenericPathGennieAmber(
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
        initial_restart=str(initial_restart),
        max_trial=pg_cfg["max_trial"],
        max_cycle=pg_cfg["max_cycle"],
        save_freq=pg_cfg.get("save_freq", 10),
    )

    trajectory_path = output_dir / cfg.get("output", {}).get("trajectory", "reactive_path.pdb")
    metrics_path = output_dir / cfg.get("output", {}).get("metrics", "metrics.csv")
    if cfg.get("output", {}).get("wrap_pbc", False):
        traj = wrap_frames_pbc(traj, topology_info)
    write_trajectory(trajectory_path, topology_info, traj)
    write_metrics_csv(metrics_path, metrics)
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
