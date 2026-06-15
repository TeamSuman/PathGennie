#!/usr/bin/env python
"""Generic PathGennie GROMACS runner driven by a case-local input.yaml.

The adaptive cycle lives in :mod:`pathgennie.core`; this module loads the case
configuration, builds a device-aware :class:`CoreGromacsEngine`, and runs the
shared :class:`~pathgennie.core.driver.PathGennieDriver` over a device pool so the
swarm spreads across all configured GPUs (each segment exports
``CUDA_VISIBLE_DEVICES`` for its assigned device and uses an isolated scratch
subdirectory).
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

import numpy as np
import yaml

from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import ThreadDevicePool
from pathgennie.core.progress import EscapeMetric, TargetMetric

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
    "cut", "gamma-ln", "igb", "imin", "ioutfm", "irest", "ntb", "ntc", "ntf",
    "ntp", "ntpr", "ntwx", "ntwr", "ntx", "ntxo", "temp0", "tempi",
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


class CoreGromacsEngine:
    """Device-aware GROMACS engine implementing the core Engine protocol.

    A *handle* is the path to a ``.gro`` structure; the matching checkpoint is the
    sibling ``.cpt`` (carried so tau2 runners can continue velocities).
    """

    def __init__(
        self,
        *,
        topology: Path,
        executable: Path,
        scratch_dir: Path,
        temperature: float,
        mdp_controls: dict[str, object],
        maxwarn: int = 1,
        grompp_args: list[str] | None = None,
        mdrun_args: list[str] | None = None,
    ):
        self.topology = str(topology)
        self.exe = str(executable)
        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = float(temperature)
        self.mdp_controls = mdp_controls
        self.maxwarn = int(maxwarn)
        self.grompp_args = grompp_args or []
        self.mdrun_args = mdrun_args or []
        self._counter = itertools.count()
        self._lock = threading.Lock()

    def _uid(self) -> str:
        with self._lock:
            n = next(self._counter)
        return f"{n}_{uuid.uuid4().hex[:8]}"

    def _device_dir(self, device):
        name = "dev_cpu" if device is None else f"dev{device}"
        path = self.scratch_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def clone_anchor(self, handle):
        src_gro = Path(handle)
        src_cpt = src_gro.with_suffix(".cpt")
        dst_gro = self.scratch_dir / f"anchor_{self._uid()}.gro"
        dst_gro.write_bytes(src_gro.read_bytes())
        if src_cpt.exists():
            dst_gro.with_suffix(".cpt").write_bytes(src_cpt.read_bytes())
        return str(dst_gro)

    def run_segment(self, handle, n_steps, *, randomize_velocities, seed, device=None):
        is_tau2 = not randomize_velocities
        workdir = self._device_dir(device)
        stem = f"seg_{self._uid()}"
        prefix = workdir / stem
        in_gro = Path(handle)
        in_cpt = in_gro.with_suffix(".cpt")
        mdp = workdir / f"{stem}.mdp"
        tpr = prefix.with_suffix(".tpr")
        out_gro = prefix.with_suffix(".gro")
        out_cpt = prefix.with_suffix(".cpt")

        write_mdp(
            mdp, int(n_steps), self.temperature, self.mdp_controls,
            generate_velocities=randomize_velocities, random_seed=int(seed),
        )

        env = os.environ.copy()
        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(device)

        grompp_cmd = [
            self.exe, "grompp", "-f", str(mdp), "-c", str(in_gro), "-p", self.topology,
            "-o", str(tpr), "-po", str(prefix.with_suffix(".mdout.mdp")),
            "-maxwarn", str(self.maxwarn), *self.grompp_args,
        ]
        if is_tau2 and in_cpt.exists():
            grompp_cmd.extend(["-t", str(in_cpt)])
        self._run(grompp_cmd, "grompp", env)

        mdrun_cmd = [
            self.exe, "mdrun", "-s", str(tpr), "-deffnm", str(prefix),
            "-c", str(out_gro), "-cpo", str(out_cpt), *self.mdrun_args,
        ]
        self._run(mdrun_cmd, "mdrun", env)
        return str(out_gro)

    def get_coords(self, handle):
        coords = read_gro_coords(handle)
        if not np.all(np.isfinite(coords)):
            raise ValueError(f"GROMACS segment produced non-finite coordinates: {handle}")
        return coords

    def release(self, handle):
        gro = Path(handle)
        for sibling in gro.parent.glob(gro.with_suffix("").name + ".*"):
            try:
                sibling.unlink()
            except OSError:
                pass

    def _run(self, cmd, stage, env):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
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

    start_cv = np.asarray(proj_fn(read_gro_coords(initial_structure), **projection_args), dtype=float)
    print(f"Initial CV: {start_cv}")

    mode = pg_cfg.get("mode", "escape")
    if mode == "target":
        if "target_projection" not in pg_cfg:
            raise ValueError("pathgennie.mode is 'target', but pathgennie.target_projection is missing")
        target_projection = np.asarray(pg_cfg["target_projection"], dtype=float).reshape(-1)
        progress = TargetMetric(proj_fn, target_projection, projection_args=projection_args)
    else:
        progress = EscapeMetric(
            proj_fn, start_cv, projection_args=projection_args,
            escape_metric=pg_cfg.get("escape_metric", "cv0"),
        )

    def convergence(coords: np.ndarray) -> bool:
        return bool(conv_fn(coords, **convergence_args))

    engine = CoreGromacsEngine(
        topology=topology,
        executable=executable,
        scratch_dir=scratch_dir,
        temperature=temperature,
        mdp_controls=mdp_controls,
        maxwarn=gmx_cfg.get("maxwarn", 1),
        grompp_args=grompp_args,
        mdrun_args=mdrun_args,
    )

    devices = pg_cfg.get("devices", gmx_cfg.get("devices"))
    workers_per_device = int(pg_cfg.get("workers_per_device", pg_cfg.get("tau1_workers", 1)))
    executor = ThreadDevicePool(devices=devices, workers_per_device=workers_per_device)

    driver = PathGennieDriver(
        engine, progress, convergence,
        executor=executor,
        sigma=pg_cfg["sigma"],
        seed=pg_cfg.get("seed"),
        reject_worse_tau2=pg_cfg.get("reject_worse_tau2", False),
        reject_worse_anchor=pg_cfg.get("reject_worse_anchor", False),
        verbosity=pg_cfg.get("verbosity", 1),
    )

    traj, metrics = driver.run(
        str(initial_structure),
        tau1=pg_cfg["tau1_steps"],
        tau2=pg_cfg["tau2_steps"],
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
