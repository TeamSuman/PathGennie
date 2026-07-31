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
from typing import Any, Mapping

import numpy as np

from pathgennie.backends.amber.utils import read_native_trajectory
from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import ThreadDevicePool, resolve_cuda_visible_device
from pathgennie.core.progress import DEFAULT_ESCAPE_METRIC, EscapeMetric, TargetMetric
from pathgennie.core.strategy import resolve_profile
from pathgennie.utils.config import load_config
from pathgennie.utils.scratch import resolve_scratch_dir

from .utils import (
    enrich_args,
    load_function,
    read_gro_coords,
    read_masses_from_topology,
    read_topology_info,
    resolve_case_path,
    write_gro_coords,
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
    controls: Mapping[str, Any],
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


def format_ref_t(controls: Mapping[str, Any], temperature: float) -> str | float:
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


def normalize_mdp_controls(controls: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
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
        mdp_controls: Mapping[str, Any],
        maxwarn: int = 1,
        grompp_args: list[str] | None = None,
        mdrun_args: list[str] | None = None,
        env_overrides: Mapping[str, Any] | None = None,
        template_gro: Path | str | None = None,
    ):
        self.topology = str(topology)
        self.exe = str(executable)
        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = float(temperature)
        self.mdp_controls = dict(mdp_controls)
        self.maxwarn = int(maxwarn)
        self.grompp_args = grompp_args or []
        self.mdrun_args = mdrun_args or []
        self.env_overrides = {str(k): str(v) for k, v in (env_overrides or {}).items()}
        self.template_gro = str(template_gro) if template_gro else None
        self._counter = itertools.count()
        self._lock = threading.Lock()

    def _uid(self) -> str:
        with self._lock:
            n = next(self._counter)
        return f"{n}_{uuid.uuid4().hex[:8]}"

    def _device_dir(self, device: int | None) -> Path:
        name = "dev_cpu" if device is None else f"dev{device}"
        path = self.scratch_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def clone_anchor(self, handle: str) -> str:
        src_gro = Path(handle)
        src_cpt = src_gro.with_suffix(".cpt")
        dst_gro = self.scratch_dir / f"anchor_{self._uid()}.gro"
        dst_gro.write_bytes(src_gro.read_bytes())
        if src_cpt.exists():
            dst_gro.with_suffix(".cpt").write_bytes(src_cpt.read_bytes())
        return str(dst_gro)

    def run_segment(self, handle: str, n_steps: int, *, randomize_velocities: bool, seed: int, device: int | None = None,
                    save_subframes: bool = False, subframe_stride: int = 1) -> "str | tuple[str, np.ndarray]":
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

        # Build MDP controls, overriding nstxout-compressed when subframes are requested.
        seg_controls = dict(self.mdp_controls)
        if save_subframes:
            seg_controls["nstxout-compressed"] = int(subframe_stride)

        write_mdp(
            mdp, int(n_steps), self.temperature, seg_controls,
            generate_velocities=randomize_velocities, random_seed=int(seed),
        )

        env = os.environ.copy()
        visible = resolve_cuda_visible_device(device, os.environ)
        if visible is not None:
            env["CUDA_VISIBLE_DEVICES"] = visible
        env.update(self.env_overrides)

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

        result_handle = str(out_gro)
        if save_subframes:
            # The protocol says the return *changes shape* when subframes are
            # requested, and the driver unpacks unconditionally. Falling through to
            # a bare handle when the .xtc is missing makes that unpack raise on a
            # path string. A stride longer than the segment leaves mdrun with
            # nothing to write, so return an empty block and keep the contract.
            traj_xtc = prefix.with_suffix(".xtc")
            if traj_xtc.exists():
                return result_handle, read_native_trajectory(traj_xtc)
            return result_handle, self._empty_subframes(result_handle)
        return result_handle

    def _empty_subframes(self, handle: str) -> np.ndarray:
        """A correctly shaped ``(0, n_atoms, 3)`` block, concatenable with real ones."""
        try:
            n_atoms = int(np.asarray(self.get_coords(handle)).reshape(-1, 3).shape[0])
        except Exception:
            n_atoms = 0
        return np.empty((0, n_atoms, 3), dtype=np.float32)

    def get_coords(self, handle: str) -> np.ndarray:
        coords = read_gro_coords(handle)
        if not np.all(np.isfinite(coords)):
            raise ValueError(f"GROMACS segment produced non-finite coordinates: {handle}")
        return coords

    def release(self, handle: str) -> None:
        gro = Path(handle)
        for sibling in gro.parent.glob(gro.with_suffix("").name + ".*"):
            try:
                sibling.unlink()
            except OSError:
                pass

    def create_handle(self, coords: np.ndarray) -> str:
        """Write coordinates to a new .gro file and return its path."""
        if self.template_gro is None:
            raise RuntimeError("Cannot create handle from coords: no template_gro set on CoreGromacsEngine")
        gro_path = self.scratch_dir / f"ckpt_{self._uid()}.gro"
        write_gro_coords(self.template_gro, gro_path, coords)
        return str(gro_path)


    def _run(self, cmd: list[str], stage: str, env: Mapping[str, str]) -> None:
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
    cfg_model = load_config(case_dir / config_name)
    cfg = cfg_model.model_dump(exclude_none=True)

    workdir = resolve_case_path(case_dir, cfg.get("workdir", "pathgennie_gmx_run"))
    scratch_dir = resolve_scratch_dir(workdir, cfg.get("scratch_root"))
    output_dir = workdir / "output"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    gmx_cfg = cfg["gromacs"]
    pg_cfg = resolve_profile(cfg["pathgennie"])
    topology = resolve_case_path(case_dir, gmx_cfg["topology"])
    initial_structure = resolve_case_path(case_dir, gmx_cfg["initial_structure"])
    executable_str = gmx_cfg["executable"]
    expanded_path = str(Path(executable_str).expanduser())
    resolved_exe = shutil.which(expanded_path) or shutil.which(executable_str)
    if resolved_exe is None:
        raise FileNotFoundError(f"GROMACS executable not found: {executable_str}")
    executable = Path(resolved_exe)

    missing = [str(path) for path in (topology, initial_structure) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s): " + ", ".join(missing))

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

    # A .gro/.pdb reference template carries no masses, so read_topology_info returns
    # placeholders. Recover the real per-atom masses from the GROMACS topology, or a
    # mass-weighted CV would silently become an unweighted centroid.
    if topology_info.get("masses_are_placeholder"):
        real_masses = read_masses_from_topology(
            topology, include_dir=gmx_cfg.get("include_dir")
        )
        expected = len(topology_info.get("atom_names", []))
        if real_masses is not None and (expected == 0 or real_masses.size == expected):
            topology_info["masses"] = real_masses
            topology_info["masses_are_placeholder"] = False
        elif real_masses is not None:
            raise ValueError(
                f"Mass count from {topology} ({real_masses.size}) does not match the "
                f"reference template {metadata_source} ({expected} atoms). The topology "
                "and the reference structure describe different systems."
            )

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

    # Guard against CPU oversubscription: with several concurrent mdrun segments
    # each would otherwise claim every core. cpu_threads_per_worker pins the
    # per-worker OpenMP thread count (env + an explicit -ntomp if not already set).
    env_overrides = {}
    cpu_threads = pg_cfg.get("cpu_threads_per_worker")
    if cpu_threads:
        n_threads = str(int(cpu_threads))
        env_overrides = {
            "OMP_NUM_THREADS": n_threads,
            "MKL_NUM_THREADS": n_threads,
            "OPENBLAS_NUM_THREADS": n_threads,
        }
        if "-ntomp" not in mdrun_args:
            mdrun_args = mdrun_args + ["-ntomp", n_threads]

    proj_fn = load_function(case_dir, cfg["projection"]["module"], cfg["projection"]["function"])
    conv_fn = load_function(case_dir, cfg["convergence"]["module"], cfg["convergence"]["function"])

    projection_args = {
        key: value
        for key, value in cfg.get("projection", {}).items()
        if key not in {"module", "function", "reference", "periodic"}
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
    else:
        progress = EscapeMetric(
            proj_fn, start_cv, projection_args=projection_args,
            escape_metric=pg_cfg.get("escape_metric", DEFAULT_ESCAPE_METRIC),
            periodic=cv_periodic,
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
        env_overrides=env_overrides,
        template_gro=initial_structure,
    )

    devices = pg_cfg.get("devices", gmx_cfg.get("devices"))
    workers_per_device = int(pg_cfg.get("workers_per_device", pg_cfg.get("tau1_workers", 1)))
    executor = ThreadDevicePool(devices=devices, workers_per_device=workers_per_device)

    trajectory_path = output_dir / cfg.get("output", {}).get("trajectory", "reactive_path.xtc")
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
        str(initial_structure),
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
    # Physical time between saved frames: each saved cycle spans tau1 + tau2
    # integrator steps, and frames are kept every save_freq cycles.
    # float() cast required: read_mdp() returns dict[str, str].
    timestep_ps = float(mdp_controls.get("dt", 0.002))
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
    parser.add_argument("--case", type=Path, default=Path.cwd(), help="Directory containing input.yaml")
    parser.add_argument("--config", default="input.yaml", help="YAML config name inside the case directory")
    args = parser.parse_args()
    run(args.case, args.config)


if __name__ == "__main__":
    main()
