#!/usr/bin/env python
"""Run a PathGennie example as an HPC smoke / multi-GPU placement test.

This drives a *real* MD backend (AMBER / GROMACS / OpenMM), so it must run on a
node where the corresponding executable and (for GPU queues) CUDA are available.
It:

  1. copies an example case into a scratch working directory,
  2. rewrites its ``input.yaml`` for a short smoke run (small ``max_cycle``,
     your executable path, and the device list you pass),
  3. optionally samples ``nvidia-smi`` while the run proceeds to record which
     physical GPUs actually hosted MD processes (this is the multi-GPU spread
     check for the AMBER/GROMACS backends),
  4. writes a JSON result and exits non-zero on failure.

Examples
--------
    # CPU smoke on a GROMACS build
    python tests/hpc/run_example.py --example examples/alanine_dipeptide/gromacs \\
        --backend gromacs --executable "$(command -v gmx)" --max-cycle 5 \\
        --out results/smoke_gromacs.json

    # Multi-GPU spread check on AMBER (expects >=2 allocated GPUs)
    python tests/hpc/run_example.py --example examples/alanine_dipeptide/amber \\
        --backend amber --executable "$(command -v pmemd.cuda)" \\
        --devices 0,1 --workers-per-device 2 --max-cycle 10 --monitor-gpu \\
        --out results/gpu_spread_amber.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# backend -> (config section holding 'executable', runner module, runner func)
BACKENDS = {
    "amber": ("amber", "pathgennie.backends.amber.pg_amber"),
    "gromacs": ("gromacs", "pathgennie.backends.gromacs.pg_gmx"),
    "openmm": ("openmm", "pathgennie.backends.openmm.pg_openmm"),
}


class GpuMonitor(threading.Thread):
    """Poll nvidia-smi compute-apps and record the set of GPUs that ran a process."""

    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()
        self.gpu_indices: set[str] = set()
        self.samples: list[str] = []
        self.available = shutil.which("nvidia-smi") is not None

    def run(self):
        if not self.available:
            return
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip()
                if out:
                    self.samples.append(out)
                # Also map UUID -> index for a readable index set.
                idx = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,gpu_uuid,utilization.gpu",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip()
                for line in idx.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3 and parts[2].rstrip(" %").isdigit():
                        if int(parts[2].rstrip(" %")) > 0:
                            self.gpu_indices.add(parts[0])
            except Exception:  # noqa: BLE001 - best-effort monitor
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


def prepare_case(example: Path, backend: str, executable: str, devices, workers,
                 max_cycle, max_trial, cpu_threads, tmp: Path):
    section, _ = BACKENDS[backend]
    root = tmp / f"pg_hpc_{backend}_{os.getpid()}"
    if root.exists():
        shutil.rmtree(root)
    # An example's projection.py may load a shared module from a SIBLING directory
    # (examples/alanine_dipeptide/{amber,gromacs}/projection.py both resolve
    # `parents[1] / "common" / "phi_psi.py"`). Copying only the backend directory
    # flattens that away, so the import failed with FileNotFoundError on
    # `<scratch>/common/phi_psi.py` -- which is why the alanine dipeptide example
    # named as the DEFAULT in every HPC template could never actually run here.
    # Stage the example under its own name and bring the siblings it can reference
    # along at the same relative depth.
    root.mkdir(parents=True)
    work = root / example.name
    shutil.copytree(example, work)
    for sibling in ("common",):
        src = example.parent / sibling
        if src.is_dir():
            shutil.copytree(src, root / sibling)

    cfg_path = work / "input.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.setdefault(section, {})["executable"] = executable
    pg = cfg.setdefault("pathgennie", {})
    pg["max_cycle"] = int(max_cycle)
    if max_trial is not None:
        pg["max_trial"] = int(max_trial)
    if devices is not None:
        pg["devices"] = devices
    if workers is not None:
        pg["workers_per_device"] = int(workers)
    if cpu_threads is not None:
        pg["cpu_threads_per_worker"] = int(cpu_threads)
    pg.setdefault("verbosity", 1)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return work


def run_backend(backend: str, case: Path):
    _, module = BACKENDS[backend]
    # Run in a child process because the runners call os.chdir and delete scratch.
    snippet = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r});"
        f"from {module} import run; from pathlib import Path;"
        f"run(Path({str(case)!r}))"
    )
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True)
    return proc, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example", type=Path, required=True)
    ap.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    ap.add_argument("--executable", required=True, help="Absolute path to the MD binary")
    ap.add_argument("--devices", default=None, help="Comma-separated logical GPU indices, e.g. 0,1")
    ap.add_argument("--workers-per-device", type=int, default=None)
    ap.add_argument("--cpu-threads-per-worker", type=int, default=None,
                    help="Pin OMP/MKL threads (and GROMACS -ntomp) per worker; CPU oversubscription guard")
    ap.add_argument("--max-cycle", type=int, default=5)
    ap.add_argument("--max-trial", type=int, default=None)
    ap.add_argument("--monitor-gpu", action="store_true", help="Sample nvidia-smi during the run")
    ap.add_argument("--tmp", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    example = (args.example if args.example.is_absolute() else REPO_ROOT / args.example).resolve()
    devices = [int(d) for d in args.devices.split(",")] if args.devices else None

    result = {
        "backend": args.backend,
        "example": str(example),
        "executable": args.executable,
        "requested_devices": devices,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    if not Path(args.executable).exists() and shutil.which(args.executable) is None:
        result.update(status="fail", detail=f"executable not found: {args.executable}")
        _emit(result, args.out)
        return 1

    case = prepare_case(example, args.backend, args.executable, devices,
                        args.workers_per_device, args.max_cycle, args.max_trial,
                        args.cpu_threads_per_worker, args.tmp)

    monitor = GpuMonitor() if args.monitor_gpu else None
    if monitor:
        monitor.start()

    proc, wall = run_backend(args.backend, case)

    if monitor:
        monitor.stop()
        monitor.join(timeout=5)
        result["gpus_used"] = sorted(monitor.gpu_indices)
        result["gpu_monitor_available"] = monitor.available

    result["wall_seconds"] = round(wall, 2)
    result["returncode"] = proc.returncode
    result["stdout_tail"] = (proc.stdout or "").strip().splitlines()[-15:]
    result["stderr_tail"] = (proc.stderr or "").strip().splitlines()[-15:]

    ok = proc.returncode == 0
    detail = "run completed" if ok else "run failed (see stderr_tail)"
    # Multi-GPU spread assertion when monitoring >=2 devices.
    if ok and monitor and devices and len(devices) >= 2:
        if len(result["gpus_used"]) >= 2:
            detail += f"; swarm spread across GPUs {result['gpus_used']}"
        else:
            ok = False
            detail = (f"run completed but only GPUs {result['gpus_used']} showed activity "
                      f"(expected >=2). If this backend is OpenMM this is EXPECTED "
                      f"(OpenMM path is single-GPU); for AMBER/GROMACS investigate device routing.")

    result["status"] = "pass" if ok else "fail"
    result["detail"] = detail

    # Clean up the scratch working copy (the runner already removed its own scratch).
    shutil.rmtree(case, ignore_errors=True)

    _emit(result, args.out)
    print(f"[{result['status'].upper()}] {args.backend}: {detail} ({wall:.1f}s)")
    return 0 if ok else 1


def _emit(result, out):
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
