#!/usr/bin/env python
"""PathGennie HPC self-check.

A dependency-light diagnostic that reports the compute environment and runs a
battery of checks that do **not** require an MD binary or a GPU, so it can be
launched on any Slurm/PBS node (CPU or GPU queue) to confirm the install and the
HPC-critical code paths before spending allocation on real MD.

It prints a human-readable summary and (with ``--out``) writes a machine-readable
JSON report. Exit code is non-zero if any check FAILs (SKIPs do not fail the run).

The JSON is designed for a follow-up automated agent (e.g. another Claude Code
session) to parse: each entry has ``name``, ``status`` (pass|fail|skip),
``detail``, and where relevant ``data``. See tests/hpc/DEBUGGING.md for how to
map a failing check to a cause and fix.

Usage
-----
    python tests/hpc/hpc_selfcheck.py                 # print summary
    python tests/hpc/hpc_selfcheck.py --out results/env.json
    python tests/hpc/hpc_selfcheck.py --example examples/alanine_dipeptide/amber
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _try(fn):
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001 - diagnostic
        return None, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# environment probes
# --------------------------------------------------------------------------- #

def probe_scheduler():
    slurm = {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}
    pbs = {k: v for k, v in os.environ.items() if k.startswith("PBS_")}
    kind = "slurm" if slurm else ("pbs" if pbs else "none")
    return {
        "kind": kind,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_gpus": os.environ.get("SLURM_GPUS") or os.environ.get("SLURM_GPUS_ON_NODE"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "pbs_gpufile": os.environ.get("PBS_GPUFILE"),
        "pbs_ncpus": os.environ.get("NCPUS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def probe_gpus():
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"nvidia_smi": False, "gpus": []}
    out, err = _try(
        lambda: subprocess.run(
            [smi, "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    )
    gpus = [line.strip() for line in (out or "").splitlines() if line.strip()]
    return {"nvidia_smi": True, "gpus": gpus, "error": err}


def probe_packages():
    versions = {}
    for mod in ("numpy", "scipy", "pydantic", "h5py", "yaml", "MDAnalysis",
                "openmm", "torch", "mpi4py", "dask", "sklearn", "openpathsampling"):
        v, _ = _try(lambda m=mod: __import__(m).__version__)
        versions[mod] = v
    return versions


def probe_backends():
    exes = {}
    for exe in ("gmx", "gmx_mpi", "pmemd", "pmemd.cuda", "pmemd.MPI",
                "pmemd.cuda.MPI", "sander", "sander.MPI"):
        exes[exe] = shutil.which(exe)
    return exes


# --------------------------------------------------------------------------- #
# functional checks (no MD binary / GPU needed)
# --------------------------------------------------------------------------- #

def check_import(results):
    sys.path.insert(0, str(REPO_ROOT))
    ok, err = _try(lambda: __import__("pathgennie"))
    results.append(_result("import_pathgennie", ok is not None, err or "imported pathgennie"))


def check_device_resolution(results):
    def run():
        from pathgennie.core.parallel import resolve_cuda_visible_device
        # Simulated scheduler mask: logical 0/1 must map inside the allocation.
        assert resolve_cuda_visible_device(0, {"CUDA_VISIBLE_DEVICES": "2,3"}) == "2"
        assert resolve_cuda_visible_device(1, {"CUDA_VISIBLE_DEVICES": "2,3"}) == "3"
        assert resolve_cuda_visible_device(2, {"CUDA_VISIBLE_DEVICES": "2,3"}) == "2"
        assert resolve_cuda_visible_device(0, {}) == "0"
        # And how THIS node's real mask would be interpreted for logical 0/1:
        real = os.environ.get("CUDA_VISIBLE_DEVICES")
        mapped = [resolve_cuda_visible_device(i) for i in range(2)]
        return {"node_cuda_visible_devices": real, "logical_0_1_map_to": mapped}
    data, err = _try(run)
    results.append(_result("device_masking", err is None,
                           err or "scheduler-aware CUDA_VISIBLE_DEVICES mapping OK", data))


def check_config(results, example_dir):
    def run():
        from pathgennie.utils.config import load_config
        cfg = load_config(Path(example_dir) / "input.yaml").model_dump(exclude_none=True)
        pg = cfg["pathgennie"]
        assert "tau1_steps" in pg, "tau1_steps was dropped by config validation"
        return {"example": str(example_dir), "tau1_steps": pg["tau1_steps"],
                "sections": sorted(cfg.keys())}
    data, err = _try(run)
    results.append(_result("config_validation", err is None,
                           err or "config keys/sections preserved", data))


def check_toy_reproducibility(results):
    def run():
        import numpy as np
        from pathgennie.core.driver import PathGennieDriver
        from pathgennie.core.parallel import ThreadDevicePool
        from pathgennie.core.progress import EscapeMetric
        from pathgennie.core.toy import ToyLangevinEngine

        def one():
            eng = ToyLangevinEngine(dt=0.002, kT=1.0)
            prog = EscapeMetric(lambda c: np.array([c[0, 0]]), start_cv=np.array([0.0]), escape_metric="cv0")
            drv = PathGennieDriver(eng, prog, convergence_fn=lambda c: False,
                                   executor=ThreadDevicePool(devices=[0, 1, 2, 3]),
                                   sigma=0.2, seed=123, verbosity=0)
            init = eng.create_state([-1.0, -1.0, 0.0])
            _, m = drv.run(init, tau1=3, tau2=3, max_trial=8, max_cycle=6, save_freq=1)
            return m, len(eng._cache)

        m1, cache1 = one()
        m2, _ = one()
        assert np.allclose(m1, m2), "threaded run not reproducible from seed"
        assert cache1 < 8 + 5, f"engine cache leaked ({cache1} entries)"
        return {"cache_entries_after_run": cache1, "cycles": len(m1)}
    data, err = _try(run)
    results.append(_result("threaded_determinism_and_leak", err is None,
                           err or "threaded run reproducible and leak-free", data))


def check_storage(results, tmp_dir):
    def run():
        import numpy as np
        from pathgennie.core.storage import HDF5Storage
        p = Path(tmp_dir) / "selfcheck.h5"
        s = HDF5Storage(p)
        for i in range(3):
            s.append("trajectory", np.full((2, 3), float(i)))
        s.close()
        import h5py
        with h5py.File(p, "r") as f:
            shape = tuple(f["trajectory"].shape)
        p.unlink(missing_ok=True)
        return {"trajectory_shape": shape}
    data, err = _try(run)
    results.append(_result("hdf5_checkpoint", err is None,
                           err or "HDF5 streaming write/read OK", data))


def check_unit_tests(results):
    """Run the repository unit suite (fast, no MD binaries)."""
    def run():
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(REPO_ROOT / "tests"),
             "--ignore", str(REPO_ROOT / "tests" / "hpc")],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=1800,
        )
        tail = (proc.stdout or "").strip().splitlines()[-1:] or [""]
        return {"returncode": proc.returncode, "summary": tail[-1]}, proc.returncode == 0
    out, err = _try(run)
    if err is not None:
        results.append(_result("unit_tests", False, err))
        return
    data, ok = out
    results.append(_result("unit_tests", ok, data["summary"], data))


def _result(name, ok, detail, data=None):
    return {"name": name, "status": "pass" if ok else "fail", "detail": detail, "data": data}


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="Write JSON report here")
    ap.add_argument("--example", type=Path,
                    default=REPO_ROOT / "examples" / "alanine_dipeptide" / "amber",
                    help="Example directory whose input.yaml is used for the config check")
    ap.add_argument("--tmp", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")),
                    help="Scratch dir for the HDF5 check")
    ap.add_argument("--skip-unit-tests", action="store_true")
    args = ap.parse_args()

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "scheduler": probe_scheduler(),
        "gpus": probe_gpus(),
        "packages": probe_packages(),
        "md_executables": probe_backends(),
        "checks": [],
    }

    checks = report["checks"]
    check_import(checks)
    check_device_resolution(checks)
    check_config(checks, args.example)
    check_toy_reproducibility(checks)
    check_storage(checks, args.tmp)
    if not args.skip_unit_tests:
        check_unit_tests(checks)

    # ---- print summary ----
    print("=" * 72)
    print(f"PathGennie HPC self-check on {report['host']}  ({report['timestamp_utc']})")
    print("=" * 72)
    sched = report["scheduler"]
    print(f"scheduler         : {sched['kind']}")
    print(f"CUDA_VISIBLE_DEVICES: {sched['cuda_visible_devices']}")
    print(f"GPUs (nvidia-smi) : {len(report['gpus']['gpus'])} visible")
    for g in report["gpus"]["gpus"]:
        print(f"    - {g}")
    have = {k: v for k, v in report["packages"].items() if v}
    print(f"packages present  : {', '.join(sorted(have)) or 'none'}")
    md = {k: v for k, v in report["md_executables"].items() if v}
    print(f"MD executables    : {', '.join(sorted(md)) or 'NONE FOUND'}")
    print("-" * 72)
    n_fail = 0
    for c in checks:
        mark = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[c["status"]]
        print(f"[{mark}] {c['name']:<32} {c['detail']}")
        if c["status"] == "fail":
            n_fail += 1
    print("-" * 72)
    print(f"{len(checks)} checks, {n_fail} failed")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"JSON report written to {args.out}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
