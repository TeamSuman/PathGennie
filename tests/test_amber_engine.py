"""Validate the device-aware AMBER engine without a real pmemd binary.

``subprocess.run`` is faked to copy the input restart to the output path (so
coordinates flow through the driver) and to record the ``CUDA_VISIBLE_DEVICES``
each segment was launched with.  This lets us assert the two behaviours the
multi-GPU refactor is responsible for: trials are spread across the configured
devices, and each device gets an isolated scratch subdirectory.
"""

import threading
from pathlib import Path

import numpy as np

import pathgennie.backends.amber.engine as amber_engine
from pathgennie.backends.amber.engine import CoreAmberEngine
from pathgennie.backends.amber.utils import default_mdin_controls
from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import ThreadDevicePool
from pathgennie.core.progress import EscapeMetric

REPO = Path(__file__).resolve().parents[1]
RST7 = REPO / "examples" / "alanine_dipeptide" / "common" / "ala_dipeptide_equilibrated.rst7"


def _install_fake_pmemd(monkeypatch, recorder, lock):
    def fake_run(cmd, check=True, capture_output=True, text=True, env=None):
        in_path = cmd[cmd.index("-c") + 1]
        out_path = cmd[cmd.index("-r") + 1]
        Path(out_path).write_bytes(Path(in_path).read_bytes())
        with lock:
            recorder.append({
                "device": (env or {}).get("CUDA_VISIBLE_DEVICES"),
                "out_dir": str(Path(out_path).parent),
            })
        return None

    monkeypatch.setattr(amber_engine.subprocess, "run", fake_run)


def test_amber_swarm_spreads_across_devices(tmp_path, monkeypatch):
    recorder = []
    lock = threading.Lock()
    _install_fake_pmemd(monkeypatch, recorder, lock)

    engine = CoreAmberEngine(
        topology=RST7,  # never read by the fake; any existing file works
        executable=Path("/bin/true"),
        scratch_dir=tmp_path / "scratch",
        temperature=300.0,
        mdin_controls=default_mdin_controls("vacuum"),
    )
    progress = EscapeMetric(lambda c: np.array([c[0, 0]]), start_cv=np.array([0.0]), escape_metric="cv0")

    driver = PathGennieDriver(
        engine, progress, convergence_fn=lambda c: False,
        executor=ThreadDevicePool(devices=[0, 1, 2, 3], workers_per_device=1),
        sigma=0.1, seed=1, verbosity=0,
    )
    driver.run(str(RST7), tau1=1, tau2=1, max_trial=8, max_cycle=2, save_freq=1)

    sampler_devices = {r["device"] for r in recorder}
    # All four GPUs should have been used by the samplers (runner uses device 0).
    assert {"0", "1", "2", "3"}.issubset(sampler_devices)
    # Each device wrote into its own scratch subdirectory.
    dev_dirs = {Path(r["out_dir"]).name for r in recorder}
    assert {"dev0", "dev1", "dev2", "dev3"}.issubset(dev_dirs)


def test_amber_single_device_sets_env(tmp_path, monkeypatch):
    recorder = []
    _install_fake_pmemd(monkeypatch, recorder, threading.Lock())
    engine = CoreAmberEngine(
        topology=RST7, executable=Path("/bin/true"),
        scratch_dir=tmp_path / "scratch", temperature=300.0,
        mdin_controls=default_mdin_controls("vacuum"),
    )
    h = engine.run_segment(str(RST7), 1, randomize_velocities=True, seed=5, device=2)
    assert Path(h).exists()
    assert recorder[-1]["device"] == "2"
