"""Validate the device-aware GROMACS engine without real gmx binaries.

Fakes ``subprocess.run`` for both ``gmx grompp`` (records the input structure
behind the produced .tpr) and ``gmx mdrun`` (copies that structure to the output
.gro and records the device).  Asserts the swarm spreads across devices and uses
isolated per-device scratch directories.
"""

import threading
from pathlib import Path

import numpy as np

import pathgennie.backends.gromacs.pg_gmx as pg_gmx
from pathgennie.backends.gromacs.pg_gmx import CoreGromacsEngine
from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import ThreadDevicePool
from pathgennie.core.progress import EscapeMetric

REPO = Path(__file__).resolve().parents[1]
GRO = REPO / "examples" / "alanine_dipeptide" / "gromacs" / "ala_dipeptide_equilibrated.gro"


def _install_fake_gmx(monkeypatch, recorder, lock):
    tpr_inputs: dict[str, str] = {}

    def fake_run(cmd, check=True, capture_output=True, text=True, env=None):
        stage = cmd[1]
        if stage == "grompp":
            in_gro = cmd[cmd.index("-c") + 1]
            tpr = cmd[cmd.index("-o") + 1]
            with lock:
                tpr_inputs[tpr] = in_gro
            Path(tpr).write_text("fake-tpr")
        elif stage == "mdrun":
            tpr = cmd[cmd.index("-s") + 1]
            out_gro = cmd[cmd.index("-c") + 1]
            out_cpt = cmd[cmd.index("-cpo") + 1]
            Path(out_gro).write_bytes(Path(tpr_inputs[tpr]).read_bytes())
            Path(out_cpt).write_text("fake-cpt")
            with lock:
                recorder.append({
                    "device": (env or {}).get("CUDA_VISIBLE_DEVICES"),
                    "out_dir": str(Path(out_gro).parent),
                })
        return None

    monkeypatch.setattr(pg_gmx.subprocess, "run", fake_run)


def test_gromacs_swarm_spreads_across_devices(tmp_path, monkeypatch):
    recorder = []
    lock = threading.Lock()
    _install_fake_gmx(monkeypatch, recorder, lock)

    engine = CoreGromacsEngine(
        topology=GRO, executable=Path("/bin/true"),
        scratch_dir=tmp_path / "scratch", temperature=300.0,
        mdp_controls={"integrator": "md", "dt": 0.002},
    )
    progress = EscapeMetric(lambda c: np.array([c[0, 0]]), start_cv=np.array([0.0]), escape_metric="cv0")
    driver = PathGennieDriver(
        engine, progress, convergence_fn=lambda c: False,
        executor=ThreadDevicePool(devices=[0, 1, 2, 3], workers_per_device=1),
        sigma=0.1, seed=1, verbosity=0,
    )
    driver.run(str(GRO), tau1=1, tau2=1, max_trial=8, max_cycle=2, save_freq=1)

    devices = {r["device"] for r in recorder}
    assert {"0", "1", "2", "3"}.issubset(devices)
    dev_dirs = {Path(r["out_dir"]).name for r in recorder}
    assert {"dev0", "dev1", "dev2", "dev3"}.issubset(dev_dirs)
