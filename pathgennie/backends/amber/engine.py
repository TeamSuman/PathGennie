"""Device-aware AMBER engine adapter for the PathGennie core driver.

Implements the :class:`pathgennie.core.engine.Engine` protocol by shelling out to
``pmemd.cuda``/``sander`` per segment.  Two problems in the original
``GenericPathGennieAmber`` are fixed here:

* **No multi-GPU:** the old ``ThreadPoolExecutor`` launched every worker with the
  inherited environment, so all trials landed on GPU 0.  ``run_segment`` now sets
  ``CUDA_VISIBLE_DEVICES`` to the device the executor assigned, so a swarm spreads
  across all GPUs (``subprocess.run`` releases the GIL, so threads genuinely run
  concurrently).
* **Scratch races:** trials wrote files with colliding names into one directory.
  Each segment now gets a unique, per-device scratch subdirectory and a unique
  file stem.

A *handle* is an absolute path to an AMBER restart (``rst7``) file.
"""

from __future__ import annotations

import itertools
import os
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from pathgennie.core.parallel import resolve_cuda_visible_device

from .utils import read_native_trajectory, read_rst7_coords, write_mdin

__all__ = ["CoreAmberEngine"]


class CoreAmberEngine:
    def __init__(
        self,
        *,
        topology: Path,
        executable: Path,
        scratch_dir: Path,
        temperature: float,
        mdin_controls: dict,
        extra_mdin_text: str = "",
        command_prefix: Optional[list] = None,
        env_overrides: Optional[dict] = None,
    ):
        self.topology = str(topology)
        self.exe = str(executable)
        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = float(temperature)
        self.mdin_controls = mdin_controls
        self.extra_mdin_text = extra_mdin_text
        self.command_prefix = command_prefix or []
        self.env_overrides = {str(k): str(v) for k, v in (env_overrides or {}).items()}
        self._counter = itertools.count()
        self._lock = threading.Lock()

    def _uid(self) -> str:
        with self._lock:
            n = next(self._counter)
        return f"{n}_{uuid.uuid4().hex[:8]}"

    def _device_dir(self, device: Optional[int]) -> Path:
        name = "dev_cpu" if device is None else f"dev{device}"
        path = self.scratch_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def clone_anchor(self, handle):
        src = Path(handle)
        dst = self.scratch_dir / f"anchor_{self._uid()}.rst7"
        dst.write_bytes(src.read_bytes())
        return str(dst)

    def run_segment(self, handle, n_steps, *, randomize_velocities, seed, device=None,
                    save_subframes=False, subframe_stride=1):
        workdir = self._device_dir(device)
        stem = f"seg_{self._uid()}"
        mdin = workdir / f"{stem}.mdin"
        out_rst = workdir / f"{stem}.rst7"
        traj_nc = workdir / f"{stem}.nc"

        # Build mdin controls, overriding ntwx when subframes are requested.
        seg_controls = dict(self.mdin_controls)
        if save_subframes:
            seg_controls["ntwx"] = int(subframe_stride)

        write_mdin(
            mdin,
            int(n_steps),
            self.temperature,
            seg_controls,
            continue_velocities=not randomize_velocities,
            random_seed=int(seed),
            extra_text=self.extra_mdin_text,
        )
        cmd = [
            *self.command_prefix, self.exe, "-O",
            "-i", str(mdin),
            "-p", self.topology,
            "-c", str(handle),
            "-r", str(out_rst),
            "-o", str(workdir / f"{stem}.out"),
            "-inf", str(workdir / f"{stem}.mdinfo"),
        ]
        if seg_controls.get("ntwx", 0):
            cmd.extend(["-x", str(traj_nc)])

        env = os.environ.copy()
        visible = resolve_cuda_visible_device(device, os.environ)
        if visible is not None:
            env["CUDA_VISIBLE_DEVICES"] = visible
        env.update(self.env_overrides)

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        except subprocess.CalledProcessError as exc:
            message = [
                f"AMBER segment failed with exit code {exc.returncode}.",
                "Command: " + " ".join(cmd),
            ]
            if exc.stdout:
                message.append("stdout:\n" + exc.stdout[-4000:])
            if exc.stderr:
                message.append("stderr:\n" + exc.stderr[-4000:])
            raise RuntimeError("\n".join(message)) from exc

        result_handle = str(out_rst)
        if save_subframes and traj_nc.exists():
            subframes = read_native_trajectory(traj_nc)
            return result_handle, subframes
        return result_handle

    def get_coords(self, handle):
        coords = read_rst7_coords(handle)
        if not np.all(np.isfinite(coords)):
            raise ValueError(f"AMBER segment produced non-finite coordinates: {handle}")
        return coords

    def release(self, handle):
        path = Path(handle)
        stem = path.with_suffix("")
        for sibling in path.parent.glob(stem.name + ".*"):
            try:
                sibling.unlink()
            except OSError:
                pass
