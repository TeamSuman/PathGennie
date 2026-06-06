import logging
import shutil
import subprocess
from pathlib import Path

import pytraj as pt  # type: ignore

logger = logging.getLogger(__name__)


class AmberEngine:
    """
    Wrapper around pmemd/sander with enhanced sampling support.
    """

    def __init__(
        self,
        topology,
        mdin,
        executable="pmemd.cuda",
        workdir="run",
        temperature=300.0,
    ):
        self.topology = topology
        self.mdin = mdin
        self.exe = executable
        self.workdir = Path(workdir)
        self.workdir.mkdir(exist_ok=True)
        self.temperature = temperature
        logger.info(f"AmberEngine initialized with T={temperature}K, exe={executable}")

    def run_segment(
        self,
        input_rst,
        output_prefix,
        randomize_velocities=True,
    ):
        """
        Run MD segment with optional velocity re-randomization.

        Parameters
        ----------
        input_rst : str
            Input restart file
        output_prefix : str
            Prefix for output files
        randomize_velocities : bool
            If True, use sander's ivel=-1 to randomize velocities at start

        Returns
        -------
        str
            Path to output restart file
        """

        out_rst = f"{output_prefix}.rst7"
        out_vel_file = f"{output_prefix}_vrand.rst7" if randomize_velocities else input_rst

        # If randomizing velocities, run a quick velocity randomization step first
        if randomize_velocities:
            self._randomize_velocities(input_rst, out_vel_file)
            actual_input = out_vel_file
        else:
            actual_input = input_rst

        cmd = [
            self.exe,
            "-O",
            "-i",
            self.mdin,
            "-p",
            self.topology,
            "-c",
            actual_input,
            "-r",
            out_rst,
            "-o",
            f"{output_prefix}.out",
            "-x",
            f"{output_prefix}.nc",
        ]

        logger.debug(f"Running: {' '.join(cmd)}")
        _ = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Log final energy from output file
        # self._log_segment_energy(f"{output_prefix}.out")

        return out_rst

    def _randomize_velocities(self, input_rst, output_rst):
        """
        Quick velocity randomization using sander.

        Uses ivel=-1 in mdin to randomize velocities at specified temperature
        without altering positions.
        """

        # Create minimal mdin for velocity randomization
        vrand_mdin = self.workdir / "_vrand_temp.mdin"
        with open(vrand_mdin, "w") as f:
            f.write(f"""Randomize velocities
 &cntrl
    ivel = -1,
    nstlim = 0,
    temp0 = {self.temperature},
    ifmbar = 0,
    ntxo = 1,
    ioutfm = 0,
 /
""")

        cmd = [
            self.exe,
            "-O",
            "-i",
            str(vrand_mdin),
            "-p",
            self.topology,
            "-c",
            input_rst,
            "-r",
            output_rst,
            "-o",
            str(self.workdir / "_vrand_temp.out"),
        ]

        _ = subprocess.run(cmd, check=True, capture_output=True)
        vrand_mdin.unlink()  # Clean up temporary mdin

    # def _log_segment_energy(self, out_file):
    #     """Extract and log final energy from sander output."""
    #     try:
    #         with open(out_file, "r") as f:
    #             lines = f.readlines()
    #             for line in reversed(lines):
    #                 if "NSTEP" in line and "Etot" in line:
    #                     logger.debug(f"Segment energy: {line.strip()}")
    #                     break
    #     except Exception as e:
    #         logger.warning(f"Could not read energy from {out_file}: {e}")

    def copy_state(self, src, dst):
        """Copy restart file to preserve state."""
        shutil.copy(src, dst)

    def load_coords(self, rst):
        """Load coordinates from restart file in Angstroms."""
        traj = pt.iterload(rst, self.topology)
        return traj.xyz[0]

    def get_velocities(self, rst):
        """Load velocities from restart file if available."""
        try:
            traj = pt.iterload(rst, self.topology)
            if hasattr(traj, "velocities") and traj.velocities is not None:
                return traj.velocities[0]
        except Exception:
            pass
        return None
