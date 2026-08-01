"""The HPC job templates must not report success when the MD run failed.

All four templates ran the MD smoke as::

    python tests/hpc/run_example.py ... || echo "smoke run FAILED (see ...)"

``|| echo`` handles the failure, so ``set -e`` never fires and the script falls
through to its final ``echo "### Done."`` and exits **0**. A user submits the job,
sees "Done", and concludes PathGennie works on their cluster -- when the one thing
the script exists to prove did not happen.

The missing-binary path had the same shape: no ``gmx``/``pmemd.cuda`` on PATH meant
the MD block was skipped entirely and the job still exited 0.

These tests execute the real scripts with a stubbed ``python`` so the exit code is
the thing under test, not a grep for a pattern.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = [
    "tests/hpc/slurm_cpu.sbatch",
    "tests/hpc/slurm_gpu.sbatch",
    "tests/hpc/pbs_cpu.pbs",
    "tests/hpc/pbs_gpu.pbs",
]


def _run(tmp_path, template, *, python_rc, env_extra):
    """Run a template with a stub `python`, in an isolated PATH and cwd."""
    work = tmp_path / "repo"
    (work / "tests" / "hpc").mkdir(parents=True)
    shutil.copy(REPO / template, work / template)

    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "python"
    stub.write_text(f"#!/bin/sh\nexit {python_rc}\n")
    stub.chmod(0o755)

    env = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "SLURM_JOB_ID": "test",
        "PBS_JOBID": "test",
        "SLURM_CPUS_PER_TASK": "8",
        "PG_REPO": str(work),
        **env_extra,
    }
    return subprocess.run(["bash", template], cwd=work, env=env,
                          capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("template", TEMPLATES)
def test_failed_md_run_fails_the_job(tmp_path, template):
    """The regression: a failing MD run must NOT exit 0."""
    r = _run(tmp_path, template, python_rc=1, env_extra={"PG_EXE": "/bin/true"})
    assert r.returncode != 0, (
        f"{template} exited 0 despite the MD run failing.\n"
        f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    )
    assert "FAILED" in r.stdout, "the failure was not reported to the user either"


@pytest.mark.parametrize("template", TEMPLATES)
def test_successful_run_still_exits_zero(tmp_path, template):
    """The fix must not make everything fail -- a clean run still passes."""
    r = _run(tmp_path, template, python_rc=0, env_extra={"PG_EXE": "/bin/true"})
    assert r.returncode == 0, (
        f"{template} failed a clean run.\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    )


@pytest.mark.parametrize("template", TEMPLATES)
def test_missing_md_binary_is_a_failure_by_default(tmp_path, template):
    """No MD binary means the script proved nothing; that is not a pass."""
    r = _run(tmp_path, template, python_rc=0, env_extra={})
    assert r.returncode != 0, (
        f"{template} exited 0 with no MD binary, having run no MD.\n"
        f"stdout:\n{r.stdout}"
    )


@pytest.mark.parametrize("template", TEMPLATES)
def test_missing_md_binary_can_be_downgraded_to_a_skip(tmp_path, template):
    """Opt-in escape hatch for sites that only want the no-MD diagnostics."""
    r = _run(tmp_path, template, python_rc=0, env_extra={"PG_ALLOW_NO_MD": "1"})
    assert r.returncode == 0, (
        f"{template} failed despite PG_ALLOW_NO_MD=1.\nstdout:\n{r.stdout}\n"
        f"stderr:\n{r.stderr}"
    )
