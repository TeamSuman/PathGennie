"""Regression tests for re-running a backend into an existing work directory.

Both ``pg_amber.run`` and ``pg_gmx.run`` wipe a leftover scratch directory near the
top of the function::

    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)

A redundant *function-local* ``import shutil`` further down made ``shutil`` a local
name for the whole function, so that earlier use raised ``UnboundLocalError`` — the
backends crashed on any second run, resume, or restart-after-crash into the same
workdir (the exact case checkpoint/restart exists to serve).  The first run on a
clean directory worked, which is why the bug survived: these entrypoints had no
test coverage at all.

These tests drive each ``run()`` far enough to execute the scratch-wipe branch.
They are expected to fail afterwards on the deliberately absent MD executable —
what matters is that the failure is *not* ``UnboundLocalError``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pathgennie.utils.scratch import resolve_scratch_dir

AMBER_CONFIG = """\
amber:
    topology: system.prmtop
    initial_restart: system.rst7
    executable: /nonexistent/pmemd.cuda
pathgennie:
    mode: escape
    tau1_steps: 2
    tau2_steps: 2
    max_trial: 2
    max_cycle: 1
projection:
    module: proj
    function: cv
convergence:
    module: proj
    function: done
workdir: pathgennie_run
"""

GROMACS_CONFIG = """\
gromacs:
    topology: topol.top
    initial_structure: conf.gro
    executable: /nonexistent/gmx
    mdp: md.mdp
pathgennie:
    mode: escape
    tau1_steps: 2
    tau2_steps: 2
    max_trial: 2
    max_cycle: 1
projection:
    module: proj
    function: cv
convergence:
    module: proj
    function: done
workdir: pathgennie_gmx_run
"""


def _make_case(tmp_path: Path, config_text: str, workdir_name: str) -> Path:
    """Write a minimal case dir whose scratch directory already exists."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "input.yaml").write_text(config_text)
    (case_dir / "proj.py").write_text(
        "import numpy as np\n"
        "def cv(coords, **kw):\n"
        "    return np.array([0.0])\n"
        "def done(coords, **kw):\n"
        "    return True\n"
    )
    # Simulate the leftover state of a previous (or interrupted) run.
    workdir = case_dir / workdir_name
    scratch_dir = resolve_scratch_dir(workdir, None)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    (scratch_dir / "stale_segment.rst7").write_text("leftover from a previous run\n")
    assert scratch_dir.exists()
    return case_dir


@pytest.fixture(autouse=True)
def _restore_cwd():
    """The backends os.chdir into the case dir; don't leak that into other tests."""
    cwd = os.getcwd()
    yield
    os.chdir(cwd)


def test_amber_run_survives_existing_scratch_dir(tmp_path):
    from pathgennie.backends.amber import pg_amber

    case_dir = _make_case(tmp_path, AMBER_CONFIG, "pathgennie_run")

    with pytest.raises(Exception) as excinfo:
        pg_amber.run(case_dir)

    assert not isinstance(excinfo.value, UnboundLocalError), (
        "pg_amber.run raised UnboundLocalError while clearing an existing scratch "
        "directory: a function-local 'import shutil' shadows the module-level import."
    )
    # Having got past the scratch wipe, it should fail on the missing executable.
    assert isinstance(excinfo.value, FileNotFoundError)


def test_gromacs_run_survives_existing_scratch_dir(tmp_path):
    from pathgennie.backends.gromacs import pg_gmx

    case_dir = _make_case(tmp_path, GROMACS_CONFIG, "pathgennie_gmx_run")

    with pytest.raises(Exception) as excinfo:
        pg_gmx.run(case_dir)

    assert not isinstance(excinfo.value, UnboundLocalError), (
        "pg_gmx.run raised UnboundLocalError while clearing an existing scratch "
        "directory: a function-local 'import shutil' shadows the module-level import."
    )
    assert isinstance(excinfo.value, FileNotFoundError)
