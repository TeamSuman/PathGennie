"""The HPC harness could not run the example every template names as its default.

``run_example.py`` staged a case with ``shutil.copytree(example, work)`` -- the
backend directory only. But ``examples/alanine_dipeptide/{amber,gromacs}/projection.py``
load their shared CV from a SIBLING directory::

    _COMMON_PROJECTION = Path(__file__).resolve().parents[1] / "common" / "phi_psi.py"

Flattening the example into ``<scratch>/`` puts ``projection.py`` one level too high,
so that resolves to ``<scratch>/common/phi_psi.py``, which was never copied:

    FileNotFoundError: .../scratch/pg_hpc_gromacs_10155/../common/phi_psi.py

Every HPC template defaults to ``PG_EXAMPLE=examples/alanine_dipeptide/<backend>``,
so the default configuration of all four templates could never work. It stayed hidden
because the templates masked the failure with ``|| echo`` and exited 0 regardless
(fixed separately) -- the run died, the job printed "Done", and nobody looked.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests" / "hpc"))
import run_example as rex  # noqa: E402

EXAMPLES = [
    pytest.param("examples/alanine_dipeptide/gromacs", "gromacs", id="gromacs"),
    pytest.param("examples/alanine_dipeptide/amber", "amber", id="amber"),
]


@pytest.mark.parametrize("example,backend", EXAMPLES)
def test_sibling_common_directory_is_staged(tmp_path, example, backend):
    work = rex.prepare_case(REPO / example, backend, "/bin/true",
                            None, None, 2, None, None, tmp_path)
    shared = work.parent / "common" / "phi_psi.py"
    assert shared.exists(), (
        "the shared projection module was not staged; projection.py resolves it as "
        "parents[1]/common/phi_psi.py and will raise FileNotFoundError"
    )


@pytest.mark.parametrize("example,backend", EXAMPLES)
def test_projection_module_actually_imports_from_the_staged_tree(tmp_path, example, backend):
    """The real failure was an import, so import it -- do not just check paths."""
    work = rex.prepare_case(REPO / example, backend, "/bin/true",
                            None, None, 2, None, None, tmp_path)
    spec = importlib.util.spec_from_file_location("_staged_proj", work / "projection.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)          # this is what blew up in the real run
    assert hasattr(module, "phi_psi_cv")


@pytest.mark.parametrize("example,backend", EXAMPLES)
def test_config_is_still_patched_after_the_layout_change(tmp_path, example, backend):
    """Staging one level deeper must not lose the config rewriting."""
    work = rex.prepare_case(REPO / example, backend, "/usr/bin/true",
                            [0, 1], 2, 7, 3, None, tmp_path)
    cfg = yaml.safe_load((work / "input.yaml").read_text())
    assert cfg["pathgennie"]["max_cycle"] == 7
    assert cfg["pathgennie"]["max_trial"] == 3
    assert cfg["pathgennie"]["devices"] == [0, 1]
    assert cfg["pathgennie"]["workers_per_device"] == 2
    section = rex.BACKENDS[backend][0]
    assert cfg[section]["executable"] == "/usr/bin/true"


def test_example_without_a_sibling_common_still_stages(tmp_path):
    """OpenMM's example is self-contained; the fix must not require a sibling."""
    ex = REPO / "examples/alanine_dipeptide/openmm"
    if not ex.is_dir():
        pytest.skip("openmm example not present")
    work = rex.prepare_case(ex, "openmm", "/bin/true", None, None, 2, None, None, tmp_path)
    assert (work / "input.yaml").exists()
