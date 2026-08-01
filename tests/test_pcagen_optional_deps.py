"""`pathgennie pcagen` crashed with a bare ModuleNotFoundError on a clean install.

`pathgennie/utils/ligcvgen.py` imported scikit-learn, matplotlib and joblib at module
level. None of the three were declared in `pyproject.toml` -- not as core dependencies
and not as an extra -- yet `pcagen` is listed in `pathgennie --help` and documented in
`docs/pca-cv.md`. A user following the docs on a clean install hit::

    ModuleNotFoundError: No module named 'sklearn'

The CLI already imported the module lazily, so only the `pcagen` subcommand broke rather
than every command. They are now declared as the ``[analysis]`` extra, and the import
site raises a message naming the install command.

The three imports sit in ONE guard on purpose: matplotlib used to be imported *above*
the sklearn check, so whichever package happened to be missing first decided the
message, and a guard placed under it could never fire.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
OPTIONAL = ["sklearn", "matplotlib", "joblib"]

_SNIPPET = '''
import sys, importlib.abc
BLOCK = {block!r}


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == BLOCK or name.startswith(BLOCK + "."):
            raise ImportError("No module named " + repr(name), name=name)


sys.meta_path.insert(0, _Blocker())
try:
    from pathgennie.utils.ligcvgen import LigPCGen
except ImportError as exc:
    print("GUARD:" + str(exc).replace(chr(10), " "))
else:
    print("NOGUARD")
'''


def _import_without(module: str) -> str:
    proc = subprocess.run([sys.executable, "-c", _SNIPPET.format(block=module)],
                          capture_output=True, text=True, cwd=REPO)
    return proc.stdout.strip()


@pytest.mark.parametrize("missing", OPTIONAL)
def test_message_names_the_extra_for_every_optional_dependency(missing):
    """Whichever one is absent, the user must be told how to fix it."""
    out = _import_without(missing)
    assert out.startswith("GUARD:"), (
        f"no informative error when {missing} is missing (got {out!r}); a bare "
        "ModuleNotFoundError is what this guard exists to replace"
    )
    assert "pathgennie[analysis]" in out, (
        f"the {missing} failure did not name the extra: {out!r}"
    )


def _installed(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


@pytest.mark.parametrize("missing", OPTIONAL)
def test_the_missing_package_is_identified_by_name(missing):
    """The message must name a genuinely missing package.

    Which one it names is only well defined when exactly ONE is absent: the guard
    reports ``_exc.name``, i.e. the FIRST import in the block that failed. In a bare
    environment (PathGennie's core CI lane installs none of the three) blocking
    ``sklearn`` still trips ``matplotlib`` first, so asserting the blocked name
    unconditionally is an assumption about the environment, not about the code --
    and it is exactly what broke this lane once.
    """
    out = _import_without(missing)
    assert out.startswith("GUARD:")
    others = [m for m in OPTIONAL if m != missing]
    if all(_installed(m) for m in others):
        assert missing in out, f"the message does not name the missing package: {out!r}"
    else:
        absent = [m for m in OPTIONAL if not _installed(m) or m == missing]
        assert any(m in out for m in absent), (
            f"the message names no genuinely missing package (absent: {absent}): {out!r}"
        )


def test_the_analysis_extra_declares_all_three():
    """A helpful message pointing at an extra that lacks the package is worse than none."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        tomllib = pytest.importorskip("tomli")
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())
    extra = cfg["project"]["optional-dependencies"]["analysis"]
    names = {d.split(">")[0].split("=")[0].split("[")[0].strip().lower() for d in extra}
    assert {"scikit-learn", "matplotlib", "joblib"} <= names, (
        f"[analysis] does not cover everything ligcvgen imports: {sorted(names)}"
    )


def test_importing_pathgennie_itself_does_not_need_the_extra():
    """Only pcagen may depend on these -- the core package must stay importable."""
    for missing in OPTIONAL:
        proc = subprocess.run(
            [sys.executable, "-c", _SNIPPET.format(block=missing).replace(
                "from pathgennie.utils.ligcvgen import LigPCGen",
                "import pathgennie; from pathgennie.cli.main import main")],
            capture_output=True, text=True, cwd=REPO)
        assert "NOGUARD" in proc.stdout, (
            f"the core package failed to import without {missing}: {proc.stdout} {proc.stderr[-300:]}"
        )
