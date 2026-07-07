"""Tests for node-local scratch placement (pathgennie.utils.scratch)."""

from __future__ import annotations

from pathlib import Path

from pathgennie.utils.scratch import resolve_scratch_dir


def test_default_scratch_under_workdir():
    wd = Path("/project/run/my_case")
    assert resolve_scratch_dir(wd, None, environ={}) == wd / "scratch"


def test_config_scratch_root_wins():
    wd = Path("/project/run/my_case")
    got = resolve_scratch_dir(wd, "/local/ssd", environ={})
    assert got == Path("/local/ssd") / "my_case_scratch"


def test_env_scratch_root_used_when_no_config():
    wd = Path("/project/run/my_case")
    got = resolve_scratch_dir(wd, None, environ={"PATHGENNIE_SCRATCH": "/tmp/job123"})
    assert got == Path("/tmp/job123") / "my_case_scratch"


def test_config_overrides_env():
    wd = Path("/project/run/my_case")
    got = resolve_scratch_dir(wd, "/local/ssd", environ={"PATHGENNIE_SCRATCH": "/tmp/job123"})
    assert got == Path("/local/ssd") / "my_case_scratch"


def test_scratch_root_is_a_config_key(tmp_path):
    from pathgennie.utils.config import load_config

    cfg = tmp_path / "input.yaml"
    cfg.write_text(
        "pathgennie:\n  max_cycle: 10\nscratch_root: /local/ssd\nworkdir: run1\n",
        encoding="utf-8",
    )
    d = load_config(cfg).model_dump(exclude_none=True)
    assert d["scratch_root"] == "/local/ssd"
