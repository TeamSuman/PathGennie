"""Regression tests for YAML config loading/validation (pathgennie.utils.config).

These guard the class of bug where the Pydantic schema silently dropped the
real keys (``tau1_steps``, ``devices``, ``downstream`` ...) and whole sections
(``md``, ``workdir``, ``output``), which made every backend crash with
``KeyError: 'tau1_steps'`` and silently ignored the user's MD parameters.
"""

from __future__ import annotations

import textwrap

import pytest

from pathgennie.utils.config import load_config


def _write(tmp_path, text):
    p = tmp_path / "input.yaml"
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


def test_real_keys_and_sections_survive_validation(tmp_path):
    cfg = _write(
        tmp_path,
        """
        amber:
            topology: sys.prmtop
            initial_restart: sys.rst7
            executable: pmemd.cuda
            devices: [0, 1]
        pathgennie:
            mode: target
            target_projection: [60.0, 40.0]
            tau1_steps: 2
            tau2_steps: 4
            max_trial: 10
            tau1_workers: 8
            devices: [0, 1, 2, 3]
            max_cycle: 5000
            sigma: 0.1
            downstream: weighted_ensemble
            checkpoint_path: run.h5
        md:
            controls:
                dt: 0.001
                cut: 8.0
        weighted_ensemble:
            n_iterations: 20
        workdir: my_run
        output:
            trajectory: path.pdb
        """,
    )
    d = load_config(cfg).model_dump(exclude_none=True)
    pg = d["pathgennie"]

    # Core keys the driver reads by name must be present and correct.
    assert pg["tau1_steps"] == 2
    assert pg["tau2_steps"] == 4
    assert pg["max_trial"] == 10
    assert pg["sigma"] == 0.1
    # Keys the older schema silently dropped.
    assert pg["tau1_workers"] == 8
    assert pg["devices"] == [0, 1, 2, 3]
    assert pg["downstream"] == "weighted_ensemble"
    assert pg["checkpoint_path"] == "run.h5"
    # Whole sections that used to vanish.
    assert d["md"]["controls"]["dt"] == 0.001
    assert d["workdir"] == "my_run"
    assert d["output"]["trajectory"] == "path.pdb"
    # The downstream-stage block (named after pathgennie.downstream) must survive.
    assert d["weighted_ensemble"]["n_iterations"] == 20


def test_profile_is_reachable_through_config(tmp_path):
    """A run profile supplies the segment lengths; its key must not be stripped."""
    from pathgennie.core.strategy import resolve_profile

    cfg = _write(tmp_path, "pathgennie:\n  profile: sampling\n  max_cycle: 10\n")
    pg = load_config(cfg).model_dump(exclude_none=True)["pathgennie"]
    assert pg["profile"] == "sampling"
    resolved = resolve_profile(pg)
    assert resolved["tau1_steps"] == 50  # from the SAMPLING profile


@pytest.mark.parametrize(
    "block",
    [
        "pathgennie:\n  sigma: 0\n  max_cycle: 10\n",
        "pathgennie:\n  max_trial: -1\n  max_cycle: 10\n",
        "pathgennie:\n  temperature: 0\n  max_cycle: 10\n",
        "pathgennie:\n  mode: targett\n  max_cycle: 10\n",
    ],
)
def test_malformed_values_fail_fast(tmp_path, block):
    from pydantic import ValidationError

    cfg = _write(tmp_path, block)
    with pytest.raises(ValidationError):
        load_config(cfg)


def test_empty_or_non_mapping_config_rejected(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(empty)
