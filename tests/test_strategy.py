import logging

import numpy as np

from pathgennie.core.strategy import (
    DISCOVERY,
    SAMPLING,
    check_learned_cv_segment_length,
    get_profile,
    resolve_profile,
)
from pathgennie.sampling import PathEnsemble, SamplingResult, SamplingStage


def test_no_profile_passthrough():
    cfg = {"tau1_steps": 2, "max_trial": 7}
    assert resolve_profile(cfg) == cfg


def test_profile_fills_defaults():
    resolved = resolve_profile({"profile": "discovery"})
    assert resolved["tau1_steps"] == DISCOVERY.tau1_steps
    assert resolved["cv"] == "geometric"
    assert resolved["downstream"] is None


def test_explicit_overrides_profile():
    resolved = resolve_profile({"profile": "sampling", "tau1_steps": 5})
    assert resolved["tau1_steps"] == 5            # explicit wins
    assert resolved["cv"] == SAMPLING.cv           # profile default kept
    assert resolved["downstream"] == "weighted_ensemble"


def test_get_profile_unknown_raises():
    try:
        get_profile("nope")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_learned_cv_segment_guard(caplog):
    # 2+8 steps * 0.002 ps = 0.02 ps < 0.1 ps -> warn, return False.
    with caplog.at_level(logging.WARNING):
        ok = check_learned_cv_segment_length(2, 8, 0.002, min_ps=0.1)
    assert ok is False
    assert any("Learned CV" in rec.message for rec in caplog.records)
    # 50+100 steps * 0.002 ps = 0.3 ps -> fine.
    assert check_learned_cv_segment_length(50, 100, 0.002, min_ps=0.1) is True


def test_path_ensemble_and_stage_contract():
    frames = np.zeros((3, 4, 3))
    ens = PathEnsemble(frames=frames, metrics=np.arange(3.0))
    assert ens.n_frames == 3

    class DummyStage:
        def run(self, ensemble, engine, **kwargs):
            return SamplingResult(metadata={"n": ensemble.n_frames})

    stage = DummyStage()
    assert isinstance(stage, SamplingStage)  # structural Protocol check
    assert stage.run(ens, engine=None).metadata["n"] == 3
