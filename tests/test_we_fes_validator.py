"""Quantitative validator: toy Weighted Ensemble recovers the analytic FES.

Turns the qualitative benchmark (benchmarks/we_fes.py) into a pass/fail check.
Runs are deterministic (fixed seeds), so the recovered free-energy profile along
y must match the analytic marginal F(y) = -kT ln integral e^{-V/kT} dx to high
Pearson correlation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# benchmarks/ is not a package; add it to the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))


@pytest.mark.slow
def test_weighted_ensemble_recovers_analytic_fes():
    from we_fes import run_validation

    out = run_validation(n_iterations=200)
    assert out["finite"].sum() > 4, "too few occupied bins to validate"
    # Deterministic run reaches ~0.99; assert a robust floor well above noise.
    assert out["corr"] > 0.85, f"WE FES correlation too low: {out['corr']:.3f}"
