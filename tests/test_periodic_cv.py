"""Progress metrics must respect collective-variable periodicity.

``EscapeMetric`` and ``TargetMetric`` score with a plain Euclidean norm, which is wrong for
periodic CVs such as dihedral angles: two values either side of the +-180 deg branch cut are
scored as ~360 deg apart when they are in fact adjacent.

Observed on a real alanine-dipeptide escape run (job 189858): start (phi, psi) =
(-116.36, 165.72), cycle-0 CV = (-146.60, -178.13). psi had moved 16.1 deg, but the metric
returned 345.2 instead of the true angular distance 34.3 -- a 10x inflation that rewards the
sampler for stepping across the branch cut rather than for real progress.

The example's *convergence* function already wraps angles (``phi_psi.angular_delta_degrees``);
these tests pin the same behaviour for the *selection* metric, which is what actually drives
the search. Non-periodic CVs (distances, PCA components) must be unaffected.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathgennie.core.progress import EscapeMetric, TargetMetric


def _proj(coords, **kwargs):
    """Identity projection: the test feeds CV values directly as a 1x2 'coordinate'."""
    return np.asarray(coords, dtype=float).ravel()[:2]


# Values taken verbatim from the real run that exposed the defect.
START = np.array([-116.36, 165.72])
CYCLE0 = np.array([-146.60, -178.13])


def test_escape_metric_wraps_periodic_angles():
    metric = EscapeMetric(_proj, START, periodic=[360.0, 360.0])
    value = metric.metric(CYCLE0)

    # True angular separation: phi -30.24 deg, psi +16.15 deg.
    expected = np.hypot(-30.24, 16.15)
    assert value == pytest.approx(expected, abs=0.05), (
        f"periodic escape metric returned {value:.2f}, expected ~{expected:.2f}"
    )
    assert value < 40.0, "metric still inflated across the +-180 branch cut"


def test_escape_metric_without_periodicity_is_unchanged():
    """Distance/PCA CVs must keep the existing Euclidean behaviour."""
    metric = EscapeMetric(_proj, START)
    assert metric.metric(CYCLE0) == pytest.approx(float(np.linalg.norm(CYCLE0 - START)))


def test_target_metric_wraps_periodic_angles():
    target = np.array([60.0, 40.0])
    cv = np.array([-179.0, 40.0])          # 121 deg from target the short way round
    metric = TargetMetric(_proj, target, periodic=[360.0, 360.0])
    # TargetMetric returns the negated distance (higher is better).
    assert -metric.metric(cv) == pytest.approx(121.0, abs=0.5)


def test_partial_periodicity_supported():
    """A mixed CV space: one periodic angle plus one non-periodic distance."""
    start = np.array([170.0, 1.0])
    cv = np.array([-170.0, 4.0])           # angle wraps by 20 deg; distance moves 3
    metric = EscapeMetric(_proj, start, periodic=[360.0, None])
    assert metric.metric(cv) == pytest.approx(np.hypot(20.0, 3.0), abs=1e-6)


def test_periodic_length_must_match_cv():
    with pytest.raises(ValueError):
        EscapeMetric(_proj, START, periodic=[360.0, 360.0, 360.0]).metric(CYCLE0)


def test_wrapping_is_symmetric_across_the_branch_cut():
    """Distance must not depend on which side of the cut a point sits."""
    m = EscapeMetric(_proj, np.array([179.0, 0.0]), periodic=[360.0, 360.0])
    a = m.metric(np.array([-179.0, 0.0]))   # 2 deg across the cut
    b = m.metric(np.array([177.0, 0.0]))    # 2 deg on the same side
    assert a == pytest.approx(b, abs=1e-9)
    assert a == pytest.approx(2.0, abs=1e-9)
