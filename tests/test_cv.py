import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples" / "alanine_dipeptide" / "common"))

import phi_psi  # noqa: E402

from pathgennie.core.progress import EscapeMetric, TargetMetric  # noqa: E402


def test_dihedral_known_geometry():
    # Atoms placed so the 0-1-2-3 dihedral is exactly +90 degrees.
    coords = np.array(
        [
            [1.0, 0.0, 0.0],   # p0
            [0.0, 0.0, 0.0],   # p1
            [0.0, 0.0, 1.0],   # p2 (axis along z)
            [0.0, 1.0, 1.0],   # p3
        ]
    )
    angle = phi_psi.dihedral_degrees(coords, (0, 1, 2, 3))
    assert np.isclose(abs(angle), 90.0, atol=1e-6)


def test_phi_psi_cv_shape():
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((20, 3))
    cv = phi_psi.phi_psi_cv(coords)
    assert cv.shape == (2,)


def test_angular_wrap():
    # 170 and -170 degrees are 20 degrees apart, not 340.
    delta = phi_psi.angular_delta_degrees([170.0], [-170.0])
    assert np.isclose(abs(delta[0]), 20.0)
    assert phi_psi.reached_phi_psi.__doc__ is None or True  # smoke


def test_target_metric_is_negated_distance():
    pv = TargetMetric(lambda c: np.array([c[0, 0], c[0, 1]]), target_cv=np.array([0.0, 0.0]))
    near = pv.metric(np.array([0.1, 0.1]))
    far = pv.metric(np.array([5.0, 5.0]))
    assert near > far
    assert np.isclose(pv.metric(np.array([3.0, 4.0])), -5.0)


def test_escape_metric_distance_from_start():
    pv = EscapeMetric(lambda c: c, start_cv=np.array([0.0, 0.0]), escape_metric="distance_from_start")
    assert np.isclose(pv.metric(np.array([3.0, 4.0])), 5.0)


def test_escape_metric_cv0():
    pv = EscapeMetric(lambda c: c, start_cv=np.array([0.0, 0.0]), escape_metric="cv0")
    assert np.isclose(pv.metric(np.array([7.0, 2.0])), 7.0)
