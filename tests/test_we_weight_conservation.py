"""Weight-conservation and reproducibility regression tests for the ``we`` (wepath) package.

Weighted Ensemble is unbiased *only* because resampling conserves total probability — a rate
constant is literally a sum of walker weights arriving at the target. ``redistribute_excess_weight``
used to strip each walker's excess above ``cap`` **before** checking that a recipient existed, so
when every walker was above the cap the excess was silently discarded (measured: 1.0 -> 0.3, a
70% loss of probability). ``_recycle_weights`` calls it after every recycling event, and weights
are renormalised only at iteration zero, so any loss compounds.

``we`` is a separate distribution (``wepath``) that is not installed with ``pathgennie``, so these
tests add its ``src`` directory to ``sys.path`` and skip cleanly if it cannot be imported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

WE_SRC = Path(__file__).resolve().parents[1] / "we" / "src"
if str(WE_SRC) not in sys.path:
    sys.path.insert(0, str(WE_SRC))

wepath_base = pytest.importorskip("wepath.base")
wepath_resampler = pytest.importorskip("wepath.resampler")


class _W:
    """Minimal walker stand-in exposing the only attribute the routine touches."""

    def __init__(self, weight: float):
        self.weight = float(weight)


def _total(walkers) -> float:
    return sum(w.weight for w in walkers)


def _redistribute(walkers, cap=0.1):
    # Called unbound: the routine only touches the walker list and cap.
    wepath_base.WeightedEnsembleBase.redistribute_excess_weight(None, walkers, cap=cap)


@pytest.mark.parametrize(
    "weights",
    [
        [0.30, 0.30, 0.40],   # every walker above the cap -> the old leak
        [0.34, 0.33, 0.33],   # 3 source walkers, as in the 1opj production configs
        [0.50, 0.01, 0.01],   # mixed, with headroom below the cap
        [0.05, 0.03, 0.02],   # all already under the cap: no-op
    ],
)
def test_redistribute_excess_weight_conserves_total(weights):
    walkers = [_W(w) for w in weights]
    before = _total(walkers)

    _redistribute(walkers)

    after = _total(walkers)
    assert after == pytest.approx(before, rel=0, abs=1e-12), (
        f"redistribute_excess_weight changed total probability {before} -> {after} "
        f"(lost {before - after}); Weighted Ensemble requires exact weight conservation."
    )


def test_redistribute_enforces_cap_when_feasible():
    """When the cap *can* be met, it should be met — conservatively."""
    walkers = [_W(0.5), _W(0.0), _W(0.0), _W(0.0), _W(0.0),
               _W(0.0), _W(0.0), _W(0.0), _W(0.0), _W(0.0)]
    before = _total(walkers)

    _redistribute(walkers, cap=0.1)

    assert _total(walkers) == pytest.approx(before, abs=1e-12)
    assert max(w.weight for w in walkers) <= 0.1 + 1e-9


def test_redistribute_never_creates_negative_weights():
    walkers = [_W(0.9), _W(0.05), _W(0.05)]
    _redistribute(walkers, cap=0.5)
    assert all(w.weight >= 0.0 for w in walkers)


def test_resampler_rng_is_seedable_and_reproducible():
    """Resampling used NumPy's global RNG, so runs could not be reproduced."""
    Resampler = wepath_resampler.Resampler

    a = Resampler(bins=None, target_per_bin=3, seed=1234)
    b = Resampler(bins=None, target_per_bin=3, seed=1234)
    c = Resampler(bins=None, target_per_bin=3, seed=9999)

    draws_a = [float(a.rng.random()) for _ in range(20)]
    draws_b = [float(b.rng.random()) for _ in range(20)]
    draws_c = [float(c.rng.random()) for _ in range(20)]

    assert draws_a == draws_b, "same seed must reproduce the same resampling decisions"
    assert draws_a != draws_c, "different seeds must give different decisions"
