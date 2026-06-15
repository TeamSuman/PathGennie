"""Enhanced-sampling stages built on top of PathGennie output.

PathGennie generates candidate reactive paths cheaply; turning those into
free-energy surfaces or rate constants is the job of a downstream
*enhanced-sampling stage*.  This package defines the consistent contract those
stages share — :class:`PathEnsemble` (what a PathGennie run hands downstream) and
:class:`SamplingStage` (how a stage consumes it) — so weighted ensemble (WE) and
OPES can be added as interchangeable implementations rather than bespoke
scripts.
"""

from .base import (
    PathEnsemble,
    SamplingResult,
    SamplingStage,
    build_path_ensemble,
)
from .weighted_ensemble import GridBinner, Walker, WeightedEnsembleStage, resample

__all__ = [
    "PathEnsemble",
    "SamplingResult",
    "SamplingStage",
    "build_path_ensemble",
    "WeightedEnsembleStage",
    "GridBinner",
    "Walker",
    "resample",
    "make_stage",
]


def make_stage(name: str, **cfg):
    """Construct an enhanced-sampling stage by ``downstream`` name.

    Recognised: ``weighted_ensemble``.  ``opes`` is reserved for Phase 6b and
    currently raises :class:`NotImplementedError`.
    """

    key = str(name).lower()
    if key in ("weighted_ensemble", "we", "wess"):
        return WeightedEnsembleStage(**cfg)
    if key == "opes":
        raise NotImplementedError("OPESStage is not implemented yet (Phase 6b)")
    raise KeyError(f"unknown downstream stage {name!r}; choose from ['weighted_ensemble']")
