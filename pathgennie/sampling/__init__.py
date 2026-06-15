"""Enhanced-sampling stages built on top of PathGennie output.

PathGennie generates candidate reactive paths cheaply; turning those into
free-energy surfaces or rate constants is the job of a downstream
*enhanced-sampling stage*.  This package defines the consistent contract those
stages share — :class:`PathEnsemble` (what a PathGennie run hands downstream) and
:class:`SamplingStage` (how a stage consumes it) — so weighted ensemble (WE) and
OPES can be added as interchangeable implementations rather than bespoke
scripts.
"""

from .base import PathEnsemble, SamplingResult, SamplingStage

__all__ = ["PathEnsemble", "SamplingResult", "SamplingStage"]
