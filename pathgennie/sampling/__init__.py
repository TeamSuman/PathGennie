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
from .opes import OPESStage, build_plumed_opes_input
from .path_sampling import (
    CVRangeState,
    PathSamplingStage,
    extract_transition_path,
    prepare_ops_seed,
    tis_interfaces,
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
    "OPESStage",
    "build_plumed_opes_input",
    "PathSamplingStage",
    "CVRangeState",
    "extract_transition_path",
    "prepare_ops_seed",
    "tis_interfaces",
    "make_stage",
]


def make_stage(name: str, *, executor=None, **cfg):
    """Construct an enhanced-sampling stage by ``downstream`` name.

    Recognised: ``weighted_ensemble`` (aliases ``we``/``wess``), ``opes``, and
    ``tps``/``tis`` (OpenPathSampling, for kinetics).

    ``executor`` (a :class:`~pathgennie.core.parallel.ParallelExecutor`) is only
    forwarded to Weighted Ensemble, whose walker propagation parallelises across
    it; other stages ignore it. A ``None`` executor leaves WE on its serial default.
    """

    key = str(name).lower()
    if key in ("weighted_ensemble", "we", "wess"):
        if executor is not None:
            cfg["executor"] = executor
        return WeightedEnsembleStage(**cfg)
    if key == "opes":
        return OPESStage(**cfg)
    if key in ("tps", "tis"):
        return PathSamplingStage(mode=key, **cfg)
    raise KeyError(
        f"unknown downstream stage {name!r}; choose from "
        "['weighted_ensemble', 'opes', 'tps', 'tis']"
    )
