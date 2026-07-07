"""Path Refinement module.

Only the dependency-free primitives (``PathCV``, ``PrincipalCurve``) import
eagerly. The analytic ``*Potential`` classes pull in ``openmm`` and
``PathRefiner`` pulls in ``openmm`` + ``torch``; those are optional
dependencies, so they load lazily on first access. This means ``import
pathrefinement`` succeeds on a base install and only touching a heavy symbol
without its dependency raises a clear, actionable error.
"""

import importlib

from .pathcv import PathCV
from .principal_curve import PrincipalCurve

__all__ = [
    "PathRefiner",
    "PathRefinementConfig",
    "RefinementResult",
    "Potential2D",
    "MullerBrownPotential",
    "ThreeHolePotential",
    "PathCV",
    "PrincipalCurve",
]

# Attribute -> submodule that defines it (imported lazily; each pulls in an
# optional dependency: 'refiner' needs openmm + torch, 'potentials' needs openmm).
_LAZY = {
    "PathRefiner": "refiner",
    "PathRefinementConfig": "refiner",
    "RefinementResult": "refiner",
    "Potential2D": "potentials",
    "MullerBrownPotential": "potentials",
    "ThreeHolePotential": "potentials",
}


def __getattr__(name):  # PEP 562 lazy loader
    module_name = _LAZY.get(name)
    if module_name is not None:
        try:
            module = importlib.import_module(f".{module_name}", __name__)
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise ImportError(
                f"pathrefinement.{name} requires optional dependencies (OpenMM "
                "and/or PyTorch) that are not installed. Install them, e.g. "
                "`pip install 'pathgennie[ml]'` plus OpenMM from conda-forge."
            ) from exc
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
