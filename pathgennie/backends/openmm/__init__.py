"""OpenMM backend for PathGennie."""

from typing import TYPE_CHECKING

from .pg_omm import PathGennieMD

__all__ = ["PathGennieMD", "run"]

if TYPE_CHECKING:
    from .pg_openmm import run


def __getattr__(name: str):
    if name == "run":
        from .pg_openmm import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
