"""GROMACS backend for PathGennie."""

from typing import TYPE_CHECKING

__all__ = ["GenericGromacsEngine", "GenericPathGennieGromacs", "run"]

if TYPE_CHECKING:
    from .pg_gmx import GenericGromacsEngine, GenericPathGennieGromacs, run


def __getattr__(name: str):
    if name in __all__:
        from . import pg_gmx

        return getattr(pg_gmx, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
