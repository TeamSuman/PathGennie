"""AMBER backend for PathGennie."""

from typing import TYPE_CHECKING

__all__ = ["GenericAmberEngine", "GenericPathGennieAmber", "run"]

if TYPE_CHECKING:
    from .pg_amber import GenericAmberEngine, GenericPathGennieAmber, run


def __getattr__(name: str):
    if name in __all__:
        from . import pg_amber

        return getattr(pg_amber, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
