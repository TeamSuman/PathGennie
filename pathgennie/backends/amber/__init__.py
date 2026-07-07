"""AMBER backend for PathGennie."""

from typing import TYPE_CHECKING

__all__ = ["CoreAmberEngine", "run"]

if TYPE_CHECKING:
    from .engine import CoreAmberEngine
    from .pg_amber import run


def __getattr__(name: str):
    if name == "CoreAmberEngine":
        from .engine import CoreAmberEngine

        return CoreAmberEngine
    if name == "run":
        from . import pg_amber

        return pg_amber.run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
