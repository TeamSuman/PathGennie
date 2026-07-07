"""Scratch-directory placement for the subprocess backends.

Each MD segment writes small control/restart/output files. Over the thousands
of ultrashort segments an adaptive run generates, that is heavy *metadata* I/O
that cripples a shared Lustre/NFS filesystem. On HPC the fix is to place scratch
on **node-local** disk (the scheduler's ``$TMPDIR``) and keep only the final
outputs on the shared filesystem.

``resolve_scratch_dir`` implements an opt-in redirect: set ``scratch_root`` in
``input.yaml`` (or the ``PATHGENNIE_SCRATCH`` environment variable, which a job
script can point at ``$TMPDIR``) and scratch goes there; otherwise it stays under
the run's ``workdir`` (the previous default, unchanged).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

__all__ = ["resolve_scratch_dir"]


def resolve_scratch_dir(
    workdir: Path | str,
    scratch_root: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Path:
    """Return the scratch directory for a run.

    Precedence: explicit ``scratch_root`` (config) > ``$PATHGENNIE_SCRATCH`` >
    ``workdir/scratch``. When a root is used, a per-workdir subdirectory keeps
    concurrent runs that share a node-local root from colliding (schedulers give
    each job its own ``$TMPDIR``, so this is belt-and-braces).
    """

    environ = os.environ if environ is None else environ
    workdir = Path(workdir)
    root = scratch_root or environ.get("PATHGENNIE_SCRATCH")
    if root:
        return Path(root).expanduser() / f"{workdir.name}_scratch"
    return workdir / "scratch"
