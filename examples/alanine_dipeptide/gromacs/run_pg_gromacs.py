#!/usr/bin/env python
"""Run the alanine dipeptide PathGennie GROMACS smoke test."""

from pathlib import Path
import sys


CASE_DIR = Path(__file__).resolve().parent

try:
    from pathgennie.backends.gromacs.pg_gmx import run
except ModuleNotFoundError:
    sys.path.insert(0, str(CASE_DIR.parents[2]))
    from pathgennie.backends.gromacs.pg_gmx import run


if __name__ == "__main__":
    run(CASE_DIR)
