#!/usr/bin/env python
"""Run the alanine dipeptide PathGennie OpenMM smoke test."""

from pathlib import Path
import sys


import argparse

CASE_DIR = Path(__file__).resolve().parent

try:
    from pathgennie.backends.openmm.pg_openmm import run
except ModuleNotFoundError:
    sys.path.insert(0, str(CASE_DIR.parents[2]))
    from pathgennie.backends.openmm.pg_openmm import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Alanine Dipeptide OpenMM case")
    parser.add_argument("--case", type=Path, default=CASE_DIR, help="Case directory")
    parser.add_argument("--config", default="input_target.yaml", help="YAML config name inside case directory")
    args = parser.parse_args()
    run(args.case, args.config)

