#!/usr/bin/env python
"""Run the AMBER QM/MM alanine-dipeptide conformational example."""

from pathlib import Path
import argparse
import sys


CASE_DIR = Path(__file__).resolve().parent

try:
    from pathgennie.backends.amber.pg_amber import run
except ModuleNotFoundError:
    sys.path.insert(0, str(CASE_DIR.parents[2]))
    from pathgennie.backends.amber.pg_amber import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="input_c7eq.yaml")
    args = parser.parse_args()
    run(CASE_DIR, args.config)
