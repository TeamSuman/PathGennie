#!/usr/bin/env python
"""Run this CLN025 PathGennie AMBER case with the packaged runner."""

from pathlib import Path
import sys


CASE_DIR = Path(__file__).resolve().parent

try:
    from pathgennie.backends.amber.pg_amber import run
except ModuleNotFoundError:
    sys.path.insert(0, str(CASE_DIR.parents[2]))
    from pathgennie.backends.amber.pg_amber import run

print(CASE_DIR)
if __name__ == "__main__":
    run(CASE_DIR)
