#!/usr/bin/env python
"""Stage 1 -- discover reactive QM/MM paths for Cl- + CH3Cl -> ClCH3 + Cl-.

Runs the PathGennie driver once per seed. Each seed is an independent reactive
path; the spread across seeds is what stage 2 refines into a single PathCV, so a
handful of seeds is the minimum and 10-20 is comfortable.

    python 1_generate_paths.py --seeds 10 --outdir ensemble

Requires ``sander`` on PATH (source AMBER's ``amber.sh``) built with DFTB support,
and the DFTB 3ob-3-1 parameter set reachable via ``$AMBERHOME/dat/slko``.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = ("sn2.prmtop", "sn2.rst7", "sn2_cv.py")


def run_one(case: Path) -> tuple[bool, str]:
    """Run the driver inside ``case`` and report whether it reacted."""
    import contextlib
    import io

    from pathgennie.backends.amber import pg_amber

    cwd = Path.cwd()
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            pg_amber.run(case, "input.yaml")
    except Exception as exc:  # a failed seed must not kill the ensemble
        log.write(f"\nFAILED: {type(exc).__name__}: {exc}\n")
    finally:
        import os

        os.chdir(cwd)
    text = log.getvalue()
    (case / "run.log").write_text(text)
    m = re.search(r"Converged at cycle (\d+)", text)
    return (m is not None), (f"converged at cycle {m.group(1)}" if m else "did not react")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--outdir", type=Path, default=HERE / "ensemble")
    p.add_argument("--base-seed", type=int, default=20260730)
    args = p.parse_args()

    template = (HERE / "input.yaml").read_text()
    if not re.search(r"^\s*seed:", template, flags=re.M):
        print("input.yaml has no `seed:` key -- every seed would run identically.")
        return 2
    args.outdir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    for i in range(1, args.seeds + 1):
        case = args.outdir / f"seed_{i}"
        case.mkdir(exist_ok=True)
        for asset in ASSETS:
            shutil.copy(HERE / asset, case / asset)
        # 7919 is prime, so consecutive seeds stay far apart in the PRNG stream.
        seed = args.base_seed + i * 7919
        (case / "input.yaml").write_text(
            re.sub(r"^(\s*seed:).*$", rf"\g<1> {seed}", template, flags=re.M)
        )
        ok, msg = run_one(case)
        n_ok += ok
        print(f"  seed {i:>3}: {msg}", flush=True)

    print(f"\n{n_ok} / {args.seeds} seeds produced a reactive path -> {args.outdir}")
    print("next: python 2_refine_pathcv.py --ensemble", args.outdir)
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
