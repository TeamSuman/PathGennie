#!/usr/bin/env python
"""Stage 5 -- an independent reference path (NEB), and a DFT barrier.

Stages 1-3 produce a path and a free energy from *sampling*. This stage computes
the minimum-energy path by a completely different route -- a nudged elastic band
-- so the refined PathCV can be checked against something that shares none of its
machinery.

The band is run at the **same level of theory as the refinement**. A reference
computed at a different level measures the method difference, not the path.

    python 5_neb_reference.py --beads 16                    # NEB at DFTB3
    python 5_neb_reference.py --beads 16 --rescore          # + DFT single points

Needs ``sander.MPI`` and ``mpirun``; ``--rescore`` additionally needs ``quick``.
Cost on this 6-atom system: ~30 s for the band, ~90 s for the DFT re-scoring.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sn2_cv import C, CL_ATTACK, CL_LEAVE  # noqa: E402

ELEMENTS = ["C", "Cl", "H", "H", "H", "Cl"]        # sn2.prmtop atom order
HARTREE_TO_KCAL = 627.509474

# cut=99, not 999: the larger value is fine for single points and short MD but
# blows up the nonbond grid under imin=1 ("SANDER BOMB ... volume of ucell too big").
# maxcyc=300/ncyc=150 are validated values -- do not "improve" them without
# re-checking the geometry. At 500/200 the conjugate-gradient phase walks the
# shallow gas-phase ion-dipole minimum apart and the product endpoint dissociates
# to d(C-Cl) ~ 900 A, which then poisons the whole band.
MIN_MDIN = """endpoint minimisation
 &cntrl
  imin=1, maxcyc=300, ncyc=150, ntmin=1,
  ntb=0, cut=99.0, ntpr=100, ntxo=1,
  ifqnt=1,
 /
 &qmmm
   qmmask=':1-2', qmcharge=-1, qm_theory='DFTB3',
 /
"""

# temp0=0 with heavy friction damps the band onto the MEP. skmin=skmax => uniform
# springs, which is the right default when you have no reason to bias bead spacing.
NEB_MDIN = """NEB quench, DFTB3
 &cntrl
  imin=0, irest=0, ntx=1, nstlim={nstlim}, dt=0.0005,
  ntc=1, ntf=1, ntb=0, cut=99.0,
  ntpr=250, ntwx=0, ntwr={nstlim}, ntxo=1,
  ntt=3, gamma_ln=1000.0, temp0=0.0, tempi=0.0, ig={seed},
  ineb=1, skmin={sk}, skmax={sk},
  tgtfitmask=':1-2', tgtrmsmask=':1-2',
  ifqnt=1,
 /
 &qmmm
   qmmask=':1-2', qmcharge=-1, qm_theory='DFTB3',
 /
"""


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def build_endpoints(start_rst: Path, work: Path) -> None:
    """Reactant as given; product = mirrored in z **and** with the Cl indices swapped.

    Mirroring alone preserves every interatomic distance, so it just reproduces the
    reactant. The swap is what exchanges the leaving and attacking roles.
    """
    from pathgennie.backends.amber.utils import read_rst7_coords, write_rst7_coords

    x = np.asarray(read_rst7_coords(str(start_rst)), dtype=float)
    write_rst7_coords(work / "react_in.rst7", x)
    p = x.copy()
    p[:, 2] *= -1.0
    p[[CL_LEAVE, CL_ATTACK]] = p[[CL_ATTACK, CL_LEAVE]]
    write_rst7_coords(work / "prod_in.rst7", p)

    def xi(c):
        return (np.linalg.norm(c[C] - c[CL_LEAVE])
                - np.linalg.norm(c[C] - c[CL_ATTACK]))

    print(f"  reactant xi = {xi(x):+.3f}   product xi = {xi(p):+.3f}")
    if abs(xi(x) + xi(p)) > 1e-6:
        raise SystemExit("product xi is not the negative of reactant xi -- "
                         "the endpoints are not mirror images")


def minimise(work: Path, topology: Path, sander: str, max_bond: float = 10.0) -> None:
    """Minimise both endpoints, then check the geometry is still a molecule.

    Existence of the restart is NOT enough. A gas-phase minimisation that runs too
    long dissociates the ion and still writes a perfectly well-formed file; the
    band built from it then relaxes into nonsense with no error anywhere.
    """
    from pathgennie.backends.amber.utils import read_rst7_coords

    (work / "min.mdin").write_text(MIN_MDIN)
    for tag in ("react", "prod"):
        out = work / f"{tag}_min.out"
        rst = work / f"{tag}_min.rst7"
        sh([sander, "-O", "-i", "min.mdin", "-p", str(topology),
            "-c", f"{tag}_in.rst7", "-o", out.name, "-r", rst.name,
            "-inf", f"{tag}.inf"], cwd=work)
        if not rst.exists() or rst.stat().st_size == 0:
            tail = "\n".join(out.read_text().splitlines()[-6:]) if out.exists() else "(no output)"
            raise SystemExit(f"{tag} minimisation wrote no restart:\n{tail}")

        c = np.asarray(read_rst7_coords(str(rst)), dtype=float)
        d1 = float(np.linalg.norm(c[C] - c[CL_LEAVE]))
        d5 = float(np.linalg.norm(c[C] - c[CL_ATTACK]))
        print(f"  {tag} minimised: d(C-Cl1)={d1:.3f}  d(C-Cl5)={d5:.3f}  xi={d1 - d5:+.3f}")
        if max(d1, d5) > max_bond:
            raise SystemExit(
                f"{tag} endpoint dissociated during minimisation "
                f"(d(C-Cl) up to {max(d1, d5):.1f} A > {max_bond} A).\n"
                "The gas-phase ion-dipole minimum is shallow; reduce maxcyc/ncyc in "
                "MIN_MDIN.\nA dissociated endpoint still writes a valid restart file, "
                "so this is checked explicitly rather than trusted.")


def interpolate(work: Path, n: int) -> None:
    from pathgennie.backends.amber.utils import read_rst7_coords, write_rst7_coords

    a = np.asarray(read_rst7_coords(str(work / "react_min.rst7")), dtype=float)
    b = np.asarray(read_rst7_coords(str(work / "prod_min.rst7")), dtype=float)
    for i in range(n):
        t = i / (n - 1)
        write_rst7_coords(work / f"bead{i + 1:02d}.rst7", (1.0 - t) * a + t * b)
    print(f"  {n} beads by linear interpolation")


def run_neb(work: Path, topology: Path, n: int, nstlim: int, sk: float, seed: int) -> None:
    (work / "neb.mdin").write_text(NEB_MDIN.format(nstlim=nstlim, sk=sk, seed=seed))
    lines = [
        f"-O -i neb.mdin -p {topology} -c bead{i:02d}.rst7 "
        f"-o neb{i:02d}.out -r nebr{i:02d}.rst7 -inf nebi{i:02d}.inf"
        for i in range(1, n + 1)
    ]
    (work / "groupfile").write_text("\n".join(lines) + "\n")
    # Stream rather than capture: MPI launch failures are the most confusing thing
    # that can go wrong here, and burying them in a pipe helps nobody.
    subprocess.run(["mpirun", "-np", str(n), "sander.MPI", "-ng", str(n),
                    "-groupfile", "groupfile"], cwd=work)
    if not (work / "nebr01.rst7").exists():
        tail = "\n".join((work / "neb01.out").read_text().splitlines()[-12:]) \
            if (work / "neb01.out").exists() else "(no bead output produced)"
        raise SystemExit(f"NEB produced no restarts:\n{tail}")
    print(f"  band relaxed ({nstlim} steps)")


def replicate_energies(path: Path, n: int) -> list:
    """Per-bead energies from the final 'NEB replicate breakdown' block.

    Do NOT scrape the last ``EPtot``: AMBER ends its output with A V E R A G E S
    and R M S  F L U C T U A T I O N S blocks, so the last EPtot is a *fluctuation*
    (~0.3 kcal/mol here) that looks exactly like a plausible small barrier.
    """
    energies = {}
    for line in path.read_text().splitlines():
        if line.startswith("Energy for replicate"):
            try:
                energies[int(line.split("replicate")[1].split("=")[0])] = float(line.split("=")[1])
            except (IndexError, ValueError):
                pass
    return [energies.get(i) for i in range(1, n + 1)]


def rescore_dft(work: Path, n: int, quick: str, basis: str, functional: str) -> list:
    """Single points at DFT on the DFTB3 geometries (``DFT//DFTB3``).

    Diffuse functions are not optional: the reactant is Cl-, and a basis without
    them describes an anion badly and distorts the barrier.
    """
    from pathgennie.backends.amber.utils import read_rst7_coords

    out = []
    for i in range(1, n + 1):
        c = np.asarray(read_rst7_coords(str(work / f"nebr{i:02d}.rst7")), dtype=float)
        inp = work / f"q{i:02d}.in"
        body = [f"{functional} BASIS={basis} CHARGE=-1 ENERGY", ""]
        body += [f"{el:<2} {x:14.8f} {y:14.8f} {z:14.8f}"
                 for el, (x, y, z) in zip(ELEMENTS, c)]
        inp.write_text("\n".join(body) + "\n")
        sh([quick, inp.name], cwd=work)
        qout = work / f"q{i:02d}.out"
        e = None
        if qout.exists():
            for line in qout.read_text().splitlines():
                if "TOTAL ENERGY" in line:
                    try:
                        e = float(line.split("=")[-1])
                    except ValueError:
                        pass
        if e is None:
            raise SystemExit(f"QUICK produced no energy for bead {i}; see {qout}")
        out.append(e)
    return out


def report(work: Path, n: int, dft: list | None) -> None:
    from pathgennie.backends.amber.utils import read_rst7_coords

    e_dftb = replicate_energies(work / "neb01.out", n)
    geom = [np.asarray(read_rst7_coords(str(work / f"nebr{i:02d}.rst7")), dtype=float)
            for i in range(1, n + 1)]
    d1 = np.array([np.linalg.norm(c[C] - c[CL_LEAVE]) for c in geom])
    d5 = np.array([np.linalg.norm(c[C] - c[CL_ATTACK]) for c in geom])

    cols = [("DFTB3", np.array([e if e is not None else np.nan for e in e_dftb]))]
    if dft is not None:
        cols.append(("DFT//DFTB3", np.array(dft) * HARTREE_TO_KCAL))
    cols = [(name, e - np.nanmin(e)) for name, e in cols]

    header = f"{'bead':>5} " + " ".join(f"{name:>12}" for name, _ in cols)
    print(f"\n{header} {'d(C-Cl1)':>9} {'d(C-Cl5)':>9}")
    for i in range(n):
        vals = " ".join(f"{e[i]:>12.2f}" for _, e in cols)
        print(f"{i + 1:>5} {vals} {d1[i]:>9.3f} {d5[i]:>9.3f}")

    for name, e in cols:
        top = int(np.nanargmax(e))
        mean = 0.5 * (d1[top] + d5[top])
        print(f"\n  {name}: barrier {np.nanmax(e):.2f} kcal/mol at bead {top + 1}, "
              f"mean d(C-Cl) = {mean:.4f} A")
        # An identity reaction must give a symmetric band -- a free error estimate.
        worst = max((abs(e[k] - e[n - 1 - k]) for k in range(n // 2)), default=np.nan)
        print(f"    band symmetry: worst |E(k) - E(n+1-k)| = {worst:.4f} kcal/mol")

    # The relaxed band must still be a molecule. A blown-up band produces a
    # confident-looking table of large numbers, so say so instead of saving it.
    if max(d1.max(), d5.max()) > 10.0:
        raise SystemExit(
            f"the relaxed band is unphysical (d(C-Cl) up to {max(d1.max(), d5.max()):.1f} A). "
            "Check the minimised endpoints first -- a dissociated endpoint poisons the "
            "whole band.")

    np.save(work / "neb_path_2d.npy", np.column_stack([d5, d1]))
    print(f"\n  saved {work / 'neb_path_2d.npy'} (d_attack, d_leave)")
    print("  overlay it on the sampled paths with:")
    print(f"    python 4_plot_2d_cv.py --extra 'NEB MEP={work / 'neb_path_2d.npy'}'")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--beads", type=int, default=16)
    p.add_argument("--nstlim", type=int, default=3000)
    p.add_argument("--spring", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=20260731)
    p.add_argument("--topology", type=Path, default=HERE / "sn2.prmtop")
    p.add_argument("--start", type=Path, default=HERE / "sn2.rst7")
    p.add_argument("--outdir", type=Path, default=HERE / "results" / "neb")
    p.add_argument("--sander", default="sander")
    p.add_argument("--rescore", action="store_true",
                   help="add DFT single points on the relaxed geometries")
    p.add_argument("--quick", default="quick")
    p.add_argument("--basis", default="6-31+G*")
    p.add_argument("--functional", default="B3LYP")
    args = p.parse_args()

    for tool in ("mpirun", "sander.MPI"):
        if shutil.which(tool) is None:
            print(f"{tool} not on PATH -- source AMBER's amber.sh (with `set +u`).")
            print("If sander.MPI links a different MPI than the mpirun on PATH, prepend")
            print("that MPI's bin/lib first; the ABI mismatch fails confusingly.")
            return 2
    if args.rescore and shutil.which(args.quick) is None:
        print(f"--rescore needs {args.quick!r} on PATH")
        return 2

    work = args.outdir
    work.mkdir(parents=True, exist_ok=True)
    topology = args.topology.resolve()

    print("building endpoints")
    build_endpoints(args.start.resolve(), work)
    print("minimising endpoints")
    minimise(work, topology, args.sander)
    print("interpolating")
    interpolate(work, args.beads)
    print("relaxing the band")
    run_neb(work, topology, args.beads, args.nstlim, args.spring, args.seed)

    dft = None
    if args.rescore:
        print(f"re-scoring at {args.functional}/{args.basis}")
        dft = rescore_dft(work, args.beads, args.quick, args.basis, args.functional)

    report(work, args.beads, dft)
    return 0


if __name__ == "__main__":
    sys.exit(main())
