#!/usr/bin/env python
"""Path refinement driven by any backend -- OpenMM, AMBER, GROMACS, or the toy engine.

`PathRefiner` originally hard-wired an OpenMM walker for its exploration step, so
refinement was unavailable to AMBER (the only QM/MM-capable backend) and GROMACS.
It now accepts an injected ``sampler``, and
:class:`pathrefinement.samplers.EngineSampler` implements that contract on the core
``Engine`` protocol.

The point of this example is what *doesn't* change between engines: only
``build_engine`` below is backend-specific. The sampler, the refiner, the config,
and the analysis are identical in all four cases.

    python refine_with_engine.py --engine toy          # no MD binary needed
    python refine_with_engine.py --engine openmm       # needs openmm
    python refine_with_engine.py --engine amber   --topology sys.prmtop --start sys.rst7
    python refine_with_engine.py --engine gromacs --topology sys.top    --start sys.gro \
                                 --mdp md.mdp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# The only backend-specific code in this file.
# --------------------------------------------------------------------------- #
def build_engine(args):
    """Return ``(engine, initial_handle, feature_fn)`` for the chosen backend."""
    scratch = args.scratch / args.engine

    if args.engine == "toy":
        from pathgennie.core.toy import ToyLangevinEngine

        engine = ToyLangevinEngine(dt=0.002, kT=1.0)
        # The toy engine's "coordinates" are already the 2-D feature space.
        return engine, engine.create_state([-1.0, 0.0]), lambda c: np.asarray(c).ravel()[:2]

    if args.engine == "openmm":
        import openmm
        from openmm import unit
        from openmm.app import Element, Simulation, Topology

        from pathgennie.backends.openmm.engine import OpenMMEngine

        system = openmm.System()
        for _ in range(2):
            system.addParticle(12.0)
        # Two particles in a shallow double well along x, weakly tethered in y/z:
        # enough structure for a path to exist without needing an input file.
        force = openmm.CustomExternalForce("2.0*(x*x-0.09)^2 + 20.0*(y*y+z*z)")
        force.addParticle(0, [])
        force.addParticle(1, [])
        system.addForce(force)

        top = Topology()
        chain = top.addChain()
        res = top.addResidue("LIG", chain)
        for _ in range(2):
            top.addAtom("C", Element.getBySymbol("C"), res)
        integrator = openmm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
        )
        sim = Simulation(top, system, integrator,
                         openmm.Platform.getPlatformByName(args.platform))
        engine = OpenMMEngine(sim, temperature=300.0, reproducible=True)
        handle = engine.create_state([[-0.3, 0.0, 0.0], [0.3, 0.0, 0.0]] * unit.nanometer)
        return engine, handle, lambda c: np.asarray(c, dtype=float).reshape(-1, 3)[:, 0][:2]

    if args.engine == "amber":
        from pathgennie.backends.amber.engine import CoreAmberEngine

        engine = CoreAmberEngine(
            topology=args.topology,
            executable=args.executable or "sander",
            scratch_dir=scratch,
            temperature=300.0,
            mdin_controls=dict(dt=0.002, ntc=2, ntf=2, ntb=1, cut=9.0,
                               ntpr=100000, ntwx=0, ntwr=100000, ntxo=1),
        )
        # An AMBER handle is a path to an rst7 file.
        return engine, str(Path(args.start).resolve()), _distance_pair(args.atoms)

    if args.engine == "gromacs":
        from pathgennie.backends.gromacs.pg_gmx import CoreGromacsEngine, read_mdp

        # Read a real .mdp rather than hand-building a dict: grompp rejects an
        # incomplete parameter set, and this is what the backend itself does.
        engine = CoreGromacsEngine(
            topology=args.topology,
            executable=args.executable or "gmx",
            scratch_dir=scratch,
            temperature=300.0,
            mdp_controls=read_mdp(Path(args.mdp)),
            maxwarn=args.maxwarn,
            mdrun_args=["-ntmpi", "1", "-ntomp", "1", "-nb", "cpu", "-pin", "off"],
            # Needed for create_handle: a .gro carries topology metadata that
            # bare coordinates do not.
            template_gro=args.start,
        )
        # A GROMACS handle is a path to a .gro file.
        return engine, str(Path(args.start).resolve()), _distance_pair(args.atoms)

    raise SystemExit(f"unknown engine {args.engine!r}")


def _distance_pair(atoms):
    """Feature map for a molecular system: two interatomic distances.

    Refinement needs a low-dimensional space to fit a curve in; raw Cartesians are
    both too high-dimensional and not translation/rotation invariant.
    """
    a, b, c, d = atoms

    def feature_fn(coords):
        xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
        return np.array([np.linalg.norm(xyz[a] - xyz[b]),
                         np.linalg.norm(xyz[c] - xyz[d])])

    return feature_fn


# --------------------------------------------------------------------------- #
# Everything below is backend-independent.
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True,
                   choices=["toy", "openmm", "amber", "gromacs"])
    p.add_argument("--topology", type=Path, help="prmtop (amber) or top (gromacs)")
    p.add_argument("--start", type=Path, help="rst7 (amber) or gro (gromacs)")
    p.add_argument("--executable", help="sander / gmx; default is the engine's own")
    p.add_argument("--mdp", type=Path, help="GROMACS .mdp (required for --engine gromacs)")
    p.add_argument("--maxwarn", type=int, default=2, help="grompp -maxwarn")
    p.add_argument("--atoms", type=int, nargs=4, default=[0, 1, 0, 5],
                   metavar=("A", "B", "C", "D"),
                   help="two atom pairs defining the 2-D feature space")
    p.add_argument("--platform", default="CPU", help="OpenMM platform (CPU/CUDA)")
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--walkers", type=int, default=3)
    p.add_argument("--nodes", type=int, default=16)
    p.add_argument("--scratch", type=Path, default=Path("scratch"))
    p.add_argument("--outdir", type=Path, default=None)
    args = p.parse_args()

    if args.engine in ("amber", "gromacs") and not (args.topology and args.start):
        p.error(f"--engine {args.engine} requires --topology and --start")
    if args.engine == "gromacs" and not args.mdp:
        p.error("--engine gromacs requires --mdp")

    from pathrefinement.refiner import PathRefiner, PathRefinementConfig
    from pathrefinement.samplers import EngineSampler

    engine, handle, feature_fn = build_engine(args)
    print(f"engine: {type(engine).__name__}")

    sampler = EngineSampler(
        engine,
        initial_handle=handle,
        feature_fn=feature_fn,
        tau1=10, tau2=10,
        max_trial=10, max_cycle=100,
        sigma=0.1, tol=0.05,
    )

    # A straight line between the endpoints is a deliberately poor starting guess;
    # refinement's job is to bend it onto the real channel.
    start = np.atleast_1d(feature_fn(engine.get_coords(handle)))
    end = start + np.array([1.2, -1.2])[: len(start)]
    initial_path = np.linspace(start, end, args.nodes)

    cfg = PathRefinementConfig(
        n_iterations=args.iterations,
        n_trajectories=args.walkers,
        nn_epochs=800,
        device="cpu",
        seed=42,
        verbosity=1,
    )
    result = PathRefiner(potential=None, config=cfg, sampler=sampler).refine(initial_path)

    moved = float(np.abs(result.refined_path - result.initial_path).max())
    print(f"\nconverged={result.converged} after {result.n_iterations_run} iterations")
    print(f"max node displacement: {moved:.3f}")
    if args.outdir:
        result.save(str(args.outdir))
        print(f"written to {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
