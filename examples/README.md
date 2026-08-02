# PathGennie examples

Each directory is a self-contained case: inputs, a runner script, and a config. Nothing
here writes outside its own `workdir`.

## Start here

**`toy_langevin/`** needs no MD engine at all — it runs on the analytic Wolfe–Quapp
surface using PathGennie's built-in toy engine. Measured: **6.9 s, converges at cycle 4**,
and it exercises the same driver, checkpointing and output path as every real backend.
If you are evaluating PathGennie, run this first.

```bash
cd examples/toy_langevin && python run_toy.py
```

## The examples

| example | backends | mode | what it demonstrates | needs |
|---|---|---|---|---|
| [`toy_langevin`](toy_langevin/) | built-in | target | the driver end-to-end with no MD engine | nothing |
| [`alanine_dipeptide`](alanine_dipeptide/) | AMBER, GROMACS, OpenMM | target | the same φ/ψ problem across all three backends | `pmemd` / `gmx` / OpenMM |
| [`CLN025`](CLN025/) | AMBER, GROMACS | target | folding of the chignolin miniprotein on an end-to-end CV | `pmemd` / `gmx` |
| [`OAMe-G2`](OAMe-G2/) | GROMACS, OpenMM | escape | host–guest unbinding driven by a PLUMED-computed CV | `gmx` / OpenMM, PLUMED |
| [`qmmm_alanine_conformation`](qmmm_alanine_conformation/amber/) | AMBER QM/MM | target | conformational change with a QM solute | `sander` |
| [`qmmm_reactive_sn2`](qmmm_reactive_sn2/amber/) | AMBER QM/MM | target | a full reactive workflow: path generation → path-CV refinement → free energy → NEB reference | `sander` (DFTB3) |
| [`path_refinement_engines`](path_refinement_engines/) | AMBER, GROMACS, OpenMM | — | driving `PathRefiner` from any backend through `EngineSampler` | one MD engine |

`mode: target` drives towards a known product state; `mode: escape` drives away from the
starting basin when no product is known in advance.

## Running one

Examples with an `input.yaml` run either through their script or the CLI:

```bash
cd examples/alanine_dipeptide/amber
python run_pg_amber.py                 # or:  pathgennie amber --case .
```

Set `executable` in the config to your MD binary if it is not on `PATH`. Outputs land in
the `workdir` named by the config and are regenerable — they are git-ignored, not shipped.

## A note on runtimes

Only `toy_langevin` has a meaningful quoted runtime, because it is the only case whose
cost does not depend on your MD build, hardware and GPU. For the others the configs state
`max_cycle`, `tau1_steps` and `tau2_steps`, which is what actually determines the work:
each cycle runs `max_trial` sampler segments of `tau1_steps` plus one runner segment of
`tau2_steps`. Lower `max_cycle` to shorten a trial run.
