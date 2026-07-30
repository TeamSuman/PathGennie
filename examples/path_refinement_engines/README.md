# Engine-agnostic path refinement

`PathRefiner` used to hard-wire an OpenMM walker for its exploration step. That made refinement
unavailable to the two backends that need it most: **AMBER**, the only backend that can run a QM
Hamiltonian, and **GROMACS**.

It now accepts an injected `sampler`, and `pathrefinement.samplers.EngineSampler` implements that
contract on the core `Engine` protocol — so refinement works with any backend, and with anything
else that satisfies the protocol.

```python
from pathrefinement.refiner import PathRefiner, PathRefinementConfig
from pathrefinement.samplers import EngineSampler

sampler = EngineSampler(engine, initial_handle=handle, feature_fn=my_features,
                        tau1=10, tau2=10, max_trial=30, max_cycle=300)
result = PathRefiner(potential=None, config=cfg, sampler=sampler).refine(initial_path)
```

`refine_with_engine.py` runs that identical code against all four engines; only the `build_engine`
function differs between them.

```bash
python refine_with_engine.py --engine toy                       # no MD binary needed
python refine_with_engine.py --engine openmm --platform CPU     # needs openmm
python refine_with_engine.py --engine amber   --topology sys.prmtop --start sys.rst7 --atoms 0 1 0 5
python refine_with_engine.py --engine gromacs --topology sys.top    --start sys.gro  --atoms 0 1 0 5
```

The `toy` and `openmm` cases are self-contained — they build their own system in memory, so you can
run them immediately. The `amber` and `gromacs` cases need your own topology and starting structure;
[`examples/qmmm_reactive_sn2/amber`](../qmmm_reactive_sn2/amber/README.md) is a complete worked AMBER
case including the QM/MM settings.

## What the sampler does

`EngineSampler` runs one PathGennie walker in **target mode on the path progress coordinate `s`**,
driving it toward the far end of the current PathCV. Each refinement iteration calls it once per
walker; the returned trajectories, expressed in feature space, are what the principal-curve and
neural consensus fit.

Because it only calls `clone_anchor` / `run_segment` / `get_coords` / `release`, anything satisfying
the `Engine` protocol works — including a stub, which is how the contract is regression-tested in
`tests/test_sampler_multi_engine.py`.

## Choosing the feature space

Refinement fits a curve, so it needs a low-dimensional space. Raw Cartesian coordinates are both too
high-dimensional and not invariant to translation or rotation. Use a handful of internal coordinates
— distances, angles, or a learned CV — that distinguish reactant from product. `PathRefiner` is
currently specialised to **2-D** feature spaces.

`feature_fn` receives an `(n_atoms, 3)` array and must return a fixed-length 1-D array.

## Engine-specific notes

| Backend | Handle is | Watch for |
| --- | --- | --- |
| `toy` | an integer id | State lives in memory; nothing to clean up. |
| `openmm` | an integer id into the context pool | Pass `reproducible=True` for seeded reruns — re-seeding a live `Context` has no effect until it is reinitialised. |
| `amber` | a path to an `.rst7` | `create_handle` writes zero velocities; τ₁ randomises them anyway. Point `scratch_dir` at fast local disk. |
| `gromacs` | a path to a `.gro` | The matching `.cpt` is carried alongside so τ₂ runners continue velocities. |

Both subprocess backends write one directory per device under `scratch_dir` and can generate a lot
of small files. Keep that off a shared network filesystem where possible, and keep it out of the
repository.
