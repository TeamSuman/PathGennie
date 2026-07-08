# Path Refinement

`pathrefinement/` turns a **rough** initial transition path (for example, the
reactive path PathGennie discovers, or a linear/TMD interpolation) into a
**smooth, physically representative** minimum-energy-like path. It does so by
alternating short MD exploration around the current path with a machine-learned
principal-curve consensus, using [path collective variables](path-cv.md) (`s`,
`z`) to keep exploration on-channel.

> **Optional dependencies.** The refiner needs OpenMM (via `PathGennieMD`) and
> PyTorch (the neural principal-curve model). Install with
> `pip install 'pathgennie[ml]'` plus OpenMM from conda-forge. The
> dependency-free primitives (`PathCV`, `PrincipalCurve`) import without them —
> `import pathrefinement` succeeds on a base install and the heavy pieces load
> lazily on first use.

## The idea

Each outer iteration:

1. **Explore** — launch several short PathGennie trajectories seeded along the
   current path (staying near it in `(s, z)` space).
2. **Smooth** — fit a `PrincipalCurve` / neural `t → x` map to the pooled
   trajectories to get a consensus path.
3. **Refine** — replace the path with the consensus and repeat until it stops
   moving (max-norm change below a tolerance).

## Components

| Module | Role |
| ------ | ---- |
| `pathcv.py` (`PathCV`) | Branduardi `s`/`z` path CVs — see [Path CVs](path-cv.md). |
| `principal_curve.py` (`PrincipalCurve`) | Expectation-maximisation curve smoother (NumPy-only). |
| `ensemblerefiner.py` (`EnsemblePathRefinerFast`) | PyTorch model that learns a smooth `t → x` map from an ensemble of trajectories. |
| `refiner.py` (`PathRefiner`, `PathRefinementConfig`, `RefinementResult`) | Orchestrates the explore→smooth→refine loop. |
| `potentials.py` | Analytic 2-D test potentials (Müller-Brown, three-hole) with OpenMM `CustomExternalForce` builders. |

## Quick start (Müller-Brown toy)

```python
from pathrefinement import MullerBrownPotential, PathRefiner, PathRefinementConfig

potential = MullerBrownPotential()
bad_path = potential.make_bad_initial_path("A", "C", n_images=20, noise=0.0)

cfg = PathRefinementConfig(
    n_iterations=10,
    n_trajectories=5,
    nn_epochs=2000,
    device="cpu",     # or "cuda"
    seed=42,
)
refiner = PathRefiner(potential, cfg)
result = refiner.refine(bad_path)
result.save("results/refinement")   # writes the refined path + metadata/timing
```

`PathRefinementConfig` also exposes the PathGennie exploration budget
(`pathgennie_tau1`, `pathgennie_tau2`, `pathgennie_max_trial`,
`pathgennie_max_cycle`, `pathgennie_tol_target`), endpoint handling
(`keep_endpoints`), and parallelism (`n_workers`, `worker_device`).

## Runnable examples

Each case is a short **numbered pipeline** under `pathrefinement/examples/`:

```bash
cd pathrefinement/examples/muller_brown
python 1_generate_initial_path.py    # sample a rough path (writes results/initial_path/)
python 2_run_refinement.py           # ensemble principal-curve refinement
python 3_analyze_refinement.py       # convergence + FES / Ramachandran plots
```

- `muller_brown/` — the analytic toy; also ships
  `verify_mathematical_correctness.py`, which checks the `s`/`z` math against
  hand-derived values.
- `AlaD/` and `CLN025/` — molecular cases (alanine dipeptide, chignolin) that
  require OpenMM/PyTorch. Edit the paths in each case's `common.py`
  (`$GMXFFDIR` for the GROMACS force-field directory) before running.

Generated `results/` are not tracked in git; they are recreated by the scripts.

## Scope & status

The `s`/`z` PathCV math and the `PrincipalCurve` smoother are general and
validated on the toy potentials. The full explore→smooth→refine loop is
production-tested on the 2-D potentials; the molecular cases (`AlaD`, `CLN025`)
are worked examples rather than turnkey pipelines — expect to adapt the feature
selection (`common.py`) to your system. Roadmap items for a general molecular
path-CV feature map are tracked in [HPC Review & Roadmap](HPC_REVIEW.md).
