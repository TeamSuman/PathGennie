# Tutorial 10 — Path refinement on the Müller-Brown potential

This tutorial turns a deliberately bad initial path between two Müller-Brown
minima into a smooth, representative transition path, and shows the `s`/`z`
[path collective variables](../path-cv.md) that drive it. It runs on CPU in a
few minutes.

## Prerequisites

```bash
pip install 'pathgennie[ml]'    # PyTorch for the neural principal-curve model
# plus OpenMM (conda-forge) for the toy-potential MD driver
```

## Run the pipeline

The example is a short numbered pipeline:

```bash
cd pathrefinement/examples/muller_brown
python 1_generate_initial_path.py    # sample a rough path -> results/initial_path/
python 2_run_refinement.py           # explore -> smooth -> refine loop
python 3_analyze_refinement.py       # convergence + s/z + FES plots
```

Outputs land in `pathrefinement/examples/muller_brown/results/` (not tracked in
git). `3_analyze_refinement.py` produces the convergence curve (max-norm path
change per iteration) and the `(s, z)` projection of the refined path.

## What is happening

Each outer iteration (see [Path Refinement](../path-refinement.md)):

1. seeds short PathGennie trajectories along the current path, kept near it in
   `(s, z)` space;
2. fits a principal-curve / neural `t → x` consensus to the pooled trajectories;
3. replaces the path with the consensus and repeats until the path stops moving.

## Programmatic equivalent

```python
from pathrefinement import MullerBrownPotential, PathRefiner, PathRefinementConfig

potential = MullerBrownPotential()
bad = potential.make_bad_initial_path("A", "C", n_images=20, noise=0.0)
result = PathRefiner(potential, PathRefinementConfig(n_iterations=10, seed=42)).refine(bad)
result.save("results/refinement")
print("converged:", result.converged, "in", result.n_iterations_run, "iterations")
```

## Verify the math

```bash
python pathrefinement/examples/muller_brown/verify_mathematical_correctness.py
```

checks the Branduardi `s`/`z` implementation against hand-derived values on a
known path — a good sanity check before trusting the CVs on a real system.

## Next steps

- Apply the same pipeline to a molecule: see `pathrefinement/examples/AlaD/` and
  `pathrefinement/examples/CLN025/` (adapt `common.py` to your force field and
  CV atoms).
- Feed a refined path back into PathGennie as a progress CV — return `s` (and
  `z`) from your `projection.py`, as shown in [Path CVs](../path-cv.md).
