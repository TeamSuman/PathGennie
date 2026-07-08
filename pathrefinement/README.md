# Path Refinement

The `pathrefinement` module provides tools to iteratively refine collective variable (CV) pathways using short-burst molecular dynamics (PathGennie) and principal curve neural network smoothing. 

This implementation aligns with the methodology described in:
**PathGennie: Rapid Generation of Rare Event Pathways via Direction-Guided Adaptive Sampling Using Ultrashort Monitored Trajectories**

## Key Features
- **Generic Path Refiner**: Iterative refinement algorithm using PathGennieMD trajectory sampling and NN smoothing.
- **Toy Potentials**: `MullerBrownPotential` and `ThreeHolePotential` for quick 2D validation of refinement algorithms.
- **PathCV**: Dimension-agnostic implementation of the Branduardi et al. path collective variables ($s, z$).
- **Principal Curve**: Endpoint-pinned curve smoothing for generating reference paths.

## Structure
- `refiner.py`: Contains the `PathRefiner` and `PathRefinementConfig`.
- `potentials.py`: Defines 2D analytical toy potentials and generates OpenMM simulations for them.
- `pathcv.py`: Path Collective Variable implementation.
- `ensemblerefiner.py`: PyTorch-based neural network model that learns the continuous map $t \to \mathbf{x}$.
- `principal_curve.py`: Expectation-maximization based curve smoothing.
- `examples/`: Runnable, numbered scripts for the Müller-Brown toy potential
  (`muller_brown/`) and molecular systems (`AlaD/`, `CLN025/`). Each case runs as
  a short pipeline: `1_generate_initial_path.py` → `2_run_refinement.py` →
  `3_analyze_refinement.py`.

## Getting Started

Run the Müller-Brown toy example to see how refinement improves an initially poor
path estimate (this stage requires only NumPy):

```bash
conda activate pathgennie

cd pathrefinement/examples/muller_brown
python 1_generate_initial_path.py   # build a deliberately bad initial path
python 2_run_refinement.py          # ensemble principal-curve refinement
python 3_analyze_refinement.py      # convergence + FES plots
```
Outputs are written under `pathrefinement/examples/muller_brown/results/`. The
`AlaD/` and `CLN025/` cases follow the same numbered sequence but require OpenMM
(and, for the refiner, PyTorch — `pip install 'pathgennie[ml]'`); edit the paths
in each case's `common.py` for your environment before running.

### Using the Refiner

```python
from pathrefinement import MullerBrownPotential, PathRefiner, PathRefinementConfig

# 1. Setup system and initial path
potential = MullerBrownPotential()
bad_path = potential.make_bad_initial_path("A", "C", n_images=20, noise=0.0)

# 2. Configure refinement
config = PathRefinementConfig(
    n_iterations=5,
    n_trajectories=10,
    pathgennie_tau1=100,
    pathgennie_tau2=100,
    pathgennie_max_trial=10,
    pathgennie_max_cycle=100,
    keep_endpoints=True,
    verbosity=1
)

# 3. Refine
refiner = PathRefiner(potential, config)
result = refiner.refine(bad_path)

# 4. Save and plot
result.plot("refined_path.png", potential=potential)
```

## How It Works (The Algorithm)

The iterative path refinement is a fixed-point iteration:
1. **Initialize**: Begin with an initial path and construct a `PathCV`.
2. **Sample**: Run PathGennieMD in target mode along the `PathCV` to generate an ensemble of short reactive trajectories.
3. **Smooth**: Apply `PrincipalCurve` to the raw trajectories, then train an `EnsemblePathRefinerFast` (neural network) to learn a consensus mapping $t \to \mathbf{x}$.
4. **Update**: Construct a new `PathCV` from the refined neural network path.
5. **Iterate**: Repeat until the reference path stops changing significantly.
