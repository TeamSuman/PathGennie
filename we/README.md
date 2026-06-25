# WEPath

Research code accompanying the manuscript **"Pathway-resolved kinetics from
pre-seeded weighted ensemble simulations guided by neural-network path
collective variables"** by Dibyendu Maity, Shaheerah Shahid, Sayari
Bhattacharya, Rupak Majumdar, and Suman Chakrabarty.

WEPath performs pathway-conditioned, steady-state weighted-ensemble (WE)
simulations along path collective variables (PathCVs). The method is designed
to estimate the target flux associated with a specified transition tube while
preserving the underlying unbiased molecular dynamics.

## Method

The complete workflow described in the manuscript is:

1. Generate an ensemble of reactive trajectories with PathGennie.
2. Cluster trajectories into mechanistic families using multidimensional
   dynamic time warping and hierarchical clustering.
3. Smooth and reparameterize each family as a reference path using a neural
   network.
4. Construct PathCV progress (`s`) and tube-distance (`z`) coordinates from
   each reference path.
5. Pre-seed WE walkers along each path, propagate unbiased dynamics, and
   periodically split or merge walkers within PathCV bins.
6. Recycle target-reaching walkers to the source and estimate the
   pathway-conditioned rate from steady-state target flux.

This repository implements the WE portion of steps 5 and 6, including
propagation, binning, resampling, recycling, and target-flux logging. The toy
example also builds upper and lower reference paths with a zero-temperature
string relaxation.

## Applications in the Manuscript

- **Three-hole model potential:** temperature-dependent crossover between two
  competing transition channels.
- **Benzamidine--trypsin (3PTB):** kinetic separation of three unbinding
  pathways.
- **Imatinib--Abl kinase (1OPJ):** redistribution of flux between the
  P-loop/hinge and alpha-C-helix channels in wild-type Abl and the N368S
  mutant.


## Repository Layout

```text
src/wepath/          WE driver, walkers, bins, resampling, workers, and utilities
examples/toy/        Runnable OpenMM three-hole-potential examples
examples/1opj/       WT and N368S Abl--imatinib production templates
example_inputs/      Local location for non-versioned molecular inputs
environment.yml      Conda environment definition
pyproject.toml       Installable Python package metadata
```

## Installation

The recommended installation uses conda-forge because OpenMM and the molecular
analysis dependencies require compiled packages:

```bash
conda env create -f environment.yml
conda activate wepath
```

The environment installs this repository in editable mode. Alternatively, in
an existing compatible environment:

```bash
python -m pip install -e .
```

Python 3.9 or newer is required. GPU production runs require an OpenMM build
with CUDA support; the toy channel tutorial defaults to the OpenMM CPU
platform.

## Quick Start

Run commands from the repository root.

### Three-hole potential

Run the short, pre-seeded lower-channel test at 200 K:

```bash
python examples/toy/run_lower_channel_t200.py
```

Compare both channels at low and high temperature using matched settings:

```bash
python examples/toy/run_lower_channel_t200.py --channel lower --temperature 50  --iterations 100 --steps 50
python examples/toy/run_lower_channel_t200.py --channel upper --temperature 50  --iterations 100 --steps 50
python examples/toy/run_lower_channel_t200.py --channel lower --temperature 200 --iterations 100 --steps 50
python examples/toy/run_lower_channel_t200.py --channel upper --temperature 200 --iterations 100 --steps 50
```

Each run generates its channel reference path and writes flux, bin-population,
and walker logs below `outputs/toy_<channel>_t<temperature>/`. Use
`--single-source` to initialize only from the left basin instead of pre-seeding
all path-coordinate bins. See [the toy example documentation](examples/toy/README.md)
for the remaining controls.

### Abl--imatinib (1OPJ)

The 1OPJ scripts are production templates for four pathway/system pairs:

- wild type, P-loop pathway;
- wild type, alpha-C pathway;
- N368S, P-loop pathway;
- N368S, alpha-C pathway.

They require local coordinates, topology files, path frames, and PathCV point
arrays that are too large or unsuitable for this source repository. Place
those files under `example_inputs/1OPJ/` using the structure documented in
[the 1OPJ example README](examples/1opj/README.md). Review all paths, atom
selections, platform settings, and iteration counts before launching a
production run.

## Output and Rate Estimation

The main text outputs are:

- `flux*.txt`: one record for each target-reaching walker, including iteration
  number, walker weight, and progress coordinates;
- `bin*.txt`: weighted bin populations for every WE iteration;
- `walker*.txt`: walker progress coordinates and weights by iteration.

For a WE interval of duration `tau`, estimate a steady-state conditional
channel rate as

```text
k_channel = mean(target weight arriving per iteration) / tau
```

Use only a stationary analysis window after the pre-seeded transient has
decayed. Pre-seeding fills transition-region and target-adjacent bins at
iteration zero, so early target flux is an initialization artifact and must
not be interpreted as steady state. Assess running averages and late-window
drift, and use independent WE replicates for uncertainty estimates. Walkers
created by splitting are correlated and are not independent samples.

Rates from separate pathway-conditioned simulations are conditional channel
fluxes. Do not sum them into a global rate unless the channels share a common
source normalization or independently justified source weights. Channel
completeness, overlapping PathCV tubes, force-field dependence, and analysis
window selection remain system-dependent limitations.

## Contact

Suman Chakrabarty, Department of Chemical and Biological Sciences,
S. N. Bose National Centre for Basic Sciences, Kolkata, India  
Email: `sumanc@bose.res.in`

