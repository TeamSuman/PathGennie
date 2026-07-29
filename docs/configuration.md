# Configuration reference (`input.yaml`)

Each case is driven by an `input.yaml` with a backend block plus `pathgennie`,
`projection`, and `convergence` sections. This page documents every key,
including the ones added in v0.2.0. Keys not listed keep their pre-v0.2.0
behaviour, and omitting the new keys reproduces the original defaults.

## Top-level layout

```yaml
workdir: pathgennie_run          # output/scratch root (per backend default)

amber:        { ... }            # exactly one backend block:
# gromacs:    { ... }            #   amber | gromacs | openmm
# openmm:     { ... }

pathgennie:   { ... }            # adaptive-sampling settings
projection:   { ... }            # coords -> CV
convergence:  { ... }            # when to stop
md:           { ... }            # optional: MD control overrides
output:       { ... }            # optional: output file names
```

## `pathgennie` block

| Key | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `escape` \| `target` | `escape` | Directed (`target`) vs blind/escape search. |
| `tau1_steps` | int | — | Sampler segment length (steps). |
| `tau2_steps` | int | — | Runner segment length (steps). |
| `max_trial` | int | — | Swarm size `N` per cycle. |
| `max_cycle` | int | — | Maximum cycles. |
| `sigma` | float | — | Selection temperature (small = greedy, large = random). |
| `temperature` | float | 300 | MD temperature (K). |
| `save_freq` | int | 10 | Save a frame every this many cycles. |
| `target_projection` | list[float] | — | Required when `mode: target`. |
| `escape_metric` | `distance_from_start` \| `cv0` | `distance_from_start` | Escape scoring. Shared by all three backends since v1.4; previously OpenMM hardcoded `distance_from_start` while AMBER/GROMACS used `cv0`. |
| `reject_worse_tau2` | bool | false | Keep the sampler if the runner regresses. |
| `reject_worse_anchor` | bool | false | Keep the old anchor if the candidate regresses. |
| **`devices`** | list[int] | none | **v0.2.0** — GPU indices to spread the swarm across. |
| **`workers_per_device`** | int | 1 | **v0.2.0** — concurrent segments per GPU (replaces `tau1_workers`). |
| **`seed`** | int | none | **v0.2.0** — master RNG seed (selection + per-segment seeds). |
| **`profile`** | `discovery` \| `sampling` | none | **v0.2.0** — goal preset; fills defaults below explicit keys. |
| `tau1_workers` | int | — | Legacy alias for `workers_per_device`. |

### Multi-GPU example

```yaml
pathgennie:
  mode: escape
  tau1_steps: 5
  tau2_steps: 10
  max_trial: 10
  devices: [0, 1, 2, 3]     # 10 samplers spread across 4 GPUs
  workers_per_device: 1
  seed: 12345               # reproducible run
  max_cycle: 5000
  save_freq: 2
  sigma: 0.25
  temperature: 300
```

### Profiles

`profile` selects a `RunProfile` whose values are used **only where you did not
set a key explicitly** (explicit always wins). See
[strategy-profiles.md](strategy-profiles.md).

```yaml
pathgennie:
  profile: discovery        # fast candidate paths (greedy, ultrashort, geometric CV)
  devices: [0, 1]           # explicit keys still override / extend the profile
```

## `projection` block

```yaml
projection:
  module: phi_psi           # a .py file in the case directory
  function: phi_psi_cv      # called as function(coords, **other_keys)
  # any additional keys are passed through as kwargs
```

`module`/`function` name a Python file in the case directory; the remaining keys
become keyword arguments. The function maps `(n_atoms, 3)` Angstrom coordinates
to a CV vector.

## `convergence` block

```yaml
convergence:
  module: phi_psi
  function: reached_phi_psi
  target: [60.0, 40.0]      # extra kwargs passed to the function
  tolerance: 15.0
```

The function returns `True` when the path is done.

## `md` block (optional)

Backend-specific MD control overrides, e.g. AMBER `mdin` controls or GROMACS
`mdp` controls:

```yaml
md:
  system: explicit          # AMBER: explicit | implicit | vacuum
  controls: { ntpr: 0 }     # merged over the backend defaults
  extra_text: ""            # appended verbatim to mdin (AMBER)
```

## `output` block (optional)

```yaml
output:
  trajectory: reactive_path.dcd
  metrics: metrics.csv
  wrap_pbc: false           # AMBER: wrap frames into the primary cell
```

## Backend blocks (essentials)

```yaml
amber:
  topology: system.prmtop
  initial_restart: start.rst7
  executable: pmemd.cuda
  # mpi_launcher / mpi_ranks / mpi_launcher_args (optional)

gromacs:
  topology: topol.top
  initial_structure: start.gro
  mdp: md.mdp
  executable: gmx
  # maxwarn / grompp_args / mdrun_args (optional)

openmm:
  # system/topology files and platform settings (see examples/)
```

> **Downstream stages (`weighted_ensemble` / `opes`).** A `downstream` field
> exists on the run profile, but the backend `run()` loaders do **not** yet
> auto-launch a downstream stage from `input.yaml`. Run Weighted Ensemble through
> the Python API — see [weighted-ensemble.md](weighted-ensemble.md).
