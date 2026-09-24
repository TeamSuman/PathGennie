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
log:          { ... }            # optional: streaming cycles.csv / trials.csv
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
| **`scheme`** | `parallel` \| `serial` | `parallel` | Trial scheme, see below. |
| **`serial`** | mapping | — | Options of the serial scheme, see below. |

### Trial scheme: `parallel` or `serial`

`parallel` (the default, unchanged) runs `max_trial` samplers per cycle and draws
one by softmax over their metrics (`sigma`).

`serial` is the probe-until-accepted scheme of the original PathGennie (0.0.0):
samplers of `tau1_steps` are started from the anchor one after another, each with
fresh velocities, until one beats the acceptance threshold; that sampler is
extended by `tau2_steps` and committed. There is no softmax, so `sigma` and
`max_trial` are not used. The number of probes a cycle needs is geometric in the
per-probe success probability, which makes it an observable of the landscape
(logged as `n_trials` in `cycles.csv`).

```yaml
pathgennie:
  scheme: serial
  serial:
    accept: record      # record: beat the best committed metric so far (0.0.0) | anchor: beat the current anchor
    max_try: 1000       # probes before the cycle is censored
    batch_size: auto    # probes run concurrently (auto = executor workers; 1 = strictly one at a time)
    on_max_try: best    # censored cycle: commit the best probe | stay: keep the anchor
    min_gain: 0.0       # accept only if metric > threshold + min_gain
```

Probes from one anchor are independent, so running `batch_size` of them at once
and accepting the *first success in index order* has the same statistics as a
one-at-a-time loop; probes after the accepted one are wasted compute but are
logged (`after_accept`). Per-probe seeds are derived from one master draw per
cycle, so for a seeded run `batch_size` changes the speed, not the result.

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
  periodic: [360.0, 360.0]  # optional: per-component CV period
  # any additional keys are passed through as kwargs
```

`module`/`function` name a Python file in the case directory; the remaining keys
become keyword arguments. The function maps `(n_atoms, 3)` Angstrom coordinates
to a CV vector.

### `periodic` — required for dihedral CVs

| Value | Meaning |
|---|---|
| omitted | every component non-periodic (previous behaviour) |
| `360.0` | that component is an angle in degrees |
| `6.283185` | that component is an angle in radians |
| `null` | that component is non-periodic (distance, PCA projection, …) |

`periodic` is consumed by the progress metric, not passed to your projection
function. Without it, two angles either side of the ±180° branch cut are scored as
~360° apart when they are adjacent: on a real alanine-dipeptide run a ψ change of
16.1° was scored as 343.9°, inflating the metric ~10× and rewarding the sampler for
crossing the cut rather than making progress. Mixed spaces are supported, e.g.
`periodic: [360.0, null]` for an angle plus a distance.

## `convergence` block

```yaml
convergence:
  module: phi_psi
  function: reached_phi_psi
  target: [60.0, 40.0]      # extra kwargs passed to the function
  tolerance: 15.0
```

The function returns `True` when the path is done.

!!! warning "Stop on the product, not on progress"

    It is tempting to reuse the `projection` CV as the convergence test — stop
    once the progress coordinate passes a threshold. That is safe only when the
    intended product is the **sole** accessible route to that threshold.

    A criterion written as a *difference of distances*,
    `d(A-B) - d(A-C) > threshold`, is satisfied by `d(A-B)` growing alone; it does
    not require the second bond to form. On a system with a competing channel,
    every run can report `Converged` while none produces the intended product —
    this was observed on a tertiary substitution test, where all 10 seeds
    converged and none formed the product bond (the leaving group departed while
    the nucleophile stayed 2.7–3.0 Å away, never bonding).

    Prefer a condition on the product state:

    ```python
    def reacted(coords, **kwargs):
        return bool(d(A, C) < 2.1 and d(A, B) > 3.0)   # both bonds must be right
    ```

    Drive on a progress CV; stop on a product-specific condition. See the
    [QM/MM workflow](qmmm-workflow.md#convergence-criteria-drive-on-progress-stop-on-product)
    for the full case.

## `md` block (optional)

Backend-specific MD control overrides, e.g. AMBER `mdin` controls or GROMACS
`mdp` controls:

```yaml
md:
  system: explicit          # AMBER: explicit | implicit | vacuum
  controls: { ntpr: 0 }     # merged over the backend defaults
  extra_text: ""            # appended verbatim to mdin (AMBER)
```

## `log` block (optional)

Streams two CSV files into the output directory while the run proceeds, for both
trial schemes (absent block = no log, as before):

```yaml
log:
  directory: .              # relative to the output directory
  cycles: cycles.csv        # one row per cycle
  trials: trials.csv        # one row per evaluated trial; false to disable
  flush_every: 1            # flush to disk every N cycles
  observables:              # optional: f(coords_A, **args) -> dict, evaluated on each committed frame
    module: projection
    function: log_observables
    cutoff: 8.0             # extra keys are passed as arguments (topology-derived ones added as for projection)
```

`cycles.csv`: `cycle, scheme, n_trials, n_failed, threshold, anchor_metric,
record, n_above_anchor, n_above_record, chosen_trial, chosen_metric,
chosen_prob, tau2_metric, committed_metric, accepted, censored, wall_s`, then one
column per observable key. `threshold` is empty for `parallel`, `chosen_prob`
for `serial`; `chosen_trial + 1` is the number of probes a serial cycle needed.
`trials.csv`: `cycle, trial, batch, metric, above_anchor, above_record, prob,
chosen, after_accept` — `max_trial` rows per parallel cycle, a variable number
per serial cycle. Observable keys that first appear after the header was written
go to `cycles_extra.csv` (`cycle, key, value`). On a checkpoint resume the files
are appended to.

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

> **Downstream stages (`weighted_ensemble` / `opes`).** Set
> `pathgennie.downstream: weighted_ensemble` plus a top-level block of that name and
> the backend `run()` loaders launch the stage automatically after path discovery,
> writing `free_energy.csv` (and `rate_constants.json` when recycling is enabled).
> The Python API remains available — see
> [weighted-ensemble.md](weighted-ensemble.md).
