# PathGennie

PathGennie is a lightweight Python toolkit for generating reactive molecular
dynamics paths with short adaptive MD bursts. It supports AMBER, GROMACS, and
OpenMM backends through small, case-local `input.yaml` files plus user-defined
collective variables and convergence checks.

The workflow is designed for fast path discovery: launch multiple short
`tau1` trials from the current anchor state, bias the next anchor toward
progress in a projection space, continue with `tau2`, and repeat until the
case-specific convergence criterion is met.

## Examples at a Glance

The repository includes compact examples for conformational change, peptide
folding/unfolding, host-guest unbinding, and QM/MM steering.

| Example                              | Backend(s)             | What it demonstrates                                           |
| ------------------------------------ | ---------------------- | -------------------------------------------------------------- |
| `examples/alanine_dipeptide`         | AMBER, GROMACS, OpenMM | Targeted phi/psi transition between Ramachandran basins        |
| `examples/CLN025`                    | AMBER, GROMACS         | Chignolin path generation with end-to-end distance projections |
| `examples/OAMe-G2`                   | AMBER, GROMACS         | Host-guest unbinding with COM-COM distance projections         |
| `examples/qmmm_alanine_conformation` | AMBER                  | QM/MM alanine-dipeptide conformational steering                |

### Movie Examples

These animations are tracked in the repo and can be used as visual references
for the kinds of paths PathGennie produces.

<table>
  <tr>
    <td width="33%" align="center">
      <!-- <strong>Host-guest unbinding</strong><br> -->
      <img src="assets/unbind.gif" alt="Host-guest unbinding animation" width="100%">
    </td>
    <td width="33%" align="center">
      <!-- <strong>Reactive path example</strong><br> -->
      <img src="assets/reaction.webp" alt="Reactive path animation" width="100%">
    </td>
    <td width="33%" align="center">
      <!-- <strong>Molecular transition example</strong><br> -->
      <img src="assets/movie.webp" alt="PathGennie molecular transition" width="100%">
    </td>
  </tr>
</table>

## Repository Layout

```text
pathgennie/
  core/         Backend-independent driver, selection, progress, device pool,
                Engine protocol, and a toy Wolfe-Quapp Langevin engine
  cv/           Data-driven CVs: NumPy featurization + on-the-fly SPIB
                (learned CV + emergent metastable states; needs PyTorch)
  backends/
    amber/      Device-aware AMBER engine + runner
    gromacs/    Device-aware GROMACS engine + runner
    openmm/     OpenMM in-process engine + runner
tests/          pytest suite (selection, CV, I/O, OpenMM, device dispatch)
benchmarks/
  scaling.py    Device-pool scaling benchmark
examples/
  alanine_dipeptide/
  CLN025/
  OAMe-G2/
  qmmm_alanine_conformation/
assets/
  unbind.gif
  unbind.webp
  reaction.webp
  movie.webp
environment.yml
```

## Installation

Create the supplied Conda environment:

```bash
conda env create -f environment.yml
conda activate pathgennie
```

Install the package from this checkout:

```bash
pip install .
```

For development, install it in editable mode:

```bash
pip install -e ".[examples]"
```

Backend executables are configured per example. Update paths such as
`amber.executable` or `gromacs.executable` in the relevant `input.yaml` before
running on your machine.

## Running Examples

### AMBER: Alanine Dipeptide

```bash
cd examples/alanine_dipeptide/amber
python run_pg_amber.py
```

Outputs are written to `pathgennie_amber_run/output/` by default:

- `reactive_path.pdb`
- `metrics.csv`

### GROMACS: Alanine Dipeptide

```bash
cd examples/alanine_dipeptide/gromacs
python run_pg_gromacs.py
```

Outputs are written to `pathgennie_gmx_run/output/` by default:

- `reactive_path.xtc`
- `metrics.csv`

### OpenMM: Alanine Dipeptide

```bash
cd examples/alanine_dipeptide/openmm
python run_pg_openmm.py
```

Outputs are written to `pathgennie_openmm_run/output/` by default:

- `reactive_path.dcd`
- `metrics.csv`

### AMBER QM/MM: Alanine-Dipeptide Conformation

```bash
cd examples/qmmm_alanine_conformation/amber
python run_pg_amber.py --config input_c7eq.yaml
python run_pg_amber.py --config input_c7ax.yaml
```

See
[`examples/qmmm_alanine_conformation/amber/README.md`](examples/qmmm_alanine_conformation/amber/README.md)
for the QM/MM region, targets, and regeneration notes.

## Configuration Model

Each case is driven by an `input.yaml` with four main parts:

- Backend block: `amber`, `gromacs`, or `openmm` input files and executable
  settings.
- `pathgennie`: adaptive sampling settings such as `mode`, `tau1_steps`,
  `tau2_steps`, `max_trial`, `max_cycle`, `sigma`, and `temperature`.
  Multi-GPU / reproducibility keys: `devices` (list of GPU indices to spread the
  swarm across), `workers_per_device` (concurrent segments per GPU; replaces the
  legacy `tau1_workers`), and `seed` (master RNG seed for the selection and
  velocity draws).
- `projection`: Python module and function that map coordinates to a
  collective-variable vector.
- `convergence`: Python module and function that decide when the generated path
  has reached the desired state.

Example target-mode configuration:

```yaml
pathgennie:
    mode: target
    target_projection: [60.0, 40.0]
    tau1_steps: 2
    tau2_steps: 4
    max_trial: 10
    max_cycle: 1000
    save_freq: 10
    sigma: 0.1
    temperature: 300

projection:
    module: phi_psi
    function: phi_psi_cv

convergence:
    module: phi_psi
    function: reached_phi_psi
    target: [60.0, 40.0]
    tolerance: 15.0
```

Example escape-mode configuration:

```yaml
pathgennie:
    mode: escape
    tau1_steps: 5
    tau2_steps: 10
    max_trial: 10
    devices: [0, 1, 2, 3]   # spread the 10 samplers across 4 GPUs
    workers_per_device: 1
    seed: 12345             # reproducible selection / velocity draws
    max_cycle: 5000
    save_freq: 2
    sigma: 0.25
    temperature: 300

projection:
    module: projection
    function: com_com_distance_cv
    group_a_resname: OCB
    group_b_resname: MOL

convergence:
    module: projection
    function: dissociated
    group_a_resname: OCB
    group_b_resname: MOL
    threshold: 10.0
```

## Data-Driven CVs (SPIB)

Instead of a hand-crafted progress variable, PathGennie can learn one on the fly
with **SPIB** (State Predictive Information Bottleneck, `pathgennie.cv.spib`),
which jointly learns a low-dimensional CV and an emergent set of metastable
states from the frames the run visits. `SPIBProgress` bootstraps from a coarse
geometric CV, buffers the path, retrains periodically (the iterative
path-learning cycle), and then steers using the learned latent. It is an adaptive
`ProgressVariable`, so it plugs into the same driver as the built-in metrics.
Requires the `ml` extra (`pip install -e .[ml]`, i.e. PyTorch).

## Writing a New Case

1. Copy the closest example directory for your backend.
2. Replace topology, coordinate, restart, and MD-control files.
3. Update the backend executable path in `input.yaml`.
4. Implement a projection function that accepts coordinates and returns a NumPy
   array-like CV.
5. Implement a convergence function that returns `True` when the path is done.
6. Tune `tau1_steps`, `tau2_steps`, `max_trial`, `sigma`, and `max_cycle` for the
   system size and available compute.

## Outputs

PathGennie writes two primary artifacts:

- Reactive path trajectory: backend-dependent format such as `.pdb`, `.dcd`,
  `.xtc`, or `.nc`.
- Metrics CSV: one metric sample per adaptive cycle.

For AMBER and GROMACS runs, scratch files are created under the configured
`workdir` and removed after a successful run. Generated `tau1` and `tau2` MD
control files are kept in the work directory for inspection.

## License

This project is distributed under the terms of the
[`LICENSE`](LICENSE) file.
