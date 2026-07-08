# PathGennie HPC test suite

A self-contained battery for validating PathGennie on real HPC clusters —
**Slurm and PBS**, **CPU-only and GPU** queues. It is designed so you can run it
on the target machine and hand the `results/` directory (plus
[`DEBUGGING.md`](DEBUGGING.md)) to an automated agent (e.g. another Claude Code
session) that will interpret the JSON and, if needed, debug the code or scripts.

## What is here

| File | Purpose |
| ---- | ------- |
| `hpc_selfcheck.py` | Dependency-light diagnostics + no-MD functional checks (device masking, config validation, threaded determinism, HDF5 checkpoint, unit suite). Runs anywhere, no GPU/MD binary needed. Emits JSON. |
| `run_example.py`   | Drives a **real** MD backend for a short smoke run; on GPU queues it samples `nvidia-smi` to prove the swarm spread across the allocated GPUs. Emits JSON. |
| `slurm_cpu.sbatch` | Slurm, CPU-only queue. |
| `slurm_gpu.sbatch` | Slurm, GPU queue (single node, multi-GPU). |
| `pbs_cpu.pbs`      | PBS/Torque/PBSPro, CPU-only queue. |
| `pbs_gpu.pbs`      | PBS/Torque/PBSPro, GPU queue (single node, multi-GPU). |
| `DEBUGGING.md`     | Symptom → cause → fix map for a follow-up agent to act on the results. |
| `results/<jobid>/` | Per-job JSON + logs (created at run time). |

## Quick start

1. Get the code onto the cluster and create the environment:
   ```bash
   git clone <repo> && cd PathGennie
   conda env create -f environment.yml && conda activate pathgennie
   pip install -e ".[dev]"          # add [ml] for SPIB / on-the-fly CV
   ```
2. **Edit the `# EDIT:` lines** in the job script for your cluster: `module load`
   lines, `conda activate`, the partition/queue/account directives, and the
   `PG_BACKEND` / `PG_EXE` / `PG_EXAMPLE` variables (path to your `gmx` /
   `pmemd.cuda` / etc.).
3. Submit from the repository root:
   ```bash
   sbatch tests/hpc/slurm_gpu.sbatch      # or slurm_cpu / pbs_gpu / pbs_cpu
   ```
4. Inspect `tests/hpc/results/<jobid>/`:
   - `selfcheck.json` — environment + no-MD checks.
   - `smoke_cpu.json` / `gpu_spread.json` — the real MD run result.

You can also run the diagnostics on a login node without submitting:
```bash
python tests/hpc/hpc_selfcheck.py --out tests/hpc/results/login/selfcheck.json
```

## What each check proves (and why it matters at scale)

- **`device_masking`** — PathGennie interprets `devices: [0,1,…]` as *logical*
  indices mapped **onto the scheduler's `CUDA_VISIBLE_DEVICES` allocation**, so a
  job never touches a GPU it was not granted. This is the #1 correctness issue on
  shared GPU nodes; the check confirms the mapping on the node's real mask.
- **`config_validation`** — the case `input.yaml` survives validation with
  `tau1_steps` and all sections (`md`, `output`, downstream blocks) intact. A
  regression here silently ignores your MD parameters.
- **`executable_resolution`** — the case's configured MD `executable` resolves the
  way the backend does (`shutil.which`), so a bare `module load` name (`gmx`,
  `pmemd.cuda`) works. A missing binary on a login node is reported, not failed.
- **`threaded_determinism_and_leak`** — a seeded multi-worker run is reproducible
  and does not leak scratch/handles (bounded engine cache).
- **`concurrent_openmm`** — the OpenMM backend's single-GPU **concurrent-Context
  pool** builds and runs segments (CPU platform, no GPU needed). This is how
  OpenMM saturates one card: several walkers run at once instead of serially.
- **`hdf5_checkpoint`** — streaming checkpoint writes/reads and surfaces writer
  errors instead of silently dropping frames.
- **`unit_tests`** — the full unit suite passes on the node's Python/toolchain.
- **`gpu_spread` (GPU jobs, AMBER/GROMACS)** — with ≥2 allocated GPUs, the swarm
  actually distributes MD segments across them (verified via `nvidia-smi`).
  *OpenMM is single-GPU by design* — it instead **saturates one card** by running
  `workers_per_device` concurrent walkers (`auto` sizes from cores + free GPU
  memory). See `DEBUGGING.md`.

## Notes / current limitations exercised here

- **Scaling model.** Path generation runs on **one GPU**, saturated by concurrent
  walkers (`workers_per_device`). The downstream **Weighted Ensemble** is what
  spreads across multiple GPUs/cores — it reuses the same device pool, so listing
  several `devices` distributes its walkers. For multi-node, run independent
  pathways/replicates as **Slurm/PBS array jobs**.
- There is **no in-process multi-node executor** — for multi-node throughput use
  independent Slurm/PBS array jobs (above). A work-queue manager for a single
  tightly-coupled multi-node run is on the roadmap (`ROADMAP.md`).
- Scratch defaults to the case's `workdir`. On production runs point it (and
  `$TMPDIR`) at **node-local SSD**, not shared Lustre/NFS — thousands of
  ultrashort segments hammer a shared metadata server. See `DEBUGGING.md`.
