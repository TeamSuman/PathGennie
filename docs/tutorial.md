# Running PathGennie on HPC Clusters

This tutorial demonstrates how to scale PathGennie on high-performance computing (HPC) clusters: **single-GPU saturation**, **multi-GPU Weighted Ensemble**, **HDF5 checkpointing**, and **Pydantic validation**.

## 1. Scaling PathGennie on HPC

PathGennie is designed to run **within a single node**. Path generation itself is
lightweight — short segments on small-to-medium systems — and targets **one GPU**,
which it saturates by running many swarm walkers concurrently. The heavier,
embarrassingly parallel workload is the downstream pre-seeded Weighted Ensemble,
and *that* is what spreads across multiple GPUs or CPU cores.

**Saturate a single GPU (path generation).** Set `workers_per_device` to run
several trial segments concurrently on the same card (the AMBER/GROMACS backends
launch concurrent MD processes; the OpenMM backend runs concurrent Contexts).
Size it to fit the GPU's memory and the node's cores:

```yaml
pathgennie:
  devices: [0]            # a single GPU
  workers_per_device: 8   # or "auto" (OpenMM) to size from free GPU memory + cores
```

**Spread the Weighted Ensemble across GPUs/cores (downstream).** When a
`downstream: weighted_ensemble` stage is configured, the WE walker propagation
reuses the same device pool, so listing several devices distributes WE walkers
across them with no algorithm change:

```yaml
pathgennie:
  devices: [0, 1, 2, 3]   # WE walkers spread across 4 GPUs
  workers_per_device: 2
  downstream: weighted_ensemble
```

**Multi-node.** Run independent pathways and replicates as **Slurm/PBS array
jobs** — the same pattern the path-resolved-kinetics workflow uses to compute
per-channel rates separately. There is no in-process multi-node executor; a
work-queue model for a single tightly-coupled run is on the roadmap. See
[the HPC guide](hpc.md).

## 2. Asynchronous HDF5 Trajectory Streaming

To prevent out-of-memory errors on long simulations, PathGennie now streams trajectory frames to HDF5 asynchronously using a background thread, removing I/O bottlenecks from the critical path.

Enable this by specifying a `checkpoint_path` in your `input.yaml`:

```yaml
pathgennie:
  mode: "escape"
  tau1_steps: 500
  tau2_steps: 500
  max_cycle: 10000
  save_freq: 10
  # New! Stream frames directly to an HDF5 dataset without keeping them all in RAM
  checkpoint_path: "output/trajectory_checkpoint.h5"
```

## 3. Robust Input Validation

All `input.yaml` configurations are now strictly validated against a Pydantic schema before execution. This means if you misspell a parameter (e.g., `max_cyles` instead of `max_cycle`), PathGennie will immediately raise a helpful error instead of silently ignoring it and using a default value.
