# Running PathGennie on HPC clusters

This guide covers running PathGennie at scale on Slurm and PBS clusters, on both
CPU-only and GPU queues. For a ready-to-submit test battery, see
[`tests/hpc/`](https://github.com/TeamSuman/PathGennie/tree/main/tests/hpc)
(`README.md` + `DEBUGGING.md` there). For the architectural analysis, known
limitations, and roadmap, see [HPC Review & Roadmap](HPC_REVIEW.md).

## Mental model

Each adaptive cycle launches `max_trial` short MD **segments** (the swarm), scores
them by progress in CV space, softmax-selects one, extends it (`tau2`), and
commits a new anchor. The swarm is the unit of parallelism:

- **AMBER & GROMACS** run each segment as a subprocess (`pmemd.cuda` / `gmx
  mdrun`). Because `subprocess.run` releases the GIL, a thread pool
  (`ThreadDevicePool`) runs several segments concurrently and pins each to a
  device — this is genuine **single-node multi-GPU**.
- **OpenMM** runs in-process against one `Simulation`/Context and is therefore
  **single-GPU** (OpenMM Contexts are not thread-safe). Use it for one GPU or CPU.

## GPU placement (respecting your allocation)

Set the logical device list in `input.yaml`:

```yaml
pathgennie:
    devices: [0, 1, 2, 3]     # spread the swarm across 4 GPUs
    workers_per_device: 1     # concurrent segments per GPU
```

`devices` are **logical** indices interpreted *relative to the GPUs the scheduler
gave you*. PathGennie reads the job's `CUDA_VISIBLE_DEVICES` mask and maps
`devices[i]` onto it, so it never targets a GPU owned by another job:

- **Slurm**: request `--gres=gpu:4`; Slurm sets `CUDA_VISIBLE_DEVICES` (e.g.
  `"2,3,6,7"`). `devices: [0,1,2,3]` then maps to those four physical GPUs.
- **PBS/PBSPro**: request `ngpus=4`. If your site exposes the allocation only via
  `$PBS_GPUFILE`, derive the mask first (the provided `tests/hpc/pbs_gpu.pbs`
  shows how):
  ```bash
  export CUDA_VISIBLE_DEVICES="$(awk -F- '{print $NF}' "$PBS_GPUFILE" | paste -sd, -)"
  ```

`workers_per_device > 1` only helps when a single segment under-fills the GPU
(small systems). For large systems keep it at `1`.

## CPU queues (avoid oversubscription)

Running `workers_per_device` concurrent `gmx mdrun` / `pmemd` processes, each of
which defaults to grabbing **every** core, will oversubscribe the node and slow
everything down. Pin per-worker threads:

```yaml
pathgennie:
    workers_per_device: 4        # 4 concurrent segments
    cpu_threads_per_worker: 4    # 4 threads each -> 16 cores total
```

This exports `OMP_NUM_THREADS`/`MKL_NUM_THREADS` per worker and adds `-ntomp` for
GROMACS. Choose `workers_per_device × cpu_threads_per_worker ≈ cores`.

## Scratch: use node-local disk

Each segment writes small control/restart files. Over thousands of ultrashort
segments this is heavy **metadata** I/O that cripples shared Lustre/NFS. Point the
run's `workdir` at node-local SSD and copy results back at the end:

```bash
export TMPDIR=/local/scratch/$SLURM_JOB_ID     # or $PBS_JOBID
# set workdir under $TMPDIR in input.yaml, then after the run:
cp -r "$TMPDIR/pathgennie_run/output" "$SLURM_SUBMIT_DIR/results_$SLURM_JOB_ID"
```

## Reproducibility

Set `pathgennie.seed`. The selection draw and per-segment velocity seeds are
derived from it deterministically — including across the multi-GPU thread pool.
Note: with a **stochastic GPU integrator** (e.g. OpenMM Langevin on CUDA),
bit-for-bit trajectories are not guaranteed because the integrator's noise stream
is fixed at Context creation; the selection and control flow remain reproducible.

## Long campaigns / checkpointing

Stream frames and metrics to HDF5 as the run proceeds (bounded memory) with:

```yaml
pathgennie:
    checkpoint_path: run_checkpoint.h5
```

Writer-thread errors (full disk, bad path) are surfaced, not silently swallowed.
Full **resume-from-checkpoint** across a walltime boundary is not yet implemented
(see [HPC Review & Roadmap](HPC_REVIEW.md)); for now size `--time` to complete a
run, or split into stages.

## Example Slurm submission (GPU, AMBER)

```bash
#!/bin/bash
#SBATCH --job-name=pg-run
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

module load cuda amber
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pathgennie
cd "$SLURM_SUBMIT_DIR/my_case"       # contains input.yaml
pathgennie amber                     # or: python run_pg_amber.py
```

`input.yaml` sets `amber.executable: pmemd.cuda`, `pathgennie.devices: [0,1,2,3]`.

## Multi-node

There is no in-process multi-node executor, and none is needed for the common
case. Scale *within* a node: path discovery saturates a single GPU with
concurrent walkers, and the downstream Weighted Ensemble spreads across the
device pool. For more throughput, run independent pathways/replicates as
**separate Slurm/PBS array-job tasks** — the practical multi-node pattern, and
the one the path-resolved-kinetics workflow already uses to compute per-channel
rates separately. A work-queue model for a single tightly-coupled multi-node run
(like WESTPA's ZeroMQ manager) is on the [roadmap](roadmap.md).
