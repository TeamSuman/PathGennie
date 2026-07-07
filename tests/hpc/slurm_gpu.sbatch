#!/bin/bash
# =============================================================================
# PathGennie HPC test — SLURM, GPU queue (single node, multi-GPU)
# -----------------------------------------------------------------------------
# Verifies the install, the scheduler-aware GPU masking, and that the swarm
# actually spreads across the ALLOCATED GPUs (AMBER/GROMACS). Submit with:
#
#     sbatch tests/hpc/slurm_gpu.sbatch
#
# Results land in tests/hpc/results/<jobid>/. NOTE: the multi-GPU spread check
# only applies to the AMBER and GROMACS backends — the OpenMM backend is
# single-GPU by design (see tests/hpc/DEBUGGING.md).
# =============================================================================
#SBATCH --job-name=pg-gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2                # EDIT: number of GPUs to allocate (>=2 for spread check)
#SBATCH --cpus-per-task=8           # EDIT: CPU cores (host side of GPU MD)
#SBATCH --time=00:30:00             # EDIT: walltime
#SBATCH --output=pg-gpu-%j.out
#SBATCH --error=pg-gpu-%j.err
###SBATCH --partition=gpu           # EDIT: uncomment + set your GPU partition
###SBATCH --account=YOUR_ACCOUNT    # EDIT: uncomment + set your allocation

set -euo pipefail

# --- EDIT: environment setup -------------------------------------------------
# module purge
# module load cuda/12.4 amber/24    # or gromacs/2024-cuda
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate pathgennie
# -----------------------------------------------------------------------------

# --- EDIT: what to run -------------------------------------------------------
PG_REPO="${SLURM_SUBMIT_DIR:-$PWD}"
PG_BACKEND="${PG_BACKEND:-amber}"            # amber | gromacs  (GPU multi-device path)
PG_EXAMPLE="${PG_EXAMPLE:-examples/alanine_dipeptide/amber}"
PG_EXE="${PG_EXE:-$(command -v pmemd.cuda || true)}"   # GPU MD binary
# Logical device list; scheduler-aware masking maps these onto the allocation.
# With --gres=gpu:2 you were given 2 GPUs, so devices 0,1 is correct.
DEVICES="${DEVICES:-0,1}"
WORKERS="${WORKERS:-2}"                       # concurrent segments per GPU
# -----------------------------------------------------------------------------

cd "$PG_REPO"
RESULTS="tests/hpc/results/${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS"

echo "### PathGennie SLURM GPU test — job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "### CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}  backend=$PG_BACKEND exe=$PG_EXE devices=$DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# 1) Diagnostics + functional checks (records how the scheduler mask is interpreted).
python tests/hpc/hpc_selfcheck.py --example "$PG_EXAMPLE" --out "$RESULTS/selfcheck.json"

# 2) Multi-GPU spread check: run a short swarm across the allocated GPUs and
#    confirm via nvidia-smi that >1 physical GPU actually did work.
if [ -n "$PG_EXE" ]; then
  python tests/hpc/run_example.py \
    --backend "$PG_BACKEND" --example "$PG_EXAMPLE" --executable "$PG_EXE" \
    --devices "$DEVICES" --workers-per-device "$WORKERS" --max-cycle 10 \
    --monitor-gpu --out "$RESULTS/gpu_spread.json" || echo "GPU run FAILED (see $RESULTS/gpu_spread.json)"
else
  echo "PG_EXE not set / GPU MD binary not found — skipping the GPU run." | tee "$RESULTS/gpu_spread.skipped"
fi

echo "### Done. Results in $RESULTS/"
