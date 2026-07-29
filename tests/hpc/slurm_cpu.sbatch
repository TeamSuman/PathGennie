#!/bin/bash
# =============================================================================
# PathGennie HPC test — SLURM, CPU-only queue
# -----------------------------------------------------------------------------
# Verifies the install + HPC-critical code paths and runs a short CPU MD smoke
# with per-worker thread pinning (the CPU oversubscription guard). Submit with:
#
#     sbatch tests/hpc/slurm_cpu.sbatch
#
# Results (JSON + logs) land in tests/hpc/results/<jobid>/. Read that directory
# alongside tests/hpc/DEBUGGING.md to interpret failures and debug.
# =============================================================================
#SBATCH --job-name=pg-cpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # EDIT: cores on the node you want to use
#SBATCH --time=00:30:00             # EDIT: walltime
#SBATCH --output=pg-cpu-%j.out
#SBATCH --error=pg-cpu-%j.err
###SBATCH --partition=cpu           # EDIT: uncomment + set your CPU partition
###SBATCH --account=YOUR_ACCOUNT    # EDIT: uncomment + set your allocation

set -euo pipefail

# --- EDIT: environment setup -------------------------------------------------
# module purge
# module load gromacs/2024        # or amber/24
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate pathgennie
# -----------------------------------------------------------------------------

# --- EDIT: what to run -------------------------------------------------------
PG_REPO="${SLURM_SUBMIT_DIR:-$PWD}"          # repo root (submit from repo root)
PG_BACKEND="${PG_BACKEND:-gromacs}"          # gromacs | amber | openmm
PG_EXAMPLE="${PG_EXAMPLE:-examples/alanine_dipeptide/gromacs}"
PG_EXE="${PG_EXE:-$(command -v gmx || true)}"  # CPU MD binary (gmx / pmemd / sander)
WORKERS="${WORKERS:-4}"                       # concurrent segments (swarm width on CPU)
THREADS_PER_WORKER="${THREADS_PER_WORKER:-$(( ${SLURM_CPUS_PER_TASK:-16} / WORKERS ))}"
# -----------------------------------------------------------------------------

cd "$PG_REPO"
RESULTS="tests/hpc/results/${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS"

echo "### PathGennie SLURM CPU test — job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "### backend=$PG_BACKEND exe=$PG_EXE workers=$WORKERS threads/worker=$THREADS_PER_WORKER"

# Keep BLAS/OpenMP from oversubscribing the login/analysis Python itself.
export OMP_NUM_THREADS="$THREADS_PER_WORKER"

# 1) Diagnostics + no-MD functional checks (also runs the unit suite).
python tests/hpc/hpc_selfcheck.py --example "$PG_EXAMPLE" --out "$RESULTS/selfcheck.json"

# 2) Short real MD smoke with per-worker thread pinning.
if [ -n "$PG_EXE" ]; then
  python tests/hpc/run_example.py \
    --backend "$PG_BACKEND" --example "$PG_EXAMPLE" --executable "$PG_EXE" \
    --workers-per-device "$WORKERS" --cpu-threads-per-worker "$THREADS_PER_WORKER" \
    --max-cycle 5 --out "$RESULTS/smoke_cpu.json" || echo "smoke run FAILED (see $RESULTS/smoke_cpu.json)"
else
  echo "PG_EXE not set / MD binary not found — skipping the MD smoke run." | tee "$RESULTS/smoke_cpu.skipped"
fi

echo "### Done. Results in $RESULTS/"
