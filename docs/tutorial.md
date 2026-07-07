# Running PathGennie on HPC Clusters

This tutorial demonstrates how to use the newly implemented high-performance computing (HPC) features in PathGennie: **MPI Parallelism**, **HDF5 Checkpointing**, and **Pydantic Validation**.

## 1. Multi-Node Parallelism with MPI and Dask

When exploring complex free-energy landscapes, simulating multiple trial trajectories concurrently is essential. PathGennie now includes an `MPIExecutor` and a `DaskExecutor` to distribute swarm evaluations across multiple cluster nodes.

**Prerequisites:**
You need `mpi4py` or `dask[distributed]` installed. You can install them using the optional HPC flag:
```bash
pip install pathgennie[hpc]
```

**Running with MPI:**
By invoking PathGennie with `mpiexec` or `srun` (on Slurm), PathGennie automatically identifies the MPI pool.

```bash
# Example: Run on 4 nodes with 4 tasks per node
srun -N 4 --ntasks-per-node=4 pathgennie-openmm --case ./my_system
```

**Running with Dask:**
When you provide a Dask scheduler address via `DaskExecutor`, PathGennie connects to the cluster and distributes the workload using the Dask task graph.
```python
from pathgennie.core.parallel import DaskExecutor
executor = DaskExecutor(address="tcp://scheduler:8786")
```

## 2. Asynchronous HDF5 Trajectory Streaming

To prevent out-of-memory errors on long simulations, PathGennie now streams trajectory frames to HDF5 asynchronously using a background thread, removing I/O bottlenecks from the critical path.

Enable this by specifying a `checkpoint_path` in your `input.yaml`:

```yaml
pathgennie:
  mode: "escape"
  tau1: 500
  tau2: 500
  max_cycle: 10000
  save_freq: 10
  # New! Stream frames directly to an HDF5 dataset without keeping them all in RAM
  checkpoint_path: "output/trajectory_checkpoint.h5"
```

## 3. Robust Input Validation

All `input.yaml` configurations are now strictly validated against a Pydantic schema before execution. This means if you misspell a parameter (e.g., `max_cyles` instead of `max_cycle`), PathGennie will immediately raise a helpful error instead of silently ignoring it and using a default value.
