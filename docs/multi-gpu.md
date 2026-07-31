# Multi-GPU scalability

PathGennie's swarm is embarrassingly parallel: the `N` sampler segments in a
cycle are independent. v0.2.0 spreads them across every GPU on the machine
through one backend-agnostic abstraction, the `ParallelExecutor`.

## What was wrong before

The AMBER/GROMACS backends launched their swarm through a bare
`ThreadPoolExecutor` with **no device assignment**, so every worker inherited the
same environment and contended for GPU 0; the extra GPUs sat idle. Trials also
wrote files with colliding names into a single scratch directory.

## How it works now

`ThreadDevicePool` round-robins trial `i` onto `devices[i % len(devices)]` and
runs them in a thread pool. Each MD segment:

- exports `CUDA_VISIBLE_DEVICES=<device>` for its assigned GPU, and
- writes into an **isolated per-device scratch subdirectory** (`dev0/`, `dev1/`,
  …) with a unique file stem.

The subprocess backends (`pmemd.cuda`, `gmx mdrun`) spend their wall-time inside
`subprocess.run`, which releases the GIL, so the threads genuinely run
concurrently across GPUs. For the in-process OpenMM engine, construct one engine
per worker process (one CUDA context per GPU).

`SerialExecutor` is the single-device reference path; with one device a
`ThreadDevicePool` reproduces it exactly for a fixed seed.

## Enabling it

In `input.yaml`:

```yaml
pathgennie:
  max_trial: 16
  devices: [0, 1, 2, 3]     # spread 16 samplers across 4 GPUs
  workers_per_device: 1     # raise for small systems that under-fill a GPU
  seed: 42
```

`workers_per_device` allows more than one concurrent segment per GPU (useful when
a single small system cannot saturate a card).

!!! warning "The context pool does nothing without CUDA MPS"

    Measured on one A100 with solvated alanine dipeptide (~2.5k atoms), sweeping
    `workers_per_device` from 1 to 32:

    | workers/device | MPS off | MPS on | gain |
    | --- | --- | --- | --- |
    | 1 | 8.60 cyc/s | 8.95 | 1.04x |
    | 2 | 7.67 | 13.79 | 1.80x |
    | 4 | 9.01 | 20.12 | 2.23x |
    | **8** | 8.97 | **23.47** | **2.61x** |
    | 16 | 8.85 | 23.17 | 2.62x |
    | 32 | 8.74 | 23.16 | 2.65x |

    **Without MPS the pool is flat** — 7.7-9.0 cycles/s whatever you set. The CUDA
    driver serializes kernels from separate contexts in one process, so extra
    Contexts wait rather than overlap. With MPS the same sweep reaches **2.7x** the
    single-worker baseline and saturates at **8 workers**; beyond that there is
    nothing further to gain.

    Start MPS inside the job script before the run:

    ```bash
    export CUDA_MPS_PIPE_DIRECTORY=$PWD/mps_pipe CUDA_MPS_LOG_DIRECTORY=$PWD/mps_log
    mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
    nvidia-cuda-mps-control -d
    ```

    Setting `workers_per_device: 20` without MPS buys nothing; it is not a
    misconfiguration, it simply has no effect. The legacy `tau1_workers` key is
still accepted as an alias.

## In Python

```python
from pathgennie.core.parallel import ThreadDevicePool, SerialExecutor
from pathgennie.core.driver import PathGennieDriver

executor = ThreadDevicePool(devices=[0, 1, 2, 3], workers_per_device=2)
driver = PathGennieDriver(engine, progress, convergence_fn,
                          executor=executor, sigma=0.2, seed=7)
# ... driver.run(...)
```

## Reproducibility

A single master `seed` seeds the driver's RNG, which produces both the selection
draw and every per-segment seed. With a deterministic integrator, `G`-device and
serial runs match bit-for-bit. (A stochastic thermostat's noise stream is not
reliably re-seedable per segment in OpenMM, so only the selection/velocity draws
are guaranteed reproducible there — see `backends/openmm/engine.py`.)

## Benchmarking

`benchmarks/scaling.py` measures cycles/second versus the number of devices using
a GIL-releasing sleep engine that models a GPU segment of a known cost, so the
dispatch overhead of the executor itself is what's measured:

```bash
python benchmarks/scaling.py --devices 1 2 4 8 --max-trial 16 --max-cycle 20 --segment-ms 15
```

Representative output (overhead-dominated toy, not a physics result):

```
 devices   wall (s)   cycles/s   speedup
       1      3.894       3.85      1.00x
       2      2.080       7.21      1.87x
       4      1.170      12.82      3.33x
       8      0.730      20.54      5.33x
```

Real speedup is bounded by `min(N, G)` and by the serial runner (τ2) step.
