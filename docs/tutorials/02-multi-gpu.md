# Tutorial 02 — Multi-GPU runs on the MD backends

The swarm is independent across trials, so it spreads across all your GPUs through
one `ParallelExecutor`. This tutorial shows the config knobs (for real AMBER/
GROMACS/OpenMM runs) and the in-process API (runnable here).

## In `input.yaml` (real backends)

```yaml
pathgennie:
  mode: escape
  tau1_steps: 5
  tau2_steps: 10
  max_trial: 16
  devices: [0, 1, 2, 3]     # 16 samplers spread across 4 GPUs
  workers_per_device: 1     # raise for small systems that under-fill a GPU
  seed: 42
  max_cycle: 2000
  save_freq: 2
  sigma: 0.25
```

Then:

```bash
pathgennie-amber   --case my_case/      # or
pathgennie-gromacs --case my_case/      # or
pathgennie-openmm  --case my_case/
```

Each segment exports `CUDA_VISIBLE_DEVICES` for its assigned GPU and writes into
an isolated `scratch/devN/` directory. `seed` makes the run reproducible.

## In Python (runnable with the toy engine)

Swap `SerialExecutor` for a `ThreadDevicePool`; nothing else changes:

```python
from pathgennie.core.parallel import ThreadDevicePool
from pathgennie.core.driver import PathGennieDriver

executor = ThreadDevicePool(devices=[0, 1, 2, 3], workers_per_device=1)
driver = PathGennieDriver(engine, progress, converged,
                          executor=executor, sigma=0.2, seed=0)
# driver.run(...) exactly as in tutorial 01
```

With one device a `ThreadDevicePool` reproduces `SerialExecutor` for a fixed seed.

## Benchmark the scaling

```bash
python benchmarks/scaling.py --devices 1 2 4 8 --max-trial 16 --max-cycle 20 --segment-ms 15
```

This uses a GIL-releasing sleep engine to model a GPU segment and reports
cycles/second and speedup vs the number of devices. See
[multi-gpu.md](../multi-gpu.md) for the full discussion.
