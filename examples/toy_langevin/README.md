# Toy Langevin (Wolfe–Quapp)

The fastest way to see PathGennie work. No MD engine, no topology, no GPU — overdamped
Langevin dynamics on the analytic Wolfe–Quapp surface using the built-in toy engine.

```bash
python run_toy.py
```

**Measured: 6.9 s, converges at cycle 4** (of `max_cycle: 100`).

## What it demonstrates

The same driver, selection rule, checkpointing and output path that every real backend
uses. If this runs, your install is sound.

- `mode: target` with `target_projection: [1.124, -1.485]` — the Wolfe–Quapp minimum the
  swarm is asked to reach
- `tau1_steps: 20` sampler segments, `max_trial: 8` per cycle, `tau2_steps: 40` runner
- `save_subframes: true` with `subframe_stride: 5` — intra-segment frame capture
- `checkpoint_freq: 10` — kill it mid-run and re-launch to see resume work
- `seed: 42` — the run is reproducible

## Output

Written to `pathgennie_toy_run/`:

| file | contents |
|---|---|
| `output/reactive_path.npy` | the reactive path, shape `(frames, 1, 3)` |
| `output/metrics.csv` | per-cycle CV and objective values |
| `checkpoint.h5` | resume state |

The directory is git-ignored and regenerable; delete it freely.
