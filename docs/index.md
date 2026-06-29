# PathGennie

Welcome to PathGennie! PathGennie is a rapid generation rare-event sampling framework.

## Features
- **HPC Parallelism**: Easily distribute sampling with MPI and Dask on multi-node clusters.
- **Asynchronous Storage**: Stream huge trajectories via `h5py` so you never run out of memory.
- **Robust Configuration**: Clean, validated input configurations using `pydantic`.

PathGennie is a **direction-guided adaptive sampling** method for molecular
dynamics. From an anchor configuration it launches a *swarm* of short, unbiased
MD segments, scores each by progress in a collective-variable (CV) space,
softmax-selects one, extends it, updates the anchor, and repeats — generating
rare-event pathways (ligand unbinding, (un)folding, conformational change)
cheaply, with only a *selection* bias on natural dynamics (no bias potential).

This manual documents the v0.2.0 architecture and the features added on top of
the original three-backend runners. For a high-level project overview and the
example gallery, see the top-level [`README.md`](../README.md); for a
chronological list of changes see [`CHANGELOG.md`](../CHANGELOG.md).

## Contents

**Reference**
- [Architecture](architecture.md) — the core driver, the `Engine` protocol,
  progress variables, and the parallel executor.
- [Configuration](configuration.md) — the full `input.yaml` schema, including the
  new `devices` / `workers_per_device` / `seed` / `profile` / `downstream` keys.
- [Multi-GPU scalability](multi-gpu.md) — how the device pool spreads the swarm
  across GPUs, and how to benchmark it.
- [Strategy profiles](strategy-profiles.md) — switching behaviour by *goal*
  (`discovery` vs `sampling`) and the learned-CV trajectory-length guard.
- [Data-driven CVs (SPIB)](data-driven-cv.md) — learning a CV and metastable
  states on the fly.
- [Non-linear search (RRT)](non-linear-search.md) — RRT / RRT-Connect for
  pathways the greedy metric cannot follow.
- [Roadmap graph](roadmap-graph.md) — all-pairs pathways between metastable
  states (Dijkstra + Yen).
- [Agentic controller](agent.md) — automating swarm size and segment lengths.
- [Weighted Ensemble](weighted-ensemble.md) — path-informed free-energy /
  rate-constant stage.
- [OPES via PLUMED](opes.md) — free-energy surfaces along a CV.
- [Path sampling (TPS/TIS)](path-sampling.md) — kinetics via OpenPathSampling
  on PathGennie seed paths.
- [Project roadmap](roadmap.md) — what is implemented vs planned.
- [Future roadmap](../ROADMAP.md) — strategic plan toward a PLUMED/WESTPA-class
  framework (tiers, milestones, priority queue).

**Tutorials** (all runnable with no MD binary or GPU unless noted)
- [01 — Quickstart on the toy engine](tutorials/01-quickstart-toy.md)
- [02 — Multi-GPU runs on the MD backends](tutorials/02-multi-gpu.md)
- [03 — A learned CV with SPIB](tutorials/03-spib-cv.md)
- [04 — Free energies & rates with Weighted Ensemble](tutorials/04-weighted-ensemble.md)
- [05 — Goal-driven strategy profiles](tutorials/05-strategy-profiles.md)
- [06 — Non-linear search with RRT-Connect](tutorials/06-rrt-search.md)
- [07 — Metastable states, roadmap & the agent](tutorials/07-roadmap-agent.md)
- [08 — Free-energy surfaces with OPES](tutorials/08-opes.md)
- [09 — Kinetics with TPS/TIS (OpenPathSampling)](tutorials/09-path-sampling.md)

## Installation

```bash
pip install -e .            # core (OpenMM backend deps included)
pip install -e .[dev]       # + pytest
pip install -e .[ml]        # + PyTorch, required for the SPIB data-driven CV
pip install -e .[examples]  # + ParmEd for some example setups
```

Run the test suite to confirm a working install:

```bash
pytest -q                   # 47 tests; SPIB tests skip if torch is absent
```

## The 60-second mental model

```
                ┌─────────────────────────── PathGennieDriver ───────────────────────────┐
 anchor ──▶ clone ─▶ run N samplers (τ1) ─▶ project→metric ─▶ softmax_select ─▶ runner (τ2)
                          │  (ParallelExecutor: 1..G GPUs)        │ (selection.py)     │
                          ▼                                       ▼                    ▼
                    Engine.run_segment                     ProgressVariable      update anchor → repeat
   (OpenMM / AMBER / GROMACS / toy)             (geometric Escape/Target, or learned SPIB)
                                                                                       │
                                                              ┌────────────────────────┘
                                                              ▼
                                       PathEnsemble ──▶ SamplingStage (Weighted Ensemble) ──▶ FES / rates
```

Every box is a small, swappable interface:

| Concern | Interface | Built-ins |
|---|---|---|
| MD engine | `core.engine.Engine` | OpenMM, AMBER, GROMACS, `ToyLangevinEngine` |
| Progress / CV | `core.progress.ProgressVariable` | `EscapeMetric`, `TargetMetric`, `cv.spib.SPIBProgress` |
| Parallelism | `core.parallel.ParallelExecutor` | `SerialExecutor`, `ThreadDevicePool` |
| Goal preset | `core.strategy.RunProfile` | `discovery`, `sampling` |
| Downstream | `sampling.base.SamplingStage` | `WeightedEnsembleStage` |

Because these are protocols, you can mix any engine with any CV, any degree of
parallelism, and any downstream stage without touching the others.
