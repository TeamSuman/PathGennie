# Changelog

All notable changes to PathGennie are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Hardening pass that unifies the feature branches onto one line and makes the
HPC paths correct and runnable. See `docs/HPC_REVIEW.md` for the full review.

### Fixed
- **Config validation (release-blocking).** `pathgennie/utils/config.py` declared
  `tau1`/`tau2` while the backends read `tau1_steps`/`tau2_steps`, and used
  `extra="ignore"` — so every run crashed with `KeyError: 'tau1_steps'` and the
  `md`/`workdir`/`output` sections (and `devices`, `downstream`, `profile`, …)
  were silently dropped. Rewrote the schema with real field names, bounds/enum
  validation, and `extra="allow"`; added `tests/test_config.py`.
- **On-the-fly SPIB** crashed on cycle 0 (`SPIBProgress.project` missing the
  `cycle` argument the driver passes). Fixed.
- **Scheduler-aware GPU placement.** Engines overwrote `CUDA_VISIBLE_DEVICES`
  with absolute indices, colliding with other jobs on shared Slurm/PBS nodes.
  Added `resolve_cuda_visible_device()` mapping logical indices onto the
  allocation; wired into the AMBER/GROMACS engines and the `we` GPU worker.
- **Reproducibility under the device pool**: per-trial seeds are pre-drawn on the
  main thread (numpy's Generator is not thread-safe), so a seeded multi-GPU run
  matches serial.
- **Scratch/handle leak**: the per-trial cloned anchor is now released (driver and
  RRT), and handle release compares by value not identity.
- **HDF5 checkpoint** writer-thread errors are surfaced instead of silently
  dropping frames.
- **`import pathrefinement`** works on a base install (lazy torch/openmm imports);
  fixed a `NameError` in `pathiter`.
- Removed committed git merge-conflict markers from `README.md`; corrected stale
  example references.

### Added
- **CPU oversubscription guard** `pathgennie.cpu_threads_per_worker` (pins
  OMP/MKL threads and GROMACS `-ntomp` per worker).
- **HPC test suite** (`tests/hpc/`): PBS + Slurm submission scripts for CPU and
  GPU queues, a dependency-light self-check, a real-backend/multi-GPU runner, and
  a debugging guide for interpreting results.
- **Docs**: `docs/hpc.md` (Slurm/PBS scaling guide) and `docs/HPC_REVIEW.md`
  (review, WESTPA comparison, SOTA positioning, roadmap); full mkdocs nav.

## [1.2.0] — 2026-06-29

This release consolidates the high-performance computing (HPC) parallel scaling features, asynchronous streaming checkpointing, and robust input validation with the newly merged **PathCV** (Path Collective Variables), **Path Refinement**, and standalone **Weighted Ensemble** (WE) frameworks.

### Added

**High-Performance Computing (HPC) & Scalability**
- **MPI & Dask parallel executors (`pathgennie/core/parallel.py`):** Added `MPIExecutor` and `DaskExecutor` to distribute swarm evaluations across cluster nodes. See [docs/tutorial.md](docs/tutorial.md#1-multi-node-parallelism-with-mpi-and-dask).
- **Asynchronous trajectory streaming (`pathgennie/core/storage.py`):** Added `HDF5Storage` class utilizing a background thread to stream frames/metrics to HDF5 without keeping them in memory. See [docs/tutorial.md](docs/tutorial.md#2-asynchronous-hdf5-trajectory-streaming).
- **Robust input validation (`pathgennie/utils/config.py`):** Replaced manual parsing with a comprehensive Pydantic schema validation model (`PathGennieConfig` / `AppConfig`). See [docs/tutorial.md](docs/tutorial.md#3-robust-input-validation).

**Path Refinement & Standalone Weighted Ensemble**
- **Path Refinement library (`pathrefinement/`):** Added an ensemble-based principal curve pathway refiner (`pathrefinement/refiner.py`), mathematical verification on the Müller-Brown potential, and tutorials. See [pathrefinement/README.md](pathrefinement/README.md).
- **Standalone Weighted Ensemble framework (`we/`):** Added the standalone Huber-Kim Weighted Ensemble resampler (`we/src/wepath/`) with examples for toy systems and the 1OPJ GPCR system. See [we/README.md](we/README.md).
- **Unified Command-line Interface (`pathgennie/cli/main.py`):** Exposed a unified command-line entrypoint `pathgennie` to drive runs and setup setups. See [docs/index.md](docs/index.md).
- **Conformation utilities (`pathgennie/utils/`):** Added `ligconfgen.py` and `ligcvgen.py` for ligand conformation generation and collective variable analysis.

**Enhanced OpenMM Driver Support**
- **Dynamic PCA dimension changes:** Added support to `PathGennieDriver` and progress metrics (`EscapeMetric` and `TargetMetric` in `pathgennie/core/progress.py`) to handle dimension reductions on the fly using NaN masking and implicit shape alignment.
- **Equilibration steps:** Added support to run equilibration steps prior to path generation (via `equilibration_steps` key under `md:` in `input.yaml`). See [README.md](README.md#4-md-parameters-md).
- **PLUMED integration:** Added support for PLUMED-based force fields via `plumed_file` parameter in OpenMM.
- **Custom system builders:** Added support to dynamically load custom OpenMM system maker functions via `system_file` config.
- **GROMACS files support:** Added GROMACS `.top`/`.itp`/`.gro` file parsing and coordinate loading support inside OpenMM runner.

### Changed
- Replaced `yaml.safe_load` config loading in all backends (AMBER, GROMACS, OpenMM) with Pydantic `load_config` validator to fail-fast on malformed parameters.
- Re-architected `PathGennieDriver` and `CallableProjection` to forward `cycle` index parameters to custom CV projection functions (enabling time/cycle-dependent CV spaces).

## [0.2.0] — 2026-06-15

This release re-architects PathGennie around a single, backend-independent core
so the adaptive-sampling cycle is implemented **once** instead of three times,
makes the swarm genuinely **multi-GPU**, and adds three new capability layers on
top: a data-driven CV (SPIB), goal-driven run profiles, and a path-informed
Weighted Ensemble stage. Existing `input.yaml` cases continue to run unchanged.

### Added

**Backend-independent core (`pathgennie/core/`).**
- `engine.py` — `Engine` protocol (`clone_anchor`, `run_segment`, `get_coords`,
  `release`) that every backend implements. A *handle* is an opaque token (a
  restart-file path or an in-process state-cache id).
- `selection.py` — single source of truth for the Boltzmann/softmax selection
  (`selection_probs`, `softmax_select`); degenerate (all-equal) batches fall back
  to a uniform draw, and the largest argument is shifted to 0 so `exp` cannot
  overflow.
- `progress.py` — `ProgressVariable` protocol plus the built-in `EscapeMetric`
  (maximise distance from start, or legacy `cv0`) and `TargetMetric` (minimise
  distance to a target CV).
- `driver.py` — `PathGennieDriver`, the one adaptive loop (swarm → select →
  runner → anchor update → convergence), parameterized by an engine, a progress
  variable, a convergence function, and a parallel executor.
- `parallel.py` — `ParallelExecutor` abstraction with `SerialExecutor` and
  `ThreadDevicePool` (round-robins trials across GPUs); `resolve_devices` helper.
- `toy.py` — pure-NumPy `ToyLangevinEngine` on the Wolfe–Quapp surface, so the
  *entire* driver runs in CI in seconds without an MD binary or GPU.
- `strategy.py` — goal-driven `RunProfile` presets (`discovery`, `sampling`),
  `resolve_profile`, and `check_learned_cv_segment_length`.

**True multi-GPU scalability.**
- Device-aware `CoreAmberEngine` (`backends/amber/engine.py`) and
  `CoreGromacsEngine` (`backends/gromacs/pg_gmx.py`): each segment exports
  `CUDA_VISIBLE_DEVICES` for its assigned GPU and uses an isolated per-device
  scratch subdirectory with unique file stems.
- OpenMM backend rewired onto the shared driver via an `OpenMMEngine`
  (`backends/openmm/engine.py`).
- New `pathgennie` config keys: `devices` (GPU index list),
  `workers_per_device`, and `seed`. Legacy `tau1_workers` is still honoured.
- `benchmarks/scaling.py` — device-pool scaling benchmark.

**Data-driven collective variables (`pathgennie/cv/`).**
- `features.py` (NumPy) — `pairwise_distances`, `contact_features`,
  `dihedral_features`, and a `Featurizer` with online standardization.
- `spib.py` (PyTorch, optional) — State Predictive Information Bottleneck: a
  learned CV with an *emergent* number of metastable states, exposed as the
  adaptive `SPIBProgress` progress variable that bootstraps from a coarse CV,
  buffers the path, retrains periodically, then steers with the learned latent.
- `PathGennieDriver` gained an optional per-cycle `observe()` hook so adaptive
  progress variables can retrain on the fly (no-op for static CVs).

**Non-linear path search (`pathgennie/search/`).**
- `rrt.py` — Rapidly-exploring Random Trees (`RRT`) and bidirectional
  `rrt_connect` over the swarm, for pathways the greedy metric cannot follow
  (backtracking, direction changes, orthogonal CVs). Reuses the `Engine`,
  `ParallelExecutor`, and `softmax_select`.
- `roadmap.py` — the conformational `Roadmap` graph (edges weighted by
  `-log(transition fraction)`) with `dijkstra_path` (minimum-free-energy path)
  and Yen's `k_shortest_paths` (competing parallel pathways) for all-pairs
  pathway extraction between metastable states.
- `core/policy.py` — the `ExplorerPolicy` protocol (`GreedyPolicy`) shared by the
  driver and the RRT searchers.

**Agentic controller (`pathgennie/agent/`).**
- `RuleBasedController` — adapts swarm size `N` and segment lengths
  `tau1`/`tau2` from the recent progress rate (escalate on stall, relax on
  progress), count-based frontier selection (anti-trapping), a CV-refresh
  schedule, and a plateau-based stop recommendation. Implements a `Controller`
  surface a future RL/LLM meta-controller can replace.

**Enhanced-sampling stages (`pathgennie/sampling/`).**
- `base.py` — the downstream contract: `PathEnsemble`, `SamplingResult`,
  `SamplingStage`, and `build_path_ensemble`.
- `weighted_ensemble.py` — path-informed `WeightedEnsembleStage` (Huber–Kim
  split/merge `resample`, `GridBinner`, `Walker`); reuses the `Engine` and
  device pool, with optional recycling for steady-state rate constants.
- `opes.py` — `OPESStage` with a **PLUMED interface** (`build_plumed_opes_input`
  generates an `OPES_METAD` input and drives a PLUMED-capable engine) plus a
  dependency-free, CI-verified OPES core (`OPESBias`, `OPESSimulation`) validated
  on the toy Wolfe–Quapp surface.
- `path_sampling.py` — OpenPathSampling (OPS) bridge: `PathSamplingStage` runs
  **TPS/TIS** on a PathGennie seed path (an alternative to WE for kinetics), plus
  dependency-free, CI-verified seed preparation (`CVRangeState`, `label_frames`,
  `extract_transition_path`, `tis_interfaces`, `prepare_ops_seed`). Needs the
  `pathsampling` extra (`openpathsampling`) and an OPS engine to run.
- `make_stage(name, **cfg)` factory keyed on the `downstream` name
  (`weighted_ensemble`, `opes`, `tps`, `tis`).
- `runner.py` — `run_downstream` glue that builds a `PathEnsemble` and runs the
  configured stage; wired into all three backends behind `pathgennie.downstream`.
- `driver.run(..., collect_seeds=True)` returns restartable seed handles aligned
  with the saved frames, so a run can hand a stage informed seeds.
- `benchmarks/we_fes.py` — validates the WE free-energy profile against the
  analytic Wolfe–Quapp marginal.

**Tests & docs.**
- New `tests/` suite (47 tests): selection, CVs/featurization, I/O round-trips,
  per-backend device dispatch, SPIB recovery, strategy profiles, full-driver
  smoke test on the toy engine, and the Weighted Ensemble stage.
- `docs/` manual and tutorials (this release).
- `pyproject.toml` optional extras: `dev` (pytest) and `ml` (torch).

### Changed
- The OpenMM/AMBER/GROMACS backends are now thin adapters (an `Engine`
  implementation + a `run()` config loader) that delegate to the core driver;
  the duplicated cycle/selection code was removed.
- Per-segment state handling restores periodic box vectors (OpenMM) and guards
  against non-finite coordinates (all backends).

### Fixed
- **Multi-GPU:** swarm trials no longer all contend for GPU 0 (the old
  `ThreadPoolExecutor` performed no device assignment).
- **Scratch races:** concurrent trials no longer write colliding filenames into a
  shared directory.
- **Reproducibility:** a single master `seed` drives both selection and
  per-segment seeds (selection previously used NumPy's global RNG).
- **Converged frame:** the final converged frame is always saved (the OpenMM loop
  could skip it when `cycle % save_freq != 0`).
- **Subprocess errors:** AMBER/GROMACS failures surface stdout/stderr with the
  failing command instead of an opaque traceback.

### Notes / environment caveats
- All new layers (RRT, roadmap, controller, WE, OPES, downstream wiring) are
  verified on the toy/synthetic systems in CI. The **real-MD** code paths
  (multi-GPU AMBER/GROMACS runs, and downstream stages launched from a real
  backend) cannot be executed in a CPU-only sandbox without `pmemd`/`gmx`, so
  they are exercised through the shared, tested helpers rather than a live run.
- **OPES on real MD requires PLUMED.** `OPESStage(mode="plumed")` generates the
  `OPES_METAD` input and calls `engine.run_plumed(...)`; the MD engines do not
  yet implement `run_plumed`, so a PLUMED-patched engine must be supplied. The
  OPES *algorithm* is verified via `OPESStage(mode="toy")` on an analytic
  potential. See `docs/opes.md`.

## [0.1.0]
- Initial PathGennie release: direction-guided adaptive sampling with separate
  OpenMM, AMBER, and GROMACS runners driven by per-case `input.yaml` files.

[1.2.0]: https://github.com/TeamSuman/PathGennie/compare/v0.2.0...v1.2.0
[0.2.0]: https://github.com/TeamSuman/PathGennie/releases/tag/v0.2.0
[0.1.0]: https://github.com/TeamSuman/PathGennie/releases/tag/v0.1.0
