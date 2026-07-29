# Changelog

All notable changes to PathGennie are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Every re-run of the AMBER and GROMACS backends crashed (release-blocking).**
  `pg_amber.run` and `pg_gmx.run` import `shutil` at module scope and then again
  *inside* the function, which made `shutil` a function-local name for the whole
  body — so the earlier `shutil.rmtree(scratch_dir)` raised
  `UnboundLocalError: cannot access local variable 'shutil'` whenever a scratch
  directory already existed. The first run into a clean directory worked, so this
  only bit on a second run, a resume, or a restart after a crash — precisely the
  case checkpoint/restart exists to serve. These entrypoints had no test coverage;
  `tests/test_backend_rerun.py` now covers the re-run path for both backends.
- **Mass-weighted collective variables silently became unweighted centroids.**
  A `.gro`/`.pdb` metadata file carries no masses, so `read_gro_topology_info` /
  `read_pdb_topology_info` filled in `np.ones(...)`, and `enrich_args` injected
  those into the user's CV whenever `group_a_resname`/`group_b_resname` were set.
  A centre-of-mass CV therefore degraded to an arithmetic centroid with no error
  and no warning (measured: 3.5193 Å vs a true 3.7278 Å on the OAMe-G2 host–guest
  COM–COM distance — a 5.6 % systematic bias). AMBER was unaffected because
  `parse_prmtop` reads the real `%FLAG MASS`. New
  `read_masses_from_topology()` recovers real masses from the topology (ParmEd,
  then MDAnalysis); the GROMACS and OpenMM backends use it, and `enrich_args` now
  **raises** rather than passing placeholder masses to a mass-weighted CV.
- **Weighted Ensemble destroyed probability weight.**
  `redistribute_excess_weight` stripped each walker's excess above `cap` *before*
  checking that a recipient existed, so when every walker was above the cap the
  excess was discarded (measured: 1.0 → 0.3, a 70 % loss). WE is unbiased only
  because resampling conserves total weight — a rate constant is a sum of walker
  weights — and there is no per-iteration renormalisation to mask the loss. The
  routine now classifies donors/recipients before mutating anything, and declines
  to act (with a message) when the cap is mathematically unsatisfiable.

### Changed
- **`escape_metric` is honoured by all three backends and shares one default.**
  OpenMM previously hardcoded `distance_from_start` while AMBER and GROMACS
  defaulted to the legacy `cv0`, so an identical `input.yaml` optimised a
  different quantity depending on the engine. The shared default is now
  `DEFAULT_ESCAPE_METRIC = "distance_from_start"`, the objective the method is
  published with. **This changes AMBER/GROMACS behaviour** — set
  `escape_metric: cv0` explicitly to restore the previous default. The dead
  `escape_direction` argument was removed from `PathGennieMD`.
- **Weighted Ensemble resampling is reproducible.** `Resampler` and the survivor
  scheme drew from NumPy's *global* RNG, so a WE run could not be reproduced even
  from identical inputs. Both now use a dedicated `Generator` seeded from a new
  `seed` config key; omitting it preserves the previous non-deterministic
  behaviour.
- **SPIB on-the-fly CV** caches features incrementally (was re-featurizing the
  whole buffer each refresh, ~O(N²)) with an optional bounded sliding window.
- **`we/examples/1opj` scripts** no longer enable walker cleaning by default. The
  published results were produced *without* it (it was added later to debug a poor
  initial reference path), so the committed scripts ran a different algorithm from
  the one behind the paper.

### Added
- **Single-GPU saturation (OpenMM).** `OpenMMEngine` backs a pool of concurrent
  Contexts on one card; `workers_per_device` (an int, or `auto` sized from cores
  and free GPU memory) runs that many swarm walkers at once instead of serially.
- **Downstream Weighted Ensemble parallelism.** The backend device pool is
  forwarded to the WE stage, so its walker propagation spreads across GPUs/cores.
- **Intra-segment frame capture** (`save_subframes`, `subframe_stride`): the
  committed τ1+τ2 segment is replayed to harvest intermediate frames, giving a
  continuous trajectory instead of one frame per `save_freq` cycles. *Caveat:
  the replay only reproduces the committed segment for a deterministic
  integrator — see Known issues.*
- **Checkpoint restart and output-overwrite protection** (`checkpoint_freq`,
  `checkpoint_path`, `overwrite`): a run resumes from the last checkpoint, and
  existing outputs are no longer silently clobbered.
- **Correct trajectory timestamps**: written frames carry real simulation times
  derived from the integrator timestep.

### Known issues
- **A seeded run is not reproducible on a stochastic integrator.**
  `OpenMMEngine.run_segment` sets the per-segment seed on an already-created
  `Context` and nothing calls `reinitialize()`, so the Langevin noise stream is
  unaffected. Two runs with an identical `seed` diverge from the first cycle.
  Reproducibility currently holds only for deterministic integrators (e.g.
  Verlet). Consequently `save_subframes` writes a *different realisation* than
  the segment that was actually selected.
- **Progress metrics ignore CV periodicity.** `EscapeMetric` / `TargetMetric`
  score with a plain Euclidean norm, which is wrong for periodic CVs such as
  dihedrals: a φ/ψ pair straddling the ±180° branch cut was scored 345.2 when the
  true angular distance was 34.3 (a 10× inflation). Distance and PCA CVs are
  unaffected.

### Removed
- **`MPIExecutor` / `DaskExecutor` and the `[hpc]` extra.** They were
  non-functional (per-cycle work closing over live engine state cannot be pickled
  across nodes) and wired to no backend. The supported scaling patterns cover
  every practical case: single-GPU saturation, a multi-GPU/CPU downstream WE, and
  independent pathways/replicates as Slurm/PBS array jobs. A work-queue model for
  a single tightly-coupled multi-node run remains on the roadmap.

## [1.3.0] — 2026-07-07

Hardening + consolidation release. Unifies the feature branches (`devel`,
`pathcv`, the GPU-optimization branch, and `v2`) onto one line, makes the
HPC paths correct and runnable, and documents every major capability
(including **Path CVs** and **Path Refinement**). See `docs/HPC_REVIEW.md`
for the full code review, WESTPA comparison, and roadmap.

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
- **Node-local scratch** `scratch_root` (or `$PATHGENNIE_SCRATCH`) redirects
  per-segment scratch to node-local disk (`$TMPDIR`); outputs stay in `workdir`.
- **CI** (`.github/workflows/tests.yml`): base lane + a torch lane that unskips
  the SPIB/CV tests, plus the HPC self-check.
- **FES validator**: `benchmarks/we_fes.py::run_validation` + a test asserting
  the toy Weighted Ensemble recovers the analytic free energy (`r > 0.85`).
- **HPC test suite** (`tests/hpc/`): PBS + Slurm submission scripts for CPU and
  GPU queues, a dependency-light self-check, a real-backend/multi-GPU runner, and
  a debugging guide for interpreting results.
- **Docs**: `docs/hpc.md` (Slurm/PBS scaling guide) and `docs/HPC_REVIEW.md`
  (review, WESTPA comparison, SOTA positioning, roadmap); full mkdocs nav.

### Documentation
- **Path CVs and Path Refinement are now fully documented**: `docs/path-cv.md`
  (Branduardi `s`/`z` path collective variables), `docs/path-refinement.md`
  (the ensemble principal-curve refiner and its numbered example pipeline), and
  `docs/pca-cv.md` (the `pcagen` artificial PCA distance-CV space). Added a
  `docs/tutorials/10-path-refinement.md` walkthrough and a README "Path CVs &
  Path Refinement" section. These merged-in features previously had code but no
  site documentation.

### Notes
- `devel` and `main` are reconciled to the same commit as of this release.

## [1.2.0] — 2026-06-29

This release consolidates the high-performance computing (HPC) parallel scaling features, asynchronous streaming checkpointing, and robust input validation with the newly merged **PathCV** (Path Collective Variables), **Path Refinement**, and standalone **Weighted Ensemble** (WE) frameworks.

### Added

**High-Performance Computing (HPC) & Scalability**
- **MPI & Dask parallel executors (`pathgennie/core/parallel.py`):** Added `MPIExecutor` and `DaskExecutor` to distribute swarm evaluations across cluster nodes. See [docs/tutorial.md](docs/tutorial.md#1-multi-node-parallelism-with-mpi-and-dask).
- **Asynchronous trajectory streaming (`pathgennie/core/storage.py`):** Added `HDF5Storage` class utilizing a background thread to stream frames/metrics to HDF5 without keeping them in memory. See [docs/tutorial.md](docs/tutorial.md#2-asynchronous-hdf5-trajectory-streaming).
- **Robust input validation (`pathgennie/utils/config.py`):** Replaced manual parsing with a comprehensive Pydantic schema validation model (`PathGennieConfig` / `AppConfig`). See [docs/tutorial.md](docs/tutorial.md#3-robust-input-validation).

**Path Collective Variables & Path Refinement**
- **Path Collective Variables (`pathrefinement/pathcv.py`):** Added a
  dimension-agnostic implementation of Branduardi *s*/*z* path CVs
  (Branduardi et al., *JCP* 126, 054103 (2007)) with log-sum-exp stabilisation,
  automatic λ selection, optional mass weighting, and an equidistance check —
  usable both as a progress CV and inside path refinement. See
  [docs/path-cv.md](docs/path-cv.md).
- **Path Refinement library (`pathrefinement/`):** Added an ensemble-based principal curve pathway refiner (`pathrefinement/refiner.py`), mathematical verification on the Müller-Brown potential, and tutorials. See [pathrefinement/README.md](pathrefinement/README.md) and [docs/path-refinement.md](docs/path-refinement.md).
- **Artificial PCA distance-CV space (`pathgennie/utils/ligcvgen.py`, `pcagen` CLI):**
  Added `LigPCGen` (with `ligconfgen.py`) to generate protein–ligand
  conformations, build a robust PCA distance-CV space, and pick the dimension of
  maximum separation — exposed as `pathgennie pcagen`. See [docs/pca-cv.md](docs/pca-cv.md).
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

[1.3.0]: https://github.com/TeamSuman/PathGennie/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/TeamSuman/PathGennie/compare/v0.2.0...v1.2.0
[0.2.0]: https://github.com/TeamSuman/PathGennie/releases/tag/v0.2.0
[0.1.0]: https://github.com/TeamSuman/PathGennie/releases/tag/v0.1.0
