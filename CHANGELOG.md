# Changelog

All notable changes to PathGennie are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **CI lane that installs OpenMM and MDAnalysis (`test-md`).** 11 tests were guarded
  by `pytest.importorskip` and no lane installed either dependency, so they skipped in
  **every** CI run while the suite reported green: all 5 of `test_openmm_engine.py`
  (the only coverage the in-process OpenMM engine has), the OpenMM cases in
  `test_sampler_multi_engine.py` and `test_run_segment_return_contract.py`, and the 4
  MDAnalysis tests in `test_io.py` — MDAnalysis being a *declared hard dependency* that
  nothing installed. Both are pip-installable, so the lane needs no conda. It asserts
  the imports succeed, then runs the four files, then re-runs with `-rs` and **fails if
  anything still reports a dependency skip** — otherwise the lane could silently degrade
  back into the state it exists to prevent. (The `we`/`wepath` tests were checked and do
  *not* need a lane: `test_we_weight_conservation.py` puts `we/src` on `sys.path` itself.)

### Fixed
- **The HPC job templates reported success when the MD run had failed.** All four
  of `tests/hpc/{slurm_cpu.sbatch,slurm_gpu.sbatch,pbs_cpu.pbs,pbs_gpu.pbs}` ran the
  MD step as `python tests/hpc/run_example.py ... || echo "smoke run FAILED (...)"`.
  The `|| echo` *handles* the non-zero status, so `set -e` never fires, the script
  falls through to its closing `echo "### Done."` and the job exits **0**. A user
  submits the template to validate PathGennie on a new cluster, sees "Done", and
  concludes it works — when the one thing the script exists to prove did not happen.
  The missing-binary path had the same shape: no `gmx`/`pmemd.cuda` on `PATH` skipped
  the MD block entirely and still exited 0. All four now accumulate a `PG_STATUS`,
  report `### FAILED`, and `exit "$PG_STATUS"`; a missing MD binary is a failure
  unless `PG_ALLOW_NO_MD=1` is set explicitly. The self-check no longer aborts the
  script either, so one queued job returns the complete picture instead of only its
  first failure. Covered by `tests/test_hpc_templates.py`, which executes the real
  scripts with a stubbed `python` and asserts on exit codes; mutation-verified
  (8 of its 16 tests fail against the shipped templates).
- **AMBER restarts for periodic systems carried no box.** `write_rst7_coords`
  wrote coordinates only, so sander rejected the file outright —
  `peek_ewald_inpcrd: Box info not found in inpcrd`. `create_handle` builds
  restarts through that function, and the driver calls it when resuming from a
  checkpoint, as does the Weighted Ensemble stage when seeding walkers from raw
  frames: **both were broken for any solvated AMBER system.** Gas-phase runs were
  unaffected because `ntb=0` needs no box, which is why it survived an entire
  gas-phase campaign and surfaced only on the first solvated one.
  `write_rst7_coords` now takes an optional `box` (3 lengths, angles defaulting
  to 90°, or all 6 values), `read_rst7_box()` reads one back, `CoreAmberEngine`
  carries it, and `pg_amber.run` reads it from the initial restart. A periodic
  engine constructed without a box now warns rather than failing later inside
  sander. Covered by `tests/test_amber_restart_box.py`; mutation-verified.
- **A single failed trial no longer kills the whole run.** The swarm exists
  because short segments are unreliable — integrators go unstable, MD
  subprocesses get killed, scratch writes fail — and all three engines already
  raise `ValueError` from `get_coords` on non-finite coordinates. But the driver's
  trial loop had no guard, so one casualty out of `max_trial` propagated out of
  `executor.map` and ended the run, discarding every completed cycle of a
  multi-hour job. Failed trials are now quarantined (handles released) and the
  cycle selects from the survivors; a cycle in which *nothing* survives still
  raises, since that indicates a systematic problem rather than an unlucky
  segment. The τ2 runner gets the same treatment, falling back to the chosen
  sampler — the state `reject_worse_tau2` already keeps. The number of
  quarantined trials is always reported, not gated on verbosity, because silently
  dropping trials would change the effective swarm size without the user knowing.
  Covered by `tests/test_driver_trial_quarantine.py`; mutation-verified.
- **Checkpoints are written through the writer thread's own file handle.**
  `save_checkpoint` opened the HDF5 file a second time while the writer thread
  still had it open. That is not a supported pattern — it worked only because
  HDF5 caches file identifiers within a process, an implementation detail rather
  than a guarantee, and the sort of thing that breaks under a different HDF5
  build or with file locking enabled. The write is now queued behind pending
  appends and performed by the writer thread, which also makes the recorded
  `n_frames` exact for free: the queue is FIFO, so every frame appended before
  the checkpoint is on disk when it is written. The file is flushed afterwards —
  a checkpoint still sitting in a buffer does not survive the crash it exists to
  survive. Covered by `tests/test_storage_single_handle.py`.
- **`run_segment` broke its own return contract on the subprocess backends.** The
  `Engine` protocol states that when `save_subframes` is True the return "changes
  to `(Handle, subframes)`", and the driver unpacks unconditionally. AMBER and
  GROMACS returned that tuple only when the trajectory file *existed*, falling
  through to a bare handle otherwise — and a handle there is a file path, so the
  unpack raises `ValueError` and kills the run. The trigger is mundane: a
  `subframe_stride` longer than the segment leaves the MD engine with nothing to
  write, so it may not create the file at all. Both backends now always return the
  tuple, with a correctly shaped `(0, n_atoms, 3)` empty block that stays
  concatenable with real ones. The toy and OpenMM engines were already correct.
  The protocol docstring now also records that engines may differ on the tail —
  OpenMM steps in `min(stride, remaining)` chunks and captures the segment end,
  the toy engine's strict modulo captures nothing — so callers must not rely on an
  exact frame count. Covered by `tests/test_run_segment_return_contract.py`;
  mutation-verified.
- **Resuming from a checkpoint duplicated trajectory frames.** Frames stream every
  `save_freq` cycles but checkpoints are written only every `checkpoint_freq`
  cycles, so a crash between two checkpoints — the case checkpointing exists for —
  leaves frames on disk for cycles *after* the last checkpoint. The driver loaded
  the whole stored trajectory and then re-ran from `checkpoint_cycle + 1`, so
  those cycles were both loaded and regenerated. The extra frames come from a
  discarded branch, so they are wrong rather than merely redundant. Nothing on
  disk identified where to cut — frames carry no cycle index — so `save_checkpoint`
  now records `n_frames`, and `load_checkpoint` truncates both the returned arrays
  and the file (otherwise the resumed run appends after the stale rows and the
  next resume repeats the problem). Checkpoints written before this attribute
  existed still load, with the previous behaviour. Covered by
  `tests/test_checkpoint_resume_frames.py` plus an end-to-end kill-and-resume
  check that asserts the resumed **trajectory** is identical to an uninterrupted
  run, not merely the same length; mutation-verified.
- **Weighted Ensemble could release an engine handle still in use.** The stage
  compared handles with `is not`, while `driver.py` documents the opposite rule:
  compare by value, because handles may be plain ints (CPython caches only
  -5..256) or file paths that have been re-derived or round-tripped through an
  executor. An engine returning an *equal* handle — the same value as a distinct
  object — therefore had its handle released while the walker still pointed at
  it. AMBER and GROMACS handles are file paths, so the released artefact is a
  file on disk. Covered by `tests/test_we_handle_identity.py`; mutation-verified
  (restoring `is not` fails 3 of its 4 tests).
- **A dead HDF5 writer hung the run instead of reporting.** `save_checkpoint`
  called `self._queue.join()` and only then checked for a writer error.
  `Queue.join()` waits for `task_done()` on every queued item, and the writer
  thread calls that only for items it has *popped* — so if it died with a backlog
  (failed file open, full disk) the join blocked forever. A reportable error
  became a silent hang, at exactly the moment checkpointing exists to protect a
  long run. Draining now polls for writer liveness and surfaces the error, naming
  how many writes will never be flushed. Covered by
  `tests/test_storage_checkpoint_deadlock.py`; mutation-verified — restoring the
  bare `join()` makes the test suite hang, which is the defect itself.
- **Weighted Ensemble free energies and rates included the startup transient.**
  Bin occupancy was averaged from iteration 0, so the relaxation away from the
  seeded initial distribution was mixed into the estimate. Seeding is commonly
  one walker per bin — a *uniform* distribution, maximally unlike the Boltzmann
  one being estimated — so this flattened profiles and biased barriers low. Only
  the *total* weight was traced, so it could not be corrected after the fact
  either. `WeightedEnsembleStage` gains `burn_in` (an `int` count or a `float`
  fraction of the run; default `0` preserves the old behaviour) applied to both
  the free energy and the rate, and now records per-iteration, per-bin occupancy
  in `metadata["bin_weight_trace"]` plus `metadata["flux_trace"]` — so a burn-in
  can be chosen, and convergence checked by comparing windows, without re-running.
  Covered by `tests/test_we_burn_in.py`; mutation-verified (neutralising the
  burn-in makes 4 of its 8 tests fail).
- **Weighted Ensemble on AMBER died on its first segment.** `create_handle`
  writes an rst7 containing coordinates only — no velocity block — but the WE
  stage seeds walkers from raw frames through exactly that call and defaults to
  `continue_velocities=True`, which makes the engine ask `sander` to restart with
  `irest=1, ntx=5`. `sander` aborts with "I could not find enough velocities",
  surfacing only as a generic non-zero exit. `run_segment` now detects a
  coordinates-only restart and generates Maxwell-Boltzmann velocities instead,
  warning once per engine. New `rst7_has_velocities()` implements the check
  (box-line aware, so a trailing box on a small system is not mistaken for
  velocities). The GROMACS backend was unaffected — it passes `-t <cpt>` only
  when the checkpoint exists. Found by running the full QM/MM workflow end to
  end; covered by `tests/test_amber_restart_velocities.py`.
  `write_rst7_coords`'s docstring also claimed velocities were "set to zero"
  when in fact no velocity block is written at all.
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
- **A seeded run is now reproducible on a stochastic integrator.**
  `OpenMMEngine.run_segment` set the per-segment seed on an already-built `Context`,
  where it has no effect — an integrator's RNG stream is created with the Context.
  Two alanine-dipeptide runs with an identical `seed` were measured diverging from
  the very first cycle. The engine now reinitialises the Context (preserving state)
  after re-seeding; the same twin-run test then produces byte-identical metrics.
  Because `reinitialize()` costs real time per segment it is opt-in via
  `OpenMMEngine(reproducible=True)`, enabled automatically when the case supplies a
  `seed`. Unseeded runs are unaffected. This also makes `save_subframes` faithful:
  the replayed segment now reproduces the one that was actually selected.
- **Progress metrics respect CV periodicity.** `EscapeMetric`/`TargetMetric` scored
  with a plain Euclidean norm, which is wrong for dihedral CVs: a (phi, psi) pair
  straddling the +-180 deg branch cut was scored 345.2 when the true angular distance
  was 34.3, a 10x inflation that rewarded the sampler for crossing the cut. Declare
  per-component periods with the new `projection.periodic` key; omitting it preserves
  the previous behaviour, so distance and PCA CVs are unaffected.

### Changed
- **Release consolidation.** `ROADMAP.md` claimed v0.2.0 while the package is
  1.3.0; `mkdocs.yml` still carried `yourusername` placeholders in `site_url`
  and `repo_url`, so the published docs pointed at a non-existent site; and four
  documents quoted four different test counts (96 / 47 / 69 / 90) against an
  actual 192.
- **`test_overwrite_check_logic` replaced.** It rebuilt the overwrite condition
  inline and then raised `FileExistsError` *itself* inside `pytest.raises`, so it
  exercised none of the production code and would have passed with the feature
  deleted. It now drives `pg_amber.run` against a case complete enough to reach
  the guard, plus a companion asserting `overwrite: true` gets past it.
  Mutation-verified: deleting the guard now fails the test.
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
- **Engine-agnostic path refinement** (`pathrefinement.samplers.EngineSampler`).
  `PathRefiner`'s exploration step was hard-wired to an OpenMM walker, which
  excluded the two backends that need refinement most: AMBER — the only backend
  that can run a QM Hamiltonian, so QM/MM refinement was impossible — and
  GROMACS. `PathRefiner` now accepts an injected `sampler`, and `EngineSampler`
  implements that contract on the core `Engine` protocol, driving the walker in
  target mode on the path progress coordinate `s`. Anything satisfying the
  protocol works; `tests/test_sampler_multi_engine.py` exercises the toy engine,
  a bare stub, OpenMM, and protocol conformance for AMBER and GROMACS.
  `examples/path_refinement_engines/refine_with_engine.py` runs identical
  refinement code against all four backends.
- **Independent NEB reference stage** (`examples/qmmm_reactive_sn2/amber/5_neb_reference.py`):
  relaxes a nudged elastic band at the *same* level of theory as the sampling, then
  optionally re-scores the relaxed geometries at DFT via QUICK. On the shipped
  S<sub>N</sub>2 case it reproduces a 3.56 kcal/mol DFTB3 barrier with the TS
  geometry matching an independent symmetric-stretch scan to 0.0007 A, and an
  8.23 kcal/mol B3LYP/6-31+G* barrier on the same geometries, in ~2 minutes. It
  also quantifies the refined PathCV against the NEB path in the CV plane
  (0.044 A mean, versus 0.059 A for the unrefined seed consensus), which is the
  workflow's end-to-end check.
- **Complete reactive QM/MM workflow** (`examples/qmmm_reactive_sn2/amber`, plus
  `docs/qmmm-workflow.md`): bond-breaking/forming path discovery → refinement
  into a PathCV → free energy along `s` by Weighted Ensemble → 2-D mechanism
  plot, all at one level of theory. Documents the measured QM-method tradeoff
  (DFTB3 at 20 ms/step for sampling; *ab initio* reserved for single points,
  because its ~8 s process launch is paid per swarm segment), why reactive
  barriers need *shorter* swarm segments than conformational ones, and the
  validation checks worth running (an independent TS reference at the same level
  of theory; a geometric mechanism signature such as Walden inversion).
- **Guidance on convergence criteria** (`docs/configuration.md`,
  `docs/qmmm-workflow.md`). A convergence function written as a difference of
  distances is satisfied by one distance growing alone and does not require the
  product bond to form. On a tertiary substitution test every seed reported
  `Converged` while none produced the product. Drive on a progress CV; stop on a
  product-specific condition. Several shipped example configs use the
  distance-difference pattern and are safe only because the intended product is
  the sole route to a large CV value there.
- **Single-GPU saturation (OpenMM).** `OpenMMEngine` backs a pool of concurrent
  Contexts on one card; `workers_per_device` (an int, or `auto` sized from cores
  and free GPU memory) runs that many swarm walkers at once instead of serially.
- **Downstream Weighted Ensemble parallelism.** The backend device pool is
  forwarded to the WE stage, so its walker propagation spreads across GPUs/cores.
- **Intra-segment frame capture** (`save_subframes`, `subframe_stride`): the
  committed τ1+τ2 segment is replayed to harvest intermediate frames, giving a
  continuous trajectory instead of one frame per `save_freq` cycles. On a
  stochastic integrator this is faithful only when the run is seeded (see the
  reproducibility fix above); the replay costs a second pass over the committed
  segment, so it roughly doubles the MD work per accepted cycle.
- **Checkpoint restart and output-overwrite protection** (`checkpoint_freq`,
  `checkpoint_path`, `overwrite`): a run resumes from the last checkpoint, and
  existing outputs are no longer silently clobbered.
- **Correct trajectory timestamps**: written frames carry real simulation times
  derived from the integrator timestep.

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
