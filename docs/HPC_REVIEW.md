# PathGennie — HPC Code Review, SOTA Positioning & Roadmap

*A computational-chemistry code review focused on running scalable, large-scale
atomistic MD on HPC clusters (CPU and GPU, Slurm and PBS), plus a comparison
against state-of-the-art tools and a prioritized development roadmap.*

This document accompanies the branch-integration work that unified `devel`
(which already contained `pathcv` and the GPU-optimization branch) and `v2` onto
`main`, hardened the HPC-critical paths, and added a Slurm/PBS test suite
(`tests/hpc/`). Fixes marked **[FIXED]** are implemented and covered by tests on
this branch; **[OPEN]** items are documented for follow-up with a suggested
approach.

---

## 1. Executive summary

PathGennie's core idea — selection bias on swarms of *unbiased* ultrashort
trajectories to discover rare-event paths, then hand seeds to quantitative
enhanced sampling — is timely and competitive. The `devel` re-architecture
(single backend-independent driver, device pool, SPIB CV, WE/OPES/TPS stages,
RRT/roadmap search) is a strong foundation.

However, before this review the promoted code had **release-blocking defects**:

- **Config validation dropped every real parameter** — every AMBER/GROMACS/OpenMM
  run crashed with `KeyError: 'tau1_steps'`, and `md`/`workdir`/`output` blocks
  were silently discarded (wrong physics would have run). **[FIXED]**
- **On-the-fly SPIB crashed on the first cycle** (`project()` missing the `cycle`
  argument). **[FIXED]**
- **GPU device selection ignored the scheduler**, overwriting
  `CUDA_VISIBLE_DEVICES` with absolute indices and colliding with other jobs on
  shared nodes. **[FIXED]**
- **A leftover git merge-conflict block was committed into `README.md`.** **[FIXED]**

The dominant *architectural* limitations for large-scale MD are: (1) no working
multi-node path; (2) per-segment process re-launch overhead that dominates
ultrashort segments on large systems; (3) no resume-from-checkpoint; and (4)
several enhanced-sampling modules that are CI-verified on toys but not yet wired
to real MD end-to-end. These are the roadmap's focus.

---

## 2. What was integrated (branch reconciliation)

| Branch | Status | Notes |
| ------ | ------ | ----- |
| `devel` | merged → this branch | Superset of features; brought to `main`. |
| `pathcv` | already ⊆ `devel` | PathCV / artificial-PCA-space mode; no separate import needed. |
| `claude/…gpu-optimization…` | already ⊆ `devel` | Shared driver, device pool, WE/OPES/search. |
| `v2` | already ⊆ `main` | Its later-deleted files (OAMe-G2 AMBER example) were intentionally relocated to OpenMM in `devel`; nothing orphaned. |

Because `pathcv` and the GPU branch were already merged into `devel`, and `v2`
into `main`, a single `devel`→`main` merge unified **all** branches. `devel`'s
own commit history (preserved by the merge) is already phased by feature; the
hardening work here is layered on top as clean, single-purpose commits.

---

## 3. Bugs fixed on this branch

| # | Severity | Area | Bug | Fix |
| - | -------- | ---- | --- | --- |
| 1 | Critical | `utils/config.py` | Schema declared `tau1`/`tau2` (code reads `tau1_steps`); `extra="ignore"` dropped `tau1_steps`, `devices`, `downstream`, `profile`, and the `md`/`workdir`/`output` sections → `KeyError` and silently-ignored MD params. | Real field names + `extra="allow"` on both models; bounds/enum validation; empty-file guard; `tests/test_config.py`. |
| 2 | Critical | `cv/spib.py` | `SPIBProgress.project(self, coords)` lacked `cycle`; driver always calls `project(coords, cycle=…)` → `TypeError` on cycle 0. Hidden because the test skips without torch. | Added `cycle` param; forwarded to bootstrap CV. |
| 3 | High (HPC) | `core/parallel.py`, engines, `we/gpu_worker.py` | Engines overwrote `CUDA_VISIBLE_DEVICES` with absolute indices → targeted GPUs the job didn't own. | `resolve_cuda_visible_device()` maps logical indices onto the scheduler mask; wired into AMBER/GROMACS engines and the `we` GPU worker; `tests/test_hpc_parallel.py`. |
| 4 | High (HPC) | `core/driver.py` | Per-trial seeds drawn inside worker threads; `np.random.Generator` is not thread-safe → non-reproducible seed→trial mapping under the device pool. | Seeds pre-drawn on the main thread; threaded runs now match serial bit-for-bit. |
| 5 | Med (HPC) | `core/driver.py`, `search/rrt.py` | Cloned-anchor handle never released → one scratch restart / cache entry leaked per trial per cycle (fills node-local scratch). | Release the clone after its segment output exists. |
| 6 | Med | `core/driver.py` | Handle release used `is` identity on ints (>256) — fragile, breaks under serialization. | Compare by value (`==`). |
| 7 | Med (HPC) | `core/storage.py` | HDF5 writer-thread exceptions swallowed → every frame silently lost after the thread died. | Capture and re-raise on `append`/`close`. |
| 8 | Med | `pathrefinement/__init__.py` | Eager `torch`+`openmm` imports broke `import pathrefinement` on a base install. | Lazy (PEP 562) loading with a clear install hint. |
| 9 | Med | `pathrefinement/pathiter.py` | `NameError` (`pc` undefined) in the iterative learner. | `self.pc`. |
| 10 | Low | `README.md` | Committed merge-conflict markers; stale example table (OAMe-G2); reference to a deleted config (`input_c7eq.yaml`). | Resolved/updated; config reference expanded with the HPC keys. |
| 11 | New (HPC) | `pathgennie` config + engines | No control over CPU thread count → concurrent CPU segments each grabbed all cores. | `cpu_threads_per_worker` pins OMP/MKL threads and GROMACS `-ntomp`. |

All fixes verified by `pytest` (90 passed, 2 skipped for absent torch/openmm) and
by `tests/hpc/hpc_selfcheck.py`.

---

## 4. Open bugs & limitations (prioritized, not yet fixed)

These need domain decisions or real-hardware validation; each has a suggested
direction. Grouped by subsystem.

### Core driver / parallelism
- **[RESOLVED] Multi-node executors removed.** `MPIExecutor`/`DaskExecutor` were
  non-functional (nested closures capturing a live, unpicklable engine) and wired
  to no backend, so they were removed along with the `[hpc]` extra. Multi-node
  throughput is served by independent pathways/replicates as Slurm/PBS array jobs;
  a work-queue manager for a single tightly-coupled run stays on the roadmap.
- **[OPEN, High] OpenMM has no multi-GPU path.** `pg_omm` hardcodes
  `SerialExecutor`; the promised process pool ("one Context per GPU") is not
  implemented. *Fix:* a `ProcessDevicePool` that builds one `Simulation` per
  worker process pinned to a `CudaDeviceIndex`.
- **[OPEN, Med] OpenMM platform properties are unset** — no `CudaDeviceIndex`,
  `Precision` (defaults to `single`, hurting energy conservation on large
  systems), or `DeterministicForces`. *Fix:* thread a `platform_properties` dict
  from `openmm:` config into both `Simulation(...)` sites.
- **[OPEN, Med] Static per-cycle barrier.** The swarm is a hard barrier: the
  slowest of `max_trial` segments gates the cycle, and `ThreadDevicePool`
  round-robins statically. Heterogeneous segment costs waste GPU time. *Fix:*
  a pull-based task queue (see §6, WESTPA).

### Enhanced sampling (`sampling/`)
- **[OPEN, Med] OPES FES reweighting lacks `c(t)`** and reuses the final bias for
  all frames → statistically inconsistent; no error bars. *Fix:* store
  instantaneous bias, use the OPES `c(t)` estimator, or reweight the converged
  tail only.
- **[OPEN, Med] OPES production (PLUMED) path is unwired** — needs an
  `engine.run_plumed(...)` no backend implements. *Fix:* implement PLUMED driving
  in the OpenMM backend (OpenMM-PLUMED plugin) and/or GROMACS `-plumed`.
- **[OPEN, Med] TIS rate analysis is a silent stub** (`TISAnalysis` is not the OPS
  rate API; wrapped in a bare `except`). *Fix:* use `StandardTISAnalysis` with
  flux/interface components; drop the silent catch.
- **[OPEN, Med] WE reports a free energy under recycling**, which is a
  non-equilibrium steady state, not the equilibrium FES. *Fix:* suppress/relabel
  the FES when `recycle=True`; report flux/rate instead.
- **[OPEN, Low] WE split under a deterministic integrator** produces perfectly
  correlated copies (no variance reduction). *Fix:* warn/require stochastic
  dynamics when splitting with continued velocities.
- **[OPEN, Med] In-process (OpenMM/toy) engines can't cross process boundaries**
  for WE under MPI/Dask (integer handles index a process-local cache). *Fix:*
  return coordinates alongside handles, or restrict distributed executors to
  file-handle backends.

### CV / search
- **[OPEN, High] SPIB retraining blocks the MD loop** on the main thread (CPU
  only), with an unbounded frame buffer re-featurized every refresh. On a long
  run the whole swarm stalls and memory grows without bound. *Fix:* background
  training with atomic model swap; ring-buffer the frames; train on GPU.
- **[OPEN, Med] SPIB uses a fixed N(0,I) prior** and reinitializes each refinement
  round — it's a time-lagged deep-VIB, not full SPIB (whose learnable
  state-dependent prior drives the emergent state count). *Fix:* implement the
  mixture prior or document as "VIB-SPIB (simplified prior)"; warm-start.
- **[OPEN, Med] SPIB CV is non-stationary across refreshes** — stored
  `anchor_metric` is compared against a metric under a newly trained latent. *Fix:*
  re-project the anchor metric after each refresh; warm-start the encoder.
- **[OPEN, Low] RRT nearest-neighbour is O(n)** with a per-call `np.stack` → O(n²)
  tree build; the requested KD-tree is absent. *Fix:* `scipy.spatial.cKDTree`
  rebuilt periodically (CV dim is tiny).

### `pathrefinement/` and `we/` packages
- **[OPEN, High] `we/` is an orphaned island** with its own `pyproject`/deps that
  duplicates `pathgennie/sampling/weighted_ensemble.py`. *Fix:* consolidate on the
  `pathgennie/sampling` implementation; port any unique `we/` features
  (recycling/flux, ligand-RMSD merge) into it; excise `we/` from the release or
  clearly mark it a separate research repo.
- **[OPEN, Med] `pathrefinement` real-MD path is non-functional/oversold** —
  `PathRefiner` refines only a 2D toy particle; `pathiter` collapses a protein
  into one 3D principal curve. *Fix:* implement a genuine path-CV (s,z) feature
  map for molecular systems; scope the docs to what works today.
- **[OPEN, Med] Duplicated/dead code**: `we/main.py` vs `main2.py` vs
  `main_clean.py`; `LigandResamplerBack` returns one value but the caller unpacks
  two. *Fix:* delete the dead variants.
- **[OPEN, High] ~90 MB of committed binary artifacts** (27 MB + 25 MB PDBs,
  multi-MB result PNGs/DCDs, large notebooks with outputs) and **hardcoded
  `/home/dm/...` paths** in `pathrefinement`/`we` examples. *Fix:* `git rm` the
  generated `results/`, add `.gitignore` rules, strip notebook outputs (consider
  `git filter-repo`/LFS for history), and parameterize the paths.

### Packaging / config surface
- **[OPEN, Med] OpenMM is a hard core dependency** — `pip install pathgennie`
  forces a heavy conda-forge package on AMBER/GROMACS-only users. *Fix:* move to
  an `[openmm]` extra and lazy-import in the backend.
- **[OPEN, Med] `pcagen` needs scikit-learn, undeclared** → `ModuleNotFoundError`
  on a clean install. *Fix:* add an `[analysis]` extra.
- **[OPEN, Low] Strategy `selection`/`cv` fields are advertised but unconsumed**
  (`SAMPLING` says `selection="beam"`, but the driver only ever does
  `softmax`). The agent controller is likewise unused at runtime. *Fix:* wire the
  policies or document them as future/no-op.

### AMBER/GROMACS backend
- **[OPEN, Med] ASCII restart (`ntxo=1`) parsing** is slow and lossy for very
  large systems; NetCDF restarts would be faster/precise. *Fix:* support and
  prefer NetCDF restarts (`ntxo=2`) via MDAnalysis/parmed.
- **[OPEN, Low] `os.chdir(case_dir)`** in the runners mutates global CWD (not
  reentrant). *Fix:* pass absolute paths; avoid `chdir`.
- **[OPEN, Low] CV/convergence use unwrapped coordinates** — COM–COM distances
  across PBC need minimum-image handling in the user's projection.

---

## 5. HPC scaling analysis

**Where PathGennie scales well today.** Single-node multi-GPU on AMBER/GROMACS is
sound: `subprocess.run` releases the GIL so `ThreadDevicePool` genuinely runs
segments concurrently, each pinned to a (now scheduler-correct) GPU with an
isolated scratch subdirectory. For discovery runs (many short segments, small/
medium systems), this saturates a GPU node well.

**The launch-overhead wall.** The subprocess backends re-launch `pmemd.cuda` /
`gmx mdrun` for **every** segment. Each launch pays GPU-context creation +
topology parse + neighbor-list build — seconds for a large (10⁵–10⁶ atom) system.
With ultrashort `tau1` (the paper's regime is tens of fs, i.e. a handful of
steps), *startup dominates useful work*: a 2-step segment on a 500k-atom system
may spend >99% of wall time in launch. This is the single biggest scaling
limitation for large systems.

*Implications / mitigations:*
- Prefer the **in-process OpenMM engine** for large systems (persistent Context,
  no per-segment launch) — but it needs the multi-GPU process pool (§4).
- Increase `tau1`/`tau2` so useful work amortizes launch (the `sampling` profile
  already does this).
- Consider a **persistent-worker** model for AMBER/GROMACS (a resident process
  fed successive segments, e.g. GROMACS via `gmxapi`, AMBER via its Python API),
  which is the real fix.

**Filesystem.** Thousands of tiny segment files on shared Lustre/NFS is a
metadata storm. Node-local `$TMPDIR` is essential (documented in `docs/hpc.md`;
enforced by the `tests/hpc` scripts). A future default should place scratch on
node-local disk automatically.

**Memory.** The driver holds the full trajectory in RAM and, with
`collect_seeds`, one State clone per saved frame; `checkpoint_path` streams to
HDF5 but is not the default and the trajectory list is still retained. For
large systems over many cycles this grows linearly. *Fix:* stream by default and
drop in-RAM frames when a checkpoint path is set.

**Load balancing.** The per-cycle barrier plus static round-robin wastes GPU time
under heterogeneous segment costs. A pull-based queue (next §) is the fix.

---

## 6. Comparison with WESTPA (Slurm/PBS job control)

[WESTPA](https://github.com/westpa/westpa) is the reference weighted-ensemble
framework and a good yardstick for HPC job control. Key differences:

| Concern | WESTPA | PathGennie (today) | Takeaway |
| ------- | ------ | ------------------ | -------- |
| Work managers | `serial`, `threads`, `processes` (single-node shared memory), **`zmq`** (ZeroMQ master + clients, multi-node), `mpi` — chosen at runtime | `SerialExecutor`, `ThreadDevicePool` (single-node); MPI/Dask stubs broken | **Borrow the ZMQ work-queue model** for real multi-node scale-out. |
| Multi-node | Slurm script starts a master; `srun` launches a client per node that pulls segments from the master's queue → dynamic load balancing across nodes | none working | Highest-value gap. |
| Load balancing | Pull-based task queue; heterogeneous segment costs balance naturally | static round-robin + per-cycle barrier | Replace round-robin with a queue. |
| Restart | Central `west.h5`, checkpointed **every iteration**; `w_run` resumes seamlessly → multi-week campaigns survive walltime limits/requeue | streaming HDF5 only; **no resume** | **Add resume-from-checkpoint** (anchor + RNG state). Critical for HPC walltime limits. |
| Per-segment RNG | `ig=RAND` entropy-seeded, replaced per segment by `runseg.sh` (best practice to avoid restart bias) | `ig=<seed>` per segment, now deterministic from a master seed | **On par / arguably better** — PathGennie is reproducible; WESTPA's entropy seed is not (by design). |
| Node-local scratch | `runseg.sh` runs each segment in `/tmp`, copies back before job end | scratch under `workdir` (often shared FS) | **Adopt node-local `$TMPDIR` by default** with copy-back. |
| Single-node multi-GPU | `processes` manager (some users prefer it over ZMQ to avoid TCP) | `ThreadDevicePool` (lighter — subprocess MD releases the GIL) | **On par**; PathGennie's is arguably simpler for this case. |
| Config / UX | `west.cfg` + `env.sh`/`node.sh`/`runseg.sh` templates per cluster | single `input.yaml` + Python CV/convergence | **PathGennie is friendlier**; keep that edge. |
| Env inheritance across nodes | Explicit `env.sh`/`node.sh` per node | inherits `os.environ` (single node) | Needed once multi-node lands. |

**Ideas to borrow, concretely:**
1. A **`work_manager` config key** (`serial|threads|processes|zmq`) selecting the
   executor uniformly across backends — decouple orchestration from allocation
   (WESTPA's biggest structural win).
2. A **ZeroMQ/task-queue executor**: a master enqueues segments; clients launched
   by `srun`/`mpirun` on each node pull work and own a process-local engine. This
   simultaneously fixes multi-node *and* dynamic load balancing, and sidesteps the
   pickling problem (send positions/params, not closures).
3. **Iteration-level checkpoint + resume** to a central HDF5, so a job that hits
   walltime requeues and continues — table-stakes for production campaigns.
4. **Node-local scratch by default** with copy-back, as WESTPA's `runseg.sh` does.

Where PathGennie already leads WESTPA: reproducible seeding, a much lower-friction
config/UX, and a broader algorithmic ambition (direction-guided discovery feeding
WE/OPES/TPS rather than WE alone).

---

## 7. SOTA positioning & further developments

PathGennie sits at the intersection of **adaptive sampling** (High-Throughput MD /
Markov State Models, FAST, AdaptiveBandit), **enhanced sampling** (WESTPA,
PLUMED/OPES, SSAGES, OpenPathSampling), and **ML collective variables**
(SPIB, VAMPnets, TICA, GNN-CVs). Its differentiator is *direction-guided
selection on unbiased ultrashort swarms* to seed the quantitative methods.

To be globally competitive:

**Correctness & trust.**
- Ship **quantitative validators** with error bars: FES vs analytic marginals on
  toys (Müller-Brown, Wolfe-Quapp) with block-averaged error, and rate/MFPT
  cross-checks against WE/TPS. Turn `benchmarks/we_fes.py`'s correlation print
  into a pass/fail with tolerances.
- A **CI lane with torch + OpenMM** so the SPIB/OpenMM paths (currently skipped)
  are actually exercised.

**Interoperability (adopt community standards, don't reinvent).**
- **PLUMED** as a first-class CV/bias engine (OpenMM-PLUMED, GROMACS `-plumed`) —
  gives OPES/metadynamics/funnel-metad for free and lets users reuse existing
  `plumed.dat`. The repo already has `plumed.dat` examples; wire them.
- **MDAnalysis/MDTraj** selection language for CVs instead of hand-indexing.
- **deeptime**/**PyEMMA** for MSM/TICA/VAMP so learned CVs and kinetics are
  validated against a mature library.
- **OpenMM `gmxapi`/AMBER Python** persistent engines to kill per-segment launch
  overhead.

**Method frontier.**
- **Modern ML-CVs**: VAMPnets, time-lagged autoencoders, and GNN-based CVs
  (`mlcolvar`) alongside SPIB; a proper learnable-prior SPIB.
- **AND** — align the discovery→refinement→sampling pipeline so a discovered path
  seeds string/FEP-style refinement and then WE/TIS for rates, end-to-end.
- **haMSM / history-augmented MSM** post-analysis (as in WESTPA 2.0) for
  unbiased kinetics from the swarm data you already generate.

**Scale & robustness (the roadmap below).**

---

## 8. Prioritized roadmap

**P0 — make the promoted release trustworthy (done).**
- [x] Fix config validation, SPIB `cycle`, GPU masking, reproducibility, leaks,
      storage errors, lazy imports, README conflict markers.
- [x] Add a CI lane (base + torch, so SPIB/CV tests run instead of skipping);
      turn the toy WE FES benchmark into a pass/fail validator (`r > 0.85`).
- [x] Untrack the ~90 MB of committed run outputs and gitignore them;
      parameterize the hardcoded home paths. *(Remaining: a `git filter-repo`/LFS
      history rewrite to reclaim the blobs already in history.)*

**P1 — HPC scale-out (the biggest competitive gap).**
- [x] Node-local scratch via `scratch_root` / `$PATHGENNIE_SCRATCH` (outputs stay
      on the shared FS). *(Next: make node-local the documented default in job
      templates — the `tests/hpc` scripts already point `$TMPDIR` there.)*
- [ ] `work_manager` config abstraction (serial/threads/processes/zmq).
- [ ] A ZeroMQ/task-queue executor for multi-node with dynamic load balancing
      (fixes multi-node + balancing + the pickling problem together).
- [ ] Iteration-level checkpoint **and resume** to a central HDF5.
- [ ] OpenMM `ProcessDevicePool` (multi-GPU in-process) + platform-properties
      (precision/deterministic forces/device index).

**P2 — eliminate the launch-overhead wall.**
- [ ] Persistent-worker AMBER/GROMACS engines (`gmxapi` / AMBER Python API) so
      large-system ultrashort segments amortize startup.
- [ ] Stream trajectories to disk by default; stop holding full trajectories in RAM.

**P3 — enhanced-sampling correctness & interoperability.**
- [ ] OPES `c(t)` reweighting with error bars; wire the PLUMED production path.
- [ ] Real OPS `StandardTISAnalysis` rates; remove silent excepts.
- [ ] Consolidate `we/` into `pathgennie/sampling`; delete dead variants.
- [ ] Background SPIB training + ring buffer + GPU; learnable prior.

**P4 — method frontier & UX.**
- [ ] PLUMED/MDAnalysis/deeptime integrations; modern ML-CVs.
- [ ] End-to-end discovery→refinement→WE/TIS pipeline with a single driver.
- [ ] First-class CLI for stages/profiles; `--version`; typo-warning on unknown
      config keys.

---

*See `tests/hpc/README.md` and `tests/hpc/DEBUGGING.md` for running the HPC test
battery and interpreting its JSON results.*
