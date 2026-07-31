# PathGennie Future Roadmap

> **Purpose.** A static, forward-looking plan to evolve PathGennie from a working
> adaptive-sampling toolkit (1.3.0) into a globally competitive, cutting-edge
> enhanced-sampling **framework** comparable to PLUMED and WESTPA.
>
> **Scope.** This document is strategy and design intent — not a commitment of
> dates. For *current* implemented-vs-planned status see
> [`docs/roadmap.md`](docs/roadmap.md). For released changes see
> [`CHANGELOG.md`](CHANGELOG.md).

## North star & positioning

PathGennie should **not** try to out-PLUMED PLUMED on CV breadth or out-WESTPA
WESTPA on weighted-ensemble plumbing. Its defensible, differentiated niche is an

> **autonomous, ML-CV-driven, multi-paradigm rare-event pipeline —
> *discover → quantify → kinetics* — across MD engines, with an agent deciding
> how to spend GPU time.**

- **PLUMED** is a *bias/CV library* patched into MD engines; it does not
  orchestrate discovery or kinetics.
- **WESTPA** is a *weighted-ensemble* framework; it is not a path-discovery or
  CV-learning tool.
- **PathGennie** already unifies greedy/RRT discovery, SPIB CV learning, WE, and
  OPES behind one `Engine`/`ParallelExecutor`. The roadmap below buys *parity*
  (Tiers 1–2) and doubles down on *the integration that no one else has*
  (Tiers 3–4).

## Competitive gap snapshot

| Capability | 1.3.0 | PLUMED | WESTPA | Target tier |
|---|---|---|---|---|
| Parallelism | single-node threads | MPI/replica | work managers, multi-node | T1 |
| Persistence / restart | none | COLVAR/STATE | HDF5 + checkpoint | T1 |
| WE binning | uniform 1-D | — | MAB/Voronoi/custom, haMSM | T2 |
| Statistical errors | point estimates | reweight + errors | block/bootstrap | T2 |
| Real biasing | OPES input only | native C++ | — | T2 |
| Kinetics / MSM | roadmap graph | — | haMSM, w_ipa | T2 |
| CV library | thin + SPIB | large | — | T3 |
| Config / validation | plain YAML dict | keyword files | YAML + tooling | T1 |
| I/O | synchronous per-frame | streamed | HDF5 streaming | T1 |
| Real-MD CI | none (toy) | extensive | extensive | T1 |

---

## Tier 1 — Production-grade engineering (parity table stakes)

### T1.1 Storage + checkpoint/restart  *(highest priority)*
- **Goal:** resumable long runs and a queryable results store (WESTPA's biggest
  lead). Container restarts currently lose all in-memory state.
- **Design:** an HDF5/Zarr store (`pathgennie/io/store.py`) holding trajectories,
  per-cycle/per-walker weights, CV trajectory, SPIB state labels, RNG state, and
  per-segment provenance. Add `checkpoint()`/`restore()` to `PathGennieDriver`,
  `WeightedEnsembleStage`, and `RRT`.
- **Deps:** `h5py`/`zarr`. **Acceptance:** kill + resume a toy WE/RRT run and
  reproduce the un-interrupted result bit-for-bit.

### T1.2 `WorkManager` abstraction (multi-node)
- **Goal:** scale past one node and beyond threads.
- **Design:** generalize `core/parallel.py` `ParallelExecutor` into
  `serial | process | thread | mpi | dask | ray` work managers; add a
  **process-pool OpenMM** path (one CUDA context per process). Straggler/retry
  handling and failed-segment quarantine.
- **Deps:** `mpi4py`/`dask`/`ray` (optional extras). **Acceptance:** WE/discovery
  run unchanged across managers; `G=1` ≡ serial for a fixed seed; a 2-process
  toy run matches serial.

### T1.3 Typed config, unified CLI, plugin registry, sandboxing
- **Goal:** a real framework UX and safety.
- **Design:** pydantic/attrs config schema + validation (replace raw `cfg[...]`),
  one CLI (`pathgennie discover|we|opes|analyze|resume`), entry-point plugins for
  engines/CVs/binners/stages, and **sandboxed loading of user `projection.py`**
  (currently `exec`'d via `backends/amber/utils.load_function` — a code-exec risk
  for a shared package).
- **Acceptance:** invalid configs fail fast with clear messages; a third-party CV
  plugin loads via entry point.

### T1.4 Async / streaming I/O
- **Goal:** stop blocking MD on disk.
- **Design:** background writer thread/process; chunked HDF5 writes replacing the
  per-frame `MDAnalysis` path in `backends/*/utils.write_trajectory`.

### T1.5 Real-MD CI, benchmarks, packaging, docs site
- **Goal:** credibility and reproducibility.
- **Design:** tiny alanine smoke tests on real `pmemd`/`gmx`/OpenMM in CI;
  regression baselines; benchmark CI (scaling + FES accuracy); conda-forge
  package; a `mkdocs-material` site built from `docs/`.

---

## Tier 2 — Sampling-method depth (match WESTPA / PLUMED)

### T2.1 Weighted Ensemble maturity
- Pluggable `Binner` interface: **MAB**, **Voronoi**, k-means, and **SPIB-state
  bins** (using `SPIBProgress.state_labels`) alongside the current `GridBinner`.
- **Walker-history tracking → haMSM** analysis; recycling/flux per bin.
- **Rate-constant error bars** (block bootstrap) + convergence diagnostics; a
  `w_ipa`-style analysis CLI. **Acceptance:** WE rate on a benchmark matches a
  published value within error bars.

### T2.2 Real PLUMED engine bridge
- Implement `engine.run_plumed(plumed_input, ensemble, **cfg)` (via the `plumed`
  Python module / engine patches) so `OPESStage(mode="plumed")`, MetaD, umbrella
  sampling, and funnel-metadynamics run end-to-end on real MD — turning today's
  OPES *interface* into a capability. **Acceptance:** alanine φ/ψ FES via OPES on
  a PLUMED-patched OpenMM matches a reference surface.

### T2.3 Estimators & analysis module (`pathgennie/analysis/`)
- MBAR/WHAM (`pymbar`), block averaging, autocorrelation / statistical
  inefficiency, bootstrap; standardize `SamplingResult` to carry error bars.

### T2.4 MSM / kinetics
- Build MSMs from swarm/WE data via `deeptime`; MFPT, committors, TPT fluxes.
- **Fuse with the roadmap graph** (`search/roadmap.py`) so all-pairs pathways
  carry *rates*, not just `-log(fraction)` weights.

### T2.5 Path sampling stages  *(partially delivered)*
- **Done:** an OpenPathSampling bridge (`sampling/path_sampling.py`) —
  `PathSamplingStage` runs TPS/TIS seeded by a PathGennie `PathEnsemble`, with
  dependency-free, CI-tested seed preparation. Needs the `pathsampling` extra +
  an OPS engine to run.
- **Remaining:** forward-flux sampling; turnkey config-driven TPS/TIS from a
  backend (an OPS engine must currently be supplied via the Python API); kinetics
  post-analysis helpers (rate error bars, crossing-probability plots).

---

## Tier 3 — CV & ML ecosystem (differentiate)

### T3.1 CV library + bridges
- RMSD, native contacts / Q, coordination numbers, secondary structure; an
  MDAnalysis/mdtraj featurizer bridge; a **PLUMED-CV bridge** (reuse PLUMED CVs
  as PathGennie progress variables).

### T3.2 Learnable CVs beyond SPIB
- One `LearnedCV` API behind which **DeepTICA, VAMPnets, committor / Deep-TDA,
  autoencoders** plug in, with an **active-learning loop** (discover → retrain CV
  → re-bias).

### T3.3 Closed-loop SPIB → OPES
- Learn the CV during discovery, then bias OPES along the *learned* latent
  automatically (the `sampling` profile already anticipates this).

---

## Tier 4 — Differentiators (own the niche)

### T4.1 Autonomous orchestration
- Upgrade `agent/controller.py` from rule-based to an RL / contextual-bandit
  policy (state = progress / energy fluctuation / CV gradient; action =
  `(N, τ1, τ2, frontier)`; reward = progress-per-GPU-second), with an optional
  **LLM meta-controller** that routes goal → method and drafts the config.

### T4.2 One-config integrated pipeline
- `discover (RRT/greedy) → quantify (WE/OPES) → kinetics (MSM/roadmap)` from a
  single config, emitting one results object. This integration *is* the product.

### T4.3 All-pairs roadmap kinetics
- Combine the roadmap graph with MSM rates into a network of states and transition
  rates with competing pathways — automated egress/entry-route discovery.

### T4.4 GPU-native ensemble engine
- Keep state on device, batched CV evaluation, CV-on-GPU (OpenMM `CustomCVForce`),
  zero host round-trips for the OpenMM swarm.

---

## Cross-cutting technical improvements
- **Reproducibility at scale:** extend the master-seed model to distributed
  workers; record full provenance (versions, config, git SHA) in the store.
- **Observability:** structured logging, metrics, and a live TUI/web dashboard
  (swarm/WE progress, per-bin occupancy, flux).
- **Robustness/fault tolerance:** restart failed walkers from checkpoints; NaN /
  blow-up quarantine with auto-recovery (extends the current finite-checks);
  disk-space guards.
- **Performance:** vectorized selection/binning, pinned float32 buffers, avoid
  OpenMM XML state serialization, batched CV evaluation.
- **API stability:** semantic versioning + a deprecation policy now that 1.3.0
  fixes the core protocols.

## Priority queue — the first five
1. **HDF5 storage + checkpoint/restart** (T1.1) — unblocks every long run and all
   analysis; closes the #1 gap vs WESTPA.
2. **`WorkManager` abstraction** (T1.2) — multi-node / process-pool scale.
3. **Adaptive WE binning + haMSM + rate error bars** (T2.1) — makes WE
   publication-grade.
4. **Real PLUMED `run_plumed` engine bridge** (T2.2) — unlocks OPES/MetaD/US on
   real MD.
5. **Analysis module (MBAR/block/bootstrap) + deeptime MSM** (T2.3, T2.4).

## Proposed release milestones (indicative, not dates)
| Version | Theme | Headline items |
|---|---|---|
| **0.3.0** | Persistence & scale | T1.1 storage/restart, T1.2 work managers, T1.4 async I/O |
| **0.4.0** | Framework UX | T1.3 typed config + CLI + plugins + sandboxing, T1.5 real-MD CI / docs site |
| **0.5.0** | WE & FES rigor | T2.1 adaptive WE + errors, T2.3 estimators, T2.2 PLUMED bridge |
| **0.6.0** | Kinetics & CV ML | T2.4 MSM, T2.5 path sampling, T3.1–T3.2 CV library + learned CVs |
| **1.0.0** | Autonomous pipeline | T4.1–T4.4 agentic orchestration, integrated pipeline, GPU-native engine |

## Non-goals / risks
- **Non-goal:** re-implementing PLUMED's CV catalogue or a new MD integrator —
  bridge to existing tools instead.
- **Risk:** breadth over depth. Each tier item must ship with tests and at least
  one validated benchmark before the next is started.
- **Risk:** real-MD paths can't be validated in CPU-only CI — gate them behind a
  GPU/MD CI lane and clearly mark unverified paths (as done in `docs/roadmap.md`).
