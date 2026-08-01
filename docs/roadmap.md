# Roadmap: implemented vs planned

A snapshot of what the current codebase delivers and what remains, against the
development plan.

## Implemented (verified in CI on toy/synthetic systems)

| Area | Module(s) | Verification |
|---|---|---|
| Shared core driver | `core/driver.py`, `engine.py`, `progress.py`, `selection.py` | full-driver smoke test on the toy engine |
| Multi-GPU device pool | `core/parallel.py`, backend `engine.py` | per-backend device-dispatch tests; G=1 ≡ serial |
| Toy Wolfe–Quapp engine | `core/toy.py` | used throughout the suite |
| Strategy profiles | `core/strategy.py` | profile-resolution + guard tests |
| Data-driven CV (SPIB) | `cv/features.py`, `cv/spib.py` | 2-state recovery + driver integration |
| Non-linear search (RRT/Connect) | `search/rrt.py`, `core/policy.py` | RRT reaches the opposite WQ basin; Connect links both |
| Roadmap graph | `search/roadmap.py` | Dijkstra + Yen k-shortest correctness |
| Agentic controller | `agent/controller.py` | escalate/relax/stop/frontier rules |
| Weighted Ensemble | `sampling/weighted_ensemble.py` | weight conservation + WQ FES recovery |
| OPES (core + PLUMED interface) | `sampling/opes.py` | toy WQ marginal recovery; PLUMED input generation |
| Path sampling (TPS/TIS) bridge | `sampling/path_sampling.py` | seed prep / reactive-path extraction / interfaces (OPS run needs the `pathsampling` extra) |
| Downstream wiring | `sampling/runner.py`, backends | toy `run_downstream` test |

Run `pytest -q` to reproduce. SPIB tests need the `ml` extra; the OpenMM and
MDAnalysis tests need those packages (CI runs them in the `test-md` lane).

## Environment caveats (not runnable in a CPU-only sandbox)

These code paths are implemented and exercised through the shared, tested helpers,
but a live run needs hardware/binaries unavailable here:

- **Real-MD multi-GPU runs** of the AMBER (`pmemd.cuda`) and GROMACS (`gmx
  mdrun`) backends.
- **Downstream stages launched from a real backend** (WE/OPES on solvated systems
  via `pathgennie.downstream`).
- **OPES on real MD**, which needs a **PLUMED-patched engine** exposing
  `run_plumed` (see [opes.md](opes.md)).
- **TPS/TIS runs**, which need the `pathsampling` extra (`openpathsampling`) and
  an OPS engine; only the dependency-free seed preparation is CI-tested (see
  [path-sampling.md](path-sampling.md)).

## Planned / future work

- A `run_plumed` capability on the MD engines (PLUMED patch / `CustomCVForce`) so
  `OPESStage(mode="plumed")` runs end-to-end on real systems.
- RRT\* rewiring of the roadmap (edge cost = `-log(transition fraction)`) to bias
  toward minimum-free-energy tubes.
- A learning controller (contextual bandit / RL) and an optional LLM
  meta-controller behind the existing `Controller` surface.
- Process-pool OpenMM execution (one CUDA context per GPU) for in-process
  multi-GPU.
