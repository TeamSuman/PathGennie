# OPES (free-energy surfaces) via PLUMED

OPES (On-the-fly Probability Enhanced Sampling; Invernizzi & Parrinello, 2020)
deposits an adaptive bias along a CV so the system samples a flatter
distribution, from which the free-energy surface is recovered by reweighting. In
production this is driven by **PLUMED** (<https://www.plumed.org/>) patched into
the MD engine; PathGennie supplies the CV definition, OPES parameters, and seed
configurations from a `PathEnsemble`.

`pathgennie/sampling/opes.py` provides two layers.

## 1. PLUMED interface (production)

`build_plumed_opes_input` generates a `plumed.dat` with an `OPES_METAD` action,
and `OPESStage(mode="plumed")` drives a PLUMED-capable engine:

```python
from pathgennie.sampling.opes import build_plumed_opes_input, OPESStage

text = build_plumed_opes_input(
    ["phi: TORSION ATOMS=5,7,9,15", "psi: TORSION ATOMS=7,9,15,17"],
    ["phi", "psi"], pace=500, barrier=40.0, temp=300.0, sigma=[0.1, 0.1],
)

stage = OPESStage(mode="plumed",
                  plumed_cv_definitions=["phi: TORSION ATOMS=5,7,9,15"],
                  plumed_arg_names=["phi"], pace=500, barrier=40.0)
# stage.run(ensemble, engine) calls engine.run_plumed(plumed_input, ensemble, ...)
```

`OPESStage(mode="plumed")` requires the engine to expose
`run_plumed(plumed_input, ensemble, **cfg)`. The current MD engines do **not**
yet implement it, so a PLUMED-patched engine must be supplied; otherwise the stage
raises an informative `NotImplementedError`. This is the documented integration
point for AMBER/GROMACS/OpenMM + PLUMED.

## 2. Verifiable OPES core (toy)

So the OPES *algorithm* (not just the interface) is testable in CI, the module
includes a dependency-free implementation on an analytic potential:

- `OPESBias` — adaptive well-tempered kernel bias along a 1-D CV (the
  `OPES_METAD` core: kernels deposited with well-tempered heights; bias capped at
  `-barrier`).
- `OPESSimulation` — biased over-damped Langevin on a `grad_fn(pos)` potential,
  with reweighted-histogram FES recovery.

```python
import numpy as np
from pathgennie.core.toy import wolfe_quapp_gradient
from pathgennie.sampling.opes import OPESBias, OPESSimulation

bias = OPESBias(kT=2.0, gamma=15.0, sigma=0.2, barrier=8.0)
sim = OPESSimulation(wolfe_quapp_gradient, cv_axis=1, kT=2.0, dt=0.005, pace=20, bias=bias)
samples = sim.run(np.array([-1.0, -1.4]), n_steps=20000)
fe = sim.fes(samples, np.linspace(-2, 2, 33))   # recovers the WQ y-marginal
```

`OPESStage(mode="toy", potential_grad=..., cv_axis=...)` wraps this behind the
`SamplingStage` contract for end-to-end use on toy systems. The bias-update and
reweighting maths are identical to the production path; only the propagator
differs.

## From `input.yaml`

```yaml
pathgennie:
  downstream: opes
opes:
  mode: plumed
  plumed_cv_definitions: ["phi: TORSION ATOMS=5,7,9,15"]
  plumed_arg_names: ["phi"]
  pace: 500
  barrier: 40.0
```

This requires a PLUMED-capable engine (see above).
