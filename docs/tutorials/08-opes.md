# Tutorial 08 — Free-energy surfaces with OPES

OPES deposits an adaptive bias along a CV to flatten sampling and recover a
free-energy surface. On real systems this runs through **PLUMED**; the algorithm
core is also runnable here on the toy surface.

## Toy mode (runnable now)

```python
import numpy as np
from pathgennie.core.toy import ToyLangevinEngine, wolfe_quapp_gradient
from pathgennie.sampling import build_path_ensemble
from pathgennie.sampling.opes import OPESStage

engine = ToyLangevinEngine(dt=0.005, kT=2.0)
initial = engine.create_state((-1.0, -1.4))
ens = build_path_ensemble(np.array([engine.get_coords(initial)]), np.array([0.0]),
                          cv_fn=lambda c: c[0, 1])

stage = OPESStage(
    mode="toy", potential_grad=wolfe_quapp_gradient, cv_axis=1,
    grid=np.linspace(-2.0, 2.0, 33), n_steps=20000, pace=20,
    gamma=15.0, sigma=0.2, barrier=8.0, kT=2.0, seed=0,
)
result = stage.run(ens, engine)
for y, f in zip(result.metadata["grid"], result.free_energy):
    print(f"y={y:+.2f}  F={f:.2f}")
```

The recovered `F(y)` matches the analytic Wolfe–Quapp marginal (FES minima in the
basins, barrier in between).

## PLUMED mode (production)

`OPESStage(mode="plumed")` generates an `OPES_METAD` input and drives a
PLUMED-capable engine:

```python
from pathgennie.sampling.opes import build_plumed_opes_input

print(build_plumed_opes_input(
    ["phi: TORSION ATOMS=5,7,9,15", "psi: TORSION ATOMS=7,9,15,17"],
    ["phi", "psi"], pace=500, barrier=40.0, temp=300.0, sigma=[0.1, 0.1],
))
```

To run it end-to-end the engine must expose `run_plumed(plumed_input, ensemble,
**cfg)` (a PLUMED-patched MD engine). Without it the stage raises an informative
`NotImplementedError`. From `input.yaml`:

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

See [opes.md](../opes.md) for the full reference and the PLUMED integration point.
