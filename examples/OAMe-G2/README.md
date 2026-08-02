# OAMe–G2 — host–guest unbinding with a PLUMED-computed CV

Guest dissociation from the octa-acid host, in `mode: escape`: no product state is named,
the swarm is driven away from the bound basin.

| directory | engine | `tau1`/`tau2` | notes |
|---|---|---|---|
| `gromacs/` | `gmx` + `NVT.mdp` | 5 / 10 | `topol.top`, `index.ndx`, `plumed.dat` |
| `openmm/` | OpenMM (in-process) | 20 / 20 | `system.prmtop`, plus analysis scripts |

Both run `max_cycle: 5000`, `max_trial: 10`.

## What makes this example different

The projection is not a plain geometric function — `plumed_cv.py::project` evaluates a
**PLUMED** CV on each frame and adds a *cooldown*: once the CV crosses 2.25 it suppresses
re-triggering for a set number of cycles. That is the pattern to copy when your progress
coordinate needs PLUMED, or when a raw CV would cause the selection to thrash near a
threshold.

It also means this example needs a working PLUMED alongside your MD engine, and the box
vectors are passed explicitly through the config (`box: [40.273098, ...]`) because the CV
needs them.

```bash
cd gromacs && python run_pg_gromacs.py
```

The OpenMM directory additionally ships `analyze_all.py` and `plot_openmm_traj.py` for
post-processing the resulting paths.
