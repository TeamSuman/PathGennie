# Alanine dipeptide — the same problem on all three backends

The standard conformational benchmark: drive the φ/ψ dihedrals from one basin to another.
The point of this example is that **one CV and one config schema work unchanged across
AMBER, GROMACS and OpenMM**, so it is the case to use when comparing backends.

| directory | engine | `max_cycle` | notes |
|---|---|---|---|
| `amber/` | `pmemd` | 5000 | `input.yaml` |
| `gromacs/` | `gmx` + `md.mdp` | 1000 | `input.yaml` |
| `openmm/` | OpenMM (in-process) | 1000 | also ships `input_escape.yaml` and `input_target.yaml` |

All three use `mode: target` with `target_projection: [60.0, 40.0]` (degrees) and
`tau1_steps: 2`, `tau2_steps: 4`, `max_trial: 10`.

```bash
cd amber && python run_pg_amber.py          # or: pathgennie amber --case .
```

Set `executable:` in the config if your MD binary is not on `PATH`. The OpenMM case needs
the `[openmm]` extra: `pip install 'pathgennie[openmm]'`.

## The `common/` directory

`common/` holds the shared system and the φ/ψ projection:

- `phi_psi.py` — `phi_psi_cv`, loaded by each backend's `projection.py`
- `generate_system.py` + `tleap.in` — rebuild the system from scratch and **copy the
  topology into each backend directory**. `common/ala_dipeptide.prmtop` is the canonical
  copy; the per-backend ones are derived from it, so edit the system here, not there.

Because `projection.py` resolves `common/` as a sibling, a backend directory copied
somewhere on its own will not import — copy the parent, or copy `common/` alongside it.
