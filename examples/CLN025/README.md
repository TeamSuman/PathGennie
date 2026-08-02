# CLN025 (chignolin) — folding on an end-to-end CV

Folding of the 10-residue chignolin variant CLN025, driven on the Cα–Cα end-to-end
distance between residues 1 and 10 (`end_to_end_cv`, atom indices 4 and 132).

| directory | engine | `max_cycle` | configs |
|---|---|---|---|
| `amber/` | `pmemd` | 5000 | `input.yaml`, `input_target.yaml`, `input_escape.yaml` |
| `gromacs/` | `gmx` + `md.mdp` | 1000 | `input.yaml` |

Both use `tau1_steps: 5`, `tau2_steps: 10`, `max_trial: 10`. The default is
`mode: target` with `target_projection: [5.0]` — an end-to-end distance of 5 Å, i.e. the
folded state, starting from `unfolded.rst7` / `unfolded.gro`.

```bash
cd amber && python run_pg_amber.py
```

The AMBER directory also ships `input_escape.yaml`, which runs the *unbiased-direction*
problem: leave the starting basin without naming a target. Useful for seeing how `escape`
and `target` modes differ on an identical system.

This is a real folding problem, not a smoke test — expect it to be the longest-running
example here. Reduce `max_cycle` for a trial run.
