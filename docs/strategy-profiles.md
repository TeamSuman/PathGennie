# Strategy profiles

PathGennie's original regime — greedy selection on **ultrashort** (tens of fs)
trajectories — is tuned for *fast candidate-path discovery*. A learned CV (SPIB)
and downstream free-energy/kinetics work generally need **longer** segments and
more data. Rather than hard-code one regime, a `RunProfile` bundles the choices
that should move together for a goal, and you switch behaviour with a single key.

## Presets (`pathgennie/core/strategy.py`)

| Profile | goal | selection | cv | tau1 | tau2 | max_trial | sigma | downstream |
|---|---|---|---|---|---|---|---|---|
| `discovery` | discovery | softmax | geometric | 2 | 8 | 15 | 0.1 | none |
| `sampling` | sampling | beam | learned | 50 | 100 | 20 | 0.2 | weighted_ensemble |

- **`discovery`** — the original PathGennie: cheap, greedy, geometric CV. Best for
  quickly enumerating candidate routes to seed later sampling.
- **`sampling`** — longer segments, a learned CV is permitted, and the run is set
  up to feed an enhanced-sampling stage. Best when the goal is a quantitative FES
  or rate.

## How resolution works

`resolve_profile(pg_cfg)` overlays the named profile's values **underneath**
whatever you set explicitly — explicit always wins, and omitting `profile`
returns the config unchanged (so old cases are unaffected). All three backends
apply it.

```yaml
pathgennie:
  profile: discovery     # fills tau1/tau2/max_trial/sigma/selection/cv/downstream
  max_trial: 24          # ...but this explicit value overrides the profile's 15
  devices: [0, 1, 2, 3]
```

In Python:

```python
from pathgennie.core.strategy import resolve_profile, get_profile
cfg = resolve_profile({"profile": "sampling", "tau1_steps": 30})
# cfg["tau1_steps"] == 30 (explicit), cfg["cv"] == "learned" (from profile)
```

## The learned-CV trajectory-length guard

A learned CV needs trajectories long enough to contain real relaxation.
`check_learned_cv_segment_length(tau1, tau2, timestep_ps, min_ps=0.1)` warns (and
returns `False`) when the per-cycle MD time `(tau1+tau2)·dt` is below `min_ps`,
flagging the ultrashort-vs-learned-CV mismatch the discovery regime can introduce:

```python
from pathgennie.core.strategy import check_learned_cv_segment_length
check_learned_cv_segment_length(2, 8, timestep_ps=0.002)   # -> warns, returns False
check_learned_cv_segment_length(50, 100, timestep_ps=0.002) # -> True
```

It is advisory — the run still proceeds — so you stay in control of the trade-off.
