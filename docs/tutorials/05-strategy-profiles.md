# Tutorial 05 — Goal-driven strategy profiles

The right settings depend on your *goal*: fast candidate paths want greedy,
ultrashort, geometric-CV runs; quantitative free energies/kinetics want longer
segments, a learned CV, and a downstream stage. A `profile` switches the whole
bundle with one key.

## In `input.yaml`

```yaml
pathgennie:
  profile: discovery      # greedy, ultrashort, geometric CV (the original regime)
  devices: [0, 1]         # explicit keys still override / extend the profile
```

or

```yaml
pathgennie:
  profile: sampling       # longer tau, learned CV, feeds weighted ensemble
  seed: 7
```

Profile values fill in only where you did **not** set a key — explicit always
wins, and omitting `profile` leaves old configs unchanged.

## In Python

```python
from pathgennie.core.strategy import resolve_profile, get_profile, PROFILES

print(list(PROFILES))                      # ['discovery', 'sampling']

cfg = resolve_profile({"profile": "sampling", "tau1_steps": 30})
print(cfg["tau1_steps"])   # 30  (explicit wins)
print(cfg["cv"])           # 'learned'  (from the profile)
print(cfg["downstream"])   # 'weighted_ensemble'

get_profile("discovery")   # the RunProfile dataclass with all defaults
```

## The learned-CV guard

If you pick a learned CV but keep ultrashort segments, the guard warns:

```python
from pathgennie.core.strategy import check_learned_cv_segment_length
check_learned_cv_segment_length(2, 8, timestep_ps=0.002)    # warns, returns False
check_learned_cv_segment_length(50, 100, timestep_ps=0.002) # True
```

See [strategy-profiles.md](../strategy-profiles.md) for the full table of preset
values.
