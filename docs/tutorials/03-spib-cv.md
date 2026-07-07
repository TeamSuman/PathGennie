# Tutorial 03 — A learned CV with SPIB

Here we let PathGennie *learn* its progress variable on the fly with SPIB instead
of hand-crafting one. Requires the `ml` extra: `pip install -e .[ml]` (PyTorch).

## Drive PathGennie with `SPIBProgress`

`SPIBProgress` bootstraps from a coarse geometric CV, buffers the frames the
driver visits, retrains periodically, then steers using the learned latent.

```python
import numpy as np
from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.parallel import SerialExecutor
from pathgennie.core.progress import TargetMetric
from pathgennie.core.toy import ToyLangevinEngine
from pathgennie.cv.features import Featurizer
from pathgennie.cv.spib import SPIBProgress

engine = ToyLangevinEngine(dt=0.005)
initial = engine.create_state((-1.174, 1.477))
target = np.array([[1.124, -1.485, 0.0]])

xy = lambda c: np.array([c[0, 0], c[0, 1]])
bootstrap = TargetMetric(xy, target_cv=np.array([1.124, -1.485]))   # coarse CV

progress = SPIBProgress(
    Featurizer(funcs=[], standardize=False),   # raw coords as features (toy)
    bootstrap,
    mode="target", target_coords=target,
    refresh_every=15, min_frames=30, dt=1,
    train_kwargs=dict(n_states_init=3, latent_dim=1, epochs=15, n_refine=2, seed=0),
)

driver = PathGennieDriver(engine, progress, lambda c: False,
                          executor=SerialExecutor(), sigma=0.2, seed=1, verbosity=0)
traj, metrics = driver.run(initial, tau1=10, tau2=20, max_trial=6,
                           max_cycle=50, save_freq=5)

print("SPIB trained:", progress.result is not None)
print("emergent metastable states:", progress.n_states)
print("learned CV dim:", progress.project(engine.get_coords(initial)).shape)
```

The driver calls `progress.observe(coords, cycle)` each cycle (a no-op for static
CVs) — that's how SPIB accumulates frames and retrains. Once trained, `project`
returns the learned latent and the driver steers in that space.

## Train SPIB directly on a trajectory

```python
from pathgennie.cv.spib import train_spib
result = train_spib(features,           # (n_frames, n_features), time-ordered
                    dt=1, n_states_init=6, latent_dim=2, epochs=60, seed=0)
result.labels, result.n_states          # emergent metastable states
```

## Caveat: trajectory length

A learned CV needs segments long enough to contain real relaxation. In the
ultrashort `discovery` regime the
`check_learned_cv_segment_length` guard will warn — prefer the `sampling` profile
or longer `tau1`/`tau2`. See [strategy-profiles.md](../strategy-profiles.md).

The emergent `state_labels` feed the [roadmap graph](../roadmap-graph.md)
(tutorial 07).
