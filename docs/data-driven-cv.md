# Data-driven CVs with SPIB

Instead of a hand-crafted progress variable, PathGennie can learn one on the fly
with **SPIB** (State Predictive Information Bottleneck; Wang & Tiwary, 2021),
which jointly learns a low-dimensional CV **and** an emergent set of metastable
states from the frames a run visits. Requires the `ml` extra (`pip install -e
.[ml]`, i.e. PyTorch).

## How SPIB works

A predictive information bottleneck:

1. an encoder `q(z|x)` maps a featurized configuration to a Gaussian latent `z`
   (the CV is its mean);
2. a classifier `p(y|z)` predicts, from `z_t`, the *future* metastable-state
   label `y_{t+dt}`;
3. training minimises `CE(p(y|z_t), y_{t+dt}) + β·KL(q(z|x)||prior)` — the IB
   trade-off between being predictive and being compressed;
4. state labels are refined self-consistently: after each round every frame is
   relabelled by `argmax p(y|z)`, empty states are dropped, and the **number of
   metastable states emerges from the data**.

## Featurization (`pathgennie/cv/features.py`, NumPy)

```python
from pathgennie.cv.features import Featurizer, pairwise_distances, contact_features, dihedral_features

feat = Featurizer(funcs=[
    lambda c: pairwise_distances(c, [[4, 6], [6, 8]]),
    lambda c: dihedral_features(c, [[4, 6, 8, 14]]),
], standardize=True)
feat.fit(frames)            # accumulate online mean/std
X = feat.transform_batch(frames)
```

`Featurizer` with no `funcs` uses the flattened coordinates (handy for the toy
system). Featurization is pure NumPy and has no heavy dependency.

## Training a model directly

```python
from pathgennie.cv.spib import train_spib
result = train_spib(features,            # (n_frames, n_features), time-ordered
                    dt=1, n_states_init=6, latent_dim=2, beta=1e-3,
                    epochs=60, n_refine=6, seed=0)
result.model        # SPIB encoder/classifier
result.labels       # per-frame metastable-state ids
result.n_states     # emergent number of states
```

## Driving PathGennie with a learned CV

`SPIBProgress` is an adaptive `ProgressVariable`: it bootstraps from a coarse
geometric CV, buffers the path the driver visits, retrains periodically (the
"iterative path-learning cycle"), then steers using the learned latent.

```python
from pathgennie.cv.features import Featurizer
from pathgennie.cv.spib import SPIBProgress
from pathgennie.core.progress import TargetMetric

bootstrap = TargetMetric(my_geometric_cv, target_cv=target)
progress = SPIBProgress(
    Featurizer(funcs=my_feature_fns),
    bootstrap,
    mode="target", target_coords=target_coords,
    refresh_every=50, min_frames=40, dt=1,
    train_kwargs=dict(n_states_init=6, latent_dim=2, epochs=40),
)
# pass `progress` to PathGennieDriver as usual
```

The driver calls `progress.observe(coords, cycle)` once per cycle (a no-op for
static CVs), which is how SPIB accumulates frames and retrains. After a refresh,
`progress.n_states` and `progress.state_labels` expose the metastable states —
these feed the [roadmap graph](roadmap-graph.md).

## Trajectory length matters

A learned CV needs segments long enough to contain real relaxation. If you are in
the ultrashort `discovery` regime, see the guard in
[strategy-profiles.md](strategy-profiles.md); prefer the `sampling` profile (or
longer `tau1`/`tau2`) when learning a CV for quantitative work.
