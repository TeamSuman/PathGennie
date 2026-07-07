# Path Collective Variables (s, z)

`pathrefinement.PathCV` implements the **Branduardi path collective variables**
(Branduardi, Gervasio & Parrinello, *J. Chem. Phys.* **126**, 054103 (2007)):
given a reference path of `P` frames, it maps any configuration to two numbers,

- **`s`** — progress *along* the path (which image you are near, in `[1, P]` or,
  optionally, normalised to `[0, 1]`), and
- **`z`** — the squared distance *off* the path (how far the configuration has
  strayed from the reference tube).

`s` and `z` are the natural CVs for steering, refining, and biasing a transition:
`s` drives A→B progress while `z` keeps the system on the discovered channel.

## API

```python
from pathrefinement import PathCV
import numpy as np

# reference path: a list/array of P frames, each (N_entities, D)
frames = [frame_0, frame_1, ..., frame_P]        # e.g. C-alpha coords, or a 2-D toy path
pcv = PathCV(
    frames,
    mass_weights=None,          # optional (N_entities,) weights
    enforce_equidistance=True,  # check the reference images are ~evenly spaced
    equidistance_tol=0.25,
    lambda_mode="auto",         # auto-select λ from the mean inter-frame MSD
    normalize_output=False,     # True -> s in [0, 1] instead of [1, P]
)

s, z = pcv.compute(coords)      # coords: (N_entities, D)
```

`PathCV` is **dimension-agnostic**: `N_entities × D` can be atoms × 3, CA beads ×
3, or points in a 2-D toy space. For a plain 2-D path there is a convenience
constructor:

```python
pcv = PathCV.from_2d_path(xy_path)   # xy_path: (P, 2)
```

Key details (all handled internally):

- **Numerical stability** — `z` is computed with a log-sum-exp shift so the
  `exp(-λ·MSD)` sum never overflows/underflows for long paths.
- **λ selection** — with `lambda_mode="auto"`, λ ≈ `2.3 / mean(MSD between
  adjacent frames)`, the Branduardi rule of thumb; pass `lambda_mode="manual",
  lambda_value=...` to override.
- **Equidistance** — unevenly spaced reference images distort `s`; the
  constructor warns (or, with `enforce_equidistance=True`, checks within
  `equidistance_tol`). Use a resampled/refined path (see
  [Path Refinement](path-refinement.md)) for best results.

## Using `s` as a PathGennie progress CV

Because PathGennie's projection is just a Python function returning a CV vector,
you can steer along a path by returning `s` (and optionally `z`) from your
`projection.py`:

```python
# projection.py
import numpy as np
from pathrefinement import PathCV

_PCV = PathCV(load_reference_frames())   # build once at import

def path_progress_cv(coords, **kwargs):
    s, z = _PCV.compute(select_cv_atoms(coords))
    return np.array([s, z])
```

Then point `input.yaml` at it and drive `s` toward the product image in
`target` mode:

```yaml
pathgennie:
    mode: target
    target_projection: [ P, 0.0 ]   # last image, on the path (z≈0)
projection:
    module: projection
    function: path_progress_cv
```

See also [Path Refinement](path-refinement.md), which builds and iteratively
improves the reference path that `PathCV` consumes.
