# Artificial PCA distance-CV space (`pcagen`)

For host–guest binding/unbinding (and similar protein–ligand problems) a good
progress CV is often *not* a single distance but a low-dimensional combination
of many contact distances. PathGennie ships a utility that builds such a CV space
automatically: it generates many trial ligand conformations, computes their
receptor–ligand distance features, fits a PCA, and identifies the direction of
maximum separation — an **artificial PCA distance-CV space** you can then steer
along.

This is exposed both as a CLI (`pathgennie pcagen`) and programmatically
(`pathgennie.utils.ligcvgen.LigPCGen`).

> **Dependencies.** This utility needs `scikit-learn`, `matplotlib` and `joblib` on
> top of PathGennie's core dependencies. They ship as an extra:
>
> ```bash
> pip install 'pathgennie[analysis]'
> ```
>
> Without them `pathgennie pcagen` raises an error naming this command.

## CLI

```bash
pathgennie pcagen structure.gro \
    -n 10000 \                     # number of trial conformations to generate
    -o output_pca.pkl \            # saved PCA model (joblib)
    --protein_sel "protein" \      # MDAnalysis selection for the receptor/host
    --ligand_sel "resname LIG" \   # MDAnalysis selection for the ligand/guest
    -v 0.95 \                      # PCA variance threshold
    --around_distance 20.0         # only use protein atoms within this cutoff (Å)
```

It prints the dimension of maximum separation and saves the fitted PCA model to
`-o`, which you can load in your `projection.py` to project live coordinates into
the learned CV.

## Programmatic use

```python
from pathgennie.utils.ligcvgen import LigPCGen

gen = LigPCGen("structure.gro", protein_selection="protein",
               ligand_selection="resname LIG")
confs = gen.generate_conformations(n=10000)
# ... fit / inspect the PCA distance-CV space, then persist it for projection.
```

The selection strings fall back gracefully: if `--protein_sel` matches nothing,
the tool uses "everything that is not the ligand" as the receptor.

## When to use which CV

| Situation | Recommended CV |
| --------- | -------------- |
| A known 1–2 geometric coordinates (a distance, a dihedral) | hand-written `projection.py` (see [Configuration](configuration.md)) |
| Binding/unbinding where the right combination of contacts is unclear | **`pcagen`** artificial PCA distance-CV space |
| A known transition path you want to follow / stay on | [Path CVs `s`/`z`](path-cv.md) |
| No good CV at all; want the model to learn one during the run | [Data-driven CV (SPIB)](data-driven-cv.md) |
