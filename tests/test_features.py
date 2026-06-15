import numpy as np

from pathgennie.cv.features import (
    Featurizer,
    contact_features,
    dihedral_features,
    pairwise_distances,
)


def test_pairwise_distances():
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
    d = pairwise_distances(coords, [[0, 1], [0, 2]])
    np.testing.assert_allclose(d, [5.0, 1.0])


def test_contact_features_monotone():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    c = contact_features(coords, [[0, 1], [0, 2]], r0=8.0)
    assert c[0] > c[1]  # closer pair has the larger contact value
    assert 0.0 <= c[1] < 0.2


def test_dihedral_features_unit_circle():
    coords = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]], dtype=float)
    f = dihedral_features(coords, [[0, 1, 2, 3]])
    assert f.shape == (2,)
    np.testing.assert_allclose(f[0] ** 2 + f[1] ** 2, 1.0, atol=1e-6)  # sin^2+cos^2


def test_featurizer_standardize():
    rng = np.random.default_rng(0)
    batch = [rng.standard_normal((4, 3)) for _ in range(200)]
    feat = Featurizer(funcs=[], standardize=True).fit(batch)
    assert feat.n_features == 12
    transformed = feat.transform_batch(batch)
    assert transformed.shape == (200, 12)
    # Standardized columns should be ~zero mean, ~unit variance.
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=0.1)
    np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=0.1)
