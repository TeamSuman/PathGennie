"""Data-driven collective variables for PathGennie.

``features`` (pure NumPy) turns coordinates into invariant feature vectors; the
optional ``spib`` module (requires PyTorch) learns a low-dimensional CV *and* a
set of metastable states from accumulated swarm frames, exposed to the driver as
an adaptive :class:`~pathgennie.core.progress.ProgressVariable`.
"""

from .features import Featurizer, contact_features, dihedral_features, pairwise_distances

__all__ = [
    "Featurizer",
    "pairwise_distances",
    "contact_features",
    "dihedral_features",
]
