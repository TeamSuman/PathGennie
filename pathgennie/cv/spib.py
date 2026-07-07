"""State Predictive Information Bottleneck (SPIB) data-driven CV.

SPIB (Wang & Tiwary, *Nat. Commun.* 2021) learns a low-dimensional collective
variable **and** a set of metastable states at the same time, directly from
trajectory data, with no hand-crafted reaction coordinate.  It is the
"self-contained, on-the-fly CV" called for in the PathGennie roadmap.

Mechanism (predictive information bottleneck):

* an encoder ``q(z|x)`` maps a featurized configuration to a Gaussian latent
  ``z`` (the CV is its mean);
* a classifier ``p(y|z)`` predicts, from ``z_t``, the *future* metastable-state
  label ``y_{t+dt}``;
* training minimises ``CE(p(y|z_t), y_{t+dt}) + beta * KL(q(z|x)||prior)`` — the
  IB trade-off between predictiveness and compression;
* state labels are refined self-consistently: after each training round every
  frame is relabelled by ``argmax p(y|z)``, empty states are dropped, and the
  number of metastable states **emerges** from the data.

The learned encoder is exposed to the PathGennie driver through
:class:`SPIBProgress`, an adaptive :class:`~pathgennie.core.progress.ProgressVariable`
that bootstraps from a coarse geometric CV, buffers the frames the driver
visits, retrains periodically, and then steers using the learned latent — the
"iterative path-learning cycle".

This module requires PyTorch; importing it without torch raises ImportError.
The rest of :mod:`pathgennie.cv` (featurization) has no such dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError("pathgennie.cv.spib requires PyTorch (`pip install torch`)") from exc

from pathgennie.core.progress import ProgressVariable

__all__ = ["SPIB", "train_spib", "SPIBResult", "kmeans_labels", "SPIBProgress"]


# --------------------------------------------------------------------------- #
# Lightweight k-means for label initialisation (avoids a scikit-learn dep).
# --------------------------------------------------------------------------- #
def kmeans_labels(x: np.ndarray, k: int, *, iters: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    k = min(k, n)
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = x[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)
    return labels


class SPIB(nn.Module):
    def __init__(self, n_features: int, n_states: int, latent_dim: int = 2, hidden: Sequence[int] = (64, 64)):
        super().__init__()
        layers: List[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.to_mean = nn.Linear(prev, latent_dim)
        self.to_logvar = nn.Linear(prev, latent_dim)
        self.classifier = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, n_states))
        self.latent_dim = latent_dim
        self.n_states = n_states

    def encode(self, x):
        h = self.encoder(x)
        return self.to_mean(h), self.to_logvar(h)

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.classifier(z), mean, logvar


@dataclass
class SPIBResult:
    model: SPIB
    labels: np.ndarray
    n_states: int
    feature_mean: np.ndarray = field(repr=False)
    feature_std: np.ndarray = field(repr=False)


def _kl_standard_normal(mean, logvar):
    # KL(N(mean, var) || N(0, I)) per sample.
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)


def train_spib(
    features: np.ndarray,
    *,
    dt: int = 1,
    n_states_init: int = 6,
    latent_dim: int = 2,
    beta: float = 1e-3,
    epochs: int = 60,
    lr: float = 1e-3,
    n_refine: int = 6,
    batch_size: int = 256,
    seed: int = 0,
) -> SPIBResult:
    """Train SPIB on a time-ordered ``(n_frames, n_features)`` array.

    Returns the trained model plus the converged per-frame state labels and the
    emergent number of states.
    """

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    features = np.asarray(features, dtype=np.float64)
    n = features.shape[0]
    if n <= dt + 1:
        raise ValueError("not enough frames for the requested lag time")

    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    x_std = (features - mean) / std
    x_all = torch.tensor(x_std, dtype=torch.float32)
    x_t = x_all[:-dt]
    fut_index = np.arange(dt, n)  # future frame index for each (t) sample

    labels = kmeans_labels(x_std, n_states_init, seed=seed)
    n_states = int(labels.max()) + 1

    model: Optional[SPIB] = None
    for _ in range(n_refine):
        model = SPIB(features.shape[1], n_states, latent_dim=latent_dim)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        y_future = torch.tensor(labels[fut_index], dtype=torch.long)

        idx = np.arange(x_t.shape[0])
        for _epoch in range(epochs):
            rng.shuffle(idx)
            for start in range(0, len(idx), batch_size):
                batch = idx[start : start + batch_size]
                xb = x_t[batch]
                yb = y_future[batch]
                logits, mu, logvar = model(xb)
                loss = F.cross_entropy(logits, yb) + beta * _kl_standard_normal(mu, logvar).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

        # Self-consistent relabelling: each frame -> argmax p(y | mean(z)).
        model.eval()
        with torch.no_grad():
            mu_all, _ = model.encode(x_all)
            new_labels = model.classifier(mu_all).argmax(dim=1).numpy()

        # Drop empty states and remap to a contiguous range.
        used = np.unique(new_labels)
        remap = {old: new for new, old in enumerate(used)}
        new_labels = np.array([remap[v] for v in new_labels], dtype=int)
        new_n_states = len(used)

        converged = new_n_states == n_states and np.array_equal(new_labels, labels)
        labels, n_states = new_labels, new_n_states
        if converged or n_states <= 1:
            break

    assert model is not None
    model.eval()
    return SPIBResult(model=model, labels=labels, n_states=n_states, feature_mean=mean, feature_std=std)


class SPIBProgress(ProgressVariable):
    """Adaptive progress variable backed by an on-the-fly SPIB model.

    Until enough frames are buffered the object delegates to a coarse
    ``bootstrap`` progress variable; once trained it steers using the learned
    latent (escape: maximise distance from the start latent; target: minimise
    distance to the target latent).  ``observe`` is called by the driver each
    cycle to grow the frame buffer and trigger periodic retraining.
    """

    def __init__(
        self,
        featurizer,
        bootstrap: ProgressVariable,
        *,
        mode: str = "escape",
        target_coords: Optional[np.ndarray] = None,
        refresh_every: int = 50,
        min_frames: int = 40,
        dt: int = 1,
        train_kwargs: Optional[dict] = None,
    ):
        if mode not in ("escape", "target"):
            raise ValueError("mode must be 'escape' or 'target'")
        if mode == "target" and target_coords is None:
            raise ValueError("target_coords required for target mode")
        self.featurizer = featurizer
        self.bootstrap = bootstrap
        self.mode = mode
        self.target_coords = None if target_coords is None else np.asarray(target_coords, dtype=float)
        self.refresh_every = int(refresh_every)
        self.min_frames = int(min_frames)
        self.dt = int(dt)
        self.train_kwargs = dict(train_kwargs or {})
        self._buffer: List[np.ndarray] = []
        self.result: Optional[SPIBResult] = None
        self._z_start: Optional[np.ndarray] = None
        self._z_target: Optional[np.ndarray] = None
        self._last_refresh = -1

    # -- driver hook ---------------------------------------------------------
    def observe(self, coords: np.ndarray, cycle: int) -> None:
        self._buffer.append(np.asarray(coords, dtype=float).copy())
        due = (cycle - self._last_refresh) >= self.refresh_every or self.result is None
        if due and len(self._buffer) >= self.min_frames:
            self._refresh()
            self._last_refresh = cycle

    def _encode(self, coords: np.ndarray) -> np.ndarray:
        assert self.result is not None
        x = (self.featurizer.raw(coords) - self.result.feature_mean) / self.result.feature_std
        with torch.no_grad():
            mu, _ = self.result.model.encode(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
        return mu.squeeze(0).numpy()

    def _refresh(self) -> None:
        features = np.stack([self.featurizer.raw(c) for c in self._buffer])
        if features.shape[0] <= self.dt + 1:
            return
        self.result = train_spib(features, dt=self.dt, **self.train_kwargs)
        self._z_start = self._encode(self._buffer[0])
        if self.mode == "target":
            self._z_target = self._encode(self.target_coords)

    # -- ProgressVariable ----------------------------------------------------
    def project(self, coords: np.ndarray, cycle: Optional[int] = None) -> np.ndarray:
        # ``cycle`` is part of the ProgressVariable protocol -- the driver always
        # calls ``project(coords, cycle=cycle)``. Accept it so on-the-fly SPIB
        # does not raise ``TypeError: project() got an unexpected keyword
        # argument 'cycle'`` on the very first evaluation. The learned encoder is
        # stationary within a cycle, so the value is only forwarded to the
        # bootstrap CV (which may be cycle-dependent).
        if self.result is None:
            return np.asarray(self.bootstrap.project(coords, cycle=cycle), dtype=float)
        return self._encode(coords)

    def metric(self, cv: np.ndarray) -> float:
        if self.result is None:
            return float(self.bootstrap.metric(cv))
        cv = np.asarray(cv, dtype=float)
        if self.mode == "escape":
            return float(np.linalg.norm(cv - self._z_start))
        return float(-np.linalg.norm(cv - self._z_target))

    @property
    def n_states(self) -> Optional[int]:
        return None if self.result is None else self.result.n_states

    @property
    def state_labels(self) -> Optional[np.ndarray]:
        return None if self.result is None else self.result.labels
