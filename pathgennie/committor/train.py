"""Committor network trained on swarm records (PyTorch).

Self-consistency loss
---------------------
With ``tau_AB`` the first time the dynamics enters A or B, the committor is the
unique function with ``q = 0`` on A, ``q = 1`` on B and

    q(x) = E[ q(X_{min(tau, tau_AB)}) | X_0 = x ]          for x outside A and B.

A swarm gives ``K`` independent endpoints ``y_1..y_K`` for one start ``x``, so

    target(x) = (1/K) sum_k  qbar(y_k),     qbar = 0 in A, 1 in B, q_target elsewhere

is an unbiased estimate of the right-hand side, and the loss is

    L = mean_x ( q_theta(x) - target(x) )^2 .

``q_target`` is a slowly moving copy of the network (an exponential moving
average) through which no gradient flows. Because the loss is *linear* in the
target, its noise at finite ``K`` averages out: the minimiser is the conditional
mean, and the true committor is the fixed point. (That would not hold for a loss
that is non-linear in the target, such as a squared difference of logarithms,
and it would not hold if gradients flowed through both ends with ``K = 1``.)

Boundary information enters only through endpoints that land in A or B, and
spreads inward by one lag per effective sweep of the fixed-point iteration. Two
practical consequences:

* the data must contain endpoints in *both* basins (checked below);
* the lag should not be tiny compared with the time a trajectory in the
  transition region needs to commit, or many sweeps are needed.

What is learned is the committor of the *recorded* dynamics as a function of the
recorded features. With fresh velocities per trial and position-only features
that is the velocity-averaged position committor at the start, evaluated with a
position-only function at the end; the two agree when the lag exceeds the
velocity decorrelation time (always true for overdamped dynamics). For
underdamped MD with lags of a few fs, include velocity features or use longer
lags -- see docs/committor.md.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .basins import LABEL_A, LABEL_B, LABEL_NONE
from .records import SwarmDataset

__all__ = ["CommittorNet", "CommittorModel", "CommittorEnsemble", "train_committor",
           "train_ensemble", "committor_scores"]


class CommittorNet(nn.Module):
    """MLP from standardized features to the committor *logit*."""

    def __init__(self, n_features: int, hidden: Sequence[int] = (64, 64)):
        super().__init__()
        layers: List[nn.Module] = []
        width = int(n_features)
        for h in hidden:
            layers += [nn.Linear(width, int(h)), nn.SiLU()]
            width = int(h)
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.n_features = int(n_features)
        self.hidden = tuple(int(h) for h in hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class CommittorModel:
    """A trained network plus the feature standardization it was trained with."""

    net: CommittorNet
    feature_mean: np.ndarray
    feature_std: np.ndarray
    history: List[float] = field(default_factory=list)

    def _tensor(self, features: np.ndarray) -> torch.Tensor:
        x = (np.asarray(features, dtype=float) - self.feature_mean) / self.feature_std
        return torch.as_tensor(x, dtype=torch.float32)

    def logit(self, features: np.ndarray) -> np.ndarray:
        """Committor logit for ``(n, d)`` (or ``(d,)``) raw features.

        The logit, not ``q`` itself, is the quantity to use as a progress
        coordinate: it stays sensitive near the basins where ``q`` saturates.
        """
        single = np.ndim(features) == 1
        with torch.no_grad():
            out = self.net(self._tensor(np.atleast_2d(features))).numpy().astype(float)
        return float(out[0]) if single else out

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Committor ``q`` in [0, 1] for raw features."""
        z = self.logit(features)
        return np.exp(-np.logaddexp(0.0, -z))

    def save(self, path: "str | Path") -> None:
        torch.save(dict(state=self.net.state_dict(), n_features=self.net.n_features,
                        hidden=self.net.hidden, feature_mean=self.feature_mean,
                        feature_std=self.feature_std, history=self.history), path)

    @classmethod
    def load(cls, path: "str | Path") -> "CommittorModel":
        blob = torch.load(path, weights_only=False)
        net = CommittorNet(blob["n_features"], blob["hidden"])
        net.load_state_dict(blob["state"])
        net.eval()
        return cls(net=net, feature_mean=blob["feature_mean"], feature_std=blob["feature_std"],
                   history=list(blob["history"]))


def train_committor(
    dataset: SwarmDataset,
    *,
    hidden: Sequence[int] = (64, 64),
    steps: int = 4000,
    batch_size: int = 512,
    lr: float = 2e-3,
    ema: float = 0.98,
    weight_decay: float = 0.0,
    seed: int = 0,
    init: Optional[CommittorModel] = None,
    require_both_basins: bool = True,
    val_fraction: float = 0.0,
    patience: int = 10,
) -> CommittorModel:
    """Fit a committor network to swarm records.

    Parameters
    ----------
    dataset:
        Swarm records with basin labels (record them with ``basins=`` set).
        Swarms whose start already lies in a basin carry no information (the
        value there is known) and are dropped.
    steps, batch_size, lr:
        Adam optimisation settings; a batch is a set of swarms.
    ema:
        Decay of the target network. Closer to 1 is more stable but slower:
        boundary information travels about ``(1 - ema) * steps`` lags.
    init:
        Warm start from a previous model (its feature standardization is kept,
        so predictions stay comparable across refreshes of an adaptive run).
    require_both_basins:
        Raise if no endpoint reached A, or none reached B. Without both, the
        relation has no anchor on one side and the fit is not a committor.
    val_fraction, patience:
        Hold out this fraction of the swarms and evaluate the self-consistency
        loss on them every 100 steps; return the weights with the lowest held-out
        loss and stop after ``patience`` evaluations without improvement. Without
        it the network is trained for ``steps`` regardless, and with enough steps
        it interpolates its (noisy) targets exactly. The held-out loss is recorded
        in ``model.val_history``. Note what this does *not* catch: held-out swarms
        come from the same starts distribution, so a region the data do not cover
        stays invisible.
    """
    use = dataset.start_label == LABEL_NONE
    if not use.any():
        raise ValueError("no swarm starts outside the basins")
    starts = dataset.starts[use]
    ends = dataset.ends[use]
    mask = dataset.mask[use]
    end_label = dataset.end_label[use]

    n_a = int(((end_label == LABEL_A) & mask).sum())
    n_b = int(((end_label == LABEL_B) & mask).sum())
    if require_both_basins and (n_a == 0 or n_b == 0):
        raise ValueError(
            f"endpoints reached A {n_a} times and B {n_b} times; the committor needs "
            "boundary information from both basins. Add swarms near the missing basin "
            "or use longer segments.")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if init is not None:
        mean, std = init.feature_mean, init.feature_std
        net = copy.deepcopy(init.net)
        # A returned model is the frozen target copy; make it trainable again.
        for p in net.parameters():
            p.requires_grad_(True)
    else:
        mean = starts.mean(axis=0)
        std = starts.std(axis=0) + 1e-8
        net = CommittorNet(starts.shape[1], hidden)
    net.train()
    target_net = copy.deepcopy(net)
    for p in target_net.parameters():
        p.requires_grad_(False)

    x0 = torch.as_tensor((starts - mean) / std, dtype=torch.float32)
    # Padded entries are NaN; zero them (they are masked out of the average).
    x1 = torch.as_tensor(np.nan_to_num((ends - mean) / std), dtype=torch.float32)
    m = torch.as_tensor(mask, dtype=torch.float32)
    lab = torch.as_tensor(end_label.astype(np.int64))
    boundary = torch.where(lab == LABEL_B, 1.0, 0.0)
    committed = (lab != LABEL_NONE)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    n_all = x0.shape[0]
    val_idx = None
    if val_fraction > 0.0 and n_all >= 20:
        order = torch.as_tensor(rng.permutation(n_all))
        n_val = max(1, int(round(val_fraction * n_all)))
        val_idx, train_idx = order[:n_val], order[n_val:]
    else:
        train_idx = torch.arange(n_all)
    n = int(train_idx.numel())

    def swarm_loss(model, idx):
        with torch.no_grad():
            q_end = torch.sigmoid(target_net(x1[idx]))
            q_end = torch.where(committed[idx], boundary[idx], q_end)
            target = (q_end * m[idx]).sum(dim=1) / m[idx].sum(dim=1)
        return ((torch.sigmoid(model(x0[idx])) - target) ** 2).mean()

    best_val, best_state, stale = float("inf"), None, 0
    val_history: List[float] = []
    batch_size = min(int(batch_size), n)
    history: List[float] = []
    running = 0.0
    for step in range(int(steps)):
        idx = train_idx[torch.as_tensor(rng.choice(n, size=batch_size, replace=False))]
        loss = swarm_loss(net, idx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for p_t, p in zip(target_net.parameters(), net.parameters()):
                p_t.mul_(ema).add_(p, alpha=1.0 - ema)
        running += float(loss.detach())
        if (step + 1) % 100 == 0:
            history.append(running / 100.0)
            running = 0.0
            if val_idx is not None:
                with torch.no_grad():
                    val = float(swarm_loss(target_net, val_idx))
                val_history.append(val)
                if val < best_val - 1e-9:
                    best_val, stale = val, 0
                    best_state = copy.deepcopy(target_net.state_dict())
                else:
                    stale += 1
                    if stale >= patience:
                        break

    # The slowly moving average is the smoother estimate; return that one (at its
    # best held-out loss when a validation split was used).
    if best_state is not None:
        target_net.load_state_dict(best_state)
    target_net.eval()
    model = CommittorModel(net=target_net, feature_mean=mean, feature_std=std,
                           history=(list(init.history) if init is not None else []) + history)
    model.val_history = val_history
    return model


class CommittorEnsemble:
    """Several networks trained on the same data from different initialisations.

    The prediction is the sigmoid of the mean logit. The spread between members
    (:meth:`std`) is large where the data do not pin the committor down -- off
    the sampled tube, or in a channel only a few swarms have seen -- which makes
    it a cheap signal for where to launch the next swarms.
    """

    def __init__(self, models: Sequence[CommittorModel]):
        if not models:
            raise ValueError("an ensemble needs at least one model")
        self.models = list(models)

    def __len__(self) -> int:
        return len(self.models)

    def member_logits(self, features: np.ndarray) -> np.ndarray:
        return np.stack([np.atleast_1d(m.logit(features)) for m in self.models])

    def logit(self, features: np.ndarray):
        single = np.ndim(features) == 1
        out = self.member_logits(features).mean(axis=0)
        return float(out[0]) if single else out

    def predict(self, features: np.ndarray):
        z = self.logit(features)
        return np.exp(-np.logaddexp(0.0, -z))

    def std(self, features: np.ndarray):
        """Standard deviation of the members' committors (0 for a single model)."""
        single = np.ndim(features) == 1
        q = np.exp(-np.logaddexp(0.0, -self.member_logits(features)))
        out = q.std(axis=0)
        return float(out[0]) if single else out


def train_ensemble(dataset: SwarmDataset, n_models: int = 4, *, seed: int = 0,
                   init: Optional[CommittorEnsemble] = None, **kwargs) -> CommittorEnsemble:
    """Train ``n_models`` networks; ``init`` warm-starts each from its predecessor."""
    if init is not None and len(init) != n_models:
        raise ValueError("init must have n_models members")
    return CommittorEnsemble([
        train_committor(dataset, seed=seed + 7919 * k,
                        init=None if init is None else init.models[k], **kwargs)
        for k in range(int(n_models))
    ])


def committor_scores(predicted: np.ndarray, reference: np.ndarray,
                     transition_band: tuple = (0.05, 0.95)) -> dict:
    """Agreement between predicted and reference committors.

    ``pearson`` and ``rmse`` over all points are the two numbers the
    committor-learning literature reports. They flatter a model when most test
    points sit next to a basin (``q`` near 0 or 1 is easy), so the same two
    numbers are also given for the points whose *reference* value lies inside
    ``transition_band`` -- the region that decides mechanism and rate.
    """
    p = np.asarray(predicted, dtype=float).ravel()
    r = np.asarray(reference, dtype=float).ravel()
    if p.shape != r.shape:
        raise ValueError("predicted and reference must have the same shape")

    def pair(sel):
        if sel.sum() < 3:
            return float("nan"), float("nan")
        return (float(np.corrcoef(p[sel], r[sel])[0, 1]),
                float(np.sqrt(np.mean((p[sel] - r[sel]) ** 2))))

    everything = np.ones(p.size, dtype=bool)
    band = (r > transition_band[0]) & (r < transition_band[1])
    pearson, rmse = pair(everything)
    pearson_ts, rmse_ts = pair(band)
    return dict(pearson=pearson, rmse=rmse, n=int(p.size),
                pearson_transition=pearson_ts, rmse_transition=rmse_ts,
                n_transition=int(band.sum()))
