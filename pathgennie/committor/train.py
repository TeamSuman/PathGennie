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

__all__ = ["CommittorNet", "CommittorModel", "train_committor", "committor_scores"]


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
        return 1.0 / (1.0 + np.exp(-z))

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
    n = x0.shape[0]
    batch_size = min(int(batch_size), n)
    history: List[float] = []
    running = 0.0
    for step in range(int(steps)):
        idx = torch.as_tensor(rng.choice(n, size=batch_size, replace=False))
        with torch.no_grad():
            q_end = torch.sigmoid(target_net(x1[idx]))
            q_end = torch.where(committed[idx], boundary[idx], q_end)
            target = (q_end * m[idx]).sum(dim=1) / m[idx].sum(dim=1)
        loss = ((torch.sigmoid(net(x0[idx])) - target) ** 2).mean()
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

    # The slowly moving average is the smoother estimate; return that one.
    target_net.eval()
    return CommittorModel(net=target_net, feature_mean=mean, feature_std=std,
                          history=(list(init.history) if init is not None else []) + history)


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
