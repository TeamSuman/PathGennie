import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class EnsemblePathRefinerFast:
    """
    Learn a smooth representative path from an ensemble of smooth trajectories.
    """

    def __init__(self, hidden_dim=64, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.model = None
        self.input_shape = None  # (N_atoms, 3)

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    @staticmethod
    def _arc_length(flat_path):
        seg = np.linalg.norm(np.diff(flat_path, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(seg)))
        return s / (s[-1] + 1e-12)

    @staticmethod
    def _resample_uniform_s(flat_path, n):
        s = EnsemblePathRefinerFast._arc_length(flat_path)
        s_target = np.linspace(0.0, 1.0, n)

        # Remove duplicate arc-length entries so interpolation stays well-defined
        s_unique, unique_idx = np.unique(s, return_index=True)
        flat_unique = flat_path[unique_idx]

        if len(s_unique) == 1:
            return s_target[:, None], np.repeat(flat_unique, n, axis=0)

        out = np.vstack([
            np.interp(s_target, s_unique, flat_unique[:, d])
            for d in range(flat_unique.shape[1])
        ]).T
        return s_target[:, None], out

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    def fit(
        self,
        trajectories,
        epochs=2000,
        lr=1e-3,
        batch_size=1024,
        samples_per_traj=100,
        start=None,
        end=None,
        patience=100,
        smoothness_weight=1e-2,
        smoothness_points=256,
        consensus_mode="median",
        verbosity=1,
    ):
        """
        trajectories: list of (T_i, N, 3) numpy arrays
        """

        # --------------------------------------------------
        # Endpoints
        # --------------------------------------------------
        avg_start = start if start is not None else np.mean([t[0] for t in trajectories], axis=0)
        avg_end   = end   if end   is not None else np.mean([t[-1] for t in trajectories], axis=0)

        self.input_shape = avg_start.shape
        dim = avg_start.size

        # --------------------------------------------------
        # Build dataset (uniform in arc length)
        # --------------------------------------------------
        aligned_paths = []
        s_target = None

        for traj in trajectories:
            flat = traj.reshape(traj.shape[0], -1)
            x_resampled, y_resampled = self._resample_uniform_s(flat, samples_per_traj)

            if s_target is None:
                s_target = x_resampled
            aligned_paths.append(y_resampled)

        aligned_paths = np.stack(aligned_paths, axis=0)

        if consensus_mode == "mean":
            consensus = np.mean(aligned_paths, axis=0)
        elif consensus_mode == "median":
            consensus = np.median(aligned_paths, axis=0)
        else:
            raise ValueError("consensus_mode must be 'mean' or 'median'.")

        spread = np.mean(np.linalg.norm(aligned_paths - consensus[None, :, :], axis=2), axis=0)
        point_weights = 1.0 / (spread + 1e-3)
        point_weights = point_weights / np.mean(point_weights)

        X = torch.from_numpy(s_target).float()
        Y = torch.from_numpy(consensus).float()
        W = torch.from_numpy(point_weights[:, None]).float()

        dataset = TensorDataset(X, Y, W)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True)

        # --------------------------------------------------
        # Model
        # --------------------------------------------------
        self.model = self._PathNet(dim, avg_start, avg_end, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        smooth_grid = torch.linspace(0.0, 1.0, smoothness_points, device=self.device).view(-1, 1)

        # --------------------------------------------------
        # Training loop
        # --------------------------------------------------
        best = np.inf
        stall = 0

        for epoch in range(epochs):
            loss_acc = 0.0
            data_acc = 0.0
            smooth_acc = 0.0
            self.model.train()

            for xb, yb, wb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                wb = wb.to(self.device)

                pred = self.model(xb)
                data_loss = torch.mean(wb * (pred - yb) ** 2)

                smooth_path = self.model(smooth_grid)
                second_diff = smooth_path[:-2] - 2.0 * smooth_path[1:-1] + smooth_path[2:]
                smooth_loss = torch.mean(second_diff ** 2)

                loss = data_loss + smoothness_weight * smooth_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                loss_acc += loss.item()
                data_acc += data_loss.item()
                smooth_acc += smooth_loss.item()

            loss_acc /= len(loader)
            data_acc /= len(loader)
            smooth_acc /= len(loader)

            if verbosity and epoch % 200 == 0:
                print(
                    f"Epoch {epoch:5d} | "
                    f"Loss {loss_acc:.3e} | "
                    f"Data {data_acc:.3e} | "
                    f"Smooth {smooth_acc:.3e}"
                )

            # early stopping
            if loss_acc < best - 1e-6:
                best = loss_acc
                stall = 0
            else:
                stall += 1
                if stall > patience:
                    if verbosity:
                        print(f"Early stopping at epoch {epoch}")
                    break

        return self

    # --------------------------------------------------
    # Generate representative path
    # --------------------------------------------------
    def transform(self, n_points=100, oversample=5):
        self.model.eval()

        M = n_points * oversample
        t = torch.linspace(0.0, 1.0, M, device=self.device).view(-1, 1)

        with torch.no_grad():
            raw = self.model(t).cpu().numpy()

        # reparametrize by arc length
        seg = np.linalg.norm(np.diff(raw, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(seg)))
        s /= s[-1]

        s_u = np.linspace(0.0, 1.0, n_points)
        out = np.vstack([
            np.interp(s_u, s, raw[:, d])
            for d in range(raw.shape[1])
        ]).T

        return out.reshape(n_points, *self.input_shape)

    # --------------------------------------------------
    # Network
    # --------------------------------------------------
    class _PathNet(nn.Module):
        def __init__(self, dim, start, end, hidden):
            super().__init__()

            self.register_buffer("start", torch.tensor(start.reshape(-1), dtype=torch.float32))
            self.register_buffer("end",   torch.tensor(end.reshape(-1),   dtype=torch.float32))

            self.net = nn.Sequential(
                nn.Linear(1, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, dim),
            )

        def forward(self, t):
            base = (1.0 - t) * self.start + t * self.end
            return base + t * (1.0 - t) * self.net(t)
