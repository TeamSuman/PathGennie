
import numpy as np


class PrincipalCurve:
    """
    Fast endpoint-pinned principal curve for 2D / 3D trajectories.
    """

    def __init__(
        self,
        n_images=50,
        lam=0.1,
        n_iter=50,
        tol=1e-5,
        verbose=False
    ):
        self.n_images = n_images
        self.lam = lam
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose

    @staticmethod
    def _arc_length(points):
        ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(ds)))
        if s[-1] > 0:
            s /= s[-1]
        return s

    def _init_path(self, points):
        s = self._arc_length(points)
        s_new = np.linspace(0.0, 1.0, self.n_images)

        path = np.vstack([
            np.interp(s_new, s, points[:, d])
            for d in range(points.shape[1])
        ]).T

        # pin endpoints
        path[0] = points[0]
        path[-1] = points[-1]
        return path

    @staticmethod
    def _project(points, path):
        """
        Vectorized nearest-node projection.
        """
        # squared distances: (N_points, N_images)
        d2 = np.sum(
            (points[:, None, :] - path[None, :, :]) ** 2,
            axis=2
        )
        return np.argmin(d2, axis=1)

    def _update_path(self, points, path, proj_idx):
        M, D = path.shape
        new_path = path.copy()

        # accumulate sums per node
        counts = np.bincount(proj_idx, minlength=M)
        sums = np.zeros((M, D))
        for d in range(D):
            sums[:, d] = np.bincount(
                proj_idx, weights=points[:, d], minlength=M
            )

        mask = counts > 0
        new_path[mask] = sums[mask] / counts[mask, None]

        # elastic smoothing (curvature penalty)
        new_path[1:-1] += self.lam * (
            path[:-2] + path[2:] - 2.0 * path[1:-1]
        )

        # endpoint pinning
        new_path[0] = path[0]
        new_path[-1] = path[-1]

        return new_path

    def _reparametrize(self, path):
        s = self._arc_length(path)
        s_new = np.linspace(0.0, 1.0, len(path))

        return np.vstack([
            np.interp(s_new, s, path[:, d])
            for d in range(path.shape[1])
        ]).T

    def fit(self, traj):
        """
        Fit principal curve to trajectory.

        Parameters
        ----------
        traj : ndarray, shape (T, D)
            Noisy trajectory in 2D or 3D

        Returns
        -------
        path : ndarray, shape (n_images, D)
            Smooth representative path
        """
        path = self._init_path(traj)

        for it in range(self.n_iter):
            old_path = path.copy()

            proj_idx = self._project(traj, path)
            path = self._update_path(traj, path, proj_idx)
            path = self._reparametrize(path)

            diff = np.max(np.linalg.norm(path - old_path, axis=1))
            if self.verbose:
                print(f"iter {it:03d} | max update = {diff:.3e}")

            if diff < self.tol:
                if self.verbose:
                    print("Converged.")
                break

        self.path_ = path
        return path
