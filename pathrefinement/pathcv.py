import warnings

import numpy as np

warnings.filterwarnings("ignore")


class PathCV:
    """
    Dimension-agnostic Path Collective Variables (s, z)
    following Branduardi et al., JCP 126, 054103 (2007).
    """

    def __init__(
        self,
        frames,
        *,
        mass_weights=None,
        enforce_equidistance=True,
        equidistance_tol=0.25,
        lambda_mode="auto",
        lambda_value=None,
        normalize_output=False,
    ):
        self.frames = [np.asarray(f) for f in frames]
        self.P = len(frames)
        self.normalize_output = normalize_output

        if self.P < 2:
            raise ValueError("Path must contain at least two frames.")

        # Infer dimensionality
        self.n_entities, self.dim = self.frames[0].shape

        for f in self.frames:
            if f.shape != (self.n_entities, self.dim):
                raise ValueError("All frames must have identical shape.")

        # Mass weighting
        if mass_weights is not None:
            mass_weights = np.asarray(mass_weights)
            if mass_weights.shape != (self.n_entities,):
                raise ValueError("mass_weights must have shape (N_entities,)")
            self.mass_weights = mass_weights / np.mean(mass_weights)
        else:
            self.mass_weights = None

        # Reference path: (P, N, D)
        self.reference_path = np.stack(self.frames, axis=0)

        # Equidistance check
        self._check_equidistance(
            enforce=enforce_equidistance,
            tol=equidistance_tol,
        )

        # Lambda
        if lambda_mode == "auto":
            self.lam = self._compute_lambda()
        elif lambda_mode == "manual":
            if lambda_value is None or lambda_value <= 0:
                raise ValueError("lambda_value must be positive.")
            self.lam = float(lambda_value)
        else:
            raise ValueError("lambda_mode must be 'auto' or 'manual'.")

    @classmethod
    def from_2d_path(cls, xy_path: np.ndarray, **kwargs):
        """Construct PathCV from a 2D path shape (N, 2)."""
        xy_path = np.asarray(xy_path)
        if xy_path.ndim != 2 or xy_path.shape[1] != 2:
            raise ValueError("xy_path must have shape (N, 2)")
        frames = [pt[np.newaxis, :] for pt in xy_path]
        return cls(frames, **kwargs)

    def __repr__(self) -> str:
        return f"PathCV(nodes={self.P}, entities={self.n_entities}, dim={self.dim}, lambda={self.lam:.3f})"

    # --------------------------------------------------
    # Internal utilities
    # --------------------------------------------------

    def _msd(self, diff):
        """
        Mean square displacement for arbitrary dimension.
        diff shape: (N, D)
        """
        sq = np.sum(diff ** 2, axis=-1)
        if self.mass_weights is not None:
            sq *= self.mass_weights
        return np.mean(sq)

    def _check_equidistance(self, *, enforce, tol):
        diffs = self.reference_path[1:] - self.reference_path[:-1]
        msds = np.array([self._msd(d) for d in diffs])

        mean = np.mean(msds)
        std = np.std(msds)

        if mean == 0:
            raise ValueError("Degenerate path: identical frames.")

        rel = std / mean

        if rel > tol:
            msg = (
                f"Path nodes are not equidistant (std/mean = {rel:.2f}). "
                "This violates assumptions of Branduardi et al."
            )
            if enforce:
                raise ValueError(msg)
            else:
                print("WARNING:", msg)

        self._mean_segment_msd = mean

    def _compute_lambda(self):
        """
        PLUMED-style automatic lambda.
        """
        return 2.3 / self._mean_segment_msd

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def compute(self, coords):
        """
        Compute (s, z) for a configuration.

        coords shape: (N_entities, D)
        """
        coords = np.asarray(coords)

        if coords.shape != (self.n_entities, self.dim):
            raise ValueError("coords shape mismatch.")

        # Vectorized MSD to all path nodes
        diff = coords - self.reference_path  # shape: (P, N_entities, D)
        sq = np.sum(diff ** 2, axis=-1)      # shape: (P, N_entities)
        if self.mass_weights is not None:
            sq *= self.mass_weights
        msd = np.mean(sq, axis=-1)           # shape: (P,)

        exponents = -self.lam * msd
        max_exp = np.max(exponents)
        weights = np.exp(exponents - max_exp)

        Z = np.sum(weights)

        indices = np.arange(1, self.P + 1)
        s = np.sum(indices * weights) / Z

        if self.normalize_output:
            s = (s - 1.0) / (self.P - 1.0)

        z = -(1.0 / self.lam) * (np.log(Z) + max_exp)

        return float(s), float(z)

    def compute_rms_distance(self, coords):
        """
        RMS distance to path (soft).
        """
        _, z = self.compute(coords)
        return np.sqrt(max(z, 0.0))
