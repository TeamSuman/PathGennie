"""OpenPathSampling (OPS) bridge: TPS / TIS on PathGennie seed pathways.

PathGennie discovers reactive A->B pathways cheaply. Transition Path Sampling
(TPS) and Transition Interface Sampling (TIS), as implemented by
**OpenPathSampling** (https://openpathsampling.org/), need an *initial reactive
trajectory* to bootstrap — exactly what a PathGennie run produces. This module is
the bridge, an alternative to Weighted Ensemble for computing kinetics.

Two layers, mirroring the OPES module:

* **Dependency-free, CI-verified prep** — turn a discovered
  :class:`~pathgennie.sampling.base.PathEnsemble` into an OPS-ready seed: define
  states as CV ranges (:class:`CVRangeState`), label frames, extract the minimal
  reactive sub-path (:func:`extract_transition_path`), and lay out TIS interfaces
  (:func:`tis_interfaces`). This is pure NumPy and always available.
* **OPS-dependent stage** — :class:`PathSamplingStage` builds the OPS CV, state
  volumes, TPS/TIS network and move scheme, seeds them with the prepared
  trajectory, runs ``PathSampling``, and returns kinetics in a
  :class:`~pathgennie.sampling.base.SamplingResult`. It lazily imports
  ``openpathsampling`` and requires an OPS-compatible engine (OPS propagates with
  its *own* engine, e.g. ``openpathsampling.engines.openmm``); without either it
  raises with guidance. Install via the ``pathsampling`` extra.

Targets the OpenPathSampling 1.x API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .base import PathEnsemble, SamplingResult

__all__ = [
    "CVRangeState",
    "label_frames",
    "extract_transition_path",
    "is_reactive",
    "tis_interfaces",
    "prepare_ops_seed",
    "PathSamplingStage",
]

# Frame state labels.
NO_STATE = -1
STATE_A = 0
STATE_B = 1


@dataclass
class CVRangeState:
    """A state defined as a closed interval ``[lo, hi]`` of a scalar CV."""

    name: str
    lo: float
    hi: float

    def contains(self, cv_value: float) -> bool:
        return self.lo <= float(cv_value) <= self.hi


# --------------------------------------------------------------------------- #
# Dependency-free seed preparation (always available, unit-tested)
# --------------------------------------------------------------------------- #
def label_frames(
    frames: np.ndarray,
    cv_fn: Callable[[np.ndarray], float],
    state_a: CVRangeState,
    state_b: CVRangeState,
) -> np.ndarray:
    """Label each frame ``STATE_A`` / ``STATE_B`` / ``NO_STATE`` by its CV."""
    labels = np.full(len(frames), NO_STATE, dtype=int)
    for i, frame in enumerate(frames):
        s = float(cv_fn(frame))
        if state_a.contains(s):
            labels[i] = STATE_A
        elif state_b.contains(s):
            labels[i] = STATE_B
    return labels


def extract_transition_path(
    frames: np.ndarray,
    cv_fn: Callable[[np.ndarray], float],
    state_a: CVRangeState,
    state_b: CVRangeState,
) -> Optional[Tuple[int, int]]:
    """Return ``(start, end)`` indices of the minimal A->B reactive sub-path.

    The seed for TPS/TIS is the segment from the *last* frame in A before the
    *first* subsequent frame in B. Returns ``None`` if the path is not reactive
    (never reaches B, or never visits A beforehand).
    """
    labels = label_frames(frames, cv_fn, state_a, state_b)
    b_indices = np.where(labels == STATE_B)[0]
    if b_indices.size == 0:
        return None
    b_first = int(b_indices[0])
    a_before = np.where(labels[:b_first] == STATE_A)[0]
    if a_before.size == 0:
        return None
    a_last = int(a_before[-1])
    return a_last, b_first


def is_reactive(
    frames: np.ndarray,
    cv_fn: Callable[[np.ndarray], float],
    state_a: CVRangeState,
    state_b: CVRangeState,
) -> bool:
    return extract_transition_path(frames, cv_fn, state_a, state_b) is not None


def tis_interfaces(lambda0: float, lambdaN: float, n_interfaces: int, *, spacing: str = "linear") -> np.ndarray:
    """Lay out ``n_interfaces`` TIS interface values in ``[lambda0, lambdaN]``.

    ``spacing='linear'`` (uniform) or ``'exp'`` (denser near ``lambda0``, where
    crossing probabilities change fastest). Always strictly increasing.
    """
    if n_interfaces < 1:
        raise ValueError("n_interfaces must be >= 1")
    if lambdaN <= lambda0:
        raise ValueError("lambdaN must be > lambda0")
    if n_interfaces == 1:
        return np.array([float(lambda0)])
    if spacing == "linear":
        return np.linspace(lambda0, lambdaN, n_interfaces)
    if spacing == "exp":
        frac = (np.expm1(np.linspace(0.0, 1.0, n_interfaces))) / np.expm1(1.0)
        return lambda0 + (lambdaN - lambda0) * frac
    raise ValueError("spacing must be 'linear' or 'exp'")


def prepare_ops_seed(
    ensemble: PathEnsemble,
    cv_fn: Callable[[np.ndarray], float],
    state_a: CVRangeState,
    state_b: CVRangeState,
    *,
    interfaces: Optional[Sequence[float]] = None,
) -> dict:
    """PathGennie-side preparation for OPS: validate and slice the seed path.

    Returns a dict with the reactive sub-path indices/frames, per-frame labels,
    the CV trajectory, and the (optional) interface set — everything OPS needs to
    be seeded, computed without importing OPS.
    """
    frames = np.asarray(ensemble.frames, dtype=float)
    labels = label_frames(frames, cv_fn, state_a, state_b)
    span = extract_transition_path(frames, cv_fn, state_a, state_b)
    seed = None
    if span is not None:
        start, end = span
        seed = frames[start : end + 1]
    return {
        "reactive": span is not None,
        "span": span,
        "seed_frames": seed,
        "labels": labels,
        "cv_trajectory": np.array([float(cv_fn(f)) for f in frames]),
        "interfaces": None if interfaces is None else np.asarray(interfaces, dtype=float),
    }


# --------------------------------------------------------------------------- #
# OPS-dependent stage (lazy import)
# --------------------------------------------------------------------------- #
def _import_ops():
    try:
        import openpathsampling as paths  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised when OPS absent
        raise ImportError(
            "TPS/TIS requires OpenPathSampling. Install it with the "
            "'pathsampling' extra (pip install pathgennie[pathsampling]) or "
            "`pip install openpathsampling`, and pass an OPS-compatible engine "
            "(OPS propagates with its own engine, e.g. "
            "openpathsampling.engines.openmm)."
        ) from exc
    return paths


class PathSamplingStage:
    """Run TPS or TIS on a PathGennie seed path via OpenPathSampling.

    Parameters
    ----------
    cv_fn:
        ``coords -> scalar`` progress coordinate (reuse the discovery projection).
    state_a, state_b:
        ``(lo, hi)`` CV intervals (or :class:`CVRangeState`) defining the states.
    mode:
        ``"tps"`` (fixed-length / flexible TPS) or ``"tis"`` (interface sampling
        for rate constants).
    interfaces:
        TIS interface values (required for ``mode="tis"``); see
        :func:`tis_interfaces`.
    ops_engine:
        An OpenPathSampling engine used to propagate shooting moves. Required to
        actually run; OPS uses its own engine, not PathGennie's swarm engine.
    n_steps:
        Number of Monte-Carlo path moves.
    storage_path:
        Optional OPS ``.nc`` storage file.
    """

    def __init__(
        self,
        cv_fn: Callable[[np.ndarray], float],
        *,
        state_a,
        state_b,
        mode: str = "tps",
        interfaces: Optional[Sequence[float]] = None,
        ops_engine=None,
        n_steps: int = 1000,
        storage_path: Optional[str] = None,
        cv_name: str = "pathgennie_cv",
    ):
        if mode not in ("tps", "tis"):
            raise ValueError("mode must be 'tps' or 'tis'")
        self.cv_fn = cv_fn
        self.state_a = state_a if isinstance(state_a, CVRangeState) else CVRangeState("A", float(state_a[0]), float(state_a[1]))
        self.state_b = state_b if isinstance(state_b, CVRangeState) else CVRangeState("B", float(state_b[0]), float(state_b[1]))
        self.mode = mode
        self.interfaces = None if interfaces is None else list(interfaces)
        self.ops_engine = ops_engine
        self.n_steps = int(n_steps)
        self.storage_path = storage_path
        self.cv_name = cv_name
        if mode == "tis" and not self.interfaces:
            raise ValueError("mode='tis' requires interfaces (see tis_interfaces)")

    def run(self, ensemble: PathEnsemble, engine=None, **_: object) -> SamplingResult:
        paths = _import_ops()
        if self.ops_engine is None:
            raise ValueError(
                "PathSamplingStage needs an OpenPathSampling engine (ops_engine=). "
                "OPS propagates shooting moves with its own engine; build one (e.g. "
                "openpathsampling.engines.openmm.Engine) and pass it in."
            )

        # PathGennie-side validation/slicing (no OPS needed).
        seed = prepare_ops_seed(ensemble, self.cv_fn, self.state_a, self.state_b, interfaces=self.interfaces)
        if not seed["reactive"]:
            raise ValueError(
                "The PathEnsemble is not a reactive A->B path under the given "
                "state definitions; cannot seed TPS/TIS. Check state_a/state_b "
                "ranges and the cv_fn."
            )

        # Build the OPS objects and run.
        cv = paths.CoordinateFunctionCV(self.cv_name, lambda snap: self.cv_fn(self._snapshot_coords(snap)))
        vol_a = paths.CVDefinedVolume(cv, self.state_a.lo, self.state_a.hi).named(self.state_a.name)
        vol_b = paths.CVDefinedVolume(cv, self.state_b.lo, self.state_b.hi).named(self.state_b.name)

        init_traj = self._to_ops_trajectory(paths, seed["seed_frames"])

        if self.mode == "tps":
            network = paths.TPSNetwork(vol_a, vol_b)
        else:
            interface_set = paths.VolumeInterfaceSet(cv, float("-inf"), self.interfaces)
            network = paths.MISTISNetwork([(vol_a, interface_set, vol_b)])

        scheme = paths.OneWayShootingMoveScheme(network, engine=self.ops_engine)
        init_conditions = scheme.initial_conditions_from_trajectories(init_traj)

        storage = paths.Storage(self.storage_path, "w") if self.storage_path else None
        sampler = paths.PathSampling(storage=storage, move_scheme=scheme, sample_set=init_conditions)
        sampler.run(self.n_steps)
        if storage is not None:
            storage.close()

        return self._analyze(paths, sampler, network)

    # -- helpers (OPS-version specific; documented integration points) -------
    @staticmethod
    def _snapshot_coords(snapshot) -> np.ndarray:
        """OPS snapshot -> (n_atoms, 3) Angstrom array for the user's cv_fn."""
        xyz = np.asarray(snapshot.xyz, dtype=float)  # OPS xyz is in nm
        return xyz * 10.0

    def _to_ops_trajectory(self, paths, frames: np.ndarray):
        """Build an OPS Trajectory of snapshots from Angstrom frames.

        Uses the engine's current snapshot as a template (topology / box /
        velocities) and overwrites coordinates (converted nm). Velocities are
        re-randomized by OPS shooting, so position-only seeding is sufficient.
        """
        template = self.ops_engine.current_snapshot
        snapshots = []
        for frame in frames:
            snap = template.copy_with_replacement(coordinates=np.asarray(frame, dtype=float) / 10.0)
            snapshots.append(snap)
        return paths.Trajectory(snapshots)

    def _analyze(self, paths, sampler, network) -> SamplingResult:
        """Extract kinetics from the completed sampling run."""
        meta = {"mode": self.mode, "n_steps": self.n_steps,
                "storage": self.storage_path, "n_states": 2}
        rate_constants = None
        if self.mode == "tis":
            try:
                analysis = paths.TISAnalysis(network)
                rate = analysis.rate_matrix(sampler.storage)
                rate_constants = {"rate_matrix": np.asarray(rate)}
            except Exception as exc:  # pragma: no cover - depends on live OPS run
                meta["analysis_error"] = str(exc)
        return SamplingResult(rate_constants=rate_constants, metadata=meta)
