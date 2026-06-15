"""OPES free-energy stage, with a PLUMED interface and a verifiable core.

OPES (On-the-fly Probability Enhanced Sampling; Invernizzi & Parrinello, 2020)
deposits an adaptive bias along a CV so the system samples a flatter
distribution, from which the free-energy surface (FES) is recovered by
reweighting.  In production this is done by **PLUMED** (https://www.plumed.org/)
patched into the MD engine; PathGennie supplies the CV definition, the OPES
parameters, and the seed configurations from a :class:`PathEnsemble`.

This module provides two layers:

* :func:`build_plumed_opes_input` + :class:`OPESStage` (``mode="plumed"``) — the
  production path.  It writes a ``plumed.dat`` with an ``OPES_METAD`` action and
  drives a PLUMED-capable engine (one exposing ``run_plumed``); if the engine
  cannot run PLUMED it raises with guidance.  This is the interface the MD
  backends plug into once PLUMED is available.
* :class:`OPESBias` + :class:`OPESSimulation` + ``OPESStage(mode="toy")`` — a
  dependency-free implementation of the OPES bias and reweighting on an analytic
  potential (e.g. the toy Wolfe-Quapp surface), so the *algorithm* is verifiable
  in CI without an MD binary.  The bias-update and reweighting maths are identical
  to the production path; only the propagator differs.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

from .base import PathEnsemble, SamplingResult

__all__ = ["build_plumed_opes_input", "OPESBias", "OPESSimulation", "OPESStage"]


# --------------------------------------------------------------------------- #
# PLUMED interface (production path)
# --------------------------------------------------------------------------- #
def build_plumed_opes_input(
    cv_definitions: Sequence[str],
    arg_names: Sequence[str],
    *,
    pace: int = 500,
    barrier: float = 40.0,
    temp: float = 300.0,
    sigma: Optional[Sequence[float]] = None,
    grid_min: Optional[Sequence[float]] = None,
    grid_max: Optional[Sequence[float]] = None,
    colvar_file: str = "COLVAR",
    state_file: str = "STATE",
    stride: int = 100,
) -> str:
    """Return a ``plumed.dat`` driving ``OPES_METAD`` along the given CV(s).

    ``cv_definitions`` are raw PLUMED action lines (e.g.
    ``"phi: TORSION ATOMS=5,7,9,15"``); ``arg_names`` are the labels OPES biases
    (e.g. ``["phi", "psi"]``). ``barrier`` is the OPES barrier parameter (kJ/mol),
    ``temp`` the temperature (K).
    """

    lines: List[str] = ["# PathGennie-generated OPES input", "UNITS LENGTH=A"]
    lines.extend(str(line) for line in cv_definitions)

    opes = [
        "OPES_METAD ...",
        "  LABEL=opes",
        f"  ARG={','.join(arg_names)}",
        f"  PACE={int(pace)}",
        f"  BARRIER={float(barrier)}",
        f"  TEMP={float(temp)}",
        f"  STATE_WFILE={state_file}",
    ]
    if sigma is not None:
        opes.append(f"  SIGMA={','.join(str(float(s)) for s in sigma)}")
    if grid_min is not None and grid_max is not None:
        opes.append(f"  GRID_MIN={','.join(str(float(g)) for g in grid_min)}")
        opes.append(f"  GRID_MAX={','.join(str(float(g)) for g in grid_max)}")
    opes.append("...")
    lines.extend(opes)

    lines.append(
        f"PRINT ARG={','.join(arg_names)},opes.bias STRIDE={int(stride)} FILE={colvar_file}"
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Verifiable OPES core (toy path)
# --------------------------------------------------------------------------- #
class OPESBias:
    """Adaptive well-tempered kernel bias along a 1-D CV (OPES_METAD core)."""

    def __init__(self, *, kT: float, gamma: float = 10.0, sigma: float = 0.1, barrier: float = 10.0):
        self.kT = float(kT)
        self.beta = 1.0 / self.kT
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.barrier = float(barrier)
        self.prefactor = 1.0 - 1.0 / self.gamma
        # Floor so the deepest bias well is ~ -barrier (in energy units).
        self._log_floor = -self.barrier / (self.prefactor * self.kT)
        self.centers: List[float] = []
        self.heights: List[float] = []

    def _prob(self, s: float) -> float:
        if not self.centers:
            return 0.0
        c = np.asarray(self.centers)
        h = np.asarray(self.heights)
        k = np.exp(-0.5 * ((s - c) / self.sigma) ** 2)
        return float((h * k).sum() / h.sum())

    def bias(self, s: float) -> float:
        p = self._prob(s)
        log_p = np.log(p) if p > 0 else self._log_floor
        log_p = max(log_p, self._log_floor)
        return self.prefactor * self.kT * log_p

    def grad(self, s: float, ds: float = 1e-3) -> float:
        return (self.bias(s + ds) - self.bias(s - ds)) / (2.0 * ds)

    def update(self, s: float) -> None:
        height = np.exp(self.beta * self.bias(s))  # well-tempered down-scaling
        self.centers.append(float(s))
        self.heights.append(float(height))


class OPESSimulation:
    """Biased over-damped Langevin on an analytic potential (toy verification)."""

    def __init__(
        self,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        *,
        cv_axis: int = 1,
        kT: float = 1.0,
        gamma_friction: float = 1.0,
        dt: float = 0.005,
        pace: int = 50,
        bias: Optional[OPESBias] = None,
        seed: int = 0,
    ):
        self.grad_fn = grad_fn
        self.cv_axis = int(cv_axis)
        self.kT = float(kT)
        self.gamma_friction = float(gamma_friction)
        self.dt = float(dt)
        self.pace = int(pace)
        self.bias = bias or OPESBias(kT=kT)
        self.rng = np.random.default_rng(seed)

    def run(self, pos0: np.ndarray, n_steps: int, *, record_stride: int = 1):
        pos = np.asarray(pos0, dtype=float).copy()
        noise_scale = np.sqrt(2.0 * (self.kT / self.gamma_friction) * self.dt)
        samples: List[float] = []
        for step in range(int(n_steps)):
            s = pos[self.cv_axis]
            force = -self.grad_fn(pos)
            force[self.cv_axis] += -self.bias.grad(s)
            pos = pos + (force / self.gamma_friction) * self.dt + noise_scale * self.rng.standard_normal(pos.shape)
            if step % self.pace == 0:
                self.bias.update(pos[self.cv_axis])
            if step % record_stride == 0:
                samples.append(float(pos[self.cv_axis]))
        return np.asarray(samples)

    def fes(self, samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Reweighted free energy on ``grid`` using the converged bias."""
        samples = np.asarray(samples)
        weights = np.exp(self.bias.beta * np.array([self.bias.bias(s) for s in samples]))
        edges = np.concatenate([
            [grid[0] - 0.5 * (grid[1] - grid[0])],
            0.5 * (grid[:-1] + grid[1:]),
            [grid[-1] + 0.5 * (grid[-1] - grid[-2])],
        ])
        hist, _ = np.histogram(samples, bins=edges, weights=weights)
        prob = hist / hist.sum() if hist.sum() > 0 else hist
        with np.errstate(divide="ignore"):
            fe = -self.kT * np.log(prob)
        finite = fe[np.isfinite(fe)]
        if finite.size:
            fe = fe - finite.min()
        return fe


class OPESStage:
    """OPES free-energy stage (implements the ``SamplingStage`` contract).

    ``mode='plumed'`` (default) generates a PLUMED ``OPES_METAD`` input and drives
    a PLUMED-capable engine (``engine.run_plumed(plumed_input, ensemble, **cfg)``).
    ``mode='toy'`` runs the verifiable analytic-potential core and requires
    ``potential_grad`` (a ``pos -> grad`` callable) and ``cv_axis``.
    """

    def __init__(
        self,
        cv_fn: Optional[Callable[[np.ndarray], float]] = None,
        *,
        mode: str = "plumed",
        # toy-mode parameters
        potential_grad: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        cv_axis: int = 1,
        grid: Optional[Sequence[float]] = None,
        n_steps: int = 20000,
        dt: float = 0.005,
        pace: int = 50,
        gamma: float = 10.0,
        sigma: float = 0.15,
        barrier: float = 10.0,
        kT: float = 1.0,
        seed: int = 0,
        # plumed-mode parameters
        plumed_cv_definitions: Optional[Sequence[str]] = None,
        plumed_arg_names: Optional[Sequence[str]] = None,
        plumed_kwargs: Optional[dict] = None,
    ):
        if mode not in ("plumed", "toy"):
            raise ValueError("mode must be 'plumed' or 'toy'")
        self.cv_fn = cv_fn
        self.mode = mode
        self.potential_grad = potential_grad
        self.cv_axis = int(cv_axis)
        self.grid = None if grid is None else np.asarray(grid, dtype=float)
        self.n_steps = int(n_steps)
        self.dt = float(dt)
        self.pace = int(pace)
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.barrier = float(barrier)
        self.kT = float(kT)
        self.seed = int(seed)
        self.plumed_cv_definitions = plumed_cv_definitions
        self.plumed_arg_names = plumed_arg_names
        self.plumed_kwargs = dict(plumed_kwargs or {})

    def run(self, ensemble: PathEnsemble, engine, **_: object) -> SamplingResult:
        if self.mode == "plumed":
            return self._run_plumed(ensemble, engine)
        return self._run_toy(ensemble)

    def _run_plumed(self, ensemble: PathEnsemble, engine) -> SamplingResult:
        if not self.plumed_cv_definitions or not self.plumed_arg_names:
            raise ValueError("plumed mode requires plumed_cv_definitions and plumed_arg_names")
        plumed_input = build_plumed_opes_input(
            self.plumed_cv_definitions, self.plumed_arg_names,
            pace=self.pace, barrier=self.barrier, **self.plumed_kwargs,
        )
        runner = getattr(engine, "run_plumed", None)
        if runner is None:
            raise NotImplementedError(
                "OPESStage(mode='plumed') needs a PLUMED-capable engine exposing "
                "run_plumed(plumed_input, ensemble, **cfg). Patch the MD backend "
                "with PLUMED, or use mode='toy' with an analytic potential_grad."
            )
        return runner(plumed_input, ensemble, **self.plumed_kwargs)

    def _run_toy(self, ensemble: PathEnsemble) -> SamplingResult:
        if self.potential_grad is None:
            raise ValueError("toy mode requires potential_grad (pos -> grad)")
        # Toy potentials are 2-D; engine coords are (n_atoms, 3). Use the first
        # atom's leading dims up to the gradient's dimensionality.
        pos0 = np.asarray(ensemble.frames[0][0, :], dtype=float)
        grad_dim = int(np.asarray(self.potential_grad(pos0[:2])).size)
        pos0 = pos0[:grad_dim]
        if self.grid is not None:
            grid = self.grid
        else:
            vals = ensemble.cv_trajectory[:, 0] if ensemble.cv_trajectory is not None else pos0[self.cv_axis]
            lo, hi = float(np.min(vals)) - 0.5, float(np.max(vals)) + 0.5
            grid = np.linspace(lo, hi, 40)
        bias = OPESBias(kT=self.kT, gamma=self.gamma, sigma=self.sigma, barrier=self.barrier)
        sim = OPESSimulation(
            self.potential_grad, cv_axis=self.cv_axis, kT=self.kT,
            dt=self.dt, pace=self.pace, bias=bias, seed=self.seed,
        )
        samples = sim.run(pos0, self.n_steps)
        fe = sim.fes(samples, grid)
        return SamplingResult(
            free_energy=fe,
            metadata={"grid": grid, "n_kernels": len(bias.centers), "n_samples": samples.size},
        )
