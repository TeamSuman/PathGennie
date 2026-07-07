"""Goal-driven run profiles for PathGennie.

PathGennie was designed as a *greedy* sampler of **ultrashort** (tens of fs)
trajectories, optimised for the rapid discovery of candidate transition paths to
seed later path-sampling.  That regime is deliberately different from what is
needed when the goal shifts to *quantitative* sampling — free-energy surfaces or
kinetics — where a data-driven CV (SPIB) and downstream enhanced sampling (WE,
OPES) typically require **longer** segments and more data.

Rather than hard-code one regime, a :class:`RunProfile` bundles the choices that
should move together for a given goal: segment lengths (``tau1``/``tau2``),
selection policy, CV type (geometric vs learned), and which downstream stage the
run feeds.  Two presets are provided; ``resolve_profile`` overlays a profile's
defaults *under* whatever the user set explicitly (explicit always wins), so
existing configs are unchanged and a user can opt in with ``profile: discovery``
or ``profile: sampling`` in ``input.yaml``.

``check_learned_cv_segment_length`` encodes the caveat that a learned CV usually
needs trajectories long enough to contain real relaxation, warning when the
configured segments look too short for that goal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "RunProfile",
    "DISCOVERY",
    "SAMPLING",
    "PROFILES",
    "get_profile",
    "resolve_profile",
    "check_learned_cv_segment_length",
]


@dataclass(frozen=True)
class RunProfile:
    """A coherent set of defaults for a sampling goal.

    Attributes
    ----------
    goal:
        ``"discovery"`` (fast candidate paths) or ``"sampling"`` (quantitative
        free energy / kinetics).
    selection:
        Per-cycle policy: ``greedy`` | ``softmax`` | ``beam`` | ``rrt``.
    cv:
        ``geometric`` (hand-crafted projection) or ``learned`` (SPIB on the fly).
    downstream:
        Enhanced-sampling stage the run feeds: ``None`` | ``weighted_ensemble``
        | ``opes``.
    min_learned_cv_segment_ps:
        Below this per-cycle MD time, a learned CV is flagged as likely
        under-resolved.
    """

    name: str
    goal: str
    selection: str = "softmax"
    cv: str = "geometric"
    tau1_steps: int = 2
    tau2_steps: int = 8
    max_trial: int = 15
    sigma: float = 0.1
    downstream: Optional[str] = None
    min_learned_cv_segment_ps: float = 0.1


# Fast candidate-path discovery: the original PathGennie regime (greedy on
# ultrashort, geometric-CV trajectories).
DISCOVERY = RunProfile(
    name="discovery",
    goal="discovery",
    selection="softmax",
    cv="geometric",
    tau1_steps=2,
    tau2_steps=8,
    max_trial=15,
    sigma=0.1,
    downstream=None,
)

# Quantitative sampling: longer segments, learned CV permitted, feeds weighted
# ensemble by default. Numbers are starting points to be tuned per system.
SAMPLING = RunProfile(
    name="sampling",
    goal="sampling",
    selection="beam",
    cv="learned",
    tau1_steps=50,
    tau2_steps=100,
    max_trial=20,
    sigma=0.2,
    downstream="weighted_ensemble",
)

PROFILES = {p.name: p for p in (DISCOVERY, SAMPLING)}


def get_profile(name: str) -> RunProfile:
    try:
        return PROFILES[str(name).lower()]
    except KeyError:
        raise KeyError(f"unknown run profile {name!r}; choose from {sorted(PROFILES)}")


def resolve_profile(pg_cfg: dict) -> dict:
    """Return ``pg_cfg`` with the named profile's defaults applied underneath.

    The profile is taken from ``pg_cfg['profile']`` (or ``['goal']``).  Explicit,
    non-null values in ``pg_cfg`` override the profile; if no profile is named the
    config is returned unchanged, preserving current behaviour.
    """

    name = pg_cfg.get("profile") or pg_cfg.get("goal")
    if name is None:
        return dict(pg_cfg)
    profile = get_profile(name)
    defaults = {
        "goal": profile.goal,
        "selection": profile.selection,
        "cv": profile.cv,
        "tau1_steps": profile.tau1_steps,
        "tau2_steps": profile.tau2_steps,
        "max_trial": profile.max_trial,
        "sigma": profile.sigma,
        "downstream": profile.downstream,
    }
    explicit = {k: v for k, v in pg_cfg.items() if v is not None}
    return {**defaults, **explicit}


def check_learned_cv_segment_length(
    tau1_steps: int,
    tau2_steps: int,
    timestep_ps: float,
    *,
    min_ps: float = 0.1,
) -> bool:
    """Warn (and return False) if segments look too short for a learned CV.

    Returns True when the per-cycle MD time ``(tau1+tau2)*dt`` is at least
    ``min_ps``.  This is advisory — the run still proceeds — to flag the
    ultrashort-vs-learned-CV mismatch the discovery regime can introduce.
    """

    segment_ps = (int(tau1_steps) + int(tau2_steps)) * float(timestep_ps)
    if segment_ps < float(min_ps):
        logger.warning(
            "Learned CV requested but per-cycle MD time is %.4f ps (< %.4f ps). "
            "Ultrashort segments may not give a learnable CV; consider a longer "
            "tau1/tau2 or the 'sampling' profile.",
            segment_ps,
            min_ps,
        )
        return False
    return True
