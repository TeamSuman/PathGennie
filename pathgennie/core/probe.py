"""Probe mode: measure how a region of the landscape responds to segment length, without selecting anything.

A probe is one batch of ``n_trials`` sampler segments of ``n_steps`` started from an anchor with fresh velocities,
each recorded every ``stride`` steps, followed by velocity-continued relaxations of the ``n_continue`` best trials.
Because sub-frames are kept, one probe gives, for every segment length on the grid,

* drift ``mu(t)`` and spread ``s(t)`` of the change of the progress metric,
* acceptance ``a(t)`` = fraction of trials that gain,
* retention ``R(t2)`` = fraction of the selected gain left after a relaxation of length ``t2``,

from which the gain and cost of a parameter set (n, tau1, tau2, rule) are predicted before any production cycle
is spent (phaseX/TUNING_STRATEGY.md, PI 17 Nov 2026). The probe needs no product and commits nothing: the anchor
is left untouched and every handle made here is released.

Limits: retention is measured from the ends of the full-length trials and is assumed to depend on ``t2`` only; the
first-accept prediction leaves out the unbiased fallback segment; a trap cannot be read from one probe (it shows
in the cycle history, as gains that do not accumulate).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .engine import Engine, Handle
from .parallel import ParallelExecutor, SerialExecutor
from .progress import ProgressVariable

__all__ = ["run_probe", "probe_stats", "predicted_gain", "expected_cost", "rank_parameter_sets", "read_regime",
           "selected_gain"]


def run_probe(
    engine: Engine,
    progress: ProgressVariable,
    anchor: Handle,
    *,
    n_trials: int,
    n_steps: int,
    stride: int,
    n_continue: int = 1,
    relax_steps: Optional[int] = None,
    seed: Optional[int] = None,
    executor: Optional[ParallelExecutor] = None,
) -> Dict[str, np.ndarray]:
    """Run one probe from ``anchor``. Returns arrays indexed [trial, sub-frame]:

    ``steps`` (segment length of each sub-frame), ``anchor_metric``, ``gain`` (metric minus the anchor's),
    ``cv``, ``frames`` (coordinates), ``seeds``; and for the continued trials ``continued`` (trial indices, best
    first), ``cont_gain`` (metric minus the anchor's), ``cont_cv``, ``cont_frames``, on the grid ``cont_steps``
    (``relax_steps`` long; default: the sampler length). A failed trial is NaN throughout.
    """
    executor = executor or SerialExecutor()
    rng = np.random.default_rng(seed)
    n_trials, n_steps, stride = int(n_trials), int(n_steps), int(stride)
    n_frames = n_steps // stride
    relax_steps = n_steps if relax_steps is None else int(relax_steps)
    n_relax = relax_steps // stride
    if n_frames < 1 or n_relax < 1:
        raise ValueError("stride is longer than the probe segment")
    seeds = [int(rng.integers(1, 2_147_483_647)) for _ in range(n_trials)]
    cont_seeds = [int(rng.integers(1, 2_147_483_647)) for _ in range(n_trials)]

    anchor_coords = np.asarray(engine.get_coords(anchor), dtype=float)
    anchor_cv = np.asarray(progress.project(anchor_coords, cycle=0), dtype=float)
    anchor_metric = float(progress.metric(anchor_cv))

    def score(frames):
        cvs = np.array([np.asarray(progress.project(f, cycle=0), dtype=float).reshape(-1) for f in frames])
        return cvs, np.array([float(progress.metric(c)) for c in cvs]) - anchor_metric

    def segment(start, n, fresh, seg_seed, device):
        """(end handle, frames, cv, gain) or None when the segment fails or returns a short block."""
        end = None
        try:
            end, frames = engine.run_segment(start, n, randomize_velocities=fresh, seed=seg_seed, device=device,
                                             save_subframes=True, subframe_stride=stride)
            frames = np.asarray(frames, dtype=float)
            if len(frames) != n // stride:
                raise RuntimeError(f"{len(frames)} sub-frames returned, {n // stride} expected")
            cvs, gain = score(frames)
            return end, frames, cvs, gain
        except Exception:  # noqa: BLE001 - one trial, not the probe
            if end is not None:
                engine.release(end)
            return None

    def sampler(i, device):
        clone = engine.clone_anchor(anchor)
        out = segment(clone, n_steps, True, seeds[i], device)
        if out is None or out[0] != clone:
            engine.release(clone)
        return out

    trials = executor.map(sampler, list(range(n_trials)))
    ok = [i for i, t in enumerate(trials) if t is not None]
    if not ok:
        raise RuntimeError(f"all {n_trials} probe trials failed")

    def stack(k):
        out = np.full((n_trials,) + np.asarray(trials[ok[0]][k]).shape, np.nan)
        for i in ok:
            out[i] = trials[i][k]
        return out

    frames, cv, gain = stack(1), stack(2), stack(3)

    best = sorted(ok, key=lambda i: -gain[i, -1])[:max(0, int(n_continue))]
    conts = executor.map(lambda i, device: segment(trials[i][0], relax_steps, False, cont_seeds[i], device), best)
    kept = [j for j, c in enumerate(conts) if c is not None]
    for c in conts:
        if c is not None:
            engine.release(c[0])
    for i in ok:
        engine.release(trials[i][0])

    result = {
        "steps": stride * np.arange(1, n_frames + 1), "anchor_metric": anchor_metric, "anchor_cv": anchor_cv,
        "gain": gain, "cv": cv, "frames": frames, "seeds": np.array(seeds),
        "continued": np.array([best[j] for j in kept], dtype=int),
        "cont_steps": stride * np.arange(1, n_relax + 1),
    }
    if kept:
        result["cont_frames"] = np.array([conts[j][1] for j in kept])
        result["cont_cv"] = np.array([conts[j][2] for j in kept])
        result["cont_gain"] = np.array([conts[j][3] for j in kept])
    else:
        result["cont_gain"] = np.empty((0, n_relax))
    return result


def probe_stats(probe) -> dict:
    """Drift ``mu``, spread ``s``, acceptance ``a`` by sampler length and retention ``R`` by relaxation length.

    ``R[j]`` = (gain left after ``steps[j]`` of relaxation) / (gain before it), both summed over the continued
    trials; NaN when nothing was continued or the continued trials had no net gain.
    """
    gain = np.asarray(probe["gain"], dtype=float)
    gain = gain[np.isfinite(gain).all(axis=1)]
    steps = np.asarray(probe["steps"])
    cont = np.asarray(probe["cont_gain"], dtype=float)
    before = float(np.sum(np.asarray(probe["gain"], dtype=float)[np.asarray(probe["continued"], dtype=int), -1])) \
        if len(cont) else 0.0
    R = cont.sum(axis=0) / before if len(cont) and before > 0 else np.full(len(steps), np.nan)
    return {
        "steps": steps, "n": len(gain), "gain": gain,
        "mu": gain.mean(axis=0),
        "s": gain.std(axis=0, ddof=1) if len(gain) > 1 else np.zeros(len(steps)),
        "a": (gain > 0).mean(axis=0),
        "R": R,
    }


def _at(stats, t, what):
    hit = np.nonzero(np.asarray(stats["steps"]) == int(t))[0]
    if len(hit) != 1:
        raise ValueError(f"{what} = {t} is not on the probe grid {list(stats['steps'])}")
    return int(hit[0])


def _retention(stats, tau2):
    return 1.0 if int(tau2) == 0 else float(stats["R"][_at(stats, tau2, "tau2")])


def predicted_gain(stats, *, n, tau1, tau2, rule="first_accept", batch=None) -> float:
    """Expected gain of one cycle. ``rule='best'``: the empirical expected maximum of ``n`` trials (the limit of
    softmax at small sigma). ``rule='first_accept'``: the mean gaining trial times the chance that one of ``n``
    gains. Both are multiplied by the retention after ``tau2``."""
    g = np.sort(stats["gain"][:, _at(stats, tau1, "tau1")])
    N, n = len(g), int(n)
    if rule == "best":
        i = np.arange(1, N + 1)
        raw = float(np.sum(g * ((i / N) ** n - ((i - 1) / N) ** n)))
    elif rule == "first_accept":
        up = g[g > 0]
        a = len(up) / N
        raw = float(up.mean() * (1.0 - (1.0 - a) ** n)) if len(up) else 0.0
    else:
        raise ValueError(f"rule must be 'best' or 'first_accept', not {rule!r}")
    return raw * _retention(stats, tau2)


def selected_gain(probe, *, n, tau2, rule="first_accept") -> float:
    """Expected gain of one cycle whose sampler length is the probe's full length, read from the relaxed ends of the
    probe's own best trials (gain relative to the anchor after ``tau2``; ``tau2 = 0``: the sampler gain itself).

    ``rule='first_accept'``: the mean over the gaining trials that were continued, times the chance that one of
    ``n`` trials gains. ``rule='best'``: each continued trial weighted by the chance that it is the maximum of ``n``
    draws from the probe's trials, renormalised over the continued ones.
    """
    end = np.asarray(probe["gain"], dtype=float)[:, -1]
    cont = np.asarray(probe["continued"], dtype=int)
    if int(tau2) == 0:
        net = end[cont]
    else:
        grid = np.asarray(probe["cont_steps"] if "cont_steps" in probe else probe["steps"])
        hit = np.nonzero(grid == int(tau2))[0]
        if len(hit) != 1:
            raise ValueError(f"tau2 = {tau2} is not on the relaxation grid {list(grid)}")
        net = np.asarray(probe["cont_gain"], dtype=float)[:, int(hit[0])]
    ok = np.isfinite(end)
    N, n = int(ok.sum()), int(n)
    if rule == "first_accept":
        up = end[cont] > 0
        a = float((end[ok] > 0).mean())
        return float(net[up].mean() * (1.0 - (1.0 - a) ** n)) if up.any() else 0.0
    if rule == "best":
        rank = np.array([int((end[ok] <= end[i]).sum()) for i in cont])      # 1 = lowest, N = highest
        w = (rank / N) ** n - ((rank - 1) / N) ** n
        return float(np.sum(w * net) / np.sum(w))
    raise ValueError(f"rule must be 'best' or 'first_accept', not {rule!r}")


def expected_cost(stats, *, n, tau1, tau2, rule="first_accept", batch=None):
    """(force calls, sequential steps) of one cycle. Under first-accept the trials run in batches of ``batch``
    (default: all ``n`` at once) and a later batch runs only if no earlier trial gained."""
    n, tau1, tau2 = int(n), int(tau1), int(tau2)
    if rule == "best":
        return n * tau1 + tau2, tau1 + tau2
    if rule != "first_accept":
        raise ValueError(f"rule must be 'best' or 'first_accept', not {rule!r}")
    a = float(stats["a"][_at(stats, tau1, "tau1")])
    batch = int(batch or n)
    calls = wall = 0.0
    for lo in range(0, n, batch):
        p_run = (1.0 - a) ** lo
        calls += p_run * min(batch, n - lo) * tau1
        wall += p_run * tau1
    return calls + tau2, wall + tau2


def rank_parameter_sets(stats, sets) -> list:
    """Rows {name, gain, calls, wall, gain_per_call, gain_per_wall, **parameters}, best gain per force call first.
    ``sets`` maps a name to the keyword arguments of :func:`predicted_gain`."""
    rows = []
    for name, kw in sets.items():
        gain = predicted_gain(stats, **kw)
        calls, wall = expected_cost(stats, **kw)
        rows.append({"name": name, **kw, "gain": gain, "calls": calls, "wall": wall,
                     "gain_per_call": gain / calls, "gain_per_wall": gain / wall})
    return sorted(rows, key=lambda r: -r["gain_per_call"])


def read_regime(stats) -> str:
    """'downhill', 'climb' or 'plateau' from the full-length drift relative to the spread (provisional
    thresholds: |mu| > 0.5 s decides the sign; acceptance above 0.8 at every length is downhill)."""
    mu, s = float(stats["mu"][-1]), float(stats["s"][-1])
    z = mu / s if s > 0 else np.sign(mu) * np.inf if mu else 0.0
    if z > 0.5 or (mu > 0 and float(np.min(stats["a"])) > 0.8):
        return "downhill"
    if z < -0.5:
        return "climb"
    return "plateau"
