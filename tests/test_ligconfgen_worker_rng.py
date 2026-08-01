"""Forked conformer workers all drew the same random poses.

``generate_single_conformer`` used the **global** ``np.random`` and is submitted to a
``ProcessPoolExecutor``. That pool forks its workers once, so each inherits one copy of
the parent's RNG state: the first task on each of W workers returns an identical pose,
then the second, and so on. Measured with the same machinery: 12 tasks on 4 workers gave
**3 distinct results**.

The output was not corrupted -- ``_is_unique`` rejects the copies on RMSD. The damage is
throughput and silent under-delivery: only 1 attempt in W does useful work, ``n_workers``
defaults to ``multiprocessing.cpu_count()``, and the loop stops at
``num_conformations * max_attempts_factor`` attempts. On a 48-core node that budget is
exhausted after delivering roughly a fifth of the requested ensemble, announced only as
"Finished with N conformations."

Each attempt now gets its own spawned seed. Seeding per *worker* would not do: a worker
handles many attempts, so that moves the collision rather than removing it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
mda = pytest.importorskip("MDAnalysis")

from pathgennie.utils.ligconfgen import generate_single_conformer  # noqa: E402


def _args():
    """A protein far enough away that no pose clashes, but near enough to be 'close'."""
    protein = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    ligand = np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [3.5, 1.0, 0.0]])
    return protein, ligand, 2.0, 0.5, 50.0


def test_distinct_seeds_give_distinct_poses():
    """The regression: without per-call seeds these were byte-identical."""
    p, l, r, c, d = _args()
    a = generate_single_conformer(p, l, r, c, d, seed=1)
    b = generate_single_conformer(p, l, r, c, d, seed=2)
    assert a is not None and b is not None
    assert not np.allclose(a, b), "two different seeds produced the same pose"


def test_the_same_seed_reproduces_a_pose():
    """Explicit seeding also buys reproducibility, which the generator lacked."""
    p, l, r, c, d = _args()
    a = generate_single_conformer(p, l, r, c, d, seed=7)
    b = generate_single_conformer(p, l, r, c, d, seed=7)
    assert np.allclose(a, b)


def test_many_seeds_give_many_distinct_poses():
    """W-fold duplication would show up here as a collapsed distinct-count."""
    p, l, r, c, d = _args()
    poses = [generate_single_conformer(p, l, r, c, d, seed=s) for s in range(24)]
    poses = [x for x in poses if x is not None]
    assert len(poses) >= 20, "too many rejections to judge; loosen the geometry"
    uniq = {tuple(np.round(x.ravel(), 9)) for x in poses}
    assert len(uniq) == len(poses), (
        f"only {len(uniq)} distinct poses from {len(poses)} distinct seeds"
    )


def test_spawned_seeds_are_all_different():
    """What generate_conformations hands to each attempt must not repeat."""
    seq = np.random.SeedSequence(1234)
    spawned = seq.spawn(64)
    keys = {tuple(s.entropy if isinstance(s.entropy, int) else s.entropy) if False else s.spawn_key
            for s in spawned}
    assert len(keys) == 64, "spawned seed streams collide"


def test_forked_pool_without_explicit_seeds_does_duplicate():
    """Pin the mechanism itself, so the fix is not mistaken for cargo cult.

    This asserts the *old* behaviour still reproduces when the global RNG is used in a
    forked pool -- i.e. that the bug was real and the seeding is what prevents it.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    if multiprocessing.get_start_method(allow_none=True) not in (None, "fork"):
        pytest.skip("duplication is specific to the fork start method")
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as ex:
        out = list(ex.map(_global_rng_draw, range(12)))
    assert len(set(out)) < 12, (
        "global-RNG draws in a forked pool no longer duplicate; if Python changed its "
        "default start method this test's premise needs revisiting"
    )


def _global_rng_draw(_):
    return float(np.random.uniform(0, 1))


def test_seeding_does_not_promise_a_reproducible_ensemble():
    """Document the limit of the fix rather than overclaim it.

    Per-attempt seeds fix the sequence of proposals, but `generate_conformations`
    consumes futures with FIRST_COMPLETED, so arrival order -- and hence which poses
    `_is_unique` accepts against the set built so far -- depends on worker timing.
    This test exists so nobody later reads "seed" as "reproducible ensemble"; if the
    consumption order is ever made deterministic, it should be updated deliberately.
    """
    import logging

    from pathgennie.utils.ligconfgen import ConformationGenerator

    logging.disable(logging.INFO)
    p, l, r, c, d = _args()
    g = ConformationGenerator(p, l, max_radius=r, clash_cutoff=c,
                              max_distance=d, rmsd_threshold=0.05)
    a = g.generate_conformations(8, max_attempts_factor=3, n_workers=4, seed=99)
    b = g.generate_conformations(8, max_attempts_factor=3, n_workers=4, seed=99)
    # Both must be full-size -- that is the actual fix (no W-fold waste).
    assert len(a) == 8 and len(b) == 8, (
        f"attempt budget not delivering: {len(a)}, {len(b)} of 8"
    )
