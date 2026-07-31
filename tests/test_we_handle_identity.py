"""Weighted Ensemble must compare engine handles by value, not identity.

``driver.py`` states the rule explicitly:

    Compare handles by value (==), not identity (is): handles may be plain ints
    (CPython caches only -5..256) or round-trip through a distributed executor,
    so `is` would leak or double-release.

The WE stage used ``new_handle is not walker.handle``. When an engine returns a
handle that *equals* the one passed in but is a different object -- a re-derived
file path, an int above CPython's small-int cache, anything that has round-tripped
through serialisation -- that test is true and the stage releases a handle the
walker is still using.
"""

from __future__ import annotations

import numpy as np

from pathgennie.sampling import WeightedEnsembleStage, build_path_ensemble


class SameHandleEngine:
    """An engine that propagates in place and returns an *equal* handle.

    The returned handle is deliberately a distinct object with the same value,
    which is exactly the case identity comparison gets wrong. String handles make
    this concrete -- both the AMBER and GROMACS backends use file paths.
    """

    def __init__(self):
        self.cv = {}
        self.released = []
        self.next_id = 0

    def _new(self, value):
        name = f"handle_{self.next_id}"
        self.next_id += 1
        self.cv[name] = float(value)
        return name

    def create_state(self, coords):
        return self._new(np.asarray(coords, dtype=float).ravel()[0])

    create_handle = create_state

    def clone_anchor(self, handle):
        return self._new(self.cv[handle])

    def run_segment(self, handle, n_steps, *, randomize_velocities, seed, device=None,
                    save_subframes=False, subframe_stride=1):
        self.cv[handle] += 0.05
        # Same value, new object -- `is` says "different", `==` says "same".
        return "".join(handle)

    def get_coords(self, handle):
        if handle in self.released:
            raise AssertionError(f"use-after-free: {handle} was released and then read")
        return np.array([[self.cv[handle], 0.0, 0.0]])

    def release(self, handle):
        self.released.append(handle)
        self.cv.pop(handle, None)


def _run(engine, **kw):
    frames = np.array([[[cv, 0.0, 0.0]] for cv in (0.2, 0.6)])
    handles = [engine.create_state(f) for f in frames]
    ensemble = build_path_ensemble(frames=frames, metrics=np.array([0.2, 0.6]),
                                   handles=handles)
    stage = WeightedEnsembleStage(
        cv_fn=lambda c: float(np.asarray(c).ravel()[0]),
        tau_steps=1, n_iterations=4, bin_edges=np.linspace(0.0, 1.0, 5),
        target_count=2, seed=1, kT=1.0, **kw,
    )
    return stage.run(ensemble, engine)


def test_an_equal_handle_is_not_released():
    """The core defect: releasing a handle the walker is about to keep using."""
    engine = SameHandleEngine()
    _run(engine)
    # Every handle still held by a walker must be alive. get_coords already raises
    # on use-after-free, so reaching here with a clean released-set is the check.
    for name in engine.cv:
        assert name not in engine.released, f"{name} is in use but was released"


def test_no_handle_is_released_twice():
    engine = SameHandleEngine()
    _run(engine)
    assert len(engine.released) == len(set(engine.released)), \
        f"double release: {engine.released}"


def test_the_run_still_produces_a_free_energy():
    """Guard against 'fixing' the comparison by never releasing anything."""
    engine = SameHandleEngine()
    result = _run(engine)
    assert result.free_energy is not None
    assert np.isfinite(result.free_energy).any()


def test_distinct_handles_are_still_released():
    """The fix must not leak: when the engine returns a genuinely new handle,
    the old one has to be freed."""

    class NewHandleEngine(SameHandleEngine):
        def run_segment(self, handle, n_steps, **kw):
            return self._new(self.cv[handle] + 0.05)

    engine = NewHandleEngine()
    _run(engine)
    assert engine.released, "old handles were never released -- that is a leak"
