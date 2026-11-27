"""Basins, swarm records and the driver hook (NumPy only)."""

import numpy as np
import pytest

from pathgennie.committor import BasinPair, BoxBasin, SphereBasin, SwarmDataset, SwarmRecorder
from pathgennie.committor.basins import LABEL_A, LABEL_B, LABEL_NONE
from pathgennie.committor.records import run_swarms
from pathgennie.core.driver import PathGennieDriver
from pathgennie.core.progress import TargetMetric
from pathgennie.core.toy import ToyLangevinEngine

A = np.array([-1.174, 1.477])
B = np.array([1.124, -1.485])
BASINS = BasinPair(SphereBasin(A, 0.4), SphereBasin(B, 0.4))


def _xy(coords):
    return np.array([coords[0, 0], coords[0, 1]])


def _run(seed, recorder=None, scheme="parallel", max_cycle=60):
    engine = ToyLangevinEngine(dt=0.005, kT=1.0)
    driver = PathGennieDriver(
        engine, TargetMetric(_xy, target_cv=B),
        lambda c: bool(np.linalg.norm(_xy(c) - B) < 0.3),
        sigma=0.3, seed=seed, verbosity=0, swarm_recorder=recorder,
        scheme=scheme, max_try=40, batch_size=4,
    )
    return driver.run(engine.create_state(A), tau1=20, tau2=40, max_trial=8,
                      max_cycle=max_cycle, save_freq=5)


def test_basin_labels():
    assert BASINS.label(A) == LABEL_A
    assert BASINS.label(B) == LABEL_B
    assert BASINS.label([0.0, 0.0]) == LABEL_NONE
    np.testing.assert_array_equal(BASINS.labels(np.array([A, B, [0.0, 0.0]])), [0, 1, -1])


def test_one_sided_box_basin():
    bound = BoxBasin([-np.inf], [3.5])
    assert bound.contains([3.0]) and not bound.contains([4.0])


def test_periodic_sphere_basin_wraps():
    basin = SphereBasin([170.0], 20.0, periodic=[360.0])
    assert basin.contains([-175.0])           # 15 degrees away across the branch cut
    assert not SphereBasin([170.0], 20.0).contains([-175.0])


def test_overlapping_basins_are_an_error():
    pair = BasinPair(SphereBasin([0.0], 1.0), SphereBasin([0.5], 1.0))
    with pytest.raises(ValueError, match="overlap"):
        pair.label([0.25])


def test_recorder_keeps_every_trial_of_every_cycle():
    rec = SwarmRecorder(featurize=_xy, basins=BASINS)
    _, metrics = _run(seed=3, recorder=rec)
    data = rec.dataset()
    assert len(data) == len(metrics)                       # one swarm per cycle
    assert data.ends.shape[1:] == (8, 2) and data.mask.all()
    np.testing.assert_array_equal(data.chosen.sum(axis=1), 1)   # exactly one selected
    np.testing.assert_allclose(data.prob.sum(axis=1), 1.0)
    assert data.total_steps == len(data) * 8 * 20
    # the first swarm starts at the reactant minimum
    assert data.start_label[0] == LABEL_A
    np.testing.assert_allclose(data.starts[0], A)
    # one runner per cycle, starting from the selected sampler's endpoint
    assert len(rec.runner_starts) == len(data)
    np.testing.assert_allclose(rec.runner_starts[0], data.ends[0][data.chosen[0]][0])


def test_recorder_does_not_change_the_run():
    traj_plain, metrics_plain = _run(seed=11)
    traj_rec, metrics_rec = _run(seed=11, recorder=SwarmRecorder(featurize=_xy))
    np.testing.assert_array_equal(metrics_plain, metrics_rec)
    np.testing.assert_array_equal(traj_plain, traj_rec)


def test_recorder_serial_scheme_keeps_all_probes():
    rec = SwarmRecorder(featurize=_xy, basins=BASINS)
    _run(seed=5, recorder=rec, scheme="serial", max_cycle=15)
    data = rec.dataset()
    assert len(data) >= 1
    assert (data.mask.sum(axis=1) >= 1).all()
    assert (data.chosen.sum(axis=1) <= 1).all()            # none when a cycle "stays"
    assert np.isnan(data.prob[data.mask]).all()            # no softmax in the serial scheme


def test_dataset_roundtrip_and_concatenate(tmp_path):
    rec = SwarmRecorder(featurize=_xy, basins=BASINS)
    _run(seed=3, recorder=rec, max_cycle=10)
    data = rec.dataset()
    path = tmp_path / "swarms.h5"
    data.save(path)
    back = SwarmDataset.load(path)
    for name in SwarmDataset._FIELDS:
        np.testing.assert_array_equal(getattr(back, name), getattr(data, name))

    engine = ToyLangevinEngine(dt=0.005)
    other = SwarmRecorder(featurize=_xy, basins=BASINS)
    run_swarms(engine, [engine.create_state([0.0, 0.0])], other, n_trials=3, n_steps=5,
               cv_fn=_xy, seed=0)
    both = SwarmDataset.concatenate([data, other.dataset()])
    assert len(both) == len(data) + 1
    assert both.ends.shape[1] == 8
    assert both.mask[-1].sum() == 3 and np.isnan(both.ends[-1, 3:]).all()
    assert both.n_trials == data.n_trials + 3


def test_run_swarms_is_reproducible_and_leaves_start_alone():
    def swarm(seed):
        engine = ToyLangevinEngine(dt=0.005)
        start = engine.create_state([0.2, -0.1])
        rec = SwarmRecorder(featurize=_xy)
        run_swarms(engine, [start], rec, n_trials=6, n_steps=10, seed=seed)
        np.testing.assert_allclose(_xy(engine.get_coords(start)), [0.2, -0.1])
        assert len(engine._cache) == 1                      # every trial handle released
        return rec.dataset().ends

    np.testing.assert_array_equal(swarm(1), swarm(1))
    assert not np.array_equal(swarm(1), swarm(2))
