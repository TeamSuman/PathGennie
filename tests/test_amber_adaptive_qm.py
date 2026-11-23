"""Per-segment adaptive QM region (pathgennie.backends.amber.adaptive_qm).

Geometry is synthetic: residue 1 is a two-atom "solute" (C at the origin, H 1.09 A along +x)
and residues 2.. are three-atom waters placed at chosen distances, so every expected selection
can be stated exactly. The engine tests fake sander by copying the input restart to the output.
"""

import threading
from pathlib import Path

import numpy as np
import pytest

import pathgennie.backends.amber.engine as amber_engine
from pathgennie.backends.amber.adaptive_qm import (
    AdaptiveQMRegion,
    format_residue_mask,
    read_qm_selection,
    selection_path,
)
from pathgennie.backends.amber.engine import CoreAmberEngine
from pathgennie.backends.amber.utils import default_mdin_controls, write_rst7_coords

BOX = np.array([30.0, 30.0, 30.0])
EXTRA = "&qmmm\n  qmmask=':1', qmcharge=0, qm_theory='DFTB3',\n/\n"


def _water(o):
    o = np.asarray(o, float)
    return [o, o + [0.96, 0.0, 0.0], o + [-0.24, 0.93, 0.0]]


def _system(water_oxygens, h=(1.09, 0.0, 0.0)):
    """Coordinates and a parse_prmtop-like dict for solute + waters at the given O positions."""
    coords = [[0.0, 0.0, 0.0], list(h)]
    names, numbers, masses = ["MOL", "MOL"], [1, 1], [12.0, 1.008]
    for k, o in enumerate(water_oxygens):
        coords += _water(o)
        names += ["WAT"] * 3
        numbers += [k + 2] * 3
        masses += [16.0, 1.008, 1.008]
    info = {"atom_residue_names": names, "atom_residue_numbers": numbers,
            "masses": np.asarray(masses)}
    return np.asarray(coords), info


def _region(info, n_solvent, **kw):
    return AdaptiveQMRegion.from_topology(info, core_residues=[1], centre_atoms=[0],
                                          n_solvent=n_solvent, **kw)


def test_mask_collapses_runs():
    assert format_residue_mask([59, 1, 2, 57, 58, 100]) == ":1-2,57-59,100"
    assert format_residue_mask([7]) == ":7"
    with pytest.raises(ValueError):
        format_residue_mask([])


def test_selects_nearest_whole_waters():
    # waters (residues 2..5) at 6, 3, 9, 4 A along -y (away from the solute H)
    x, info = _system([[0, -6, 0], [0, -3, 0], [0, -9, 0], [0, -4, 0]])
    reg = _region(info, 2, guard_cutoff=0.0)
    sel = reg.select(x)
    assert sel == (1, 3, 5)
    assert reg.n_atoms(sel) == 2 + 2 * 3          # whole molecules only


def test_minimum_image_is_used():
    # residue 2 sits 28 A away in +x, i.e. 2 A away through the periodic boundary
    x, info = _system([[-28.0, 0, 0], [0, -5, 0]])
    reg = _region(info, 1, guard_cutoff=0.0)
    assert reg.select(x, box=BOX) == (1, 2)
    assert reg.select(x) == (1, 3)                # without a box it is genuinely far


def test_hysteresis_keeps_incumbent_until_clearly_beaten():
    # outsider (res 2) at 3.0 A, incumbent (res 3) at 3.2 A -- inside the 0.3 A bonus
    x, info = _system([[0, -3.0, 0], [0, 0, -3.2]])
    reg = _region(info, 1, hysteresis=0.3, guard_cutoff=0.0)
    assert reg.select(x) == (1, 2)                       # fresh: nearest wins
    assert reg.select(x, previous=(1, 3)) == (1, 3)      # incumbent kept
    # outsider now 0.5 A closer: beyond the bonus, so it swaps in
    x2, _ = _system([[0, -2.7, 0], [0, 0, -3.2]])
    assert reg.select(x2, previous=(1, 3)) == (1, 2)


def test_selection_is_deterministic_on_ties():
    x, info = _system([[0, -3.0, 0], [0, 0, -3.0], [0, 0, 3.0]])
    reg = _region(info, 1, guard_cutoff=0.0)
    assert reg.select(x) == (1, 2)                # equal distance -> lowest residue number


def test_proton_guard_pulls_in_a_shared_proton_acceptor():
    # The solute H is put 5 A from the centre atom so that only the guard, not the
    # nearest-to-centre rule, can bring the waters around it in. n_solvent=1 takes the water
    # at 3 A; the water whose O is 1.2 A from the H must join it; the one at 1.6 A must not.
    x, info = _system([[0, -3.0, 0], [5.0, 0, -1.2], [5.0, 0, 1.6]], h=(5.0, 0.0, 0.0))
    assert _region(info, 1, guard_cutoff=1.3).select(x) == (1, 2, 3)
    assert _region(info, 1, guard_cutoff=0.0).select(x) == (1, 2)


def test_proton_guard_also_catches_mm_hydrogen_on_qm_heavy_atom():
    # an MM water pointing its H at the centre atom from 1.2 A (O is ~2.2 A away, beyond the
    # 2 A of the nearest water), so it is out by rank but in by the guard
    x, info = _system([[0, -2.0, 0], [-2.16, 0, 0]], h=(0.0, 0.0, 5.0))
    reg = _region(info, 1, guard_cutoff=1.3)
    assert reg.select(x) == (1, 2, 3)
    assert _region(info, 1, guard_cutoff=0.0).select(x) == (1, 2)


def test_apply_replaces_qmmask_only():
    x, info = _system([[0, -3, 0]])
    reg = _region(info, 1)
    out = reg.apply(EXTRA, (1, 2))
    assert "qmmask=':1-2'" in out and "qmcharge=0" in out
    with pytest.raises(ValueError):
        reg.apply("&qmmm\n qmcharge=0,\n/\n", (1, 2))


def test_core_and_solvent_must_not_overlap():
    _, info = _system([[0, -3, 0]])
    with pytest.raises(ValueError):
        AdaptiveQMRegion.from_topology(info, core_residues=[1, 2], centre_atoms=[0],
                                       n_solvent=1)


# ---------------------------------------------------------------- engine integration

def _fake_sander(monkeypatch, seen):
    lock = threading.Lock()

    def fake_run(cmd, check=True, capture_output=True, text=True, env=None, cwd=None):
        mdin = Path(cmd[cmd.index("-i") + 1]).read_text()
        src, dst = cmd[cmd.index("-c") + 1], cmd[cmd.index("-r") + 1]
        Path(dst).write_bytes(Path(src).read_bytes())
        with lock:
            seen.append((Path(src).name, mdin))
        return None

    monkeypatch.setattr(amber_engine.subprocess, "run", fake_run)


def _engine(tmp_path, reg):
    controls = default_mdin_controls("explicit")
    return CoreAmberEngine(topology=tmp_path / "x.prmtop", executable=Path("/bin/true"),
                           scratch_dir=tmp_path / "scratch", temperature=300.0,
                           mdin_controls=controls, extra_mdin_text=EXTRA, box=BOX,
                           qm_region=reg, qm_selection_log=tmp_path / "qm.log")


def _qmmask(mdin):
    return mdin.split("qmmask='")[1].split("'")[0]


def test_engine_writes_selected_mask_and_sidecar(tmp_path, monkeypatch):
    seen = []
    _fake_sander(monkeypatch, seen)
    x, info = _system([[0, -6, 0], [0, -3, 0], [0, -9, 0], [0, -4, 0]])
    start = tmp_path / "start.rst7"
    write_rst7_coords(start, x, box=BOX)
    eng = _engine(tmp_path, _region(info, 2, guard_cutoff=0.0))

    h = eng.run_segment(str(start), 10, randomize_velocities=True, seed=1)
    assert _qmmask(seen[-1][1]) == ":1,3,5"
    assert read_qm_selection(h) == (1, 3, 5)
    assert "n_atoms=8" in (tmp_path / "qm.log").read_text()

    # a child segment sees its parent's selection (hysteresis input), and release cleans it
    anchor = eng.clone_anchor(h)
    assert read_qm_selection(anchor) == (1, 3, 5)
    eng.release(anchor)
    assert not selection_path(anchor).exists() and not Path(anchor).exists()


def test_all_trials_from_one_anchor_share_the_region(tmp_path, monkeypatch):
    seen = []
    _fake_sander(monkeypatch, seen)
    x, info = _system([[0, -3.0, 0], [0, 0, -3.2], [0, 0, 7.0]])
    start = tmp_path / "start.rst7"
    write_rst7_coords(start, x, box=BOX)
    eng = _engine(tmp_path, _region(info, 1, hysteresis=0.3, guard_cutoff=0.0))
    anchor = eng.clone_anchor(eng.run_segment(str(start), 5, randomize_velocities=True, seed=0))

    threads = [threading.Thread(target=eng.run_segment, args=(anchor, 5),
                                kwargs=dict(randomize_velocities=True, seed=s)) for s in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    masks = {_qmmask(m) for name, m in seen if name == Path(anchor).name}
    assert masks == {":1-2"}


def test_engine_rejects_template_without_qmmask(tmp_path):
    _, info = _system([[0, -3, 0]])
    with pytest.raises(ValueError):
        CoreAmberEngine(topology=tmp_path / "x", executable=Path("/bin/true"),
                        scratch_dir=tmp_path / "s", temperature=300.0,
                        mdin_controls=default_mdin_controls("explicit"),
                        extra_mdin_text="&qmmm\n qmcharge=0,\n/\n", box=BOX,
                        qm_region=_region(info, 1))


def test_engine_without_region_is_unchanged(tmp_path, monkeypatch):
    seen = []
    _fake_sander(monkeypatch, seen)
    x, _ = _system([[0, -3, 0]])
    start = tmp_path / "start.rst7"
    write_rst7_coords(start, x, box=BOX)
    eng = CoreAmberEngine(topology=tmp_path / "x", executable=Path("/bin/true"),
                          scratch_dir=tmp_path / "s", temperature=300.0,
                          mdin_controls=default_mdin_controls("explicit"),
                          extra_mdin_text=EXTRA, box=BOX)
    h = eng.run_segment(str(start), 5, randomize_velocities=True, seed=1)
    assert _qmmask(seen[-1][1]) == ":1"
    assert not selection_path(h).exists()
