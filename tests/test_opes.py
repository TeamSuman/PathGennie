"""OPES tests: PLUMED input generation + verifiable toy-core FES recovery."""

import numpy as np

from pathgennie.core.toy import ToyLangevinEngine, wolfe_quapp_gradient, wolfe_quapp_potential
from pathgennie.sampling import build_path_ensemble
from pathgennie.sampling.opes import OPESBias, OPESSimulation, OPESStage, build_plumed_opes_input


def test_build_plumed_opes_input_contains_actions():
    text = build_plumed_opes_input(
        ["phi: TORSION ATOMS=5,7,9,15", "psi: TORSION ATOMS=7,9,15,17"],
        ["phi", "psi"],
        pace=500, barrier=40.0, temp=300.0, sigma=[0.1, 0.1],
    )
    assert "OPES_METAD" in text
    assert "ARG=phi,psi" in text
    assert "PACE=500" in text
    assert "BARRIER=40.0" in text
    assert "TORSION ATOMS=5,7,9,15" in text
    assert "opes.bias" in text


def test_opes_bias_deposits_and_flattens():
    bias = OPESBias(kT=1.0, gamma=10.0, sigma=0.2, barrier=8.0)
    assert bias.bias(0.0) == bias.bias(0.0)  # no kernels -> floor, well-defined
    for _ in range(50):
        bias.update(0.0)
    # After depositing at 0, the bias there should be higher (less negative) than
    # at a far, never-visited point.
    assert bias.bias(0.0) > bias.bias(3.0)
    assert len(bias.centers) == 50


def _analytic_marginal(grid, kT):
    x = np.linspace(-2.5, 2.5, 2001)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    fe = np.array([-kT * np.log(trapz(np.exp(-np.array([wolfe_quapp_potential(xi, y) for xi in x]) / kT), x))
                   for y in grid])
    return fe - fe.min()


def test_opes_simulation_recovers_wq_marginal():
    kT = 2.0
    grid = np.linspace(-2.0, 2.0, 33)
    bias = OPESBias(kT=kT, gamma=15.0, sigma=0.2, barrier=8.0)
    sim = OPESSimulation(wolfe_quapp_gradient, cv_axis=1, kT=kT, dt=0.005, pace=20, bias=bias, seed=0)
    samples = sim.run(np.array([-1.0, -1.4]), n_steps=20000)
    fe = sim.fes(samples, grid)

    finite = np.isfinite(fe)
    # FES minimum should land in a basin (|y| ~ 1.4), not the central barrier.
    assert abs(grid[finite][np.argmin(fe[finite])]) > 0.7
    # Correlated with the analytic marginal over occupied bins.
    an = _analytic_marginal(grid, kT)
    corr = np.corrcoef(fe[finite] - fe[finite].min(), an[finite] - an[finite].min())[0, 1]
    assert corr > 0.6


def test_opes_stage_toy_mode_end_to_end():
    engine = ToyLangevinEngine(dt=0.005, kT=2.0)
    initial = engine.create_state((-1.0, -1.4))
    # A tiny ensemble just to seed the stage with a starting configuration.
    frames = np.array([engine.get_coords(initial)])
    ens = build_path_ensemble(frames, np.array([0.0]), cv_fn=lambda c: c[0, 1])
    stage = OPESStage(
        mode="toy", potential_grad=wolfe_quapp_gradient, cv_axis=1,
        grid=np.linspace(-2.0, 2.0, 25), n_steps=6000, pace=20,
        gamma=15.0, sigma=0.2, barrier=8.0, kT=2.0, seed=1,
    )
    result = stage.run(ens, engine)
    assert result.free_energy.shape == (25,)
    assert result.metadata["n_kernels"] > 0


def test_opes_plumed_mode_requires_capable_engine():
    stage = OPESStage(
        mode="plumed",
        plumed_cv_definitions=["phi: TORSION ATOMS=5,7,9,15"],
        plumed_arg_names=["phi"],
    )
    ens = build_path_ensemble(np.zeros((1, 3, 3)), np.array([0.0]))
    try:
        stage.run(ens, engine=object())  # no run_plumed -> informative error
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
