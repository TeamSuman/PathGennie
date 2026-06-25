from typing import Callable, Dict, Optional

import numpy as np  # type: ignore
from openmm import unit  # type: ignore
from openmm.app import Simulation  # type: ignore
from tqdm.auto import trange


class PathGennieMD:
    """
    PathGennie implementation based on:
    'PathGennie: Rapid Generation of Rare Event Pathways via Direction-Guided Adaptive Sampling Using Ultrashort Monitored Trajectories'
    J. Chem. Theory Comput. 2016, 12, 5, 2035-2043
    """

    NM_TO_ANG = 10.0

    def __init__(
        self,
        simulation: Simulation,
        projection_fn: Callable[..., np.ndarray],
        projection_args: Optional[Dict] = None,
        mode: str = "escape",
        target_projection: Optional[np.ndarray] = None,
        convergence_fn: Optional[Callable] = None,
        convergence_args: Optional[Dict] = None,
        escape_direction: str = "auto",
        temperature: float = 300.0,
        sigma: float = 0.5,
    ):
        if mode not in ("escape", "target"):
            raise ValueError("mode must be 'escape', 'target'")
        if mode == "target" and target_projection is None:
            raise ValueError("target_projection required for target mode")
        if mode == "target" and convergence_fn is None:
            raise ValueError("convergence_fn required for target mode")
        if mode == "escape" and convergence_fn is None:
            raise ValueError("convergence_fn required for escape mode")

        self.sim = simulation
        self.mode = mode
        self.proj_fn = projection_fn
        self.proj_args = projection_args or {}
        self.target = np.asarray(target_projection) if target_projection is not None else None
        self.converge_fn = convergence_fn
        self.converge_args = convergence_args or {}
        self.escape_direction = escape_direction

        # Temperature for velocity re-randomization
        self.temperature = temperature * unit.kelvin  # type: ignore
        self.sigma = sigma

    def run(
        self,
        initial_pos: np.ndarray,
        tau1: int = 200,
        tau2: int = 200,
        max_trial: int = 20,
        max_cycle: int = 5000,
        save_freq: int = 10,
        verbosity: int = 1,
    ):

        trajectory = []
        metrics_history = []

        # ---- initialize system ----
        self.sim.context.setPositions(initial_pos)
        self.sim.context.setVelocitiesToTemperature(self.temperature)

        # Initial anchor
        anchor_state = self.sim.context.getState(getPositions=True, getVelocities=True)
        pos = self._pos()
        start_proj = self._proj(pos, cycle=0)
        current_proj = start_proj

        # Metric definition
        def metric(cv):
            if self.mode == "escape":
                s_proj = start_proj
                
                # Support NaN masking for explicit dimensional reduction
                if np.isnan(cv).any():
                    valid = ~np.isnan(cv)
                    cv = cv[valid]
                    s_proj = s_proj[valid]
                
                # Support implicit shape reduction (assume they kept the LAST elements)
                if len(cv) < len(s_proj):
                    s_proj = s_proj[-len(cv):]
                    
                return np.linalg.norm(cv - s_proj)
            else:
                # Progress is closeness to target
                return -np.linalg.norm(cv - self.target)

        current_metric = metric(current_proj)
        cycle_iter = trange(max_cycle, desc="PathGennie") if verbosity >= 2 else range(max_cycle)

        for cycle in cycle_iter:
            trial_results = []

            #  Run M trials ----
            for _ in range(max_trial):
                # Restore anchor positions but randomize velocities
                self.sim.context.setState(anchor_state)
                self.sim.context.setVelocitiesToTemperature(self.temperature)

                # sampler segment
                self.sim.step(tau1)

                trial_pos = self._pos()
                trial_proj = self._proj(trial_pos, cycle=cycle)
                trial_metric = metric(trial_proj)

                # Store the state and metric
                # We need to store the FULL state after tau1 to continue it later
                state_after_tau1 = self.sim.context.getState(getPositions=True, getVelocities=True)
                trial_results.append(
                    {"metric": trial_metric, "state": state_after_tau1, "pos": trial_pos, "proj": trial_proj}
                )

            raw_metrics = np.array([r["metric"] for r in trial_results])
            m_min = np.min(raw_metrics)
            m_max = np.max(raw_metrics)

            # Avoid division by zero if all trials are identical
            if (m_max - m_min) < 1e-9:
                probs = np.ones(len(trial_results)) / len(trial_results)
            else:
                # 1. Scale metrics between 0 and 1
                # 0 = worst trial of this batch, 1 = best trial
                scaled_metrics = (raw_metrics - m_min) / (m_max - m_min)

                # 2. Boltzmann Weighting on the scaled interval
                # Shifted by 1.0 so the max weight is always exp(0) = 1
                logits = (scaled_metrics - 1.0) / (self.sigma + 1e-12)
                weights = np.exp(logits)
                probs = weights / np.sum(weights)

            chosen_idx = np.random.choice(len(trial_results), p=probs)
            best_trial = trial_results[chosen_idx]

            # ---- runner segment ----
            self.sim.context.setState(best_trial["state"])

            self.sim.step(tau2)

            # Update anchor
            anchor_state = self.sim.context.getState(getPositions=True, getVelocities=True)
            pos = self._pos()
            current_proj = self._proj(pos, cycle=cycle)
            current_metric = metric(current_proj)
            metrics_history.append(current_metric)

            # ---- SAVE (after tau2) ----
            if cycle % save_freq == 0:
                trajectory.append(pos * self.NM_TO_ANG)
                if verbosity:
                    print(f"Cycle {cycle}: metric={current_metric:.4f}, CV={current_proj}")

            # ---- CONVERGENCE ----
            converge_fn = self.converge_fn
            if converge_fn is None:
                raise ValueError("convergence_fn is required for run()")

            if self.mode == "escape":
                if converge_fn(pos * self.NM_TO_ANG, **self.converge_args):
                    if verbosity:
                        print(f"\nEscape convergence reached at cycle {cycle}")
                    if cycle % save_freq != 0:
                        trajectory.append(pos * self.NM_TO_ANG)
                    break
            else:  # mode == "target"
                # For target mode, metric is -norm(cv - target), so norm < tol means metric > -tol
                if converge_fn(pos * self.NM_TO_ANG, **self.converge_args):
                    if verbosity:
                        print(f"\nTarget convergence reached at cycle {cycle}")
                    if cycle % save_freq != 0:
                        trajectory.append(pos * self.NM_TO_ANG)
                    break

            if verbosity >= 2 and cycle % 10 == 0 and cycle % save_freq != 0:
                print(f"Cycle {cycle:4d}: metric={current_metric:.4f} (best of {max_trial})")

        if verbosity:
            print("Final metric:", current_metric)

        return np.array(trajectory), np.array(metrics_history)

    def _get_step_size_ps(self):
        """Return integrator step size in picoseconds when available."""
        try:
            step_size = self.sim.integrator.getStepSize()
            return step_size.value_in_unit(unit.picosecond)  # type: ignore
        except Exception:
            return 1.0

    def _pos(self):
        """Returns positions as raw numpy array in nanometers"""
        p = self.sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        return p.value_in_unit(unit.nanometer)  # type: ignore

    def _proj(self, pos, cycle=None):
        """pos is raw numpy array in nm. Returns projection (CV) as numpy array."""
        pos_ang = pos * self.NM_TO_ANG
        kwargs = dict(self.proj_args)
        if cycle is not None:
            import inspect
            sig = inspect.signature(self.proj_fn)
            if 'cycle' in sig.parameters:
                kwargs['cycle'] = cycle
        return np.asarray(self.proj_fn(pos_ang, **kwargs))
