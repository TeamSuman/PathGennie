import gc
import multiprocessing as mp
import os

import numpy as np

from .util import omm2np, ommv2np


class GPUWorker(mp.Process):
    """
    Multiprocessing worker that runs one reusable OpenMM Simulation per process.
    Uses omm2np to convert positions/velocities to NumPy (Å units for positions).
    """

    def __init__(
        self,
        gpu_id,
        task_queue,
        result_queue,
        temperature,
        projection_fn,
        kwargs,
        n_steps_per_tau,
        runner_class,
        runner_kwargs=None
    ):
        super(GPUWorker, self).__init__()
        self.gpu_id = int(gpu_id)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.temperature = temperature
        self.projection_fn = projection_fn
        self.kwargs = kwargs
        self.runner_class = runner_class
        self.runner_kwargs = runner_kwargs if runner_kwargs is not None else {}
        self.n_steps_per_tau = int(n_steps_per_tau)
        self.sim_obj = None
        self.simulation = None

    def _init_simulation(self):
        """Create Simulation once per process."""
        self.sim_obj = self.runner_class(**self.runner_kwargs)
        if hasattr(self.sim_obj, "simulation") and self.sim_obj.simulation is not None:
            self.simulation = self.sim_obj.simulation
        else:
            self.sim_obj._create_simulation()
            self.simulation = self.sim_obj.simulation

    def _recreate_simulation_safely(self):
        """Delete old Simulation to free GPU memory, then re-init."""
        try:
            if hasattr(self, "simulation") and self.simulation is not None:
                if hasattr(self.simulation, "reporters"):
                    self.simulation.reporters.clear()
                del self.simulation
        except Exception:
            pass
        gc.collect()
        self._init_simulation()

    def run(self):
        # Respect a scheduler-provided GPU allocation. If CUDA_VISIBLE_DEVICES is
        # already set (Slurm --gres=gpu / PBS), treat gpu_id as a position within
        # that allocation rather than overwriting it with an absolute index --
        # otherwise the worker targets a GPU this job was not granted and
        # collides with another user's job on a shared node.
        base = os.environ.get("CUDA_VISIBLE_DEVICES", "") or ""
        tokens = [tok.strip() for tok in base.split(",") if tok.strip() != ""]
        if tokens:
            os.environ["CUDA_VISIBLE_DEVICES"] = tokens[int(self.gpu_id) % len(tokens)]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        self._init_simulation()

        while True:
            task = self.task_queue.get()
            if task == "STOP":
                break

            try:
                idx, positions, weight, velocities = task
                # print("Got:", positions)
                # Set positions (convert numpy Å → Vec3 in nm if needed)
                # print("taken:", positions)
                pos_nm = np.asarray(positions, dtype=float).reshape(
                    -1, 3
                )  # angstrom → nm
                # print("NM:", pos_nm)
                try:
                    self.simulation.context.setPositions(pos_nm)
                except Exception:
                    raise ValueError("Invalid position data")

                # Velocities: set if provided, else randomize
                if velocities is not None:
                    try:
                        self.simulation.context.setVelocities(velocities)
                        # self.simulation.context.setVelocitiesToTemperature(self.temperature)
                    except Exception:
                        self.simulation.context.setVelocitiesToTemperature(
                            self.temperature
                        )
                else:
                    self.simulation.context.setVelocitiesToTemperature(self.temperature)

                # Run MD
                try:
                    self.simulation.step(self.n_steps_per_tau)
                except Exception as e_step:
                    print(f"[GPU {self.gpu_id}] step error: {e_step}, minimizing...")
                    try:
                        self.simulation.context.setPositions(positions)
                        self.simulation.minimizeEnergy()
                        self.simulation.context.setVelocitiesToTemperature(
                            self.temperature
                        )
                        self.simulation.step(self.n_steps_per_tau)
                    except Exception as e_min:
                        print(
                            f"[GPU {self.gpu_id}] minimize failed: {e_min}, recreating simulation."
                        )
                        self._recreate_simulation_safely()
                        self.simulation.context.setPositions(positions)
                        self.simulation.context.setVelocitiesToTemperature(
                            self.temperature
                        )
                        self.simulation.step(self.n_steps_per_tau)

                # Get final state
                state = self.simulation.context.getState(
                    getPositions=True, getVelocities=True, enforcePeriodicBox=True
                )

                pos_np = omm2np(state.getPositions(asNumpy=True))  # nm
                vel_np = ommv2np(
                    state.getVelocities(asNumpy=True)
                )  # nm/ps if omm2np unchanged
                pos_np = np.array(state.getPositions(asNumpy=True))
                # vel_np = None
                progress = None
                try:
                    progress = self.projection_fn(pos_np, self.kwargs)
                except Exception as e_proj:
                    print(f"[GPU {self.gpu_id}] projection error: {e_proj}")
                # print("Finally:", pos_np)
                del state
                gc.collect()

                # FIX: Return all 5 items to match the receiver in the base class.
                self.result_queue.put(
                    (idx, pos_np, weight, vel_np, progress)
                )  # <<< CHANGED

            except Exception as e_outer:
                import traceback

                print(f"[GPU {self.gpu_id}] error: {e_outer}\n{traceback.format_exc()}")
                try:
                    # Return Nones for a failed propagation
                    self.result_queue.put((None, None, None, None, None))
                except Exception:
                    pass

        try:
            if hasattr(self, "simulation"):
                self.simulation.reporters.clear()
                del self.simulation
        except Exception:
            pass
        del self.sim_obj
        gc.collect()
