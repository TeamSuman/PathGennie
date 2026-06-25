import os
import numpy as np
import multiprocessing as mp
import gc
# Ensure these utility functions are available in your path or util.py
from .util import omm2np, ommv2np 

class CPUWorker(mp.Process):
    """
    Multiprocessing worker that runs one reusable OpenMM Simulation per process on the CPU.
    Forces the use of the CPU platform by hiding GPU devices.
    """

    def __init__(self, worker_id, task_queue, result_queue, temperature,
                 projection_fn, kwargs, n_steps_per_tau, runner_class,
                 threads=1, runner_kwargs=None):
        super(CPUWorker, self).__init__()
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.temperature = temperature
        self.projection_fn = projection_fn
        self.kwargs = kwargs
        self.runner_class = runner_class
        self.runner_kwargs = runner_kwargs if runner_kwargs is not None else {}
        self.n_steps_per_tau = int(n_steps_per_tau)
        self.threads = str(threads)
        self.sim_obj = None
        self.simulation = None

    def _init_simulation(self):
        """Create Simulation once per process."""
        # Initialize the runner. We pass '0' as device, but it will be ignored 
        # because we hide GPUs in run(), forcing the runner to fall back to CPU.
        self.sim_obj = self.runner_class(device=0, **self.runner_kwargs)
        my_seed = int.from_bytes(os.urandom(4), byteorder='little') & 0x7FFFFFFF

        if hasattr(self.sim_obj, "simulation") and self.sim_obj.simulation is not None:
            self.simulation = self.sim_obj.simulation
        else:
            self.sim_obj._create_simulation(seed = my_seed)
            self.simulation = self.sim_obj.simulation

    def _recreate_simulation_safely(self):
        """Delete old Simulation to free memory, then re-init."""
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
        # --- CPU SPECIFIC SETTINGS ---
        # 1. Hide GPUs to force OpenMM to use the CPU platform
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # 2. Prevent thread oversubscription. 
        # If running N workers on N cores, each worker should use 1 thread.
        os.environ["OPENMM_CPU_THREADS"] = self.threads
        os.environ["OMP_NUM_THREADS"] = self.threads
        # -----------------------------

        self._init_simulation()

        while True:
            task = self.task_queue.get()
            if task == "STOP":
                break

            try:
                idx, positions, weight, velocities = task
                
                # Convert input positions to numpy array (nm)
                # Ensure input format matches what omm2np/ommv2np expects or outputs
                pos_nm = np.asarray(positions, dtype=float).reshape(-1, 3)

                try:
                    self.simulation.context.setPositions(pos_nm)
                except Exception:
                    raise ValueError("Invalid position data")

                # Velocities: set if provided, else randomize
                seed = int.from_bytes(os.urandom(4), byteorder='little') & 0x7FFFFFFF
                if velocities is not None:
                    try:
                        self.simulation.context.setVelocities(velocities)
                    except Exception:
                        self.simulation.context.setVelocitiesToTemperature(self.temperature, seed = seed)
                else:
                    self.simulation.context.setVelocitiesToTemperature(self.temperature)

                # Run MD
                try:
                    self.simulation.step(self.n_steps_per_tau)
                except Exception as e_step:
                    print(f"[CPU {self.worker_id}] step error: {e_step}, minimizing...")
                    try:
                        self.simulation.context.setPositions(pos_nm)
                        self.simulation.minimizeEnergy()
                        self.simulation.context.setVelocitiesToTemperature(self.temperature)
                        self.simulation.step(self.n_steps_per_tau)
                    except Exception as e_min:
                        print(f"[CPU {self.worker_id}] minimize failed: {e_min}, recreating simulation.")
                        self._recreate_simulation_safely()
                        self.simulation.context.setPositions(pos_nm)
                        self.simulation.context.setVelocitiesToTemperature(self.temperature)
                        self.simulation.step(self.n_steps_per_tau)

                # Get final state
                state = self.simulation.context.getState(
                    getPositions=True, getVelocities=True, enforcePeriodicBox=False
                )

                # Use util functions to sanitize/convert units if needed
                pos_np = np.array(state.getPositions(asNumpy=True)) 
                vel_np = np.array(state.getVelocities(asNumpy=True))
                # Note: Ensure omm2np / ommv2np are used here if your pipeline requires unit stripping
                # Example: pos_np = omm2np(state.getPositions(asNumpy=True))
                
                progress = None
                try:
                    progress = self.projection_fn(pos_np, self.kwargs)
                except Exception as e_proj:
                    print(f"[CPU {self.worker_id}] projection error: {e_proj}")

                # Clean up State object to help GC
                del state
                
                # Return result tuple
                self.result_queue.put((idx, pos_np, weight, vel_np, progress))

            except Exception as e_outer:
                import traceback
                print(f"[CPU {self.worker_id}] error: {e_outer}\n{traceback.format_exc()}")
                try:
                    self.result_queue.put((None, None, None, None, None))
                except Exception:
                    pass
        
        # Cleanup on exit
        try:
            if hasattr(self, "simulation"):
                if hasattr(self.simulation, "reporters"):
                    self.simulation.reporters.clear()
                del self.simulation
        except Exception:
            pass
        if hasattr(self, "sim_obj"):
            del self.sim_obj
        gc.collect()
