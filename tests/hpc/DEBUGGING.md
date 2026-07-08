# PathGennie HPC results — debugging guide (for a follow-up agent)

You are reading the `results/<jobid>/` output of the PathGennie HPC test suite
(see `README.md`). Your job: decide whether the run is healthy, and if not, map
each failing check to its cause and a concrete fix in the codebase. Every result
file is JSON with `status` (`pass`/`fail`/`skip`), `detail`, and `data`.

## How to triage

1. Read `selfcheck.json`. Confirm `checks[*].status`. Also read the top-level
   `scheduler`, `gpus`, `packages`, `md_executables` blocks — most "failures" are
   environment issues (missing module, wrong conda env), not code bugs.
2. Read `smoke_cpu.json` / `gpu_spread.json`. On failure, `stderr_tail` and
   `stdout_tail` contain the MD engine's own error.
3. Only after ruling out the environment, treat it as a code/script bug and use
   the table below. Reproduce locally with the toy engine where possible
   (`tests/test_hpc_parallel.py`, `tests/test_config.py`) — no cluster needed.

## Symptom → cause → fix

| Symptom in results | Likely cause | Where to look / fix |
| ------------------ | ------------ | ------------------- |
| `import_pathgennie` FAIL | Wrong env / not installed | `pip install -e .` in the active conda env; check `packages` block for numpy/pydantic. |
| `config_validation` FAIL (`tau1_steps was dropped`) | Config schema regression | `pathgennie/utils/config.py` — `PathGennieConfig` must declare `tau1_steps`/`tau2_steps` and both models must set `extra="allow"`. Guarded by `tests/test_config.py`. |
| `executable_resolution` FAIL | The resolution logic itself raised (a missing binary is *not* a fail — it is reported as `data.found=false`) | `shutil.which` handling in `pathgennie/backends/{amber/pg_amber,gromacs/pg_gmx}.py`. On a login node without the MD module loaded, `found=false` is normal. |
| `device_masking` FAIL | `resolve_cuda_visible_device` logic changed | `pathgennie/core/parallel.py::resolve_cuda_visible_device`. Must map logical index into the `CUDA_VISIBLE_DEVICES` token list. Guarded by `tests/test_hpc_parallel.py`. |
| `concurrent_openmm` FAIL | Single-GPU concurrent-Context pool broken | `pathgennie/backends/openmm/engine.py` — the Context pool queue and the lock-guarded State cache. Guarded by `tests/test_openmm_engine.py`. A `skip` here just means OpenMM is not installed. |
| `threaded_determinism_and_leak` FAIL (not reproducible) | Seeds drawn inside worker threads again | `pathgennie/core/driver.py` — seeds must be pre-drawn on the main thread (`cycle_seeds`) before `executor.map`. |
| `threaded_determinism_and_leak` FAIL (cache leaked) | Cloned-anchor handle not released | `driver.py` worker must `release(handle)` after `run_segment`; same in `search/rrt.py`. |
| `hdf5_checkpoint` FAIL | h5py missing, or writer error not surfaced | `packages.h5py`; `pathgennie/core/storage.py` re-raises writer-thread errors on `close()`. |
| `unit_tests` FAIL | Toolchain/version drift on the node | Read the pytest summary in `data.summary`; run `pytest -q tests -k <failing>` on the node. |
| `gpu_spread` FAIL: `only GPUs [x] showed activity` **and backend is openmm** | Expected — OpenMM is single-GPU **by design** | Not a bug. OpenMM saturates **one** card with a concurrent-Context pool (`workers_per_device`, or `auto` = cores capped by free GPU memory); it does not spread across GPUs. Use AMBER/GROMACS for a multi-GPU spread test. To confirm OpenMM is using the GPU, set `openmm.platform: CUDA` and watch a single card's utilization/memory rise as `workers_per_device` increases. |
| `gpu_spread` FAIL: only 1 GPU active on **amber/gromacs** | Device routing not reaching the engine, or MD binary is not the CUDA build | 1) Confirm `PG_EXE` is the CUDA binary (`pmemd.cuda`, GROMACS built with `-DGMX_GPU=CUDA`). 2) Confirm `data.cuda_visible_devices` shows ≥2 GPUs. 3) Check the engine still sets `CUDA_VISIBLE_DEVICES` per segment via `resolve_cuda_visible_device` (`backends/amber/engine.py`, `backends/gromacs/pg_gmx.py`). 4) `nvidia-smi` monitor can miss very short segments — raise `--max-cycle`. |
| `gpu_spread` targets the wrong physical GPUs / collides with another job | `CUDA_VISIBLE_DEVICES` not honored | The engine must never overwrite the mask with an absolute id. Verify `resolve_cuda_visible_device`. On PBS, ensure the mask was derived from `$PBS_GPUFILE` (the `pbs_gpu.pbs` script does this). |
| smoke run FAIL: `Amber restart ... contains overflow coordinates` | MD blew up (bad input, too-large timestep) | The system in the example must be equilibrated for your force field; check `md.controls.dt`. Not a PathGennie bug. |
| smoke run FAIL: `executable not found` | `PG_EXE` wrong | Set `PG_EXE` to the absolute binary path in the job script. |
| smoke run hangs | MD segment stuck / shared-FS stall | Scratch on Lustre/NFS under thousands of tiny segments stalls; point `workdir`/`$TMPDIR` at node-local SSD. See "Performance" below. |
| Very slow throughput on large systems | Per-segment process launch dominates ultrashort `tau1` | Architectural: subprocess backends re-launch pmemd/gmx (GPU context + topology parse) every segment. For large systems prefer the OpenMM in-process engine, larger `tau1`, or fewer/bigger segments. Tracked in `ROADMAP.md`. |

## Performance / scaling knobs (for tuning, not failures)

- **CPU oversubscription**: set `pathgennie.cpu_threads_per_worker` so
  `workers_per_device × cpu_threads_per_worker ≈ cores`. The GROMACS engine also
  injects `-ntomp`. Without this, N concurrent `gmx mdrun` each grab all cores.
- **Node-local scratch**: run inside `$TMPDIR` (or set `workdir` there) and copy
  results back at the end. Thousands of ultrashort segments generate heavy
  metadata I/O that cripples shared filesystems.
- **GPU packing**: `workers_per_device > 1` helps only if a single segment
  under-fills the GPU (small systems). For large systems use 1 worker/GPU.

## If you change code

Re-run the fast, cluster-free guards before resubmitting a job:
```bash
pytest -q tests/test_config.py tests/test_hpc_parallel.py tests/test_robustness.py
python tests/hpc/hpc_selfcheck.py --skip-unit-tests
```
Keep any new HPC-relevant invariant covered by a test in `tests/` so it is caught
without cluster access.
