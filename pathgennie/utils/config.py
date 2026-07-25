"""Configuration validation using Pydantic.

The schema is deliberately *permissive but typed*: the fields the runtime relies
on are declared with their real names (``tau1_steps``/``tau2_steps`` etc.) and,
where sensible, bounds-checked so a malformed value fails fast with a clear
error. Everything else (backend-specific keys, ``downstream``, ``profile``,
``devices``, and the named downstream-stage blocks) is preserved via
``extra="allow"`` so nothing the user wrote is ever silently dropped.

Historical note: an earlier revision declared ``tau1``/``tau2`` (not
``tau1_steps``/``tau2_steps``) and used Pydantic's default ``extra="ignore"``.
That silently discarded every real key — ``tau1_steps``, ``devices``,
``downstream``, ``profile`` — and dropped whole top-level sections (``md``,
``workdir``, ``output``), so the user's MD parameters were ignored and every
backend crashed with ``KeyError: 'tau1_steps'``. Both problems are fixed here.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PathGennieConfig(BaseModel):
    """Adaptive-sampling parameters (the ``pathgennie:`` block of ``input.yaml``).

    Declared fields are validated when present. Optional fields default to
    ``None`` and are stripped by ``model_dump(exclude_none=True)`` so the
    backends' own ``.get(key, default)`` fallbacks (and profile defaults) still
    apply — declaring a field here never changes a backend's default behaviour.
    Unknown keys are kept (``extra="allow"``).
    """

    model_config = ConfigDict(extra="allow")

    # Core adaptive-cycle parameters. Left optional (not defaulted) so a run
    # profile (``profile:``) may supply them; a truly missing required value
    # surfaces as a clear error when the backend reads it.
    mode: Optional[Literal["escape", "target"]] = Field(None, description="Sampling mode")
    tau1_steps: Optional[int] = Field(None, gt=0, description="Swarm (tau1) MD steps per trial")
    tau2_steps: Optional[int] = Field(None, gt=0, description="Commit (tau2) MD steps")
    max_trial: Optional[int] = Field(None, gt=0, description="Swarm trials per cycle")
    max_cycle: Optional[int] = Field(None, gt=0, description="Maximum number of adaptive cycles")
    save_freq: Optional[int] = Field(None, gt=0, description="Save a frame every N cycles")
    temperature: Optional[float] = Field(None, gt=0, description="Temperature in Kelvin")
    sigma: Optional[float] = Field(None, gt=0, description="Selection softness (>0)")
    seed: Optional[int] = Field(None, description="Master RNG seed for reproducibility")

    # Mode / metric configuration.
    target_projection: Optional[List[float]] = Field(None, description="Target CV for 'target' mode")
    escape_metric: Optional[Literal["cv0", "distance_from_start"]] = Field(
        None, description="Escape-mode progress metric"
    )

    # Parallel / device placement.
    devices: Optional[List[int]] = Field(None, description="Logical GPU indices for the swarm")
    workers_per_device: Optional[Union[int, Literal["auto"]]] = Field(
        None, description="Concurrent segments per device (int), or 'auto' for OpenMM single-GPU saturation"
    )
    tau1_workers: Optional[int] = Field(None, gt=0, description="Legacy alias for workers_per_device")
    cpu_threads_per_worker: Optional[int] = Field(
        None, gt=0, description="OMP/MKL threads exported per subprocess worker (CPU oversubscription guard)"
    )

    # Downstream enhanced-sampling stage and goal-driven profiles.
    downstream: Optional[str] = Field(None, description="Downstream stage name, e.g. 'weighted_ensemble'")
    profile: Optional[str] = Field(None, description="Run profile: 'discovery' or 'sampling'")
    goal: Optional[str] = Field(None, description="Alias for profile")

    # Behaviour flags / streaming.
    reject_worse_tau2: Optional[bool] = Field(None, description="Reject a tau2 runner worse than its sampler")
    reject_worse_anchor: Optional[bool] = Field(None, description="Reject a candidate worse than the anchor")
    collect_seeds: Optional[bool] = Field(None, description="Retain restartable seeds for a downstream stage")
    checkpoint_path: Optional[str] = Field(None, description="Stream frames/metrics to this HDF5 file")
    verbosity: Optional[int] = Field(None, ge=0, description="0 = silent, 1 = per-save logging")

    # Intra-segment frame capture (Phase 2).
    save_subframes: Optional[bool] = Field(
        None, description="Capture intra-segment frames for the chosen walker"
    )
    subframe_stride: Optional[int] = Field(
        None, gt=0, description="Save a subframe every N integrator steps within each segment"
    )

    # Checkpoint / restart.
    checkpoint_freq: Optional[int] = Field(
        None, ge=0,
        description="Save a full restart checkpoint every N cycles (0 = disabled). "
                    "Typically much larger than save_freq."
    )
    overwrite: Optional[bool] = Field(
        None,
        description="Allow overwriting existing output files. "
                    "Default is False — raises FileExistsError if outputs exist."
    )


class AppConfig(BaseModel):
    """Top-level ``input.yaml`` schema.

    ``extra="allow"`` is essential: besides the declared sections, real configs
    carry ``md:``, ``workdir:``, ``output:`` and a downstream-stage block named
    after ``pathgennie.downstream`` (e.g. ``weighted_ensemble:``). These must
    survive validation, so unknown top-level keys are preserved rather than
    dropped.
    """

    model_config = ConfigDict(extra="allow")

    pathgennie: PathGennieConfig = Field(default_factory=PathGennieConfig)
    amber: Optional[Dict[str, Any]] = None
    gromacs: Optional[Dict[str, Any]] = None
    openmm: Optional[Dict[str, Any]] = None
    projection: Optional[Dict[str, Any]] = None
    convergence: Optional[Dict[str, Any]] = None
    md: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None
    workdir: Optional[str] = None
    scratch_root: Optional[str] = Field(
        None, description="Place per-segment scratch here (e.g. node-local $TMPDIR) instead of workdir/scratch"
    )


def load_config(filepath: Path | str) -> AppConfig:
    """Load and validate a PathGennie YAML configuration file."""
    filepath = Path(filepath)
    with filepath.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raise ValueError(f"Configuration file is empty: {filepath}")
    if not isinstance(raw_config, dict):
        raise ValueError(f"Configuration file must contain a top-level mapping: {filepath}")

    return AppConfig.model_validate(raw_config)
