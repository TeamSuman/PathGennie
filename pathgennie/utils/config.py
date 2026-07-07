"""Configuration validation using Pydantic."""

from typing import Any, Dict, List, Optional
import yaml
from pathlib import Path
from pydantic import BaseModel, Field


class PathGennieConfig(BaseModel):
    mode: str = Field("escape", description="Sampling mode: 'escape' or 'target'")
    tau1: int = Field(200, description="Swarm steps")
    tau2: int = Field(200, description="Commit steps")
    max_trial: int = Field(20, description="Number of parallel trials per cycle")
    max_cycle: int = Field(5000, description="Maximum number of cycles")
    save_freq: int = Field(10, description="Save frequency")
    temperature: float = Field(300.0, description="Temperature in Kelvin")
    sigma: float = Field(0.5, description="Selection softness")
    seed: Optional[int] = Field(None, description="Random seed")
    collect_seeds: bool = Field(False, description="Collect seeds for downstream")
    target_projection: Optional[List[float]] = Field(None, description="Target projection for 'target' mode")
    checkpoint_path: Optional[str] = Field(None, description="HDF5 checkpoint path")
    reject_worse_tau2: bool = Field(False, description="Reject candidates worse than the anchor")
    reject_worse_anchor: bool = Field(False, description="Reject candidates worse than the existing anchor")


class AppConfig(BaseModel):
    pathgennie: PathGennieConfig = Field(default_factory=PathGennieConfig)
    openmm: Optional[Dict[str, Any]] = None
    gromacs: Optional[Dict[str, Any]] = None
    amber: Optional[Dict[str, Any]] = None
    downstream: Optional[Dict[str, Any]] = None
    projection: Optional[Dict[str, Any]] = None
    convergence: Optional[Dict[str, Any]] = None


def load_config(filepath: Path | str) -> AppConfig:
    """Load and validate a PathGennie YAML configuration file."""
    filepath = Path(filepath)
    with filepath.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
        
    return AppConfig.model_validate(raw_config)
