"""Alanine dipeptide phi/psi projection for the AMBER example."""

from pathlib import Path
import importlib.util


_COMMON_PROJECTION = Path(__file__).resolve().parents[1] / "common" / "phi_psi.py"
_SPEC = importlib.util.spec_from_file_location("_alanine_dipeptide_phi_psi", _COMMON_PROJECTION)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load projection module: {_COMMON_PROJECTION}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

phi_psi_cv = _MODULE.phi_psi_cv
reached_phi_psi = _MODULE.reached_phi_psi
