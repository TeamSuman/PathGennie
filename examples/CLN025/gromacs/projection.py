"""CLN025 projection functions shared with the AMBER example."""

from pathlib import Path
import importlib.util


_AMBER_PROJECTION = Path(__file__).resolve().parents[1] / "amber" / "projection.py"
_SPEC = importlib.util.spec_from_file_location("_cln025_amber_projection", _AMBER_PROJECTION)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load projection module: {_AMBER_PROJECTION}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

end_to_end_cv = _MODULE.end_to_end_cv
end_to_end_escaped = _MODULE.end_to_end_escaped
rmsd_cv = _MODULE.rmsd_cv
escaped = _MODULE.escaped
load_reference = _MODULE.load_reference
