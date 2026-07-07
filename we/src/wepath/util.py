import numpy as np
from openmm import unit
import json

def omm2np(pos_arr):
    if isinstance(pos_arr, np.ndarray):
        # OpenMM's asNumpy=True gives unitless values in nm → convert to Å
        return  pos_arr.astype(np.float32, copy=False)
    return np.array(pos_arr.value_in_unit(unit.nanometer), dtype=np.float32)

def ommv2np(vel_arr):
    """Return velocities in OpenMM's default (nm/ps)."""
    if isinstance(vel_arr, np.ndarray):
        return vel_arr.astype(np.float32, copy=False)  # already nm/ps
    return np.array(vel_arr.value_in_unit(unit.nanometer/unit.picosecond), dtype=np.float32)


def make_json_serializable(obj):
    """Recursively convert numpy ints and sets to JSON-safe types."""
    if isinstance(obj, dict):
        return {make_json_serializable(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(x) for x in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    else:
        return obj

def save_json_on_the_fly(data, filename):
    """Append JSON record per iteration."""
    safe_data = make_json_serializable(data)
    with open(filename, "a") as f:
        json.dump(safe_data, f)
        f.write("\n")  # one JSON object per line
