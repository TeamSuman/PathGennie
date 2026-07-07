import numpy as np
import copy
class Walker:

    def __init__(self, positions, weight, velocities=None, progress_coord=None):
        self.positions = positions
        self.velocities = velocities
        self.weight = weight
        self.progress_coord = progress_coord

    def clone(self):
        new_positions = self.positions.copy()
        new_velocities = self.velocities.copy() if self.velocities is not None else None
        new_weight = copy.deepcopy(self.weight)
        new_walker = Walker(
            new_positions,
            new_weight,
            new_velocities
        )
        new_walker.progress_coord = self.progress_coord
        return new_walker

    def __repr__(self):
        if self.progress_coord is None:
            pc_str = "PC=None"
        elif isinstance(self.progress_coord, np.ndarray):
            pc_str = ", ".join(
                f"pc{i+1}={val:.2f}" for i, val in enumerate(self.progress_coord)
            )
        else:
            pc_str = f"PC={self.progress_coord:.2f}"

        return (f"Walker({pc_str}, W={self.weight:.4e})")
