#!/usr/bin/env python3
import multiprocessing as mp
import sys
import warnings

import numpy as np
from toy import OpenMMRunner

from wepath import WESS

warnings.filterwarnings("ignore")

def warp_criteria(positions, kwargs):
    center = np.array([1.0, 0.0, 0.0])
    dist = np.linalg.norm(positions - center)
    return dist < 0.5


def project(positions, kwargs):

    #path_cv = kwargs["path_cv"]
    #positions = positions.reshape(1, 3)
    #s, z = path_cv.compute(positions)
    #print(positions.shape)
    #if positions.shape[1] > 1:
    #    print(positions)
    #    return np.array([positions[0][0]])
    return np.array([positions.ravel()[0]])


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    # file paths

    initial_positions = [np.array([-1.0, 0.0, 0.0])]
    projection_fn = project

    projection_kwargs = {
    }

    warp_kwargs = {}

    bin_edges = [list(np.linspace(-1.0, 1.0, 45))]

    config = {
        "n_gpus": 1,
        "runner_class": OpenMMRunner,
        "enable_cleaning": False,
        "nbins" : [45],
        "bin_max" : np.array([1.0]),
        "bin_min" : np.array([-1.0]),
        "clean_threshold": 0.25,
        "source_bin_indices": np.array([[0], [1], [2], [3]]),
        "temperature": 200.0,
        #"bin_edges": bin_edges,
        "n_walkers_per_bin": 4,
        "dt": 0.002,
        "n_steps_per_tau": 50,
        "n_iterations": 5000,
        "flux_file": "flux_file_x_1_0.1_T50.txt",
        "bin_file": "bin_file_x_1_0.1_T50.txt",
        "walkers_file": "walker_x_1_0.1_T50.txt",
        "survive_empty": False,
        "warp_function": warp_criteria,
        "warp_kwargs": warp_kwargs,
        "h5_atom_indices": list(range(0, 1)),
        "traj_file": "x_1_T=50.h5",
    }

    we_sim = WESS(
        config=config,
        initial_positions=initial_positions,
        projection_fn=projection_fn,
        kwargs=projection_kwargs,
    )
    we_sim.run()
