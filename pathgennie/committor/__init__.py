"""Committor learning and validation on top of PathGennie swarms.

A PathGennie cycle launches ``max_trial`` independent, *unbiased* short
trajectories from one anchor. That is exactly the data the self-consistency
(semigroup) characterisation of the committor needs -- several independent
continuations per starting point -- so the swarm that steers discovery can also
teach a committor, with no bias force and no gradient passed to the MD engine.

Modules
-------
``basins``     reactant/product state definitions in CV space
``records``    :class:`SwarmRecorder`, the driver hook that keeps *every* trial
``shooting``   brute-force shooting committor (the ground truth for validation)
``reference``  numerically exact committor on a 2-D grid (toy potentials)
``train``      PyTorch committor network and self-consistency training

Only ``train`` needs PyTorch; the rest is NumPy (``reference`` needs SciPy).
"""

from .basins import BasinPair, BoxBasin, CallableBasin, SphereBasin
from .records import SwarmDataset, SwarmRecorder
from .shooting import ShootingResult, shooting_committor

__all__ = [
    "BasinPair",
    "BoxBasin",
    "CallableBasin",
    "SphereBasin",
    "SwarmDataset",
    "SwarmRecorder",
    "ShootingResult",
    "shooting_committor",
]
