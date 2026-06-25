import numpy as np
import plumed
import MDAnalysis as mda

class PlumedCVRunner:
    def __init__(self, n_atoms, plumed_input, cv_names, masses=None, log_file="plumed.log"):
        self.cv_names = list(cv_names)
        self.p = plumed.Plumed()

        self.p.cmd("setMDEngine", "python")
        self.p.cmd("setTimestep", 1.0)
        self.p.cmd("setKbT", 1.0)
        self.p.cmd("setNatoms", int(n_atoms))
        self.p.cmd("setLogFile", log_file)
        self.p.cmd("init")

        for line in plumed_input.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("PRINT "):
                continue
            self.p.cmd("readInputLine", line)

        self.masses = np.asarray(
            np.ones(n_atoms, dtype=np.float64) if masses is None else masses,
            dtype=np.float64,
        )
        self.forces = np.zeros((n_atoms, 3), dtype=np.float64)
        self.virial = np.zeros((3, 3), dtype=np.float64)

        self.value_bufs = {}
        for name in self.cv_names:
            shape = np.zeros(1, dtype=np.int_)
            self.p.cmd(f"getDataRank {name}", shape)
            buf = np.zeros(1, dtype=np.float64)
            self.p.cmd(f"setMemoryForData {name}", buf)
            self.value_bufs[name] = buf

    def compute(self, positions_A, box_A=None, step=0):
        pos_nm = np.asarray(positions_A, dtype=np.float64) * 0.1

        if box_A is None:
            box_nm = np.zeros((3, 3), dtype=np.float64)
        else:
            box_nm = np.asarray(box_A, dtype=np.float64) * 0.1
            if box_nm.shape == (3,):
                box_nm = np.diag(box_nm)

        self.p.cmd("setStep", int(step))
        self.p.cmd("setMasses", self.masses)
        self.p.cmd("setPositions", pos_nm)
        self.p.cmd("setForces", self.forces)
        self.p.cmd("setVirial", self.virial)
        self.p.cmd("setBox", box_nm)
        self.p.cmd("calc")

        return np.array([self.value_bufs[name][0] for name in self.cv_names], dtype=np.float64)

def compute_cvs(u, runner):
    test_cv = []
    for ts in u.trajectory:
        coords = u.atoms.positions
        box = u.dimensions[:3]
        cvs = runner.compute(coords, box_A=box, step=0)
        test_cv.append(cvs)
    return np.array(test_cv)

u = mda.Universe("system.prmtop", "system.rst7", format="RESTRT")
import os
runner = PlumedCVRunner(
    n_atoms=u.atoms.n_atoms,
    plumed_input=open("plumed.dat").read(),
    cv_names=["V1", "V2","V3", "cyl.z", "comdist"],
    masses=u.atoms.masses,
    log_file=f"plumed_{os.getpid()}.log"
)

import numpy as np

# Global tracker for when the cooldown expires
cooldown_end_cycle = -1

def project(positions, box, cycle=0):
    global cooldown_end_cycle
    cvs = runner.compute(positions, box_A=box, step=0)
    
    # 1. Trigger the cooldown if we cross the threshold and aren't already cooling down
    if cvs[1] >= 2.25 and cycle > cooldown_end_cycle:
        # Lock into the reduced space for exactly 50 cycles
        cooldown_end_cycle = cycle + 50
        print(f"*** Threshold reached at cycle {cycle}! Reducing space until cycle {cooldown_end_cycle} ***")
        
    # 2. Check if we are currently inside the cooldown window
    if cycle <= cooldown_end_cycle:
        # Return the dynamically reduced 2D space (dropping the first 3 CVs)
        # PathGennie's engine will automatically detect the shape change 
        # and safely freeze the corresponding axes in the initial_cv comparison!
        return np.array([cvs[3], cvs[4]])
    
    # 3. Otherwise, revert to the full 5D original space!
    return np.array([min(1.5, cvs[0]), cvs[1], min(2.0, cvs[2]), cvs[3], cvs[4]])
def __project__(positions, box):
    box = box
    cvs = runner.compute(positions, box_A=box, step=0)
    if cvs[1] >= 2.25:
        return np.array([cvs[3], cvs[4]]) 
    return np.array([min(1.5, cvs[0]), min(2.5, cvs[1]), min(2.0, cvs[2]), cvs[3], cvs[4]])

def convergence_fn(pos, box):
    box = box
    cv = project(pos,  box = box)
    return cv[-2] > 1.8
