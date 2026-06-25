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

u = mda.Universe("box.gro")
runner = PlumedCVRunner(
    n_atoms=u.atoms.n_atoms,
    plumed_input=open("plumed.dat").read(),
    cv_names=["V1", "V2","V3", "cyl.z"],
    masses=u.atoms.masses,
)
def project(positions, box):
    box = box
    cvs = runner.compute(positions, box_A=box, step=0)
    return cvs
def convergence_fn(pos, box):
    box = box
    cv = project(pos,  box = box)
    return np.linalg.norm(cv - target) < 0.05
