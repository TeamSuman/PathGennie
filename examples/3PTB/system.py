import os, sys
sys.path.insert(0, "../../Scripts/omm")
from omm_simulator import OpenMMSimulation

class Simulation_obj:
    def __init__(self, device_idx = 0):
        path = "./"
        crd_file = os.path.join(path, "step3_input.crd")
        top_file = os.path.join(path, "step3_input.psf")
        inp_file = os.path.join(path,"step5_production.inp")
        toppar = os.path.join(path,"toppar.str")
        sysinfo = os.path.join(path, "sysinfo.dat")

        sim = OpenMMSimulation(inpfile=inp_file, topfile=top_file, crdfile=crd_file,
                        toppar = toppar, sysinfo = sysinfo, device_idx=device_idx)

        self.simulation = sim.simulation
        self.simulation.context.reinitialize()
