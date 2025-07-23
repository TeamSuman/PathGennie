import os, sys
#sys.path.insert(0, "./omm/")

from omm.omm_readinputs import *
from omm.omm_readparams import *
from omm.omm_vfswitch import *
from omm.omm_barostat import *
from omm.omm_restraints import *
from omm.omm_rewrap import *

from openmm import *
from openmm.app import *
from openmm.unit import *

class OpenMMSimulation:
    def __init__(self, inpfile, topfile, crdfile, fftype, toppar, sysinfo, irst=None, ffdir = None):
        self.inpfile = inpfile
        self.topfile = topfile
        self.crdfile = crdfile
        self.fftype = fftype.upper()
        self.toppar = toppar
        self.sysinfo = sysinfo
        self.irst = irst
        self.ffdir = ffdir

        self.inputs = None
        self.top = None
        self.crd = None
        self.params = None
        self.system = None
        self.integrator = None
        self.simulation = None

    def setup_simulation(self):
        self.inputs = read_inputs(self.inpfile)
        self.top = read_top(self.topfile, self.fftype, self.ffdir, self.crdfile)
        self.crd = read_crd(self.crdfile, self.fftype)
        if self.fftype == "CHARMM":
            self.params = read_params(self.toppar)
            self.top = read_box(self.top, self.sysinfo)

        # Build system
        nboptions = dict(
            nonbondedMethod=self.inputs.coulomb,
            nonbondedCutoff=self.inputs.r_off * nanometers,
            constraints=self.inputs.cons,
            ewaldErrorTolerance=self.inputs.ewald_Tol,
        )
        if self.inputs.vdw == "Switch":
            nboptions["switchDistance"] = self.inputs.r_on * nanometers
        if self.inputs.vdw == "LJPME":
            nboptions["nonbondedMethod"] = LJPME
        #print(nboptions, self.params)
        if self.params:
            self.system = self.top.createSystem(self.params, **nboptions)
        else:
            self.system = self.top.createSystem(**nboptions)
        if self.inputs.vdw == "Force-switch":
            self.system = vfswitch(self.system, self.top, self.inputs)
        if self.inputs.lj_lrc == "yes":
            for force in self.system.getForces():
                if isinstance(force, NonbondedForce):
                    force.setUseDispersionCorrection(True)
                if isinstance(force, CustomNonbondedForce) and force.getNumTabulatedFunctions() != 1:
                    force.setUseLongRangeCorrection(True)

        if self.inputs.e14scale != 1.0:
            for force in self.system.getForces():
                if isinstance(force, NonbondedForce):
                    nonbonded = force
                    break
            for i in range(nonbonded.getNumExceptions()):
                atom1, atom2, chg, sig, eps = nonbonded.getExceptionParameters(i)
                nonbonded.setExceptionParameters(i, atom1, atom2, chg * self.inputs.e14scale, sig, eps)

        if self.inputs.pcouple == "yes":
            self.system = barostat(self.system, self.inputs)
        if self.inputs.rest == "yes":
            self.system = restraints(self.system, self.crd, self.inputs)

        self.integrator = LangevinIntegrator(
            self.inputs.temp * kelvin,
            self.inputs.fric_coeff / picosecond,
            self.inputs.dt * picoseconds,
        )
        platform = Platform.getPlatformByName("CUDA")
        prop = dict(CudaPrecision="single") if platform.getName() == "CUDA" else dict()
        self.simulation = Simulation(self.top.topology, self.system, self.integrator, platform, prop)
        self.simulation.context.setPositions(self.crd.positions)

        if self.irst:
            with open(self.irst, "r") as f:
                self.simulation.context.setState(XmlSerializer.deserialize(f.read()))
