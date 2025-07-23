# Modules
from openmm import *
from openmm.app import *
from openmm.unit import *
import os
#from openmmplumed import PlumedForce

class Simulation:
    def __init__(self, gro_file, top_file, include_dir, temperature = 300,
                  pressure = 1.0, dt = 0.004, platform_name='CUDA', precision='mixed', device = '0',
                  plumed = None):
        # Input Files
        file_extension = os.path.splitext(gro_file)[1]
        if file_extension == '.gro':
            self.gro = GromacsGroFile(gro_file)
            self.top = GromacsTopFile(top_file, includeDir=include_dir, periodicBoxVectors=self.gro.getPeriodicBoxVectors())
        elif file_extension == '.pdb':
            self.gro = PDBFile(gro_file)
        elif file_extension == '.xml':
            self.gro = XmlSerializer.deserialize(open(gro_file).read())
        elif file_extension == '.crd':
            self.gro = AmberInpcrdFile(gro_file)
            self.top = AmberPrmtopFile(top_file, periodicBoxVectors=self.gro.boxVectors)
        else:
            raise ValueError(f'Unsupported file format: {file_extension}')
        if gro_file is None:
            raise ValueError('No GRO file specified')
        #self.gro = GromacsGroFile(gro_file)
        #self.top = GromacsTopFile(top_file, includeDir=include_dir, periodicBoxVectors=self.gro.getPeriodicBoxVectors())

        # System Configuration
        self.nonbondedMethod = PME
        self.nonbondedCutoff = 1.0 * nanometers # type: ignore
        self.ewaldErrorTolerance = 0.0005
        self.constraints = HBonds
        self.rigidWater = True
        self.constraintTolerance = 0.000001
        self.hydrogenMass = 1.5 * amu

        # Integration Options
        self.dt = dt * picoseconds # type: ignore
        self.temperature = temperature * kelvin
        self.friction = 1.0 / picosecond # type: ignore
        self.pressure = pressure * atmospheres
        self.barostatInterval = 25
        self.plumed = plumed
        # Simulation Options
        #self.steps = 10000000
        self.equilibrationSteps = 5000
        self.platform = Platform.getPlatformByName(platform_name)
        self.platformProperties = {'Precision': precision, 'DeviceIndex': str(device)}

        #if self.dataReporter is None:
        #    self.dataReporter = StateDataReporter('log.txt', 5000, totalSteps=self.steps,
        #                                        step=True, speed=True, progress=True, potentialEnergy=True, temperature=True, separator='\t')
        #self.checkpointReporter = CheckpointReporter('checkpoint.chk', 1000000)

        # Prepare the Simulation
        self.topology = self.top.topology
        self.positions = self.gro.positions
        self.system = self.top.createSystem(nonbondedMethod=self.nonbondedMethod, nonbondedCutoff=self.nonbondedCutoff,
                                            constraints=self.constraints, rigidWater=self.rigidWater, ewaldErrorTolerance=self.ewaldErrorTolerance, hydrogenMass=self.hydrogenMass)
        self.system.addForce(MonteCarloBarostat(self.pressure, self.temperature, self.barostatInterval))
        ## Add plumed force
        if self.plumed is not None:
            if not os.path.exists(self.plumed):
                raise FileNotFoundError(f'Plumed file not found: {self.plumed}')
            script = open(plumed).read()
            #self.system.addForce(PlumedForce(script))

        self.integrator = LangevinMiddleIntegrator(self.temperature, self.friction, self.dt)
        self.integrator.setConstraintTolerance(self.constraintTolerance)
        if self.platform == 'CUDA':
            self.simulation = Simulation(self.topology, self.system, self.integrator, self.platform, self.platformProperties)
        else:
            self.simulation = Simulation(self.topology, self.system, self.integrator, self.platform)

        self.simulation.context.setPositions(self.positions)

    def minimize(self):
        print('Performing energy minimization...')
        self.simulation.minimizeEnergy()

    def equilibrate(self):
        print('Equilibrating...')
        self.simulation.context.setVelocitiesToTemperature(self.temperature)
        self.simulation.step(self.equilibrationSteps)

    def run_production(self, position = None, steps = 10000, traj = "trajectory.xtc", save_freq=5000):
        #print('Simulating...')
        if position is not None:
            self.simulation.context.setPositions(position)
            self.simulation.context.setVelocitiesToTemperature(self.temperature)
        self.Reporter = XTCReporter(traj, save_freq)
        self.simulation.reporters = [self.Reporter]
        #self.simulation.reporters.append(self.dataReporter)
        #self.simulation.reporters.append(self.checkpointReporter)
        self.simulation.currentStep = 0
        self.simulation.step(steps)

# Example usage:
# sim = MolecularDynamicsSimulation('../Data/AlaD/conf.gro', '../Data/AlaD/topol.top', '/usr/local/gromacs/share/gromacs/top')
# sim.minimize()
# sim.equilibrate()
# sim.run_production()
