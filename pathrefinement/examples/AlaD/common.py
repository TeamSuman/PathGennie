import os
import sys
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

PHI_ATOMS = (4, 6, 8, 14)  # ACE C, ALA N, ALA CA, ALA C
PSI_ATOMS = (6, 8, 14, 16)  # ALA N, ALA CA, ALA C, NME N

def load_coords(file_path, topology_file=None):
    import MDAnalysis as mda
    if file_path.endswith(".dcd"):
        if topology_file is None:
            raise ValueError("topology_file must be specified to load a DCD file.")
        u = mda.Universe(topology_file, file_path)
    else:
        u = mda.Universe(file_path)
    n_frames = len(u.trajectory)
    coords = []
    for ts in u.trajectory:
        coords.append(u.atoms.positions.copy() / 10.0) # Angstroms to nm
    coords = np.array(coords)
    if n_frames == 1:
        return coords[0] # (N, 3)
    return coords # (N_frames, N, 3)

def create_alad_system(gro_file, top_file, device=0):
    ffdir = os.environ.get("GMXFFDIR", "/usr/share/gromacs/top")  # override via $GMXFFDIR
    gro = app.GromacsGroFile(gro_file)
    top = app.GromacsTopFile(
        top_file,
        periodicBoxVectors=gro.getPeriodicBoxVectors(),
        includeDir=ffdir
    )
    
    system = top.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0*unit.nanometer,
        constraints=app.AllBonds,
        rigidWater=True,
        ewaldErrorTolerance=1e-5,
        hydrogenMass=1.5*unit.amu
    )
    
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            force.setUseDispersionCorrection(True)
            
    system.addForce(mm.CMMotionRemover(100))
    
    integrator = mm.LangevinMiddleIntegrator(
        300*unit.kelvin,
        1.0/unit.picosecond,
        4.0*unit.femtoseconds
    )
    integrator.setConstraintTolerance(1e-6)
    
    try:
        platform = mm.Platform.getPlatformByName("CUDA")
        props = {"Precision": "mixed", "DeviceIndex": str(device)}
    except Exception:
        platform = mm.Platform.getPlatformByName("CPU")
        props = {}
        
    simulation = app.Simulation(top.topology, system, integrator, platform, props)
    return simulation

def get_heavy_indices(gro_file, top_file):
    ffdir = os.environ.get("GMXFFDIR", "/usr/share/gromacs/top")  # override via $GMXFFDIR
    gro = app.GromacsGroFile(gro_file)
    top = app.GromacsTopFile(
        top_file,
        periodicBoxVectors=gro.getPeriodicBoxVectors(),
        includeDir=ffdir
    )
    indices = []
    for atom in top.topology.atoms():
        if atom.residue.index < 3:
            if not atom.name.startswith("H") and not atom.name.startswith("h"):
                indices.append(atom.index)
    return np.array(indices)

def compute_pairwise_distances(coords, indices):
    """Compute pairwise distances between selected atoms."""
    # coords shape is expected to be (N_atoms, 3) in Angstroms
    heavy_coords = coords[indices]
    diff = heavy_coords[:, np.newaxis, :] - heavy_coords[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    i, j = np.triu_indices(len(indices), k=1)
    return dist[i, j]

def dihedral_degrees(coords, atoms):
    p0, p1, p2, p3 = (coords[index] for index in atoms)
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))

def phi_psi_cv(coords, phi_atoms=PHI_ATOMS, psi_atoms=PSI_ATOMS):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 2:
        phi = dihedral_degrees(coords, phi_atoms)
        psi = dihedral_degrees(coords, psi_atoms)
        return np.array([phi, psi])
    elif coords.ndim == 3:
        cvs = []
        for frame in coords:
            phi = dihedral_degrees(frame, phi_atoms)
            psi = dihedral_degrees(frame, psi_atoms)
            cvs.append([phi, psi])
        return np.array(cvs)
    else:
        raise ValueError(f"Invalid shape for coords: {coords.shape}")

def angular_delta_degrees(values, target):
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float)
    return (values - target + 180.0) % 360.0 - 180.0

def reached_phi_psi(coords, target=(60.0, 40.0), tolerance=10.0):
    cv = phi_psi_cv(coords)
    delta = angular_delta_degrees(cv, target)
    return bool(np.linalg.norm(delta) < float(tolerance))

def save_path_pdb(template_file, path_nm, out_pdb):
    """Save trajectory to PDB using MDAnalysis and a template PDB/GRO to preserve formatting."""
    import MDAnalysis as mda
    u = mda.Universe(template_file)
    path_ang = path_nm * 10.0
    with mda.Writer(out_pdb, u.atoms.n_atoms) as W:
        for coords in path_ang:
            u.atoms.positions = coords
            W.write(u.atoms)
