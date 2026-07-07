import os
import sys
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

def load_coords(pdb_file):
    pdb = app.PDBFile(pdb_file)
    n_frames = pdb.getNumFrames()
    coords = []
    for i in range(n_frames):
        pos = pdb.getPositions(asNumpy=True, frame=i).value_in_unit(unit.nanometer)
        coords.append(pos)
    coords = np.array(coords)
    if n_frames == 1:
        return coords[0] # (N, 3)
    return coords # (N_frames, N, 3)

def create_chignolin_system(prmtop_file):
    prmtop = app.AmberPrmtopFile(prmtop_file)
    system = prmtop.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        implicitSolvent=app.GBn2,
        implicitSolventSaltConc=0.1 * (unit.moles / unit.liter),
        temperature=300 * unit.kelvin,
    )
    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond
    )
    
    try:
        platform = mm.Platform.getPlatformByName("CUDA")
        props = {"Precision": "mixed"}
    except Exception:
        platform = mm.Platform.getPlatformByName("CPU")
        props = {}
        
    simulation = app.Simulation(prmtop.topology, system, integrator, platform, props)
    return simulation

def get_calpha_indices(prmtop_file):
    prmtop = app.AmberPrmtopFile(prmtop_file)
    indices = []
    for atom in prmtop.topology.atoms():
        if atom.name == "CA":
            indices.append(atom.index)
    return np.array(indices)

def compute_pairwise_distances(coords, heavy_indices):
    """Compute pairwise distances between selected atoms."""
    # coords shape is expected to be (N_atoms, 3) in Angstroms
    heavy_coords = coords[heavy_indices]
    diff = heavy_coords[:, np.newaxis, :] - heavy_coords[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    i, j = np.triu_indices(len(heavy_indices), k=1)
    return dist[i, j]

def save_path_pdb(template_pdb, path_nm, out_pdb):
    """Save trajectory to PDB using MDAnalysis and a template PDB to preserve formatting."""
    import MDAnalysis as mda
    u = mda.Universe(template_pdb)
    
    # MDAnalysis expects coordinates in Angstroms
    path_ang = path_nm * 10.0
    
    with mda.Writer(out_pdb, u.atoms.n_atoms) as W:
        for coords in path_ang:
            u.atoms.positions = coords
            W.write(u.atoms)
