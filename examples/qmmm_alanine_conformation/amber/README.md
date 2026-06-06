# AMBER QM/MM Alanine-Dipeptide Conformational Transition

This example demonstrates a QM/MM conformational transition, not a chemical
reaction. The system is the well-studied ACE-ALA-NME alanine dipeptide in
explicit water with one `Na+` and one `Cl-` ion.

The conformational targets are Ramachandran basins:

- `input_c7eq.yaml`: target the C7eq basin, approximately `phi=-80`, `psi=80`
- `input_c7ax.yaml`: target the C7ax basin, approximately `phi=60`, `psi=-60`

The full solute, residues `1-3` (`ACE`, `ALA`, `NME`), is the QM region. Water
and ions are MM. This avoids QM/MM covalent boundary link atoms and avoids bond
formation or bond breaking.

The example is intentionally compact. It is suitable for checking AMBER QM/MM
execution and PathGennie target-mode wiring, not for production free-energy
sampling.

## Build and Equilibrate

The repository includes prepared files:

- `qmmm_ala.prmtop`
- `qmmm_ala_equilibrated.rst7`
- `qmmm_ala_equilibrated.pdb`

To regenerate them:

```bash
cd examples/qmmm_alanine_conformation/amber
python build_system.py
```

The build script:

1. builds ACE-ALA-NME with AmberTools `tleap`,
2. adds TIP3P solvent,
3. adds one `Na+` and one `Cl-`,
4. minimizes and equilibrates briefly with OpenMM,
5. writes the equilibrated AMBER restart used before QM/MM steering.

## Run PathGennie QM/MM

```bash
cd examples/qmmm_alanine_conformation/amber
python run_pg_amber.py --config input_c7eq.yaml
python run_pg_amber.py --config input_c7ax.yaml
```

The generated `tau1.mdin` and `tau2.mdin` include:

```text
&qmmm
  qmmask=':1-3',
  qmcharge=0,
  qm_theory='PM3',
  qmcut=8.0,
/
```

Outputs are written as AMBER NetCDF trajectories (`.nc`) plus CSV metrics.
