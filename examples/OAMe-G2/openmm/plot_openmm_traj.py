import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import plumed_cv

print("Loading topology and trajectory...")
u = mda.Universe("system.prmtop", "pathgennie_openmm_run/output/unbinding.dcd")

print("Computing CVs using plumed_cv.py...")
cvs = plumed_cv.compute_cvs(u, plumed_cv.runner)

# CV names from plumed_cv.py: ["V1", "V2", "V3", "cyl.z"]
# V2 is index 1, cyl.z is index 3
v2 = cvs[:, 1]
cyl_z = cvs[:, 3]

plt.figure(figsize=(8, 6))
plt.scatter(v2, cyl_z, c=np.arange(len(v2)), cmap='viridis', alpha=0.7)
plt.colorbar(label='Frame')
plt.xlabel('V2 (nm)')
plt.ylabel('cyl.z (nm)')
plt.title('OpenMM Unbinding Trajectory: cyl.z vs V2')
plt.grid(True)
plt.tight_layout()
out_png = "pathgennie_openmm_run/output/cylz_vs_v2.png"
plt.savefig(out_png, dpi=300)
print(f"Saved plot to {out_png}")
