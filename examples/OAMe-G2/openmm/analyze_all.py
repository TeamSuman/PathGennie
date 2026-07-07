import os
import glob
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import shutil

def main():
    # Automatically find all unbinding.dcd trajectories in the run folders
    dcd_files = glob.glob("pathgennie_openmm_run_*/output/unbinding.dcd")
    # Sort them numerically so the plot legend is in order
    dcd_files.sort(key=lambda x: int(x.split('_run_')[1].split('/')[0]))
    
    ref_plumed = "plumed.dat"
    ref_pdb = "conf_template.pdb"
    
    if not os.path.exists(ref_plumed):
        print(f"Error: {ref_plumed} not found in the current directory.")
        return
        
    plt.figure(figsize=(10, 6))
    processed_count = 0

    for dcd in dcd_files:
        out_dir = os.path.dirname(dcd)
        colvar_path = os.path.join(out_dir, "COLVAR")
        run_name = out_dir.split('/')[-2] # e.g. pathgennie_openmm_run_1
        
        # 1. Run PLUMED if COLVAR hasn't been generated yet
        if not os.path.exists(colvar_path):
            print(f"Running PLUMED driver for {run_name}...")
            shutil.copy(ref_plumed, out_dir)
            
            if os.path.exists(ref_pdb):
                shutil.copy(ref_pdb, out_dir)
            
            # Execute plumed driver inside the output directory
            cmd = ["plumed", "driver", "--plumed", "plumed.dat", "--mf_dcd", "unbinding.dcd"]
            try:
                subprocess.run(cmd, cwd=out_dir, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running PLUMED for {run_name}: {e.stderr.decode()}")
                continue
                
        # 2. Plot the COLVAR file
        if os.path.exists(colvar_path):
            try:
                # np.loadtxt automatically ignores lines starting with '#' (plumed header)
                cv = np.loadtxt(colvar_path)
                if len(cv) > 0:
                    plt.plot(cv[:, 2], cv[:, -1], label=run_name, alpha=0.8)
                    processed_count += 1
            except Exception as e:
                print(f"Error reading {colvar_path}: {e}")

    # 3. Finalize and save the plot
    if processed_count > 0:
        plt.xlabel("CV Index 2")
        plt.ylabel("CV Index -1 (cyl.z)")
        plt.title(f"Unbinding Path Overlays ({processed_count} Runs)")
        
        # Place legend nicely outside the plot if there are many runs
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=max(1, processed_count//20 + 1))
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        out_plot = "all_runs_analysis.png"
        plt.savefig(out_plot, dpi=300)
        print(f"\nSuccessfully plotted {processed_count} runs to {out_plot}!")
    else:
        print("\nNo COLVAR files were processed.")

if __name__ == "__main__":
    main()
