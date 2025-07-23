# 🧬 PathGennie: Rapid Rare Event Pathway Generator

This repository contains code and examples for the article:
**"Rapid Generation of Rare Event Pathways Using Direction-Guided Adaptive Sampling: From Ligand Unbinding to Protein (Un)Folding."**

PathGennie is a general-purpose **steering framework** for guiding molecular simulations along data-driven(or physical) collective variables (CVs) to rapidly sample rare event transitions such as:

* Ligand **unbinding**
* Protein **folding and unfolding**

It leverages high-performance libraries like **OpenMM** and **MDAnalysis**, and includes tooling for CV construction, adaptive sampling, and optimized trajectory generation.

---

## 🧩 Core Components

```
pathgennie/
│
├── README.md               # Documentation and usage guide
├── LICENSE                 # MIT or compatible license
├── environment.yml         # Conda environment for reproducibility
│
├── Scripts/                # Scripts for various path generation tasks
│   ├── unbind              # Ligand unbinding module
│   ├── unfold              # Protein unfolding module
│   └── fold                # Protein folding / reverse folding module
│
├── examples/               # example systems
│   ├── 3PTB/               # Example: Bovine Trypsin Inhibitor
│   │   ├── native.pdb
│   │   ├── start.gro
│   │   ├── topol.top
│   │   └── system.py
│   │
│   └── 2JOF/               # Example: Trp-cage protein system
│       ├── native.pdb
│       ├── start.gro
│       ├── topol.top
│       └── system.py
```

---

## 🛠️ Installation

Clone the repository and set up the Conda environment:

```bash
# Clone the repository
git clone https://github.com/dmighty007/PathGennie.git    

# Navigate to the project folder
cd PathGennie

# Create and activate the environment
conda env create -f environment.yml
conda activate pathgennie
```

This installs all required dependencies, including:

* `openmm`
* `mdanalysis`
* `numpy`, `numba`, `tqdm`, and more

> ✅ **Note**: Ensure [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) is installed before proceeding.

---

## 🚀 Use Case: Ligand Unbinding via PCA-Guided Steering

This framework enables **ligand unbinding simulations** by steering MD along principal components derived from distance features between a ligand and its binding site. These components form a low-dimensional **CV space** for guided sampling.

---

### 🧪 Step 0: Prepare Your System

1. Create a working directory:

   ```bash
   mkdir Test && cd Test
   ```

2. Add your input files:

   ```
   pbcmol.gro       # Structure with solvent and PBC
   topol.top        # GROMACS-compatible topology
   ```

3. Perform **energy minimization** and **equilibration** using GROMACS or OpenMM. These outputs will be used as initial configurations.

---

### 🔧 Step 1: Generate PCA-Based Collective Variables

Generate a PCA model based on ligand–protein contacts:

```bash
python pcagen.py pbcmol.gro \
    --ligand_sel "resname LIG" \
    --output pca.pkl
```

This script:

* Computes distance features between the ligand and surrounding protein atoms
* Performs PCA on the feature matrix
* Stores the principal components in `pca.pkl`

---

### ⚙️ Step 2: Define the Simulation Object

Create a `system.py` file to build the OpenMM simulation system:

```python
from openmm.app import *
from openmm import *
from openmm.unit import *

class Simulation_obj:
    def __init__(self):
        gro = GromacsGroFile('pbcmol.gro')
        top = GromacsTopFile('topol.top',
                             periodicBoxVectors=gro.getPeriodicBoxVectors(),
                             includeDir='/usr/local/gromacs/share/gromacs/top')
        system = top.createSystem(nonbondedMethod=PME,
                                  nonbondedCutoff=1*nanometer,
                                  constraints=HBonds)
        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
        self.simulation = Simulation(top.topology, system, integrator)
```

> 📘 Refer to the [OpenMM documentation](https://openmm.org/documentation/latest/userguide/application/02_running_sims.html) for advanced setup options.

---

### 🚦 Step 3: Run the Unbinding Simulation

Run the unbinding driver with:

```bash
../../Scripts/unbind \
    --structure_file pbcmol.gro \
    --verbose \
    --relax1 10 \
    --relax2 15 \
    --max_probes 50 \
    --temperature 300 \
    --model_file pca.pkl
```

Output:

* `trajectory.xtc`: Reactive trajectory file showing unbinding progression

---

### 🔄 CLI Options

| Parameter            | Description                                  | Default |
| -------------------- | -------------------------------------------- | ------- |
| `--ligand_name`      | Ligand residue name                          | LIG     |
| `--selection_radius` | Distance (Å) to include nearby protein atoms | 20.0    |
| `--relax1`           | MD steps for trial probe                     | 10      |
| `--relax2`           | MD steps for relaxation after acceptance     | 15      |
| `--max_probes`       | Number of parallel probes per cycle          | 50      |
| `--temperature`      | Temperature in Kelvin                        | 290     |
| `--no_save`          | If set, disables output trajectory           | False   |

> 💡 For full options, run:

```bash
../../Scripts/unbind -h
```

---

## 🔁 Use Case: Protein Folding and Unfolding

The same logic applies to protein folding/unfolding simulations.

### Unfolding Example

```bash
../../Scripts/unfold \
    --ref_config folded.gro \
    --start_config folded.gro \
    --verbose \
    --relax1 10 \
    --relax2 15 \
    --max_probes 50 \
    --temperature 290
```

### Folding Example

```bash
../../Scripts/fold \
    --ref_config folded.gro \
    --start_config equili.gro \
    --verbose \
    --relax1 10 \
    --relax2 15 \
    --max_probes 50 \
    --temperature 290
```

> 📌 Make sure `system.py` is present in the working directory for these commands.

---

## 📂 Example Datasets

The `examples/` directory includes two protein systems with:

* Native PDB structure
* Starting `gro` configuration
* GROMACS topology (`topol.top`)
* Compatible `system.py`

You can directly run folding/unfolding or unbinding simulations on these examples.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋 Questions?

Feel free to open an issue or submit a pull request!

---
