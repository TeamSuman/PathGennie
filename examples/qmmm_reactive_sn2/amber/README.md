# Reactive QM/MM: the full path workflow

Gas-phase identity S<sub>N</sub>2, **Cl⁻ + CH₃Cl → ClCH₃ + Cl⁻**, at DFTB3/3ob-3-1 with all six
atoms in the QM region (net charge −1). Bonds genuinely break and form here — unlike
[`qmmm_alanine_conformation`](../../qmmm_alanine_conformation/amber/README.md), which is a QM/MM
*conformational* change and is the right place to start if you only want to check that your AMBER
QM/MM build works.

This is the reference for the complete pipeline:

| Stage | Script | Produces |
| --- | --- | --- |
| 1. Discover | `1_generate_paths.py` | a multi-seed ensemble of reactive QM/MM paths |
| 2. Refine | `2_refine_pathcv.py` | one smooth path → a `PathCV` (`s`, `z`) |
| 3. Free energy | `3_free_energy.py` | *F*(`s`) along that PathCV, by Weighted Ensemble |
| 4. Plot | `4_plot_2d_cv.py` | the paths on the 2-D distance plane |
| 5. Check | `5_neb_reference.py` | an independent NEB reference path + a DFT barrier |

Every stage runs the **same QM Hamiltonian**. That is not a stylistic choice: a path refined at a
different level of theory than it was discovered at, or a free energy computed at a third level, is
not a profile of the path you generated.

## Requirements

- AMBER with `sander` built with DFTB support, and the 3ob-3-1 Slater–Koster set reachable through
  `$AMBERHOME/dat/slko`. Source `amber.sh` **with `set +u`** — it references unbound variables such
  as `PERL5LIB` and will abort an otherwise-correct job script under `set -u`.
- PathGennie with the `[ml]` extra (stage 2 trains a small PyTorch model).
- No GPU is needed. DFTB3 on this system costs ≈20 ms/step on one CPU core.

## Stage 1 — discover

```bash
python 1_generate_paths.py --seeds 10 --outdir ensemble
```

Each seed is an independent PathGennie run in target mode along
ξ = d(C–Cl_leaving) − d(C–Cl_attacking) (`sn2_cv.py`). Typical cost per seed is **tens of seconds**;
converged runs report `Converged at cycle 11` or thereabouts.

**The settings in `input.yaml` matter more than usual.** The first attempt at this system used
τ₁ = τ₂ = 100 with 8 workers and produced **no reaction in 300 cycles** — the molecule simply
vibrated in the reactant well, because thermal fluctuation over a 50 fs segment cannot deliver the
concerted ~0.5 Å stretch plus ~0.7 Å approach needed to reach the transition state. Short segments
with many trials and a greedy selection are what work for a stiff enthalpic barrier:

```yaml
tau1_steps: 10          # short bursts -> high selection pressure per unit time
tau2_steps: 10          # minimal relaxation back into the reactant well
max_trial: 30           # more chances at a productive fluctuation
sigma: 0.05             # greedier softmax selection
reject_worse_anchor: true
```

That is the opposite of the usual advice for diffusive conformational barriers, where longer
segments pay off. Enthalpic barrier → short τ.

Sanity checks on a converged path: d(C–Cl_leaving) goes 1.87 → 3.05 Å, d(C–Cl_attacking) goes
2.90 → 1.82 Å, and the CH₃ umbrella coordinate flips sign (−1.000 → +0.965). That sign flip is
**Walden inversion**, the mechanistic signature of backside attack, and it is the check worth
running — the distances alone cannot distinguish backside from frontside.

## Stage 2 — refine into a PathCV

```bash
python 2_refine_pathcv.py --ensemble ensemble --iterations 6
```

Refinement runs through `pathrefinement.samplers.EngineSampler`, which drives the **core `Engine`
protocol** rather than OpenMM. That is what makes QM/MM refinement possible at all: AMBER is the
only backend that can run a QM Hamiltonian, and `PathRefiner`'s built-in walker is OpenMM-only. See
[Path Refinement](../../../docs/path-refinement.md#engine-agnostic-refinement) for using the same
sampler with GROMACS or OpenMM.

The feature space is the 2-D pair (d(C–Cl_attacking), d(C–Cl_leaving)), which is both what
`PathRefiner` expects and the plane the mechanism is conventionally drawn in. It is defined **once**,
as `path_features` in `sn2_cv.py`, and imported by stages 2, 3 and 4 — during development two of
them disagreed on component order, which silently transposed the axes of the final figure and was
nearly invisible precisely *because* this reaction is symmetric.

Seed trajectories are combined by **arc-length resampling before averaging** — averaging raw frames
would be meaningless, since the seeds have different lengths and different dwell times, so frame *k*
of one is not frame *k* of another.

## Stage 3 — free energy along the path

```bash
python 3_free_energy.py --refined results/refinement/refined_path.npy
```

Weighted Ensemble binned on the path progress coordinate `s`. WE is unbiased — walkers propagate
under the plain QM/MM Hamiltonian and only their statistical *weights* are split and merged — so the
profile needs no reweighting.

One walker is seeded per `s` bin from the stage-1 frames. This matters: WE only splits walkers that
*reach* a bin, so a run started entirely in the reactant well spends its whole budget crawling out.

`burn_in` defaults to 30 % here. Seeding one walker per bin is a *uniform* distribution — the
opposite of the Boltzmann one being estimated — so averaging from iteration 0 flattens the profile
and biases the barrier low.

### What a real run actually shows, and the limit you will hit

2500 iterations (125 ps/walker, 16 bins × 6 walkers, ~15 min on 8 cores) gives a clean symmetric
double well:

| s | 0.031 | 0.094 | 0.156 | 0.219 | 0.281 | … | 0.781 | 0.844 | 0.906 | 0.969 |
|---|---|---|---|---|---|---|---|---|---|---|
| F (kcal/mol) | 0.00 | 2.51 | 3.44 | 4.41 | 5.21 | **unsampled** | 5.40 | 4.25 | 2.90 | 0.01 |

The wells agree to **0.01 kcal/mol** — a strong check, since an identity reaction must be exactly
symmetric.

**But the seven bins spanning the barrier top hold no weight, and 5.40 kcal/mol is a lower bound,
not the barrier.**

WE climbs a barrier by a **ratchet**: a walker straying into the next bin up is split there, so its
descendants are more numerous and probability creeps upward. A bin going empty is normal — the
ratchet refills it. What matters is whether the two sides **meet**. Here they did not: the climb
reached s = 0.281 from the reactant side and s = 0.781 from the product side, leaving seven bins
unbridged.

The ratchet was working — the bin at s = 0.094 emptied and refilled **191 times**, and bins further
up refilled 46, 13 and 9 times, with re-entries as late as iteration 2291 of 2500. It **stalled**
rather than being unable to repopulate. WE is also behaving correctly while it does: adjacent-bin
occupancy ratios are 0.211, 0.196, 0.263 → ΔF = +0.93, +0.97, +0.80 kcal/mol, i.e. textbook
exp(−ΔF/kT). The empty bins simply have tiny equilibrium weight, and the climb rate falls off
exponentially with height.

The script detects the unbridged gap, reports how often the boundary bins refilled, prints the
per-bin ratios, and relabels the maximum as a lower bound.

To reach the top:

- **more walkers per bin, or finer bins across the stalled region** — the standard WE remedies,
  since the problem is climb *rate*, not a structural impossibility;
- **a longer τ** — 50 fs gives each iteration little chance to advance a bin;
- **a bias along `s`** (umbrella sampling or OPES), which sidesteps the climb entirely;
- **recycling** (`recycle=True` with `source_cv`/`target_cv`) if a steady-state *rate* is what you
  want rather than a profile.

**Use the symmetry as an error bar.** For this identity reaction, deviation from mirror symmetry
about `s = 0.5` is pure sampling error. It grows with height — 0.01, 0.39, 0.81, 0.99 kcal/mol —
so even the sampled flanks carry ~1 kcal/mol of error near the top. Most systems do not hand you a
free internal error estimate like this; when yours does, use it.

## Stage 4 — plot

```bash
python 4_plot_2d_cv.py --refinement results/refinement --ts 2.3565
```

`--ts` marks a reference transition state on the d₁ = d₂ symmetry line. For this system two
independent determinations agree: a constrained symmetric-stretch scan gives **2.3565 Å** and an
AMBER NEB (`ineb=1`) midpoint at the same DFTB3 level gives **2.3579 Å**.

> When locating a symmetric TS by scanning the symmetric stretch, take the **minimum** of the scanned
> energy, not the maximum. A saddle point is a maximum along the *reaction* coordinate but a
> **minimum** along every orthogonal one, and the symmetric stretch is orthogonal here.

Mechanisms separate visually on this plane: S<sub>N</sub>2 crosses the diagonal near the symmetry
line; S<sub>N</sub>1 detours through the top-right corner (both bonds long — a free carbocation)
before the new bond forms; eliminations never reach short d(C–X) at all. Overlay a second substrate
with `--extra 'tert-butyl (E2)=path.npy'`.

## Stage 5 — check it against something that shares no machinery

```bash
python 5_neb_reference.py --beads 16 --rescore     # ~2 min total
```

Stages 1–3 produce a path and a free energy by *sampling*. This computes the minimum-energy path a
completely different way — a nudged elastic band — **at the same level of theory**, so the comparison
measures the path rather than the method. Then it re-scores the relaxed geometries at DFT.

Needs `sander.MPI` and `mpirun`; `--rescore` also needs `quick`.

Measured on this system:

| level | barrier | TS mean d(C–Cl) |
| --- | --- | --- |
| DFTB3 (the sampling Hamiltonian) | 3.56 kcal/mol | 2.3558 Å |
| B3LYP/6-31+G\* // DFTB3 | **8.23 kcal/mol** | (same geometries) |
| CCSD(T) reference | ≈13 kcal/mol | — |

**DFTB3 gets the geometry, not the barrier.** The TS it finds agrees with an independent
symmetric-stretch scan to **0.0007 Å**, but its barrier is ~4× too small. Re-scoring the same
geometries at DFT more than doubles it; the residual gap is the known B3LYP underestimate for
anionic S<sub>N</sub>2. So: **discover and refine at DFTB3, then re-score the band** — that last step
costs 90 seconds, not a new campaign. Note this is `DFT//DFTB3`, a single point on a DFTB3 geometry,
not a DFT-optimised path.

And the payoff for the whole workflow — distance to the NEB path in the CV plane:

| path | mean | max |
| --- | --- | --- |
| initial (seed consensus) | 0.059 Å | 0.086 Å |
| **refined PathCV** | **0.044 Å** | **0.066 Å** |

Refinement moves the path measurably *closer* to an independently computed MEP rather than merely
smoothing it. Overlay them with
`python 4_plot_2d_cv.py --extra 'NEB MEP=results/neb/neb_path_2d.npy'`.

> **Do not raise `maxcyc` in the endpoint minimisation.** The gas-phase ion–dipole minimum is
> shallow; at `maxcyc=500` the conjugate-gradient phase walks the product endpoint apart to
> d(C–Cl) ≈ 900 Å, and the whole band relaxes into nonsense. A dissociated endpoint still writes a
> perfectly valid restart file, so the script checks the *geometry*, not just that the file exists.

## A caution on convergence criteria

`sn2_cv.py` stops on ξ > 1.0, a *difference of distances*. That is safe **only** because for methyl
chloride the sole accessible route to large ξ is substitution.

It is not safe in general. Running the identical protocol on the tertiary substrate
(CH₃)₃C–Cl + Br⁻ made all 10 seeds report `Converged` while **none produced tert-butyl bromide**: the
chloride left (d(C–Cl) 3.6–3.8 Å) while Br⁻ loitered at 2.70–2.99 Å without ever bonding to carbon
(a C–Br bond is ~1.99 Å). A difference of distances is satisfied by one distance growing alone.

**Drive on a progress coordinate; stop on a product-specific condition.** `sn2_cv.py` ships that
form as `reacted_product_specific` (`d(C-Cl_attacking) < 2.1` **and** `d(C-Cl_leaving) > 3.0`); switch
`input.yaml` to it whenever a competing channel could also drive ξ up:

```yaml
convergence:
    module: sn2_cv
    function: reacted_product_specific
```

(The tertiary run is chemically instructive in its own right: PathGennie found **E2 elimination in
8 of 10 seeds** — two independent signatures, a formed H–Br bond and a formed C=C bond — while being
driven on a substitution coordinate. Tertiary substrates eliminate; they do not undergo backside
substitution. The sampler followed the physics rather than the coordinate it was given.)

## Files

| File | Role |
| --- | --- |
| `sn2.prmtop`, `sn2.rst7`, `sn2.pdb` | prepared system (6 atoms) |
| `ch3cl.mol2`, `ch3cl.frcmod`, `leap.in` | `antechamber`/`tleap` inputs used to build it |
| `sn2_cv.py` | reaction coordinate and convergence test |
| `input.yaml` | driver configuration for one seed |

To rebuild the system from scratch: `antechamber -i ch3cl.pdb -fi pdb -o ch3cl.mol2 -fo mol2 -c bcc
-nc 0` then `tleap -f leap.in`. In `tleap` the bromide unit is `BR` — uppercase, no minus sign — if
you adapt this to the tert-butyl case.
