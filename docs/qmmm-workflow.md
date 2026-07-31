# QM/MM Reactive Workflow

PathGennie can generate **bond-breaking, bond-forming** paths under a QM or QM/MM
Hamiltonian, refine them into a path collective variable, and compute a free
energy along that CV — all at one level of theory.

This page covers the complete pipeline. For a QM/MM *conformational* change (no
chemistry), start instead with
[`examples/qmmm_alanine_conformation`](https://github.com/TeamSuman/PathGennie/tree/main/examples/qmmm_alanine_conformation),
which is the cheapest way to confirm your AMBER QM/MM build works at all.

## Why PathGennie suits reactive systems

The sampler biases **selection**, not forces: swarms of short *unbiased* segments
are propagated, scored on a progress CV, and the best one is extended. Nothing is
added to the Hamiltonian.

That matters more for chemistry than for conformational change. A biasing
potential on a reactive coordinate perturbs the electronic problem you are trying
to solve, and any force-based bias has to be removed again before the energetics
mean anything. Here the dynamics are the QM dynamics; only the *choice* of which
segment to keep is biased, and the free energy is recovered by a separate
unbiased stage (Weighted Ensemble) rather than by reweighting.

The practical consequence is also useful: because segments are independent, the
expensive QM force evaluations parallelise across walkers with no communication.

## Backend support

**AMBER (`sander`) is the only backend that can run a QM Hamiltonian.** QM/MM is
enabled by `ifqnt=1` plus a `&qmmm` namelist, which PathGennie injects verbatim
through `md.extra_text`:

```yaml
md:
    controls:
        ifqnt: 1
        dt: 0.0005          # 0.5 fs -- see "Timestep" below
        ntc: 1              # no constraints: SHAKE on a breaking bond is wrong
        ntf: 1
        ntb: 0              # gas phase; use 1 with a periodic box
    extra_text: |
        &qmmm
          qmmask=':1-2',
          qmcharge=-1,
          qm_theory='DFTB3',
        /
```

Refinement and the free-energy stage reach AMBER through the engine-agnostic
[`EngineSampler`](path-refinement.md#engine-agnostic-refinement). Before that
existed, refinement was OpenMM-only and this workflow was not possible.

### Choosing a QM method

Measured on a 6-atom gas-phase system, one CPU core:

| Method | Cost/step | Use for |
| --- | --- | --- |
| DFTB3 / 3ob-3-1 | **20 ms** | discovery, refinement, and long free-energy runs |
| QUICK (GPU, *ab initio*) | 1.07 s (+~8 s launch) | single-point refinement of key structures |

The launch overhead is the deciding factor, not the per-step cost: PathGennie
issues one short segment per walker per cycle, so an 8-second process start is
paid *per segment*. That makes semiempirical methods the right choice for the
path-generation and sampling stages, with *ab initio* reserved for refining
individual structures. DFTB3 with the 3ob-3-1 set covers H, C, N, O, P, S and the
halogens.

### Timestep and constraints

Use **`dt = 0.0005`** (0.5 fs) and **`ntc = 1`, `ntf = 1`**. Constraining bonds
with SHAKE is standard practice in classical MD and actively wrong here — the
bond you constrain may be the bond that has to break.

## The four stages

The worked reference is
[`examples/qmmm_reactive_sn2/amber`](https://github.com/TeamSuman/PathGennie/tree/main/examples/qmmm_reactive_sn2/amber)
(identity S<sub>N</sub>2, Cl⁻ + CH₃Cl → ClCH₃ + Cl⁻, DFTB3, all atoms QM).

### 1. Discover

Run the driver in target mode on a reaction coordinate, once per seed. Each seed
gives an independent reactive path; the ensemble is what stage 2 refines.

**Reactive barriers need different swarm settings than conformational ones.** For
the S<sub>N</sub>2 case, the usual long-segment settings (τ₁ = τ₂ = 100, 8
workers) produced **no reaction in 300 cycles** — the molecule vibrated in the
reactant well, because thermal fluctuation over a 50 fs segment cannot deliver the
concerted ~0.5 Å stretch and ~0.7 Å approach needed to reach the transition state.
Short, greedy, many-trial settings converged at cycle 11 in 26 seconds:

```yaml
tau1_steps: 10          # short bursts -> high selection pressure per unit time
tau2_steps: 10          # minimal relaxation back into the reactant well
max_trial: 30           # more chances at a productive fluctuation
sigma: 0.05             # greedier softmax selection
reject_worse_anchor: true
```

The rule of thumb: **stiff enthalpic barrier → short τ**; diffusive entropic
barrier → long τ. This is the opposite of the guidance that suits conformational
transitions.

### 2. Refine into a PathCV

Alternate short QM/MM exploration around the current path with a
principal-curve/neural consensus fit until the path stops moving. Use
`EngineSampler` with the *same* `&qmmm` namelist as stage 1.

Combine seeds by **arc-length resampling before averaging**. Averaging raw frames
is meaningless — seeds have different lengths and different dwell times, so frame
*k* of one is not frame *k* of another.

### 3. Free energy along `s`

Bin Weighted Ensemble on the path progress coordinate `s`:

```python
def s_of(coords):
    s, _z = path_cv.compute(np.atleast_2d(feature_fn(coords)))
    return float(s)

stage = WeightedEnsembleStage(cv_fn=s_of, tau_steps=100, n_iterations=200,
                              bin_edges=np.linspace(0, 1, 17), target_count=8,
                              kT=0.0019872041 * 300)
```

WE propagates walkers under the plain QM/MM Hamiltonian and splits/merges only
their statistical *weights*, so the profile needs no reweighting.

Seed one walker per `s` bin. WE only splits walkers that *reach* a bin, so a run
started entirely in the reactant well spends its whole budget crawling out.

### 4. Validate against an independent reference

`5_neb_reference.py` in the worked example does both of the following in ~2 minutes
(`--beads 16 --rescore`).

Two checks are worth the cost:

- **A transition state from an independent method.** A constrained
  symmetric-stretch scan and an AMBER NEB (`ineb=1`, with `skmin`/`skmax` and
  `tgtfitmask`/`tgtrmsmask`, run under `sander.MPI` as a multi-replica job) gave
  2.3565 Å and 2.3579 Å for the S<sub>N</sub>2 case — agreement to 0.001 Å. Run
  the reference **at the same level of theory** as the path, or the comparison
  measures the method difference rather than the path.
- **A geometric signature of the mechanism.** For S<sub>N</sub>2 the decisive
  test is the sign flip of the CH₃ umbrella coordinate — Walden inversion.
  Distances alone cannot distinguish backside from frontside attack.

> When locating a *symmetric* transition state by scanning the symmetric stretch,
> take the **minimum** of the scanned energy, not the maximum. A saddle point is a
> maximum along the reaction coordinate but a **minimum** along every orthogonal
> one, and the symmetric stretch is orthogonal.

### Semiempirical geometry, DFT energetics

Measured on the S<sub>N</sub>2 case, all on the same 16-bead NEB band:

| level | barrier | TS geometry |
| --- | --- | --- |
| DFTB3 | 3.56 kcal/mol | 2.3558 Å — matches an independent scan to **0.0007 Å** |
| B3LYP/6-31+G\* // DFTB3 | 8.23 kcal/mol | (single points on the same geometries) |
| CCSD(T) reference | ≈13 kcal/mol | — |

**DFTB3 gets the structure right and the barrier wrong** — about fourfold low here, which is typical
of semiempirical methods for S<sub>N</sub>2. Re-scoring the relaxed geometries at DFT more than
doubles the barrier and costs ~90 s for 16 single points; the residual gap is the known B3LYP
underestimate for anionic transition states. This is what makes the two-tier design work: sample
where geometry matters and cost is 20 ms/step, then re-score once at the end.

Two things to keep straight: `DFT//DFTB3` is a single point on someone else's geometry, not a DFT
minimum-energy path; and **diffuse functions are not optional** for an anionic reaction — use at
least `6-31+G*`.

> **Watch shallow gas-phase minima during endpoint preparation.** Minimising the S<sub>N</sub>2
> product endpoint for 500 cycles instead of 300 dissociated the ion–dipole complex to
> d(C–Cl) ≈ 900 Å — and still wrote a perfectly valid restart file, which then poisoned the entire
> band with no error raised anywhere. Check the *geometry* a stage produces, not just that it
> produced a file.

## Convergence criteria: drive on progress, stop on product

This is the failure mode most likely to produce a wrong result quietly.

A convergence function written as a *difference of distances* —
ξ = d(C–X_leaving) − d(C–Y_attacking) > threshold — is satisfied by one distance
growing alone. It does not require the new bond to form.

Running the S<sub>N</sub>2 protocol unchanged on the tertiary substrate
(CH₃)₃C–Cl + Br⁻, **all 10 seeds reported `Converged` and none produced
tert-butyl bromide**: chloride left (d(C–Cl) 3.6–3.8 Å) while Br⁻ loitered at
2.70–2.99 Å without ever bonding to carbon (a C–Br bond is ~1.99 Å). A
lower-barrier channel existed and the criterion could not tell the difference.

State convergence as a condition on the **product**:

```python
def reacted(coords, **kwargs):
    return bool(d(C, Y_attacking) < 2.1 and d(C, X_leaving) > 3.0)
```

A progress CV is the right thing to *drive* on and a poor thing to *stop* on.
Several shipped example configs use the distance-difference pattern; it is safe
in those cases only because the intended product is the sole accessible route to
a large CV value. Check that this holds for your system before reusing it.

(The tertiary run is instructive on its own: PathGennie found **E2 elimination in
8 of 10 seeds** — confirmed by two independent signatures, a formed H–Br bond and
a formed C=C bond — while being *driven* on a substitution coordinate. Tertiary
substrates eliminate rather than undergo backside substitution, so this is the
chemically correct answer, and evidence that the sampler follows the physics
rather than obediently satisfying the coordinate it was handed.)

## Reading mechanisms off a 2-D plot

Plotting paths on the plane of the forming and breaking bond distances separates
mechanisms visually, which is the quickest way to see what a run actually did:

| Mechanism | Signature on d(C–Y_attacking) vs d(C–X_leaving) |
| --- | --- |
| S<sub>N</sub>2 | crosses the diagonal near the d₁ = d₂ symmetry line (concerted) |
| S<sub>N</sub>1 | an L-shaped detour through the both-bonds-long corner (free carbocation), then capture |
| E1 / E2 | never reaches short d(C–Y); the trace is truncated because the path leaves this plane entirely |

Add the elimination coordinates (H–Y distance, C=C distance) as a second pair of
axes to resolve E1 from E2. `4_plot_2d_cv.py` in the S<sub>N</sub>2 example takes
`--extra 'label=path.npy'` to overlay a second substrate in the same axes.

Define the feature map **once** and import it into every stage. Two stages that
disagree on component order silently transpose the axes of the final figure, and
for a symmetric reaction that is nearly invisible.

## Cluster notes

- Source AMBER's `amber.sh` **with `set +u`**. It references unbound variables
  such as `PERL5LIB` and will abort an otherwise-correct job script under
  `set -u`.
- `sander.MPI` (needed for NEB) links the system OpenMPI. If a conda environment
  puts MPICH on `PATH` first, the launcher and the library disagree and the job
  fails confusingly — prepend the correct MPI `bin`/`lib` before running.
- AMBER restart files default to NetCDF (`ntxo = 2`). Set `ntxo: 1` for ASCII if
  you need to read them with PathGennie's own `read_rst7_coords`; otherwise read
  them with `parmed`.
- QM/MM runs are CPU-bound and single-threaded per walker. Request one core per
  worker and scale by walker count, not by threads.
