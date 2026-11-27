# Committor learning from swarms

!!! note "Status"
    Milestones **M0**, **M1** and **M3** are validated on the toy Wolfe-Quapp
    potential. **M2** has been run on alanine dipeptide in vacuum with CHARMM36 as a
    stand-in for CHARMM27: the gate is met, but by a protocol chosen after seeing the
    results, so it counts as provisional until repeated with a fixed protocol. The
    rest is planned work.

## The idea

The committor `q(x)` is the probability that a trajectory started at `x` reaches the
product state B before the reactant state A. It is the ideal reaction coordinate: its
level sets order configurations by true progress, `q = 1/2` is the transition-state
ensemble, and rates follow from it.

A PathGennie cycle launches `max_trial` independent, **unbiased** short trajectories
from one anchor. Outside A and B the committor satisfies

```
q(x) = E[ q(X_tau) | X_0 = x ]
```

so the average of `q` over the endpoints of one swarm is an unbiased estimate of the
right-hand side. The swarm that steers discovery therefore also supplies training data
for a committor network — with no bias force, and without ever passing a gradient to
the MD engine. That last point is what separates this from control-based committor
learning (REACT, [arXiv:2604.13213](https://arxiv.org/abs/2604.13213); Pharos,
[arXiv:2611.26765](https://arxiv.org/abs/2611.26765)), which injects `grad log q` into
the dynamics every few steps: here the engine only ever runs plain MD, so any backend
works, including AMBER QM/MM.

Two rules follow directly from the relation and are enforced by the code:

1. **Every trial is recorded, not just the selected one.** The selected trial was
   chosen *because* of where it ended, so its endpoint is not a fair sample. On the toy
   system a target built from the selected endpoint alone is more than twice as far
   from the truth as the all-trial average (`tests/test_committor_train.py`).
2. **The tau2 runner is not a training sample.** It starts from a selected sampler
   with its selected velocities. It is stored for path analysis only.

Where the anchors sit does not matter for correctness — the relation holds point by
point — so PathGennie's selection bias only decides *where* the committor is learned.

## Usage

```python
import numpy as np
from pathgennie.committor import BasinPair, SphereBasin, SwarmRecorder
from pathgennie.committor.train import train_committor

basins = BasinPair(A=SphereBasin(cv_reactant, 0.4), B=SphereBasin(cv_product, 0.4))
recorder = SwarmRecorder(featurize=my_features, basins=basins)

driver = PathGennieDriver(engine, progress, converged, swarm_recorder=recorder, ...)
driver.run(initial, tau1=20, tau2=40, max_trial=16, max_cycle=400)

recorder.save("swarms.h5")                       # every trial of every cycle
model = train_committor(recorder.dataset())      # needs PyTorch
q = model.predict(features)                      # committor in [0, 1]
s = model.logit(features)                        # use this as a progress coordinate
```

* `featurize(coords) -> 1-D array` is the network input. Use internal coordinates
  (distances, dihedral sin/cos) for molecules so the input is invariant to translation
  and rotation; the default (flattened Cartesian coordinates) is only for toy systems.
* Basins are defined on the CV the driver already computes; pass `basin_cv=` to define
  them in a different CV.
* Combine runs with `SwarmDataset.concatenate`. Runs in **both directions** are needed:
  training refuses to start unless some endpoints reached A and some reached B.
* `run_swarms(engine, handles, recorder, n_trials=..., n_steps=...)` adds swarms at
  configurations of your choice, without selection.

### Closed loop

```python
from pathgennie.committor.loop import CommittorLoop, LoopConfig

loop = CommittorLoop(
    engine, basins, cv_fn, featurize,
    start_a=handle_in_A, start_b=handle_in_B,
    bootstrap_progress=lambda target: my_geometric_progress(target),   # "A" or "B"
    config=LoopConfig(tau1=20, tau2=40, n_trials=16, episodes_per_round=16),
)
loop.run(8)                 # bootstrap, then 8 rounds
loop.model                  # CommittorEnsemble: .predict, .logit, .std
loop.recorder.save("swarms.h5")
```

Each round restarts from visited configurations spread evenly over the current
committor estimate, runs short episodes until a basin is entered, and refits. An
episode targets B with probability `q` (else A); its progress metric is the
log-probability of reaching that target and trials are selected with the driver's new
`selection="boltzmann"` weights, i.e. with probability proportional to the committor at
their endpoints (`sigma = 1`). In the limit of many trials this is importance
resampling of the transition kernel conditioned on reaching the target — the same
conditioned process that control-based methods approximate with a force, obtained here
by selection alone.

The bootstrap progress variable only has to move the system out of each basin; it
does not need to reach the other side (see the results below).

### Validation tools

```python
from pathgennie.committor import shooting_committor
res = shooting_committor(engine, handle, basins, cv_fn, n_shots=200, check_every=10)
res.committor, res.stderr, res.n_open, res.total_steps
```

`shooting_committor` works with any engine and is the ground truth for molecular
systems. `pathgennie.committor.reference.grid_committor` solves the committor exactly
on a 2-D grid for toy potentials (needs SciPy).

`committor_scores` reports Pearson correlation and RMSE over all test points — the
numbers the literature quotes — and the same two restricted to the transition region
(reference `q` between 0.05 and 0.95). Quote the second pair: most test points sit next
to a basin, where any model looks good.

## Results so far (toy Wolfe-Quapp, overdamped, kT = 1)

Reproduce with `python benchmarks/committor_toy.py` (about four minutes on CPU).

**M0 — shooting agrees with the exact committor.** Six points, 400 shots each: largest
deviation 1.4 standard errors.

**M1 — committor from ordinary discovery runs.** Runs A→B and B→A with `tau1 = 20`,
16 trials per cycle; scored on 637 anchors of independent runs, 144 of them in the
transition region (TS). Averages over three network initialisations.

| discovery runs (each way) | MD steps | r | RMSE | r (TS) | RMSE (TS) |
|---|---|---|---|---|---|
| 1 | 98,240 | 0.924 | 0.173 | 0.522 | 0.360 |
| 2 | 166,400 | 0.946 | 0.153 | 0.758 | 0.318 |
| 4 | 224,640 | 0.995 | 0.045 | 0.989 | 0.091 |
| 8 | 461,120 | 0.999 | 0.022 | 0.996 | 0.044 |
| 16 | 715,840 | 1.000 | 0.011 | 0.999 | 0.021 |
| *geometric CV, no learning* | – | 0.943 | 0.208 | 0.838 | 0.311 |

Reaching the 8-run accuracy by shooting at the same 144 points would take roughly
2.9 million steps, about six times the swarm data.

What the table also says:

* **One or two runs are not enough.** This potential has two reaction channels; a
  couple of runs cover one of them and the model is poor in the transition region
  even though the overall correlation still looks respectable (0.92–0.95).
* **Swarm length matters.** With 8 runs each way, `tau1 = 2` gives r (TS) = 0.76,
  `tau1 = 5` gives 0.98, `tau1 = 20` gives 0.996. Boundary information spreads inward
  by one lag per sweep of the fixed-point iteration, so lags that are tiny compared
  with the time to commit (about 180 steps from the barrier here) learn slowly. The
  comparison is not at equal cost — shorter swarms also used fewer MD steps — but the
  practical message stands: the fs-scale `discovery` profile is too short to learn
  from, and learning runs should use longer swarms or add `run_swarms` data.

### M3 — closed loop from a poor CV

Reproduce with `python benchmarks/committor_loop_toy.py` (about 30 minutes on CPU).

The loop was bootstrapped with the **x coordinate alone**, which cannot distinguish the
product from an off-path minimum, and scored on a grid over the whole thermally
relevant landscape: 450 transition-region points, 279 in the lower channel and 171 in
the upper one. This is a harder test than the M1 table, which scored only along the
tube the runs had visited. Eight rounds, about 260,000 MD steps in total (trials and
runners).

| | seeds | r (TS) median (worst) | RMSE (TS) median (worst) | both channels | MD steps to RMSE (TS) < 0.10 |
|---|---|---|---|---|---|
| **closed loop, x-only bootstrap** | 10 | 0.998 (0.995) | 0.033 (0.054) | 10 / 10 | 110,000 |
| good-CV bootstrap | 5 | 0.998 (0.997) | 0.028 (0.032) | 5 / 5 | 96,000 |
| no ensemble-guided restarts | 5 | 0.997 (0.990) | 0.039 (0.064) | 5 / 5 | 43,000 |
| single network | 5 | 0.995 (0.988) | 0.040 (0.078) | 4 / 5 | 86,000 |

Fixed good CV plus offline training (the M1 protocol, ensemble of 4), on the same grid:

| discovery runs (each way) | MD steps | r (TS) | RMSE (TS) | lower channel | upper channel |
|---|---|---|---|---|---|
| 2 | 146,520 | 0.914 | 0.117 | 0.034 | 0.172 |
| 4 | 227,400 | 0.991 | 0.061 | 0.051 | 0.068 |
| 8 | 507,720 | 0.998 | 0.030 | 0.034 | 0.018 |

All three gates pass: (a) every seed reaches r (TS) > 0.95 and RMSE (TS) < 0.10 from the
poor CV; (b) the loop needs about half the MD of the fixed-CV protocol, both to cross
RMSE (TS) = 0.10 (110,000 against between 147,000 and 227,000 steps) and to reach
RMSE ≈ 0.03 (260,000 against 508,000); (c) both channels are resolved in 10 of 10 seeds.

What the ablations do and do not show:

* **The starting CV hardly matters.** A good bootstrap CV gives nearly the same result
  as the x-only one. The loop does not inherit the weakness of the CV it starts from.
* **The ensemble helps robustness.** A single network is worse in the worst case and
  missed a channel in one of five seeds.
* **Restarts guided by ensemble disagreement are not shown to help.** Without them the
  final accuracy is slightly worse, but the 0.10 threshold was crossed sooner in these
  five seeds. With this few seeds that is inconclusive; the feature stays on by default
  but should not be claimed as a contribution until tested on a harder system.
* A bug this benchmark caught: a bootstrap episode can leave its basin on the runner
  and end without one trial endpoint inside it, leaving no boundary data on that side.
  Bootstrap now launches further episodes until both basins are represented.

### M2 — alanine dipeptide in vacuum (CHARMM36 stand-in)

Reproduce with `examples/committor_alanine_dipeptide/m2_study.py` (stages `discover`,
`reference`, `enrich`, `swarms`, `stopped`, `refreshed`, `analyze`; about six hours on
four CPU cores). The numbers are in `results_charmm36/m2_summary.json`. System, basins
and features follow the Pharos benchmark (300 K, 1 fs step, friction 1/ps, basins of
±10° around C7eq and C7ax, 45 heavy-atom distances as input) **except the force field**:
CHARMM27 is not bundled with OpenMM, so these runs use `charmm36_2024.xml` and are not
comparable with published numbers.

**Reference.** 450 held-out configurations from discovery runs that share no seed with
the training runs, 20 to 100 shots each, 20.0 million MD steps. The committor switches
sharply — over roughly φ = 95–160° — so a uniform draw put only 13 of 300 configurations
in the transition region (TS); a second batch drawn from that φ window (a geometric
criterion, blind to the shooting outcome) raised it to 62. Shooting noise alone
contributes 0.038 to an RMSE (TS) against this reference.

**Training data.** 24 discovery runs (12 each way, 20 fs swarms, 16 trials) give the
`discovery` set. Selection-free swarms were then launched at the *same* 600
configurations drawn from those runs, at every lag, so that coverage is held fixed and
only the swarm length changes. `stop ≤ N` are swarms of at most N fs whose trials stop
when found inside a basin (checked every 20 fs).

| protocol | MD steps | r | RMSE | r (TS) | RMSE (TS) |
|---|---|---|---|---|---|
| *geometric CV, no learning* | – | 0.886 | 0.252 | 0.341 | 0.306 |
| discovery swarms (20 fs, selection on) | 2,223,360 | 0.934 | 0.213 | 0.814 | 0.204 |
| lag 2 fs | 19,200 | 0.655 | 0.384 | −0.073 | 0.358 |
| lag 5 fs | 48,000 | 0.679 | 0.370 | 0.107 | 0.338 |
| lag 20 fs | 192,000 | 0.514 | 0.509 | −0.253 | 0.472 |
| lag 20 fs, 8× training steps | 192,000 | 0.637 | 0.432 | −0.088 | 0.424 |
| lag 50 fs | 480,000 | 0.868 | 0.248 | 0.263 | 0.350 |
| lag 200 fs | 1,920,000 | 0.972 | 0.136 | 0.794 | 0.202 |
| lag 500 fs | 4,800,000 | 0.988 | 0.106 | 0.917 | 0.167 |
| stop ≤ 200 fs | 1,491,140 | 0.990 | 0.064 | 0.900 | 0.150 |
| stop ≤ 500 fs | 2,697,160 | 0.967 | 0.130 | 0.791 | 0.238 |
| stop ≤ 1000 fs | 3,835,060 | 0.936 | 0.182 | 0.746 | 0.342 |
| stop, all three | 8,023,360 | 0.986 | 0.084 | 0.843 | 0.180 |
| **discovery + stop ≤ 1000 fs** | 6,058,420 | 0.996 | 0.043 | 0.955 | 0.103 |

**Gate** (r > 0.9, RMSE (TS) < 0.15, less MD than the shooting reference): met by
`discovery + stop ≤ 1000 fs`, with 3.3 times less MD than the reference. Two reasons to
treat this as provisional. First, that protocol was picked out of thirteen after
scoring all of them on the same test set. Second, every row is one data set and one
ensemble, and the rows do not order sensibly (`stop ≤ 1000` alone is worse than
`stop ≤ 200` alone), so differences between neighbouring rows are not reliable. The
repeat with CHARMM27 will fix the protocol in advance: discovery swarms plus stopped
swarms of at most 1 ps at 600 of their anchors, ensemble of four, 10,000 steps.

**The velocity question.** Trials start with fresh velocities, so what a position-only
model converges to is the committor of the chain whose velocities are redrawn every
swarm length. That committor was measured directly, with no model, by shooting with
velocities redrawn every τ at 24 TS configurations (40 shots each):

| velocities redrawn every | RMS difference from physical committor | expected from sampling noise | mean difference | r | MD steps per shot |
|---|---|---|---|---|---|
| 20 fs | 0.095 | 0.075 | +0.039 ± 0.015 | 0.963 | 11,125 |
| 200 fs | 0.079 | 0.075 | +0.017 ± 0.015 | 0.963 | 3,235 |
| never (physical) | – | – | – | – | 865 |

Redrawing velocities every 20 fs barely changes the committor: the difference is close
to sampling noise, with a small shift toward the product that is 2.6 standard errors
from zero. **Velocity resampling is therefore not what makes short swarms fail**, which
is the opposite of what was expected when M2 was set up. This is for a 22-atom molecule
in vacuum at friction 1/ps and 24 configurations; it should be repeated in solvent.

**What does make them fail.**

* *The relation carries almost no signal at short lags.* Under 20 fs resampling a
  trajectory needs 11,125 steps to commit — 556 swarm lengths — so boundary values must
  be passed inward through hundreds of sweeps, each resting on a difference in `q` of
  a few hundredths between a start and its endpoints. Swarms of 2–50 fs did worse than
  the geometric CV, and eight times more training steps did not help.
* *Unstopped long swarms overshoot the basins.* At 500 fs only 89 endpoints lay in the
  10° basin A, against 2,057 once trials were stopped on entry. Stopping at 200 fs
  improved RMSE (TS) from 0.202 to 0.150 with a quarter less MD.
* *Coverage of the transition region dominates.* Models fit held-out starts from the
  same pool well (RMSE 0.05 against their 16-trial outcomes) yet miss the TS test
  points, because a uniform draw of 600 starts puts few of them where `q` changes. The
  best protocol is the one that adds the 6,176 discovery anchors along the paths.
  Weight decay also helped in a side test (0.34 → 0.18 for `stop ≤ 1000`); the trainer
  has no validation-based stopping yet.

**Consequences for the design.**

1. Steering and learning need different swarms. Short swarms are fine for selection;
   learning data should be stopped swarms of a few hundred fs to 1 ps.
2. Those learning swarms should be launched where `q` is changing — which is what the
   closed loop's `q`-stratified restarts do. M3b is therefore the real test, and
   `CommittorLoop` needs a separate learning-swarm length.
3. The plain M1 protocol (learn from the discovery swarms as they are) does not carry
   over to MD: r (TS) 0.81, RMSE (TS) 0.20.
4. The trainer needs a held-out split with early stopping, and the reference set
   should always be enriched in the transition region before any score is quoted.

## Known limits

* **Velocities.** Trials start with fresh Maxwell-Boltzmann velocities and the network
  sees positions only, so the fixed point is the committor of the velocity-refreshed
  chain rather than of the physical dynamics. On alanine dipeptide in vacuum the two
  agree to within sampling noise (see M2); this is untested in solvent and says nothing
  about *rates*, which the refreshed chain distorts strongly (it commits 13 times more
  slowly at 20 fs).
* **Coverage.** The model is only meaningful where swarms were launched. Off the
  sampled tube it extrapolates.
* **Basin entry is checked at segment ends in driver cycles.** A trial that enters a
  basin and leaves again within one segment is missed. `run_swarms(check_every=...)`
  stops trials on entry and should be used for any swarm that is not short compared
  with the residence time in the smaller basin.
* **No validation-based stopping.** Training losses go to ~1e-5, i.e. the networks
  interpolate their targets; accuracy then depends on where the starts are.
* **Serial scheme.** Probing stops at the first success, so the number of probes
  depends on their outcomes and the endpoint average is slightly biased. Use the
  default parallel scheme for learning runs.
* **Exploration is not solved.** On the toy both channels are found because thermal
  motion at kT = 1 reaches them. Nothing in the loop yet drives it toward a channel
  that is many kT away from everything visited; that needs the tempering or
  state-discovery work planned after M4.
* **Episodes are short here by construction.** At unit selection temperature an
  episode lasts about as long as an unbiased commitment (a few cycles on the toy). For
  a molecular system with ns commitment times `sigma < 1` will be needed, and what that
  does to the learned committor is untested.

## Milestones

Each milestone is a capability with a pass/fail gate. "TS" means the transition region,
reference `q` in (0.05, 0.95). Gates on molecular systems are scored against shooting
committors computed with the tool from M0.

| | Capability | Test system | Gate | Status |
|---|---|---|---|---|
| **M0** | Ground truth: exact grid committor, shooting committor, basin and swarm-record API | Wolfe-Quapp | shooting within 4 standard errors of exact at every test point | **passed** |
| **M1** | Offline committor from recorded discovery swarms | Wolfe-Quapp | r (TS) > 0.95 and RMSE (TS) < 0.10 at 8 runs each way | **passed** |
| **M2** | Same on real MD; resolve the velocity/lag question; internal-coordinate features | alanine dipeptide, vacuum, C7eq ↔ C7ax | r > 0.9 and RMSE (TS) < 0.15 against shooting at ≥ 200 configurations, using less MD than shooting them directly; accuracy against lag from 2 fs to 1 ps | **provisional pass** with CHARMM36 (protocol chosen post hoc); repeat with CHARMM27 and a fixed protocol. YAML wiring for the backends still open |
| **M3** | Closed loop: log-committor as the progress variable, selection weights proportional to `q`, target B with probability `q` else A, restarts stratified in `q`, network ensemble | Wolfe-Quapp | (a) started from a deliberately poor CV, r (TS) > 0.95 and RMSE (TS) < 0.10 over the whole landscape; (b) fewer MD steps than fixed-CV discovery plus offline training; (c) both channels in ≥ 9 of 10 seeds | **passed** (toy) |
| **M3b** | The same loop on real MD | alanine dipeptide | reaches the M2 accuracy from a single-dihedral CV with less MD than M2 | after M2 |
| **M4** | Closed loop on a folding and an unbinding problem; first HPC runs | CLN025, OAMe-G2 | r > 0.9 against shooting at ≥ 300 configurations each; fewer cycles to first transition than the hand-picked CV | HPC |
| **M5** | High-dimensional path refinement: path nodes as conditional means of features in `q`-bins, per channel; path CVs `s`, `z` in any dimension | Wolfe-Quapp (both channels), alanine dipeptide, CLN025 | toy: nodes within 0.1 of the minimum-free-energy path of each channel; molecules: `s` monotonic in shooting `q` (Spearman > 0.95) | |
| **M6** | Export: TorchScript CV usable from PLUMED; `q`-interfaces for TIS/RETIS; `q`-bins for both weighted-ensemble codes; umbrella/OPES on `logit q` | alanine dipeptide | free-energy barrier along `logit q` within 0.5 kcal/mol of a reference; exported model reproduces Python values to 1e-5 | |
| **M7** | Self-contained rates: committor-weighted flux (Hill relation) from velocity-continuous segments | Wolfe-Quapp (exact rate from the grid operator), ion-pair or alanine dipeptide (brute-force rate) | rate within a factor of 2 of the reference with a bootstrap interval that covers it | |
| **M8** | Paper benchmarks against published budgets (Pharos) and an uncontrolled baseline | CLN025, Trp-cage, OAMe-G2; one QM/MM reaction | simulated time to r = 0.9, including seeding | HPC |

Second paper, after M8: orthogonal slow CVs (committor-distribution test on the
`q ≈ 1/2` surface, channel-separating coordinates, slow modes orthogonal to `q`) and
discovery without a known product state through Trails-MD metastable states.
