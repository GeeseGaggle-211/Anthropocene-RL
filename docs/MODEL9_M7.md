# model9 M7: PPO hero runs, stages 2-4

Status: complete (18 October 2026). Contract 8: simulator `feature/elder-model9-native` a682994, pin
`configs/simulator/elder-model9-native8.json`, learner `autonomous_v7`. Checkpoints and trajectories are in
`gs://anthropocene-rl-datasets-508912/model9/ppo-m7/`.

## Pilot rebuild (18 October 2026)

The pilot applied the changes agreed after M7 and reran stages 2-3 locally.

**Simulator** (`feature/elder-model9-native` 4ba0de1, 3a5db2e; `Docs/HerdLabour.md`):
- **Animal textile need.** Portable-shelter herding camps pay tent fabric ahead of the reserve, herd growth for that
  need is granted before Recovery, and herding daughters keep an animal. Herding-only fixture QoL rose from
  0.13-0.15 to 0.32-0.44, with no regression on default worlds.
- **Herd transition limit.** A non-distress cull is capped at half the herd per turn. With the heuristic, culls of
  at least half a herd fell from 70 to 38 per 32 worlds; 58 of the 70 had coincided with taking up farming, and
  none of either set reached zero.
- **Data.** Teacher data was regenerated as `teacher-v5`: 374k TRAIN decisions, with teacher welfare up 8.6%.
  Stage 1 passes on seeds 0 and 2 (`MODEL9_STAGE1.md`), and M7 starts from seed 2.

**Learner** (46310e9, 3fdf818):
- **(a)** Advantages are standardized per world regime, and each regime gets equal weight (decision weights capped
  at 4x).
- **(b)** Optional capability-conditioned expert adapters (`--experts`): one zero-initialized residual adapter per
  economy, averaged over the economies a village can practise.
- **(c)** `stage_eval` fails a candidate whose default welfare is below the init, any regime more than 3% below it,
  or with more extinctions. `best.pt` maximizes the geometric mean of per-regime ratios on separate selection seeds
  (2020-2025), which removes the winner's curse.
- A switching cost of 0.02 x total-variation mix change x welfare.

**Runs.**
- Stage 2: 50 iterations from stage 1 seed 2.
- Stage 3: 40 iterations from stage-2 best, twice with the same world draws, with shared heads (a) and with expert
  adapters (a+b).

Gate (welfare against the stage-1 init; DEVELOPMENT 2000-2019 default, fixtures 2000-2009, held-out grassland 10):

| Regime | Stage 2 best | Stage 3 (a) | Stage 3 (a+b) | Pre-pilot stage 2 | Pre-pilot stage 3b |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default (20) | **1.036** | 0.980 | 0.995 | 1.037 | 1.057 |
| Herding-only (10) | 0.996 | 0.992 | 1.021 | 1.040 | 0.986 |
| Farming-only (10) | 1.076 | 1.013 | 1.045 | 1.092 | 1.173 |
| Fishing-only (10) | **1.045** | 1.054 | 1.080 | 0.938 | 0.860 |
| Held-out grassland (10) | **1.018** | 0.978 | 0.974 | 0.956 | 0.961 |
| Gate | **PASS** | fail (default) | fail (default, by 0.5%) | - | - |

Against the teacher, stage-2 best scores:
- default 1.036, herding 1.016, farming 1.042, fishing 1.213 and grassland 1.069;
- QoL 0.83 on default worlds and 0.41 on herding-only worlds (0.14 before the simulator fix);
- no extinctions in any regime.

**Findings.**
1. **Balancing and regime selection stop the minority-regime erosion.** Fishing moved from -6% to +4.5% and
   grassland from -4% to +1.8%, while default worlds held at +3.6%.
2. **Expert adapters help modestly where the regimes differ.** With the same world draws, (a+b) beat (a) in stage 3
   on default (+1.5%), herding (+2.9%), farming (+3.2%) and fishing (+2.5%), and roughly tied on grassland. That is
   one seed, against evaluation noise of about ±3%, so it is suggestive, not established.
3. **Stage 3 still loses welfare.** In both variants it fell below its stage-2 start on default worlds. Its two
   additions are unanchored clothing and shelter routes, and grassland and fixture rehearsal; the pre-pilot 3b run
   recovered only at its last evaluation. The next run should isolate which addition causes the loss: native
   routes or a route anchor. It should not be tuned further blind.
4. **The switching cost is too slight to matter.** The deterministic policies' mix change per turn is the cloned
   heuristic's (0.079 on default worlds, 6% abrupt switches). Herd liquidation is governed by the simulator's cull
   cap.

### Stage-3 regression isolated

Two more stage-3 runs used expert adapters, the same stage-2 start and the same world draws:
- `mixed-native-routes`: clothing and shelter routes stay native, and the route heads are untrained;
- `mixed-anchored-routes`: the routes are learned under a teacher KL anchor.

Gate (against the stage-1 init):

| Regime | Stage 2 best | Free routes (a) | Free routes (a+b) | Native routes (a+b) | Anchored routes (a+b) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default (20) | 1.036 | 0.980 | 0.995 | **1.032** | 1.025 |
| Herding-only | 0.996 | 0.992 | 1.021 | 0.979 | 1.025 |
| Farming-only | 1.076 | 1.013 | 1.045 | 1.053 | 1.026 |
| Fishing-only | 1.045 | 1.054 | 1.080 | 0.985 | 0.967 |
| Held-out grassland | 1.018 | 0.978 | 0.974 | 0.987 | 0.997 |
| Gate | PASS | fail | fail | **PASS** | fail (fishing 0.967) |

**Cause.** Learning the routes freely cost 4-5% on default worlds; with routes native or anchored, default
welfare held at 1.025-1.032. The mixed-use curriculum itself is not the cause, but it adds nothing measurable over
stage 2: the native-routes run is within noise of stage-2 best on every regime and slightly below it on four of
five.

The 6-world selection evaluation was misleading:
- it scored both new runs below their stage-2 start on default worlds (0.94-0.98);
- the 20-world gate puts them at 1.03.

Selection needs at least the gate's size.

**The game model is not the limit.** `--model9-oracle` (simulator 77ecd3d) ran 6 TRAIN worlds (1100-1105) for 25
turns. In each arm, every village that can farm and herd follows one fixed directive for the whole horizon:

| World | Best fixed arm | World welfare vs heuristic | Mixed-village welfare across arms |
| ---: | --- | ---: | --- |
| 1100 | farm-lean | 1.047 | 0.95-1.10 |
| 1101 | balanced | 1.110 | 0.90-1.14 |
| 1102 | balanced, high development | 1.030 | 0.80-0.98 |
| 1103 | balanced, high development | 1.051 | 0.85-1.07 |
| 1104 | pure farm | 1.013 | 0.76-1.03 |
| 1105 | (heuristic best) | 0.942 | 0.85-0.94 |

- In 5 of 6 worlds a fixed allocation beats the heuristic, by 1-11%. The best of 8 noisy arms is optimistic, but a
  state-dependent policy should do at least as well.
- The surface is steep: about 25% between arms. The best arm differs by world, a larger development budget is the
  most consistent lever, and pure herding or zero development are consistently worst.
- QoL stays at 0.78-0.90 in every arm. There is no collapse or pathology.

The choice space is therefore solvable and rewards good choices. The remaining gap is on the learning side.

The critic's values correlate 0.82 with realized returns on default worlds, against 0.20-0.65 on the fixtures,
where stage 2 still improved. So critic fit does not explain the default-world loss either.

### Seed noise and the stage-3 answer (18-19 October 2026)

**Stage-2 seed noise.** Three PPO seeds of the stage-2 recipe from stage-1 seed 2 (gate, against the init):

| Regime | Seed 0 | Seed 1 | Seed 2 | Mean | Sigma |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default (20) | 1.036 | 1.037 | 1.005 | 1.026 | 0.018 |
| Herding-only | 0.996 | 0.988 | 1.004 | 0.996 | 0.008 |
| Farming-only | 1.076 | 0.961 | 0.987 | 1.008 | 0.060 |
| Fishing-only | 1.045 | 1.051 | 1.017 | 1.038 | 0.018 |
| Grassland | 1.018 | 0.965 | 1.009 | 0.997 | 0.028 |
| Gate | pass | fail (farming, grassland) | pass | 2 of 3 | |

**Free against native routes, two pairs.** Each pair starts from the same stage-2 checkpoint (seeds 0 and 2), with
expert adapters and the same world draws:

| Regime | Pair 1 free | Pair 1 native | Pair 2 free | Pair 2 native | Mean free - native |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default | 0.995 | 1.032 | 1.018 | 1.037 | -2.8% |
| Herding-only | 1.021 | 0.979 | 0.996 | 0.983 | +2.8% |
| Farming-only | 1.045 | 1.053 | 1.101 | 1.059 | +1.7% |
| Fishing-only | 1.080 | 0.985 | 1.072 | 1.068 | +5.0% |
| Grassland | 0.974 | 0.987 | 0.983 | 1.010 | -2.0% |

**Conclusions.**
- **Free route learning costs default and grassland worlds 2-3%, in both pairs.** That is about 1.5 standard errors
  given the measured seed spread: real but modest, smaller than the single-run -4%.
- **Free routes help the single-economy fixtures,** where one textile and shelter route is natural.
- **The mechanism is not route churn.** The free-route policy picks the heuristic's route 98% of the time and
  switches less often. It ends with fewer and smaller settlements: 10% fewer village-turns on default worlds, at
  slightly higher QoL.
- **Stage 3 with native routes adds value.** From stage-2 seed 2 (default 1.005) it reaches 1.037, and both pair-2
  runs pass every gate.
- **Recommendation:** keep routes native, or anchored, in mixed and default worlds. Route learning belongs with
  capital commitments (`ElderCapitalCommitmentsDesign.md`), where a route is a held decision.

### Capital commitments, first learning test (19 October 2026)

Setup:
- stage 2 with `--commitments` (contract 9: simulator c37ea55, learner 7568ced), three seeds from stage-1 seed 2;
- control: the three contract-8 stage-2 seeds (same init, seeds and settings; not bit-identical, see the upgrade
  note);
- welfare compared against the teacher, because the contract-9 init re-evaluates slightly differently.

| Regime | Control | Commitments | Difference |
| --- | ---: | ---: | ---: |
| Default (20) | 1.026 (sd 0.018) | 1.016 (sd 0.029) | -0.010 |
| Herding-only | 1.016 (sd 0.008) | 1.005 (sd 0.006) | -0.011 |
| Farming-only | 0.976 (sd 0.058) | 0.950 (sd 0.011) | -0.026 |
| Fishing-only | 1.204 (sd 0.021) | 1.185 (sd 0.082) | -0.020 |
| Grassland | 1.048 (sd 0.030) | 1.021 (sd 0.009) | -0.027 |

- No commitment seed passes every regime gate (control: 2 of 3). Every difference is within about one control sd
  except farming and grassland (about 1 sd).
- **The policy did not learn to use commitments.** Over 50 iterations the span shares barely moved: holds stayed
  at 3-6% and "this turn only" at 48-57%. A third of decisions broke a holdable commitment.
- Deterministic evaluation takes the argmax span, which stayed "this turn only", so **the evaluated policies never
  commit.** The small losses reflect noisier training (random commitments of randomly drawn capital values during
  rollouts), not committed behaviour.
- **Conclusion:** the experiment does not test whether commitments help. The machinery works (simulator and
  learner checks); the exploration does not.

Next, before any retest:
- initialise the span head by behaviour cloning from the heuristic's own persistence: label a hold where the
  heuristic's capital decision matches its previous one within tolerance, and a span where it stays matched;
- or start holds at higher probability and strengthen the break cost.

Until then, keep commitments off in training. With the span head off, a contract-9 policy behaves as contract 8.

### Capital commitments, bootstrapped span head (19 October 2026)

Bootstrap (`span_bootstrap.py`, 83bcd28):
- labels come from `span_labels.py`. Where the heuristic's capital decision repeats within 0.05 (levels and mix
  shares, identical routes), the run becomes the longest allowed span, followed by holds. In the DEVELOPMENT data,
  31% of decisions are holds and 13% start a commitment.
- stage-1 seed 2 was upgraded to contract 9 and fine-tuned for 3 epochs on the BC loss plus span cross-entropy;
  output `checkpoints10/stage1-seed2-span.pt`.
- span accuracy 0.894 (always-single 0.56). Holds are taken whenever legal and 6-turn spans recall 62%; 2-4-turn
  spans are rarely predicted.
- closed loop on 20 DEVELOPMENT worlds: welfare 0.9955 of the teacher; 6.4% commitment starts and 10.6% holds;
  replay exact.

The stage-2 test was then rerun (same settings as the first test, three seeds from the bootstrapped init). Welfare is
shown against the teacher:

| Regime | Control | First test | Bootstrapped | Bootstrapped - control |
| --- | ---: | ---: | ---: | ---: |
| Default (20) | 1.026 (sd 0.018) | 1.016 (sd 0.029) | 1.035 (sd 0.027) | +0.009 |
| Herding-only | 1.016 (sd 0.008) | 1.005 (sd 0.006) | 1.026 (sd 0.006) | +0.010 |
| Farming-only | 0.976 (sd 0.058) | 0.950 (sd 0.011) | 0.996 (sd 0.105) | +0.020 |
| Fishing-only | 1.204 (sd 0.021) | 1.185 (sd 0.082) | 1.247 (sd 0.064) | +0.043 |
| Grassland | 1.048 (sd 0.030) | 1.021 (sd 0.009) | 1.035 (sd 0.009) | -0.013 |

Usage:
- **The policy now commits.** In training, holds are 30-35% of decisions and commitment starts 17-22%. Voluntary
  breaks are about 0.1-0.2%, against a third of holdable commitments in the first test.
- In deterministic evaluation, holds are 6-9% of decisions on default and farming worlds and 18-27% on herding and
  fishing worlds.

Selection artefact:
- Seed 2 wrote no `best.pt`. Every one of its evaluations had a single fishing-fixture extinction. Best-checkpoint
  selection rejects more extinctions than the init's own baseline, and this init's baseline happened to have none
  (the control and first-test inits each had one).
- Seed 1's selected checkpoint is therefore from iteration 9. Seed 2 was gated on `latest.pt` and is marked
  unselected.
- Extinctions on these fixtures are common to every arm. Selection should compare against a fixed reference rather
  than the init's own luck.

Gate: no bootstrapped candidate passes.
- seed 0: farming 0.967;
- seed 1: farming 0.928, at iteration 9;
- seed 2: one fishing extinction.

For comparison, the control passed 2 of 3 and the first test 0 of 3.

Conclusions:
- **The bootstrap fixes exploration, and commitments no longer cost welfare.** The bootstrapped arm is above the
  first test in every regime and above the control in four of five.
- **It does not show a gain.** Every difference from the control is within about one control sd, except herding
  (+0.010, about 1.3 sd); with three seeds that is not significant.
- Farming remains the weak regime for both commitment arms: two of three candidates fall below its 3% gate.
- Not yet tested: whether commitments help where they were designed to help, i.e. stage 3 mixed use with committed
  routes against the route-churn regression (tested next).

### Stage 3 with committed routes (19-20 October 2026)

Setup (`model9-data/stage3-commit.ps1`):
- all arms start from the bootstrapped stage-2 seed 0 (`stage2-span-seed0/best.pt`, default 1.029 vs teacher);
- same settings as the stage-3 p2 pair: experts, seed 3, 40 iterations;
- one seed per arm.

**Evaluation caveat.** Training without `--commitments` forces "this turn only", but deterministic evaluation always
takes the span head's argmax.
- A policy that inherited the bootstrapped span head therefore still commits at evaluation, even when PPO never
  trained its spans.
- Each free-route arm was gated twice: as trained, and with its span head forced to "this turn only" (`best-single`).
- Contract-8 controls are unaffected: their upgraded span head never commits.

Welfare against the teacher:

| Arm | Spans at evaluation | Default | Herding | Farming | Fishing | Grassland | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Free routes, trained with commitments | committing | 0.995 | 1.005 | 0.944 | 1.158 | 1.051 | farming 0.954 |
| same | single | 0.998 | 0.997 | 0.967 | 1.197 | 1.054 | pass |
| Free routes, trained single-turn | inherited bootstrap | 1.047 | 1.025 | 1.019 | 1.142 | 1.024 | pass |
| same | single | 1.042 | 1.031 | 1.047 | 1.199 | 1.048 | pass |
| Native routes, trained with commitments | committing | 1.027 | 1.016 | 0.976 | 1.103 | 1.002 | fishing 0.951, grassland 0.956 |

Interpretation (one seed per arm; stage-2 gate noise is about 0.02-0.03):

**Committed routes do not repair stage 3.**
- Training with commitments gave the weakest default-world policy (0.995-0.998). This is 4-5% below the same
  init trained single-turn, whichever way its spans are evaluated.
- The loss therefore lies in the trained weights, not in committing at evaluation.

**Committing at evaluation does not help either.**
- In both free arms, forcing "this turn only" is neutral on default worlds (+0.003 / -0.005).
- It is better by 2-6% on farming, fishing and grassland in five of six comparisons.
- The heuristic's own persistence, copied into the span head, costs a little where it differs from the policy's own
  choices.

**Free routes trained single-turn pass every regime.**
- The single-span evaluation is the best stage-3 result so far: default 1.042 against 1.029 for its stage-2 init,
  i.e. stage 3 adds about 1%, within noise.
- The earlier default-world regression (free routes 0.980) does not reproduce from this init.
- With one seed, this is not yet evidence that the regression is gone. Together with the seed-noise section, it
  supports noise plus fewer, smaller settlements, not route churn. That mechanism is one commitments were never
  designed to address.

Decision proposed:
- keep commitments off in training and force "this turn only" at evaluation; contract 9 remains in place
  (simulator, saves, learner) at no cost;
- stop pursuing commitments as the stage-3 fix;
- make evaluation honour the training stage's commitments setting, so a single-turn policy cannot commit by
  inheritance;
- confirm the stage-3 result with three seeds of free routes, single-turn.

The user approved this on 19 October. Evaluation now honours the setting (RL e156184).

### Decisive stage-3 experiment: pre-registration (written 19 October 2026, before any result)

Question: is stage 3 (mixed use, free routes) worse than simply training stage 2 for as long?

Every earlier stage-3 comparison was against an untrained init, on 1-3 seeds and about 30 worlds.

Arms:
- shared: init `stage2-span-seed0/best.pt`, expert adapters, single-turn, 40 iterations, identical arguments except
  `--stage`;
- treatment: stage 3 (`mixed`), seeds 3-7 (seed 3 is `stage3-c9-free`, evaluated single-turn);
- control: stage 2 (`capability`), seeds 3-7.
- reference: the init evaluated single-turn.

Evaluation (`stage_eval`, script `model9-data/stage3-decisive.ps1`):
- 54 default DEVELOPMENT worlds (2000-2019, 2026-2059; the selection worlds 2020-2025 are excluded);
- 20 seeds for each capability fixture (2000-2019);
- the 10 held-out grassland seeds;
- CONFIRMATION seeds untouched.

Primary measure:
- `latest.pt` (fixed training budget, no checkpoint selection);
- default-world welfare, treatment vs control;
- 95% bootstrap interval resampling training seeds and the shared worlds (`paired_compare`).

Decision rule:
- interval entirely below 1: stage 3 regresses, and the cause is investigated before scale-out;
- interval entirely above 1: stage 3 helps;
- interval containing 1: no detectable difference at this scale; if it lies within ±2%, mixed-use training is
  safe to carry into scale-out.

Secondary measures: the other four regimes on the same rule (3% tolerance), each arm against the reference, and the
same comparison on `best.pt` where selection kept one.

### Decisive stage-3 experiment: results (20 October 2026)

The run took 6.5 hours (23:09-05:43): 6 trainings in two batches, then about 1.1 hours per 1,600-rollout
evaluation. Evidence is in `eval10/decisive-{latest,best}.json` and `-compare.json`.

Treatment (stage 3) against control (stage 2 continued). The ratio is of arm-mean welfare summed over worlds, with
95% bootstrap intervals over training seeds and shared worlds:

| Regime (worlds) | `latest.pt` (primary) | `best.pt` |
| --- | --- | --- |
| Default (54) | **0.987 [0.955, 1.017]** | 0.983 [0.963, 1.003] |
| Herding (20) | 1.003 [0.983, 1.026] | 1.006 [0.985, 1.027] |
| Farming (20) | 0.979 [0.941, 1.023] | 0.990 [0.955, 1.028] |
| Fishing (20) | 1.014 [0.952, 1.082] | 0.980 [0.914, 1.046] |
| Grassland (10) | 0.984 [0.943, 1.023] | 0.988 [0.951, 1.028] |

Per seed, welfare against the teacher (`latest.pt`):
- default worlds:
  - control 0.993, 1.023, 0.998, 1.014, 1.004 (mean 1.006, sd 0.012);
  - stage 3 0.992, 1.015, 0.948, 0.994, 1.015 (mean 0.993, sd 0.027).
- stage-1 init: 0.967;
- stage-2 init, single-turn reference: 1.008.

Each arm against that reference (`latest.pt`):

| Regime | Control | Stage 3 |
| --- | --- | --- |
| Default | 0.999 [0.970, 1.027] | 0.985 [0.953, 1.018] |
| Herding | 1.016 [0.993, 1.039] | 1.020 [0.997, 1.044] |
| Farming | **1.077 [1.028, 1.132]** | 1.055 [0.998, 1.110] |
| Fishing | 0.956 [0.876, 1.043] | 0.970 [0.883, 1.069] |
| Grassland | 1.002 [0.965, 1.044] | 0.986 [0.949, 1.023] |

Gate against the stage-1 init (3% regime tolerance, no extra extinctions):

| Checkpoint | Control passes | Stage 3 passes |
| --- | --- | --- |
| `latest.pt` | 4 of 5 | 1 of 5 |
| `best.pt` | 3 of 5 | 3 of 5 |

Five of the ten failures are a single extinction on the 20 fishing worlds.

Findings:

1. **Pre-registered verdict: inconclusive.**
   - The primary interval contains 1, but it does not lie within ±2%.
   - A stage-3 regression of up to about 4.5% on default worlds is not ruled out, and neither is a 1.7% gain.
   - Both point estimates are -1.3 to -1.7%, and `best.pt` is borderline (upper bound 1.003).
   - Stage 3 is **less stable rather than uniformly worse**: four of five seeds lie in the control's range, and
     seed 5 falls to 0.948. Its default-world seed sd is twice the control's.
2. **Forty more iterations of PPO do not improve default worlds in either arm.**
   - The control against its own init gives 0.999 [0.970, 1.027].
   - At this batch size and anchor strength, learning on default worlds has plateaued at about 1% above the teacher.
   - The gains that remain are in the capability fixtures: farming +7.7% (interval excludes 1) and herding +1.6%.
   - This matters more for scale-out than the stage-3 question: longer runs of the same configuration are not
     expected to move default worlds. Batch size, the teacher anchor and the reward need testing as variables.
3. **Seed noise dominates the uncertainty.**
   - Per-world differences between the arms are small: the largest is 0.34% of total welfare.
   - The interval is set by seed-to-seed spread: default sd 0.012-0.027 per seed at `latest.pt`, 0.005-0.013 at
     `best.pt`.
   - Narrowing the primary interval to ±1% needs about 20-45 seeds per arm. That is the 64-vCPU machine's job, not
     more worlds.
4. **The extinction rule makes the gate a coin flip.**
   - One extinction in 20 fishing worlds decides half the failures, and its occurrence varies between seeds of the
     same arm.
   - The gate should use an extinction rate with an interval rather than "no more than the init".
5. **Seed 7 settles the selection artefact.**
   - With the single-turn baseline, selection kept a `best.pt` for every new seed.

Curriculum decision (user, 20 October): stage 3 and learned movement will be trained together, as a 2x2 of movement
(native or learned) against stage (capability or mixed). The prerequisites are contract-9 movement and a cloned
movement head within about 2-3% of the teacher. The standalone 20-seed stage-3 experiment is dropped; it becomes one
cell of the 2x2.

### E2: what moves default worlds? Pre-registration (written 20 October 2026, before any result)

Question: which optimisation lever lets stage-2 PPO improve default-world welfare at all?

The control above did not move in 40 iterations (0.999 against its init). The suspects are:
- the learning rate: 1e-5, with approximate KL of only 0.005-0.012 per update;
- the teacher anchor: weight 0.05;
- the batch: 12 worlds per iteration.

Design, on the 60-vCPU VM, 24 arms with `OMP_NUM_THREADS=3` (RL `--teacher-kl` flag; hyperparameters saved in
checkpoints):

| Cell | Change from baseline | Seeds | Iterations |
| --- | --- | --- | ---: |
| A baseline | none (lr 1e-5, teacher-KL 0.05, 12 worlds) | 20-24 | 50 |
| B | lr 3e-5 | 20-24 | 50 |
| C | lr 1e-4 (target-KL early stop still active) | 20-24 | 50 |
| D | teacher-KL 0.01 | 20-24 | 50 |
| E | 48 worlds per iteration | 20-23 | 12 |

- Every cell has the same decision budget, about 450-500k.
- Shared: init `stage2-span-seed0/best.pt`, `--stage capability --experts`, single-turn, with the other arguments
  as in the decisive experiment.
- Evaluation: the decisive experiment's 104 worlds, `latest.pt`, against the reference (the init, single-turn).
  Intervals come from `paired_compare`.

Primary measure: each cell's default-world welfare against the reference.

Decision rule:
- a cell "moves default worlds" if its interval lies entirely above 1;
- among moving cells, prefer the one with no regime below 0.97 of the reference, then the largest default gain;
- if no cell moves, optimisation is not the bottleneck at this budget. The economy-only ceiling (oracle on default
  worlds) then decides whether the cap is the action space.

Secondary measures:
- each cell against cell A (paired);
- the guard cost and Lagrange multiplier;
- the welfare/QoL breakdown, to flag gains bought by population at the expense of QoL;
- teacher KL at the end of training.

### Contract 10 (destination movement): built 20 October 2026

- Simulator b38c2c1: `ElderMovement`, Elder memory schema 3.
- Learner 3be0f7d and a7237f1.
- Design notes: simulator `Docs/ElderModel9Contract9MovementDesign.md`, "Implementation".

Exactness and start state:
- Contract-9 checkpoints upgrade exactly. With native movement, an upgraded `stage2-span-seed0/best-single.pt`
  reproduces the decisive experiment's per-world welfare bit for bit (DEVELOPMENT 2000 and 2001).
- The region head starts native, at probability 0.8 when all rows are legal, so its argmax is native. It needs no
  re-cloning: the heuristic's label is "native" by construction.
- At its start, a learned-movement policy makes no moves, yet its welfare differs per world: 0.76-1.28, totalling
  0.995 over DEVELOPMENT 2000-2019.
- The cause is input shift, not movement. Enabling movement changes the mask inputs, which the contract-9 policy never
  saw in training, and the world's cascades amplify the small change in its other choices.
- The 2x2's native arms therefore train with `--observe-movement`: the same observations, every decision native.

### 2x2 movement x stage: pre-registration (written 20 October 2026, before any result)

Question: does learned destination movement help, alone or together with mixed-use (stage 3) training?

Arms, 6 seeds each (30-35), on the 60-vCPU VM:

| Arm | Stage | Movement |
| --- | --- | --- |
| N-cap | `capability` | `--observe-movement` (native, same observations) |
| L-cap | `capability` | `--learned-movement` |
| N-mix | `mixed` | `--observe-movement` |
| L-mix | `mixed` | `--learned-movement` |

Shared settings:
- init `stage2-span-seed0/best.pt` (upgraded to contract 10), `--experts`, single-turn, 50 iterations;
- the optimisation settings E2 selects by its decision rule (the baseline if no cell moves default worlds).

Evaluation:
- `stage_eval --movement --region-movement` on the decisive experiment's 104 worlds, `latest.pt`;
- every policy runs in the same movement-enabled environment; native arms argmax native;
- the reference is the init in that environment;
- intervals from `paired_compare`.

Primary measure: L-mix against N-cap on default worlds (the combined stage against the stage-2 control).

Decision rule:
- carry the combined stage into scale-out if its interval's lower bound is at least 0.98 and no regime's point
  estimate is below 0.97;
- otherwise use whichever single-factor arm meets that rule against N-cap;
- if none does, stage-2 training stays the scale-out curriculum, and movement is investigated on the valley
  fixtures first.

Secondary measures:
- the movement main effect (L-cap/N-cap, L-mix/N-mix) and the stage main effect (N-mix/N-cap);
- their interaction;
- the share of decisions that move at evaluation. If the learned arms never move, the movement question is untested;
- welfare against QoL;
- valley fixtures (below): learned against native movement on each bank, and the move share.

Valley fixtures (simulator `--model9-valley-scan`, seeds 7000-7999; the heuristic plays up to 12 turns):
- The scan looks for a grassland herding village whose own region observation shows farmland away from its tile:
  - visible: a farming forecast of at least one meal per resident and 1.25 times the current tile's;
  - remembered: river land in grassland, temperate or tropical environments.
- **Strict** adds competition: a known foreign settlement within 4 tiles. 13 seeds qualify; the bank is
  `valley-strict.txt`.
- **Relaxed** requires the valley alone. 221 seeds qualify. The first 40 are held out (`valley-relaxed-heldout.txt`);
  the other 168 (`valley-relaxed-train.txt`) are reserved for curriculum.
- A fixture is (seed, first qualifying turn); evaluation plays the heuristic that many turns, then 25 policy turns.
- Openings alone never qualify: camp spacing (3) keeps every settlement at least 3 tiles away. The design's
  within-two-tiles competition was therefore impossible, and both the rows and the scan count within 4 tiles.

### Curriculum stability: is the world chaotic? (first probe, 19 October 2026)

`--model9-divergence` (simulator 48e531b) runs two copies of a world under the heuristic and, at turn 2, multiplies
one camp's food by (1 + epsilon) in copy B. On DEVELOPMENT 2000-2005 over 25 turns, with epsilon 0.001 and 0.05:
- **11 of 12 runs end identical** in population, settlements, fissions and migrations. The perturbed food persists
  in state (the fingerprints differ) but changes no decision.
- **The exception** (seed 2002, 5%) ends 1.0% lower in population, with the same settlement, fission and migration
  counts.

So the world absorbs small continuous perturbations; it is not chaotic in the butterfly sense. The large per-seed
spread of rule A/B tests (sigma about 9% on main) is a heterogeneous treatment effect: a rule acting on every camp
every turn changes discrete decisions (fission timing, destinations, site choice) differently in each world. The
spread between PPO seeds comes from training stochasticity, not from the environment.

**Discrete perturbation** (epsilon -1: one animal removed from the largest managed herd at turn 4, the same
seeds). The gap does propagate, but it is usually small:

| Seed | Population gap at turn 24 | Notes |
| ---: | ---: | --- |
| 2000 | 0.25% | |
| 2001 | 0.24% | |
| 2002 | **8.0%** | 2 more settlements and 2 more fissions |
| 2003 | 0.72% | |
| 2004 | 0.10% | |
| 2005 | 0% | no managed herd |

The world damps most shocks and occasionally amplifies one through fission timing (one seed in five). Fission
cascades are therefore the amplifier behind the per-seed spread, and the curriculum-stability question is how often a
policy's choices trigger them.

Practical consequences:
- **Simulator A/B:** judge on 20 or more paired worlds (`--model9-ab`), not 6.
- **Learner:** compare candidates on shared openings and report mean ± SE per regime; the per-world variance is
  structural, not noise.

Checkpoints are in `gs://anthropocene-rl-datasets-508912/model9/ppo-pilot/`; evidence is in `evidence/model9-pilot/`.

## Summary (pre-pilot M7)

| Stage | Iterations | Gate on default DEVELOPMENT worlds | Notes |
| --- | ---: | --- | --- |
| 2 capability | 60 | PASS, 1.037 vs stage 1 | Fishing-only worlds -6% |
| 3 mixed use, unanchored | 20 (stopped) | - | Welfare fell to 0.876; development budget drifted |
| 3b mixed use, anchored | 40 | **PASS, 1.057 vs stage 1 (1.083 vs teacher)** | Fishing -14%, held-out grassland -4% |
| 4 learned movement | 40 | **FAIL**, 0.920 vs teacher | Learned movement loses 13-15% against native movement |

- Training was stable in every stage:
  - approximate KL per update 0.005-0.018, under the 0.03 stop;
  - entropy 13.2-15.6, with no collapse;
  - no extinction on default worlds in any evaluation.
- The candidate for scale-out is **stage 3b best** with native movement and fission.
- Two pieces of evidence call for architecture changes before scale-out:
  - minority economic regimes lose ground to the majority (fishing, grassland);
  - the movement pointer head cannot match native movement.

  Both are discussed under *Architecture evidence*.

## Setup

- **Initial policy.** Stage-1 seed 0 on the fixed heuristic, with welfare 1.025 of the teacher and critic EV 0.714
  (`MODEL9_STAGE1.md`).
- **PPO** (`ppo.py`):
  - 12 worlds x 25 turns per iteration (7-9k village-decisions), with learning rate 1e-5;
  - clip 0.2, 4 epochs with early stop at approximate KL 0.03;
  - logit-scale floor -2 and entropy 1e-3.
- **Teacher KL.** The anchor weight is 0.05, on strategic and service levels in every stage. The first stage-3 run
  dropped the strategic anchor and degraded (see below).
- **Guard.** Settlements below 30 residents are a Lagrangian cost with a 1% budget; daughters in their first 3 turns
  are exempt. The multiplier's learning rate is 0.5 in stage 2, 5 in the unanchored stage-3 run and 2 from stage 3b.
- **Curriculum.**
  - Stage 2 uses the capability fixtures in equal thirds: herding-only (mask 12), farming-only (10) and fishing-only
    (6), on TRAIN seeds.
  - Stages 3-4 use half TRAIN worlds, a quarter grassland-rich openings (seeds 4000-5999 with a husbandry village on
    grassland) and a quarter rehearsed fixtures.
- **Evaluation.**
  - Every 10 iterations: deterministic closed-loop rollouts on DEVELOPMENT 2000-2005, plus the fixtures on 2000-2001
    in stage 2.
  - Stage gate (`stage_eval.py`): DEVELOPMENT 2000-2019 default worlds and 2000-2009 for each fixture, each compared
    with the stage-1 init and the teacher.
- **Hardware.** One workstation (6 cores, 14 GB). Throughput is about 75 village-decisions per second, so an
  iteration takes about 2 minutes and a 60-iteration stage (about 0.5 million decisions) about 2 hours.

## Stage 2: capability (60 iterations, about 530k decisions)

Trajectory (periodic evaluation, welfare relative to the stage-1 init):

| Iteration | 9 | 19 | 29 | 39 | 49 | 59 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Welfare ratio | 1.035 | 1.037 | 0.979 | 0.990 | 1.068 | 1.010 |
| QoL | 0.742 | 0.718 | 0.704 | 0.705 | 0.719 | 0.707 |
| Settlements below 30 per world | 0.33 | 0.42 | 0.33 | 0.33 | 0.25 | 0.17 |

The stage-1 baseline had QoL 0.747 and 0.75 settlements below 30 per world.

Training was stable throughout:
- the approximate KL per update stayed at 0.005-0.015;
- entropy held at 14.8-15.6;
- teacher KL rose smoothly from 0.01 to 0.2;
- there were no extinctions.

The guard cost ran at 1-3% against the 1% budget. The stage-2 multiplier learning rate was too slow to bind (0.19
after 60 iterations); stage 3 raises it.

Gate (best, iteration 49):

| Worlds | vs stage-1 init | vs teacher | QoL | Below 30 per world |
| --- | ---: | ---: | ---: | ---: |
| Default DEVELOPMENT (20) | **1.037** | 1.063 | 0.821 | 0 |
| Herding-only (10) | 1.040 | 0.915 | 0.14 | 1.4 |
| Farming-only (10) | 1.092 | 1.036 | 0.81 | 0 |
| Fishing-only (10) | **0.938** | 1.143 | 0.73 | 0.8 |

**Gate: PASS** on default DEVELOPMENT worlds (welfare at least that of the stage-1 policy, guard within budget there,
no extinctions). Fishing-only worlds regressed 6% against the init while remaining 14% above the teacher. Stages 3-4
therefore rehearse the fixtures.

Herding-only worlds stay at QoL about 0.14 under every policy. This is a simulator limitation, not a learning
failure: without cultivation, clothing and tents need Animal Fabric, and the planner counts herd food toward meals
first, so fabric feedstock is missing (see the simulator's `Docs/ElderModel9Contract7.md`).

## Stage 3: mixed use

### First run without the strategic anchor (stopped at iteration 20)

Stage 3 first anchored only the service levels, so the strategic controls (development budget, reserve, farming /
herding / fishing mix) could move freely. Welfare fell against the stage-1 init:

| Iteration | 9 | 19 |
| --- | ---: | ---: |
| Welfare ratio | 0.955 | **0.876** |
| QoL | 0.801 | 0.801 |
| Settlements per world | 36.3 | 32.3 |

The optimisation itself stayed healthy (approximate KL 0.009-0.017, entropy 13.5-13.7). The deterministic policy
drifted, however, and the development budget rose about 29% above the stage-2 policy. That drift was rewarded on the
sampled training worlds but lost population on held-out ones. The run was stopped and the anchor restored. Every
stage now anchors the strategic controls.

### Stage 3b: anchored (40 iterations, about 290k decisions)

Started from the stage-2 best, with multiplier learning rate 2.

| Iteration | 9 | 19 | 29 | 39 |
| --- | ---: | ---: | ---: | ---: |
| Welfare ratio | 0.972 | 0.986 | 0.983 | **1.035** |
| QoL | 0.822 | 0.820 | 0.804 | 0.813 |
| Settlements below 30 per world | 0.17 | 0 | 0.17 | 0.33 |

- Approximate KL was 0.011-0.015 and entropy 13.2-13.9.
- Teacher KL settled at 0.31-0.36.
- Guard cost was 0.7-1.3% against the 1% budget, with the multiplier near 0.22-0.27, so it binds.
- There were no extinctions.

Gate (best, iteration 39; stage-2 best shown for reference):

| Worlds | Stage 2 vs init | Stage 3b vs init | Stage 3b vs teacher | QoL | Below 30 per world |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default DEVELOPMENT (20) | 1.037 | **1.057** | 1.083 | 0.810 | 0.15 |
| Farming-only (10) | 1.092 | **1.173** | 1.113 | 0.806 | 0.2 |
| Herding-only (10) | 1.040 | 0.986 | 0.868 | 0.118 | 0.8 |
| Fishing-only (10) | 0.938 | **0.860** | 1.048 | 0.727 | 1.6 |
| Held-out grassland (10) | 0.956 | 0.961 | 1.028 | 0.807 | 0.1 |

The teacher has 0.05 settlements below 30 per world on default worlds and 0.3 on held-out grassland.

**Gate: PASS** on default DEVELOPMENT worlds, with no extinctions in any category. Gains concentrate where
cultivation dominates (default and farming-only worlds). Fishing-only worlds regress further, though they remain
above the teacher. Held-out grassland stays about 4% below the init and herding-only worlds are flat. Rehearsing a
quarter of the fixtures did not stop the minority regimes from losing ground to the majority (see *Architecture
evidence*).

## Stage 4: learned movement (40 iterations, about 260k decisions)

Started from stage-3b best. Villages now choose their movement through the pointer over site rows; the teacher keeps
native movement. During training, the periodic evaluation kept movement native (code loaded before the evaluation
fix), so it measures the other heads only:

| Iteration | 9 | 19 | 29 | 39 |
| --- | ---: | ---: | ---: | ---: |
| Welfare ratio vs stage 3b (movement native) | 0.979 | 0.914 | 0.995 | 0.962 |
| QoL | 0.825 | 0.814 | 0.829 | 0.830 |
| Guard cost (training) | 2.2% | 1.5% | 1.4% | 1.7% |
| Multiplier | 0.40 | 0.53 | 0.59 | 0.66 |

- Optimisation stayed healthy: approximate KL was 0.009-0.013, entropy 13.5-13.9 and teacher KL 0.20-0.29.
- Training-world QoL was 0.65-0.68, against about 0.70 in stage 3b, because sampled movement explores.
- The guard cost stayed above its budget for the whole stage, with the multiplier still rising.

Gate, with policies choosing movement (`stage_eval.py --movement`). The init is stage-3b best with its cloned
movement head:

| Worlds | Init (3b, learned movement) vs teacher | Stage 4 best vs init | Stage 4 best vs teacher | Stage 4 latest vs teacher |
| --- | ---: | ---: | ---: | ---: |
| Default DEVELOPMENT (20) | 0.871 | 1.056 | **0.920** | 0.896 |
| Farming-only (10) | 0.860 | 1.063 | 0.914 | 0.926 |
| Herding-only (10) | 0.867 | 0.945 | 0.819 | 0.784 |
| Fishing-only (10) | 1.083 | 0.894 | 0.968 | 0.929 |
| Held-out grassland (10) | 0.948 | 0.993 | 0.941 | 0.959 |

For comparison, the same stage-3b policy with native movement reaches 1.083 of the teacher on default worlds. Learned
movement therefore costs about 20%; stage 4 recovers a quarter of that. On default worlds, population falls from
338k (teacher) to 304k (cloned movement) and 313k (stage 4 best), over 20 worlds. Stage-4 best also had one extinct
settlement on a herding-only world.

**Gate: FAIL.** Learned movement does not reach native movement, and the trend over 40 iterations (best at iteration
29, latest below it) does not suggest that more iterations of the same model would close a 13-15% gap. Movement stays
native for scale-out.

## Architecture evidence (proposal for discussion)

1. **Minority-regime erosion.** One shared policy improves where cultivation dominates and erodes the rest. Even
   with a quarter of each batch rehearsing fixtures, results against the stage-1 init were:
   - default worlds +5.7%;
   - farming-only worlds +17%;
   - fishing-only worlds -14%;
   - held-out grassland -4%.

   The policy gradient is dominated by cultivation decisions, because cultivating villages are the majority of
   decisions and of population-weighted welfare. Options, cheapest first:
   - a. **Regime-balanced advantage weighting:** normalise advantages per capability regime, or per economy-mix
     mode, within each batch.
   - b. **Capability-conditioned heads:** route the strategic and service heads through a small mixture of experts
     gated by the capability mask and site-row economics (fishing access, farm suitability). The trunk stays shared.
   - c. **Per-regime gates** in promotion: a candidate must not regress any regime by more than 3% against the init.
     This is cheap and should be adopted regardless.
2. **Movement.** The pointer over site rows sees each candidate site in isolation and receives credit for a move
   only through the 400-year lineage return. Native movement uses the simulator's full local search. Options:
   - a. Keep movement native, which is the recommendation for scale-out.
   - b. Make movement **residual to native**: the head chooses between the native choice and k native-ranked
     alternatives, rather than a free pointer.
   - c. Give it a spatial encoder over the local hex neighbourhood, with an auxiliary site-value target taken from the
     simulator's own forecast.
3. **Pastoral societies (simulator, not learner).** Herding-only worlds hold QoL at about 0.13 under the teacher and
   under every policy. Two static calibration sweeps were tried:
   - herd food x2 to x6;
   - tent labour per craftsperson x4 to x8.

   Neither lifts QoL, because the binding constraint is feedstock arbitration. The planner counts herd animals as
   meal food first, so clothing and tent requests fail on "animal feedstock", and tents also need Construction.
   Calibration was left unchanged. The proposal is to let the DemandControl planner reserve Animal Fabric feedstock
   ahead of meals, when herding is the only textile route, before any curriculum depends on pastoral success. The
   about 250-person nomadic target is not reachable until then.

## Scale-out plan

Assuming 2(a), 1(c) and either 1(a) or 1(b) are accepted:

- **Throughput.** Locally: about 60-75 village-decisions per second, including the update. A 40-iteration stage takes
  about 70-80 minutes, and the M7 hero runs used about 1.1 million decisions in about 6 hours of wall time.
  - Rollouts dominate; the update is under 10%. Collection already uses six bridge processes; the C# planner is
    the bottleneck, not the GPU.
  - Scale-out means more CPU rollout workers (about 12 decisions per second per core). A 64-core host is about 10x
    the local rate, so about 10 million decisions take about 4 hours.
- **Seeds and selection.** Run three or more training seeds per configuration. Select on a separate selection bank
  (DEVELOPMENT 2020-2059) rather than the 6 evaluation worlds, because best.pt here was picked on the same worlds it
  was scored on (a winner's curse). With 12 worlds the evaluation noise is about ±3%; gate on 40 or more worlds per
  category.
- **Guard.** Use multiplier learning rate 2 and a budget of 1%. In stage 4 the guard ran above budget, so watch it
  again if movement returns.
- **Anchors.** Keep the strategic and service teacher KL at 0.05, because the unanchored run shows the strategic
  controls drift without it.
- **Promotion.** CONFIRMATION seeds 3000-3059 stay untouched until a candidate passes per-regime gates on the
  selection bank.
