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
