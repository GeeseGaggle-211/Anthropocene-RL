# model9 stage 1: behaviour cloning and critic pretraining (M6)

## Pilot: animal textile need and herd transition limit (18 October 2026)

The heuristic gained two simulator rules (simulator 4ba0de1 and 3a5db2e, pin
`configs/simulator/elder-model9-native8b.json`):
- portable-shelter herding camps pay tent fabric ahead of the reserve;
- herd growth for that need is granted before Recovery;
- herding daughters keep an animal;
- a non-distress cull is capped at half the herd per turn.

The contract is unchanged (8).

The data was regenerated on the same banks: TRAIN 374,252 and DEVELOPMENT 37,344 decisions, in
`gs://anthropocene-rl-datasets-508912/model9/teacher-v5/`. Teacher welfare per TRAIN world rose from 3.50 to 3.80
million. Checkpoints are in `.../checkpoints/contract8-pilot/`.

| Seed | Welfare ratio (total) | Median | Min | Critic EV | MSE / var | Bias / sigma |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.959 | 1.004 | 0.656 | 0.718 | 0.282 | -1.85% |
| 1 | 1.010 | 1.022 | 0.764 | 0.707 | 0.293 | -2.18% |
| 2 | 1.000 | 0.988 | 0.658 | 0.717 | 0.284 | -1.93% |

Seeds 0 and 2 pass. Seed 1 again misses the bias criterion, by 0.18 points. M7 starts from seed 2.

## Contract 8 on the fixed heuristic (herd fixes, 17 October 2026)

The heuristic now has three herd fixes (simulator e7c4b61, main d4f949a):
- residual-water routing for managed herds;
- the seed-herd floor;
- a growth ceiling no lower than the retained herd.

The model9 data was regenerated: TRAIN 350,097 and DEVELOPMENT 34,984 decisions, in
`gs://anthropocene-rl-datasets-508912/model9/teacher-v4/`. Checkpoints are in `.../checkpoints/contract8-herdfix/`.

| Seed | Welfare ratio (total) | Median | Min | Critic EV | MSE / var | Bias / sigma |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.025 | 1.050 | 0.797 | 0.714 | 0.287 | -1.92% |
| 1 | 1.029 | 0.978 | 0.774 | 0.706 | 0.295 | -2.07% |
| 2 | 0.985 | 0.980 | 0.713 | 0.716 | 0.284 | -1.81% |

Welfare passes on all three seeds, and the critic explained variance rose from 0.66 to above 0.70. The bias is
consistently about -2% of sigma, a slight systematic undervaluation; seed 1 misses the 2% criterion by 0.07 points.
A warm critic-only refinement made it worse (-2.28%), so the original checkpoint stands. M7 starts from seed 0.

## Contract 8 retrain (17 October 2026, before the herd fixes; superseded)

Contract 8 (simulator df19c33, pin `configs/simulator/elder-model9-native8.json`) adds the three-way economy mix and
fishing site rows. The data was regenerated on the same banks: TRAIN 339,819 and DEVELOPMENT 34,111
village-decisions, the same counts as contract 7, because the heuristic is unchanged. Shards are in
`gs://anthropocene-rl-datasets-508912/model9/teacher-v3/` and checkpoints in `.../checkpoints/contract8/`. Teacher
throughput is now 137 native decisions per second per process; the exact immigration change gave the 27% gain.

| Seed | Welfare ratio (total) | Median | Min | Critic EV | MSE / var | Bias / sigma |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.984 | 0.962 | 0.738 | 0.662 | 0.338 | -1.87% |
| 1 | 0.964 | 0.942 | 0.750 | 0.662 | 0.338 | -1.95% |
| 2 | 0.964 | 0.945 | 0.788 | 0.662 | 0.338 | -1.93% |

**Gate: PASS** on all three seeds (welfare at least 95% of the teacher; EV at least 0.60, MSE / variance at most
0.40, |bias| at most 2% of sigma).

Panel per world, teacher -> policy (seeds 0/1/2):
- population 16,898 -> 14,338 / 14,422 / 13,718;
- settlements 34.1 -> 35.6 / 37.6 / 37.5;
- villages of 600+ 13.3 -> 8.6 / 7.4 / 6.5;
- QoL 0.807 -> 0.863 / 0.861 / 0.875;
- settlements below 30 residents 0.05 -> 0.20 / 0.45 / 0.40;
- no extinctions.

Welfare ratios are 2-10 points below the contract-7 runs, with the same pattern: fewer large villages and higher
QoL. The economy-mix head clones the teacher's split closely (dev cross-entropy 0.06-0.08). Because the teacher never
develops fishing, the cloned mix's fishing share starts near zero.

## Contract 7 (superseded)

Native contract 7 (`configs/simulator/elder-model9-native7.json`, simulator `feature/elder-model9-native`
b0a72a6), learner `anthropocene_rl.autonomous_v7`. Stage 1 of the learned-Elder curriculum: clone the repaired
heuristic (DemandControl settlement planner, kernel-scored Elder split, surplus-funded births) and pretrain the
factored critic. No actor update depends on the critic.

## Data (M5)

| Bank | Seeds | Worlds | Village-decisions |
| --- | --- | ---: | ---: |
| TRAIN | 1000-1599 | 600 | 339,819 |
| DEVELOPMENT | 2000-2059 | 60 | 34,111 |
| CONFIRMATION | 3000-3059 | untouched | - |

Small/8 worlds, shipped configuration, the heuristic Elder for every village, 25 turns (500 years) after the
opening. Measured throughput: 106-110 native village-decisions per second per process (about 98 per second wall),
about 690 per second on seven processes. The plan's estimate of 800-1,000 per worker was about 10x optimistic;
5-10 million decisions would take 2-4 hours on seven processes. Shards: `gs://anthropocene-rl-datasets-508912/model9/`
(`teacher-v1/` without critic privileged inputs, `teacher-v2/` with them).

## Model

Shared-weight row encoders (64 units) with masked sum/mean/max pooling per entity set, a core encoder over
features plus presence flags, a 256-unit LayerNorm trunk. Heads: logistic-normal strategic controls and goods
service levels, masked categorical clothing/shelter routes and fission policy, a pointer over site rows for
movement. The critic has its own encoder and additionally receives the training-only privileged block (growth
headroom, lineage, episode age); a test asserts those inputs never change actor outputs. The value equation is
unchanged: u = G / P at the decision, G the TD(lambda = 0.95) return of population-weighted lineage welfare
(D4, 3-turn window) discounted by exp(-years / 400), predicted as a two-hot over 255 symlog bins.

## Results (three training seeds, DEVELOPMENT seeds 2000-2019 for closed loop)

Closed loop: every village runs the cloned policy (deterministic mode); movement and fission stay native
(stages 1-3). Teacher and policy start from the same opening; welfare is realized P x QoL x years summed over
all settlements.

| Seed | Welfare ratio (total) | Median | Min | Worlds < 0.95 | Critic EV | MSE / var | Bias (u) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.019 | 1.021 | 0.768 | 7 / 20 | 0.666 | 0.334 | -22.9 |
| 1 | 1.066 | 1.052 | 0.814 | 3 / 20 | 0.664 | 0.336 | -24.3 |
| 2 | 0.976 | 0.940 | 0.789 | 11 / 20 | 0.655 | 0.346 | -23.8 |

Panel (mean per world, teacher -> policy, seeds 0/1/2): population 16,898 -> 15,512 / 15,687 / 14,693;
settlements 34.0 -> 38.1 / 37.8 / 39.9; villages of 600+ 13.3 -> 10.0 / 10.3 / 7.9; QoL 0.807 -> 0.856 /
0.855 / 0.862; nutrition 0.996 -> 0.998 / 0.997 / 0.996; clothing 0.990 -> 0.982 / 0.984 / 0.983; hungry
camp-turns 0.98% -> 0.99% / 1.05% / 1.44%; extinctions 0 -> 0; settlements below 30 residents 0.05 -> 0.45 /
0.25 / 0.40; compiler repairs per decision 2.6 / 2.3 / 3.0 (teacher-as-action: about 0.85).

**Stage-1 gate: PASS** on all three seeds: closed-loop welfare at least 95% of the teacher, and on world-disjoint
validation critic explained variance at least 0.60, MSE / variance at most 0.40 and |bias| at most 2% of the target
standard deviation (criterion restated from model8's scaled units, agreed 17 October 2026). Bias is -1.84% / -1.97% /
-1.90% of the target standard deviation (-23 to -24 welfare-years per resident). Without the privileged block the critic stalled at
EV 0.59-0.61 (MSE / var 0.40-0.41).

Not yet promotion evidence. The cloned policy runs more, smaller settlements at higher QoL than the teacher, with
more settlements below 30 residents (a promotion guard) and wide per-world spread. These belong to the PPO stages
(M7) and the mixed-use fixture suite, not to stage 1.

## Reproduction

```
dotnet build -c Release integration/Anthropocene.Model9Bridge.csproj -p:AnthropoceneSimRoot=<simulator> -o integration/bin/model9
# teacher shards (simulator checkout)
dotnet Tools/Headless/bin/Release/net8.0/Anthropocene.Headless.dll 0 --model9-teacher=<dir>,1000,1099,25
python -m anthropocene_rl.autonomous_v7.train --train <train> --dev <dev> --bc-epochs 12 --critic-rounds 4 --critic-epochs 6 --seed 0 --out <pt>
python -m anthropocene_rl.autonomous_v7.evaluate --checkpoint <pt> --seeds 2000-2019 --turns 25 --out <json>
```

Evidence: `evidence/model9-stage1/` (evaluation reports and training logs); checkpoints in
`gs://anthropocene-rl-datasets-508912/model9/checkpoints/`.
