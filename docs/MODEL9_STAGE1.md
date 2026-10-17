# model9 stage 1: behaviour cloning and critic pretraining (M6)

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
