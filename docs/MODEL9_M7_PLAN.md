# model9 M7: PPO stages 2-4 (preparation)

Status: scaffolding implemented and smoke-tested (`autonomous_v7/ppo.py`, `tests/test_autonomous_v7_ppo.py`); no PPO
stage has been run. Contract 8 (simulator `feature/elder-model9-native` df19c33, pin
`configs/simulator/elder-model9-native8.json`) adds the three-way economy mix; stage 1 is retrained on contract-8 data
before M7 starts (see `MODEL9_STAGE1.md`).

## What changed before M7

- **Three-way economy head.** The actor's development split is one masked softmax over (cultivation, herding,
  fishing). The compiler turns the fishing share into whole Fishers beyond the survival frontier; fields and herds
  keep the existing ray. Cloning uses soft cross-entropy against the teacher's mix; PPO uses an additive
  logistic-normal density on the available coordinates. The heuristic never develops fishing, so cloning starts at a
  fishing share of 0 and PPO has to find fishing development itself.
- **Fishing camps can move.** Site rows now exist for fishing-only camps and carry fishing access, so none of the
  heuristic's moves is unrepresentable (seeds 17 and 53).
- **Immigration cost.** Two exact changes cut the physical immigration pass 19-37% with identical fingerprints
  (backported to main 40ea0a2). Teacher throughput rose from 106-110 to 137 native decisions per second per process,
  about 850 per second on seven processes.

## Stages

| Stage | Policy chooses | Native | Teacher KL anchor |
| --- | --- | --- | --- |
| 2 capability | strategic levels, service levels, economy mix | routes, movement, fission | strategic, service |
| 3 mixed | + clothing and shelter routes | movement, fission | strategic, service (none on mix or routes) |
| 4 movement | + movement over value-ordered sites | fission | strategic, service |

Common machinery (`ppo.py`):
- every village of a world uses the shared policy;
- rewards are lineage rewards (D4), with GAE(lambda = 0.95) on exp(-years / 400);
- the critic keeps u = G / P two-hot and its critic-only privileged block, which the rollouts now carry;
- the guard on settlements below 30 residents (daughters in their first 3 turns exempt) is a constraint, with its own
  cost critic and a Lagrange multiplier against a 1% budget, so the reward never pays for being small or large;
- logit-space scales are floored at -2 for sampling, recomputation and the anchor, because cloning drives them to
  the clamp.

## Decision needed: where single-capability training comes from

The TRAIN bank has no single-capability worlds: all 600 are cultivation-dominated. Decisions by available channel:

| Channels | Share of decisions |
| --- | ---: |
| cultivation only | 66.7% |
| cultivation + herding | 14.0% |
| cultivation + fishing | 13.7% |
| all three | 3.1% |
| fishing only | 2.0% |
| herding only (with or without fishing) | 0.6% |

Options:
1. **Capability-restricted fixtures (recommended).** A simulator world option disables one or two of cultivation,
   husbandry and fishing for every society, generated from the same seed banks. It gives real single-capability
   worlds for stage 2 and a fishing-only and herding-only evaluation slice. It changes no shipped behaviour (a
   training-only option, off by default), but it is a new world configuration.
2. **Village-level masking.** Keep the worlds and restrict each village's economy mix to one channel for the
   stage. It is cheap, but the worlds stay cultivation-dominated, so herding and fishing get little signal.
3. **Skip stage 2's split.** Train mixed directly and report per-capability slices. It is the fastest, but it gives
   up the curriculum the plan recorded.

## Budget and gates

- **Throughput.** Rollouts run near the teacher rate (about 850 decisions per second on seven processes), roughly
  3 million decisions per hour before update cost. At 2-5 million decisions per stage, each stage is 1-2 hours.
- **Gate per stage.** On DEVELOPMENT seeds 2000-2059:
  - closed-loop welfare at least that of the stage-1 policy, which is itself at least 95% of the teacher;
  - guarded below-30 share within the 1% budget;
  - no extinctions, and compiler fallback at most 1%;
  - the mixed-use fixture suite passes.
- **Promotion.** CONFIRMATION seeds 3000-3059 stay untouched until promotion.

## Trial on the contract-8 stage-1 checkpoint (seed 0; 6 TRAIN worlds, 8 turns)

- **Step size.** At learning rate 1e-4 one update moved the approximate KL to about 0.7. At 1e-5 it is 0.004-0.015
  over all four epochs, under the 0.03 target, and teacher KL is 0.01-0.02. The default is now 1e-5.
- **Sampling is not harmful.** Sampled actions produced at least the welfare of the deterministic mode and the
  teacher on the same worlds and horizon: 1.017 million welfare-years against 0.923 (deterministic) and 0.977
  (teacher). The result is the same at logit-scale floors from -9 to -2. Sampling already proposes a small fishing
  share (mean 0.039), which the deterministic mode rounds to zero.
- **Early-turn QoL.** The low mean QoL in short trials (about 0.5) belongs to the early turns, not to the policy.
- **Uncalibrated values.** The cost budget (1%) and the teacher-KL weight (0.05) are initial values.
