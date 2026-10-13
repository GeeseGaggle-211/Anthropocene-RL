# Autonomous Elder PPO gate v1

The bounded CPU pilot passed its preregistered numerical gate. Independent review
gave GO for the capped CPU parent-Elder canary, subject to enforced limits and a
worker/replay/checkpoint preflight before concurrency increases. No canary or
expensive training has been launched. The new parent-Elder
model plugs into the native shared WEGO boundary through `autonomous_v4`, without
changing the legacy five-head policy or its datasets/checkpoints.

## Implementation

Native action 3 / observation 4 / environment 4 / replay 4 add knowledge-masked
movement and diagnostic event accounting to the autonomous economic contract.
The physical opening-stock escrow settles before simultaneous movement; destination
production and field abandonment retain native resolution order. Actor observations
contain no diagnostic population/event telemetry or remote private inventory.

The model follows the existing independent actor/critic 256×256 tanh architecture,
with orthogonal initialization. Heads cover masked workforce/recipe Dirichlet shares,
capital/livestock Beta controls, legal movement, and conditional stock quotes. Trade
selects enable, known partner, offered good, different requested good, stock fraction
and price bucket. PPO scores all requested continuous shares and categorical choices;
it never scores rounded staffing or substitutes heuristic actions. Inactive branches
have zero density contribution and consume no sampler RNG. The sampled downstream
trade entropies are excluded from regularization to avoid a biased conditional-entropy
gradient. The native quote API still accepts arbitrary bounded continuous quantities.

PPO uses separate actor/critic Adam groups and gradient clipping, clipped likelihood
ratios, an actor KL stop, physical-time GAE, truncation bootstrap, and fixed feature
compression/reward units. `AutonomousElderPolicy` loads a provenance-checked checkpoint
and returns native actions from isolated observations; batched calls sort camp IDs.

## Frozen basic curriculum and results

Three new training world seeds × four families; four disjoint development world
seeds × two fixed policy sampling seeds × four families per initialization. All
families enable native trade, movement, immigration, and fission. These are generated
48×24 native worlds with controlled-valid starting economies, not the synthetic
single-step production fixture. This initial bank is cultivation-based; husbandry
and fishing breadth remains a later promotion gate.

The four equally weighted families are survival/resource allocation, food-import
trade, migration pressure, and fission/exit opportunity. Focal Elders train; the
reciprocal market peer uses only its own observation and a public food-for-tools
rule. Other settlements and fission daughters retain the native controller.

Each seed trained 2,048 focal transitions (32 PPO updates, eight-decision episodes).
Total training was 6,144 transitions, CPU only, in 284.64 seconds including paired
evaluations, replay and initial checkpoint checks. No event rewards, privileged
critic, teacher substitution, curriculum filtering, best-checkpoint selection,
or development-trajectory optimization was used. Evaluation selected final
checkpoints only. Welfare is accumulated native population×QoL×years per opening
focal resident, equally averaged across families.

| Initialization | Untrained welfare | Final welfare | Change | Final survival |
|---|---:|---:|---:|---:|
| 17 | 28.9916 | 90.4358 | +211.9% | 32/32 |
| 29 | 28.9887 | 119.0886 | +310.8% | 32/32 |
| 43 | 29.0102 | 89.2022 | +207.5% | 32/32 |
| Equal-seed mean | 28.9968 | 99.5756 | +243.4% | 96/96 |

All four family welfare means improved for every seed. Actor and critic changed;
every action head received nonzero gradients. No illegal/nonfinite rollout occurred.
Sampled minibatch KL maxima were .0605/.0802/.0774; the .03 actor stop prevented
further actor updates after crossing the threshold (a preceding minibatch update
can overshoot it). Critics continued updating.

Trade is still sparse: training seeds 17/29 each settled one learned quote; seed 43
settled none. No final development learned quote settled. Quote proposals and
probability gradients are exercised, but this is **not evidence of learned trade
competence**. Separate scripted curriculum witnesses produced four focal moves,
one focal settled reciprocal trade, five focal fissions and 308 focal migrants,
all through native rules and with replay verification. Background trades are
reported separately and never counted as learned settlement success.

Survival alone is weak evidence: parent population retention varies from .127 to
2.702 across final seed/family cells, and mean food coverage from .390 to .862.
Fission/emigration reduce parent population as well as mortality. The report keeps
food coverage and population retention visible; it does not claim a robust or
sustainable village economy. These limitations motivate a monitored canary.

## Validation and performance

- Native full suite: 69 PASS groups, including 471 autonomous assertions and 143
  legacy Elder assertions; 73.2 seconds, peak 1,782 MiB. Production, input/tool
  shortages, conservation, dependent recipes, masks, movement with opening trade,
  simultaneous actions and replay/partition checks pass.
- Learner full suite: 142 passed, two skipped (old v3 native opt-in and an existing
  optional test); after adding the inference plug-in test, all eight focused v4
  tests pass with the native bridge enabled. Ruff passes.
- Exact heuristic and old-five-head fingerprint parity with validated base
  `5559961164d6f9da8e6d20c1dc563ecec3eb04b7`: initial boundary, four heuristic
  steps, three old-Elder steps. Historical v3 artifacts are preserved.
- All development episodes replay-verified. All three final checkpoints separately
  reproduced the next eight native episodes and PPO update exactly with populated
  Adam state: fingerprints, likelihoods, model, optimizer, sampler/shuffle RNG,
  update metrics. Duplicate validation did not modify retained checkpoints.
- Current documentation contract checker passes: 88 config values, 20 claims.

Complete native world-step benchmark (two warmups, seven samples, fresh snapshots,
TieredCompilation=0) includes planning, economic execution, endpoint observations
and fingerprints; autonomous timing also includes replay bookkeeping. It excludes
Python policy inference/IPC and opening reset observation. Dense synthetic benchmark
worlds are useful comparisons, not a forecast for larger training worlds.

| Camps | Heuristic median ms | Autonomous v4 median ms |
|---:|---:|---:|
| 1 | 1.5823 | 1.1232 |
| 2 | 2.0273 | 1.6461 |
| 8 | 8.2559 | 7.0750 |
| 32 | 30.6881 | 25.1633 |
| 80 | 81.7751 | 62.9588 |

At 80 camps autonomous execution used 47.68 MB allocated versus 54.99 MB and was
23.0% faster at the median. Raw benchmark ranges and allocations are retained.
Unique training throughput for this entire pilot was 21.6 transitions/second;
this includes evaluation/replay/checkpoint overhead and is not collection-only FPS.

## Independently approved next gate

The independent reviewer approved a **parent-Elder CPU canary** within this envelope: retain all three
pilot final checkpoints; at most 32,768 additional focal transitions per seed
(98,304 total), four native CPU workers, one Torch thread, four hours maximum.
Evaluate every 4,096 transitions on the frozen paired development suite and retain
all intermediate checkpoints separately. Abort on any illegal/nonfinite/replay
failure, survival below 75%, welfare below 80% of its corresponding pilot final,
or unexplained memory/throughput deterioration exceeding 2×. Validate worker
partition/replay and checkpoint equivalence before raising concurrency.

Keep all four dynamics in every curriculum stage. Track focal quote proposals,
settlements, food, population, migration/fission and welfare per family. Improve
trade exposure in a separately versioned training-only market curriculum while
preserving the fixed development suite; require repeated learned settlement on
unseen worlds before calling trade successful. Add husbandry/fishing/shortage
and longer-horizon stress banks before broader promotion. Daughter control and
lineage reward accounting require a separate design gate before dynastic self-play.
Only after the canary passes should a new review approve a larger/GPU PPO campaign.
No additional GPU ecology work is involved.

## Reproduction and artifacts

See `configs/training/autonomous-ppo-gate-v1.json`,
`configs/curriculum/autonomous-ppo-gate-v1.json`, and the shared
`AUTONOMOUS_ELDER_PPO_GATE_V1.md`. New schema/manifest files are named v4 (action v3).
Compact hash-anchored evidence is in `evidence/autonomous-ppo-gate-v1/`; raw bank
snapshots remain in `datasets/autonomous-ppo-gate-v1/` and model/optimizer checkpoints
and episode/update traces in `runs/autonomous-ppo-gate-v1/`. Existing artifacts
were neither replaced nor removed. The inventory records every raw run file hash.

```bash
DOTNET_CLI_HOME=/tmp/dotnet-home dotnet build integration/Anthropocene.RLBridge.csproj -c Release -p:AnthropoceneSimRoot=/workspace/Anthropocene-elder
DOTNET_CLI_HOME=/tmp/dotnet-home dotnet run --project integration/AutonomousCurriculum.csproj -c Release -- configs/curriculum/autonomous-ppo-gate-v1.json /tmp/new-autonomous-bank
python scripts/run_autonomous_ppo_gate.py --bank /tmp/new-autonomous-bank --output /tmp/new-autonomous-run --bridge integration/bin/Release/net8.0/Anthropocene.RLBridge.dll --sim-root /workspace/Anthropocene-elder
python scripts/check_autonomous_resume.py --bank datasets/autonomous-ppo-gate-v1 --run runs/autonomous-ppo-gate-v1 --bridge integration/bin/Release/net8.0/Anthropocene.RLBridge.dll --output /tmp/resume-check.json
```

The pilot runner enforces its small budget and refuses an existing output directory.
It cannot launch the proposed canary. A separate orchestration change is needed for
parallel canary execution; the model/environment plug-in is ready for that integration.

Independent review: `evidence/autonomous-ppo-gate-v1/independent-review.md` in the learner repository. The reviewer independently reran native-enabled tests, all three populated-Adam continuation checks and event witnesses, verified raw artifact hashes, and recomputed pilot improvements. The review does not certify the future parallel collector; it must pass the stated preflight.
