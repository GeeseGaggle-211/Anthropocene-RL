# Cultivation PPO pilot: normalization-v3 scope-safe campaign

## Decision

This campaign is **NO-GO for curriculum expansion** and **GO for a bounded
optimization/normalization-layer remediation experiment**. Two of three neural
seeds passed the declared gates. Seed 2101 narrowly missed the held-out return
tolerance and produced one post-update KL excursion above three times the
target. The native simulator remained frozen at
`5559961164d6f9da8e6d20c1dc563ecec3eb04b7`; no economics, ecology, reward,
observation, action, or physics behavior was changed.

The execution used learner commit
`714903a64d71a1bc6c912de2d8d8282b104b24f0`, descended from the recorded
starting SHA `f529b11b783620637f3f4eb9551159ee6eb0e9c3`. Its raw campaign evidence
identity is `d2c0fe55274b177d6cb6cb7333f29aabb7df411590bc0bdff8328f1a90c0f185`.

| Seed | Result | Selected update | DEVELOPMENT return vs Option A | HELDOUT return vs Option A | Maximum post-update KL |
|---:|:---:|---:|---:|---:|---:|
| 1701 | PASS | 96 | -4.27% | -3.03% | 0.1036 |
| 1901 | PASS | 72 | -7.96% | -9.74% | 0.1313 |
| 2101 | FAIL | 48 | -8.14% | -10.63% | 0.1558 |

All seeds passed contract/numerical, structural-scope, inactive-gradient,
checkpoint reload/continuation, critic-finite, DEVELOPMENT-competitiveness, and
non-pathological compiler-intervention gates. Seed 2101 failed only PPO KL
stability and HELDOUT competitiveness. HELDOUT has now been consumed and must
not be used for post-hoc tuning.

## Cultivation trajectory and economic behavior

Cultivation intensity did **not** rise steadily. Across seeds, mean requested
field operation moved only from 0.4817 to 0.5054, had a slightly negative
linear slope, and increased on 55.8% of update transitions. Mean requested
Development budget fell from 0.5168 to 0.2972 and increased on 50.5% of
transitions.

The seed trajectories diverged materially:

- seed 1701 field operation fell from 0.5495 to 0.1225;
- seed 1901 field operation rose from 0.4675 to 0.8628;
- seed 2101 field operation rose from 0.4281 to 0.5309;
- Development request endpoints fell in every seed.

No seed triggered the universal-self-sufficiency diagnostic. On the two
trade-dependent HELDOUT cases, field-operation contrast versus isolated cases
was -0.0033, +0.0181, and +0.0037 for seeds 1701, 1901, and 2101. These tiny
contrasts do not demonstrate useful state-dependent adaptation; the learned
actions were much more different between seeds than between economic slices.
The correct conclusion is therefore neither “the agent learned to intensify”
nor “the agent learned trade-aware cultivation.” It learned unstable,
seed-specific operating levels within a clean cultivation-only contract.

## Bank and normalization

The scope-safe bank contains 300 accepted snapshots: F01 30, F02 35, F03 35,
F13 40, F15 50, F16 40, F17 24, and F18 46. The partition is 210 TRAIN, 46
DEVELOPMENT, and 44 HELDOUT. Complete world/lineage/counterfactual groups remain
partition-isolated. The frozen manifest identity is
`1cfabbc6057c063d2bca8d27c4bda861a090d5811e2b02f3fe51adc9fd2a6b83`.

Generation rejected 496 candidates or groups: 196 requested-stratum misses,
134 quota overflow, 98 incomplete causal groups, 28 duplicate native
identities, 28 calibration scope escapes, and 12 other native-generation
rejections. All accepted calibration observations have zero movement,
husbandry, herd, and fishing activation; every reset mask is `1010`.

Normalization-v3 is not the present blocker. It was fitted from 210 TRAIN
resets and 1,050 legal TRAIN rollout observations only. Maximum feature
saturation was 0.198% on TRAIN, 0.725% on DEVELOPMENT, and 0.379% on HELDOUT;
the sparse causal-support gate passed. DEVELOPMENT and HELDOUT did not enter
the fit.

## Architecture diagnosis

The critic, unlike the input normalizer, is strongly saturated. At selected
checkpoints, the share of critic tanh activations with absolute value at least
0.95 was 60.5--70.3% in layer 1 and 89.8--92.0% in layer 2. Mean layer-2 tanh
derivatives were only 0.044--0.062. Actor layer-2 saturation was lower but still
11.1--13.3%. Critic MSE improved for seeds 1901 and 2101 but worsened from
26.82 to 39.93 for seed 1701.

This justifies a controlled LayerNorm ablation, beginning with the critic. It
does not demonstrate a feed-forward representational failure and does not
justify a GRU, Transformer, shared trunk, or privileged critic.

ReLU is not the first remediation. It does not control preactivation scale,
can create dead units on sparse signed features, and offers no material speed
benefit here: policy inference already sustained about 520--527 transitions/s
while complete training sustained only 15.1--15.8 transitions/s. If activation
choice remains suspect after normalized tanh is tested, SiLU and ReLU should be
small predeclared ablations rather than replacements made during diagnosis.

## Recommended rectification

1. Keep normalization-v3 frozen and keep the separate feed-forward actor and
   critic.
2. Increase each update from 8 to 32 episodes so every family contributes four
   independent trajectories; the present 24-transition update is too noisy for
   a minibatch size of 64.
3. Reduce actor learning rate from `3e-4` to `1e-4`, reduce actor epochs from 8
   to 4, and retain an explicit target-KL stop. Run this optimizer-only arm
   before interpreting an architectural effect.
4. Test critic-only `Linear -> LayerNorm -> tanh` at both hidden layers, with
   critic learning rate `3e-4`; compare against the optimizer-only arm on the
   same TRAIN/DEVELOPMENT states.
5. Test actor-and-critic LayerNorm only if critic-only normalization leaves the
   seed divergence. Defer ReLU/SiLU until these arms are resolved.
6. Select with DEVELOPMENT only. Create a new experiment generation with fresh
   held-out world/lineage groups for one confirmatory evaluation after the
   remediation choice is frozen.
7. Strengthen achieved causal evaluation coverage. The bank has 50 F15 states,
   but only 10 total states met the measured trade-dependent tag and none were
   in DEVELOPMENT. Do not infer trade adaptation from two HELDOUT examples.

No architecture-level user decision is required to run this bounded matrix.
The next decision point is after the optimizer-only and critic-LayerNorm arms
show whether seed dispersion, critic saturation, and F03/F17/F18 weakness move
together.

## Storage and throughput

The immutable snapshot dataset is at
`gs://anthropocene-rl-datasets-508912/datasets/elder-v2-cultivation-pilot-normalization-v3-scope-safe/2026-10-12-manifest-1cfabbc6/manifest.json`.
The final raw diagnostics and checkpoint bundle is published through
`gs://anthropocene-rl-datasets-508912/runs/cultivation-ppo-pilot-normalization-v3-scope-safe/714903a6-20261011-final/manifest.json`.
Its 44,062,720-byte object has SHA-256
`b236f4499c414bda6c40f15bb8442ed6a491d2108a523f48ba306e0496564cff`.
A restart deduplicated it, and an independent cold range-download reproduced
the byte count and SHA-256 exactly. After verification, 23 individually
identified local files totaling 696,594,610 bytes (about 664 MiB) were safely
evicted. The content cache remains capped at 5 GiB.

Actual three-seed training took 583.8 seconds on local CPU. Per-seed throughput
was 15.13, 15.80, and 15.65 transitions/s. The existing worker benchmark
measured 5.16/10.56/19.58/24.92 transitions/s at 1/2/4/8 workers, or
1.00x/2.05x/3.79x/4.83x scaling. Local CPU remains practical; collection and
bridge orchestration, not policy inference, is the bottleneck.

`summary.json` is the compact, hash-anchored Git evidence. Full traces,
normalization diagnostics, checkpoint bytes, and snapshot bytes remain outside
Git in immutable GCS objects.
