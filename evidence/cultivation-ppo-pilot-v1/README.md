# Cultivation PPO pilot v1

This directory preserves both the rejected eight-decision execution and the
scope-corrected campaign. The native simulator remains frozen at
`5559961164d6f9da8e6d20c1dc563ecec3eb04b7`; no native economics, ecology,
reward, observation, action, or physics code was changed.

## Promotion result

The corrected campaign is **NO-GO for the next curriculum stage**. It ran from
learner commit `2378d42095491a6ff9608b138f800c6f8880d4be` over the frozen
300-state bank with manifest identity
`9d33a9e5b8be889ac1fa015ee10ae790fb390dd9b36621967252d2aa34b0228d`.
All three seeds completed 96 updates and 2,304 in-scope transitions on local
CPU. Contract, replay, structural-scope, numerical, checkpoint-reload,
checkpoint-continuation, inactive-head-gradient, and survival gates passed for
all seeds.

Promotion nevertheless fails because the outcome is not robust across the
three neural seeds:

| Seed | Result | Selected update | DEVELOPMENT return vs Option A | HELDOUT return vs Option A | Main failure |
|---:|:---:|---:|---:|---:|---|
| 1701 | PASS | 72 | -5.77% | -2.62% | none under declared tolerance |
| 1901 | FAIL | 48 | -6.45% | -6.79% | maximum post-update KL 0.1643 |
| 2101 | FAIL | 72 | -13.96% | -15.10% | welfare regression and rising compiler intervention |

The Option-A mean discounted returns were 11.859 on DEVELOPMENT and 13.782 on
HELDOUT. The learned-policy means were respectively 11.096/13.489 for seed
1701, 11.049/12.971 for seed 1901, and 9.910/11.504 for seed 2101. HELDOUT has
now been consumed for this experiment generation and must not be used for
post-hoc tuning.

There is no demonstrated feed-forward representational blocker and no evidence
supporting a GRU, Transformer, shared trunk, or privileged critic. The observed
failures are attributable first to sparse-event normalization coverage,
optimizer/seed instability, and compiler-intervention sensitivity.

## Cultivation trajectory

Cultivation intensity did **not** increase steadily. Across the three corrected
runs, requested field operation increased overall from 0.4819 to 0.6817, but
only 60% of update-to-update changes were nonnegative. Requested Development
budget fell from 0.5169 to 0.2951, with only 48.4% nonnegative changes. This is
best read as a shift toward operating existing fields rather than a monotonic
increase in both operation and capital expansion. Seed 2101 developed a
seed-specific high-intensity signal (field-operation-at-or-above-0.9 rate
0.622) alongside worse welfare and intervention distance; there was no
universal self-sufficiency policy across all seeds.

## Normalization finding

The final normalization-v2 was fitted only from 210 TRAIN resets plus 630 short
legal TRAIN rollout observations. Overall maximum saturation was 5.0% on the
TRAIN fit, 8.33% on DEVELOPMENT, and 7.78% on HELDOUT. Sparse causal slices are
not adequately scaled: maximum feature saturation reached 100% for
trade-dependent states, 68.75% for immigration, and 71.43% for emigration.
Several corresponding TRAIN scales collapsed to near zero because nonzero
events were too rare in the unconditional quantiles.

The rectification is a new experiment generation with fresh HELDOUT groups,
more rare-event-bearing TRAIN calibration trajectories, and TRAIN-only robust
or conditional-nonzero scaling for sparse trade/migration features. The frozen
v1 normalizer must not be silently refitted, and DEVELOPMENT or HELDOUT
observations must not enter the fit.

## Bank and durable storage

The accepted bank contains 300 snapshots: F01 30, F02 35, F03 35, F13 40,
F15 50, F16 40, F17 24, and F18 46; partitions are 210 TRAIN, 45 DEVELOPMENT,
and 45 HELDOUT. Generation rejected 200 candidates: 95 stratum misses, 53
incomplete groups, 38 quota overflow, 12 duplicates, and 2 native-generation
failures. The deterministic native audit passed.

The immutable bank is stored at
`gs://anthropocene-rl-datasets-508912/datasets/elder-v2-cultivation-pilot-v1/2026-10-11-manifest-9d33a9e5/manifest.json`.
The 64,706,560-byte content-addressed shard has SHA-256
`d7fdec842a63e71ae5520f8a1a59a7954740761165af3372c99c4532b012032b`.
A cold readback reproduced that SHA, and a subsequent full 300-record audit
reported 300 cache hits, zero misses, and PASS. The loader now checks the local
5 GiB LRU cache first and transparently materializes a verified record from the
GCS shard on a miss.

The complete run bundle, including raw per-seed diagnostics and checkpoints,
is published through the immutable manifest at
`gs://anthropocene-rl-datasets-508912/runs/cultivation-ppo-pilot-v1/2378d420-20261011/manifest.json`.

## Throughput

Measured end-to-end throughput was 5.16, 10.56, 19.58, and 24.92 transitions/s
at 1, 2, 4, and 8 workers, corresponding to 1.00x, 2.05x, 3.79x, and 4.83x
speedup. Native step time rose from 43.4 ms at one worker to 74.5 ms at eight,
showing CPU contention. Actual training sustained 15.30--16.05 transitions/s;
rollout/bridge collection, not PyTorch inference or PPO updates, is the local
bottleneck. Local CPU remains practical for the rectification run.

## Final validation

The final worktree validation on 2026-10-11 produced:

- Ruff lint: PASS;
- pytest: 76/76 PASS, including immutable GCS publication, bounded cache,
  loader fallback, legacy manifest-identity compatibility, and run-bundle
  tests;
- frozen simulator manifest check: PASS at
  `5559961164d6f9da8e6d20c1dc563ecec3eb04b7`;
- synthetic PPO smoke: PASS with zero movement-head gradient;
- native simulator check and bridge schema smoke: PASS;
- 300-snapshot native bank re-audit: PASS with no warnings;
- independent cultivation-readiness integrity audit: 9/9 PASS;
- GCS warm-cache full-record verification: 300/300 PASS.

The key commands were `ruff check .`, `pytest`, `check-manifest`, `ppo-smoke`,
`check-simulator`, `adapter-smoke`, `audit-snapshot-bank`,
`audit_cultivation_readiness.py`, and `dataset verify`. Test temporary files were
kept outside the repository and removed after successful validation.

## Recommended clean boundary

1. Create a v2 bank generation with new world/lineage/counterfactual groups so
   HELDOUT is pristine again.
2. Increase legal TRAIN calibration exposure to delivered trade and both signs
   of migration; measure achieved event counts before fitting.
3. Fit a new TRAIN-only normalizer using robust sparse-event scaling and audit
   conditional slices before any PPO optimization.
4. Rerun the same feed-forward architecture over three seeds with a more
   conservative actor update (lower actor learning rate and/or fewer epochs),
   retaining target-KL early stopping and predeclared gates.
5. Add a predeclared penalty or selection gate for requested-to-accepted
   intervention distance; do not alter native economics to suppress it.
6. Keep the three-decision cultivation-only horizon. A longer horizon requires
   an explicitly husbandry-inclusive curriculum or a newly validated native
   cohort in which husbandry cannot emerge.

## Rejected eight-decision execution

`failed-eight-step/` records the first complete three-seed run from learner
commit `a774a0c1a07487764c50082f4cde53af73b5281e`. Its aggregate evidence identity
is `e57c9d99a249291db54ad35d01b5ae88e03512f4a68412f858d4dd8e8b9b943c`.
Although replay, likelihood, numerical, checkpoint, KL, and reward-denominator
checks passed, the run is not promotion evidence: natural hunting experience
unlocked native husbandry during the longer rollouts, so excluded herd heads
became active and acquired nonzero gradients. The matching checkpoint bytes are
preserved outside Git in `checkpoints/cultivation-ppo-pilot-v1-failed-eight-step/`;
the paths embedded in the archived reports are their paths at execution time.

The rejected run also answers the trajectory question for that execution:
cultivation intensity did not steadily increase. Across seeds, mean requested
field operation moved from 0.4908 to 0.4763 and mean requested Development
budget moved from 0.4906 to 0.3892.

## Structural-horizon boundary

`structural-horizon-audit.json` simulates every one of the 300 frozen snapshots
for eight decisions under one fixed legal cultivation directive. All 300 are
clean through decision 3. One F15 case exposes husbandry capability at decision
4; herd ownership and herd action masks first appear at decision 6. Fishing
capability, fishing grounds, and movement remain absent throughout the audit.

The corrected campaign therefore uses three-decision episodes and evaluations,
with a fail-fast assertion on every reset and every post-step observation. The
assertion covers husbandry capability, owned herds, both herd masks, fishing
capability, fishing-ground presence, and movement. It supplements rather than
relies on the fixed-action audit. Training increases from 32 to 96 updates, so
each seed receives 2,304 in-scope transitions instead of 2,048. H=400 and the
declared optimizer remain unchanged.

The scope boundary is an experiment-design correction, not an architecture or
simulator modification. Longer-horizon future curricula must explicitly admit
husbandry or construct a new, physically valid native scenario cohort where it
cannot emerge.
