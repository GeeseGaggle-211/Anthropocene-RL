# Cultivation pilot normalization-v3 remediation

Status: **BANK AND NORMALIZER FROZEN; PPO NOT YET RUN**

## Frozen identities

- Starting learner commit: `f529b11b783620637f3f4eb9551159ee6eb0e9c3`
- Native simulator: `GeeseGaggle-211/Anthropocene@5559961164d6f9da8e6d20c1dc563ecec3eb04b7`
- Generator configuration SHA-256: `05ae6432b175289f8463cb8a8199a393f8e2279bc3be4d78facc1c150c44a56b`
- Frozen snapshot manifest SHA-256: `e8e544311040423f56e0cb1d0fe300112bd60c1d21f5c0516f1c6e16f8939f52`
- Native result SHA-256: `e695307d42eb85af72e1ac7f03c2b5eab10045c90808a833c884c902b366f756`
- Snapshot-generator DLL SHA-256: `46d5ccbfdd3b9e5e01c7d6aef19bffb89321b6cfb77d8e9bdfff7da5f604a172`
- Final normalizer file SHA-256: `f08158acd0e46268d8b6b330b6282cfca0f7fa19edba2d908981673dc6fa1e01`

## Bank result

The generator evaluated 698 deterministic candidates twice and froze 300
accepted snapshots: 210 TRAIN, 46 DEVELOPMENT, and 44 HELDOUT. Complete
world/lineage/counterfactual groups never cross partitions. Family counts are
F01 30, F02 35, F03 35, F13 40, F15 50, F16 40, F17 24, and F18 46.

Native audit passed with movement, herding, and fishing structurally absent.
The final structural action-mask signature is `1010` for all 300 snapshots.
Rejected evidence was retained rather than patched: 196 requested strata not
achieved, 10 native-generation rejections, 94 incomplete causal-group records,
32 duplicate native-identity records, and 66 valid records beyond the frozen
coverage quota.

## Normalization result

Normalization-v3 fit 1,260 TRAIN observations only: reset plus five legal
calibration steps for each of 210 TRAIN snapshots. Scenario families receive
equal total fit weight. All nine sparse causal support groups pass. Examples:

- commodity import/export nonzero counts: 46 / 75;
- core signed migration: 206 nonzero, 58 positive and 148 negative;
- contact trade flow: 139 nonzero;
- memory trade utility: 480 nonzero.

At the absolute normalized-output threshold of 0.95, maximum per-feature
saturation is 0.1931% in TRAIN, 0.7246% in DEVELOPMENT, and 0% in HELDOUT. TRAIN
trade-dependent, immigration, emigration, pre-fission, and post-fission slices
all have 0% maximum saturation.

The normalizer remains tanh-based. Dense tails use a family-weighted fit plus
an unweighted 99.5th-percentile envelope; sparse event magnitudes use the 99th
percentile of nonzero TRAIN values; finite ratio softening preserves physical
zero without confusing it with missingness.

## Clean-boundary decisions

The first fresh plan was rejected before promotion because its partition salt
could not satisfy complete-group quotas. The next plan was rejected because
native F15 yield could not preserve five pairs per causal contrast. F15 was
oversampled, not weakened. F16 was likewise oversampled after identity
deduplication left only 36 complete states. The accepted bank, native physics,
reward, observation/action schemas, and policy architecture were never relaxed.

See `generation/audit.json`, `generation/coverage.md`,
`generation/final-normalization-diagnostics.json`, and
`generation/generation-summary.json` for machine-readable evidence.
