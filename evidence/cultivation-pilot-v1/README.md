# Cultivation open-economy pilot v1

This directory freezes the first 300-snapshot, fixed-location cultivation
campaign and its TRAIN-only normalization. The Google Drive RL design document
remains the normative agent architecture, while GCS is the durable storage
backend for machine-consumed datasets and runs.

## Frozen identities

- Learner starting commit: `9a9e174f829dc0342ac92e47c8b5317a727e592e`
- Native simulator: `GeeseGaggle-211/Anthropocene@5559961164d6f9da8e6d20c1dc563ecec3eb04b7`
- Generator configuration identity: `c2a6b259cb69566e96a8cd4797be40182074606dd21508424a5766246263d0d4`
- Native result identity: `424bdea8295200cddba984d4ec3e0b2ced099e83b60209aac6bbfd34e3e262b2`
- Frozen manifest identity: `9d33a9e5b8be889ac1fa015ee10ae790fb390dd9b36621967252d2aa34b0228d`
- Final TRAIN-only normalizer file: `b4393cbf0cd20e80d1ace345f48f768bd4f8393c9fe2c09d0535fd6aee42228b`

The accepted split is 210 TRAIN / 45 DEVELOPMENT / 45 HELDOUT. Every one of
F01, F02, F03, F13, F15, F16, F17, and F18 is present in every partition.
World, lineage, and counterfactual groups were assigned before timestamp
selection and never cross partitions.

Family totals are F01=30, F02=35, F03=35, F13=40, F15=50, F16=40, F17=24,
and F18=46. The generator rejected or did not promote 200 candidate records:
95 missed requested measured strata, 53 belonged to incomplete causal groups,
12 duplicated native identities, 2 failed native generation, and 38 were valid
surplus groups beyond the frozen coverage quotas.

## Native and normalization audits

`audit.json` and `coverage.json` report PASS. Every accepted blob passed
SaveCodec round-trip, Reset, legal Step, finite positive physical time, native
replay, frozen schema/configuration identity, and measured-strata validation.
Movement, herd controls, owned herds, and fishing are absent.

The final normalization-v2 was fit only from 210 TRAIN resets plus three short
legal rollout observations per TRAIN snapshot: 840 fit observations total.
DEVELOPMENT (180 observations) and HELDOUT (180 observations) are transform-only
diagnostics and did not contribute to fitted scales. See
`final-normalization-diagnostics.json` for feature and causal-slice statistics.

## Durable storage

The bank is published at
`gs://anthropocene-rl-datasets-508912/datasets/elder-v2-cultivation-pilot-v1/2026-10-11-manifest-9d33a9e5/manifest.json`.
Its content-addressed 64,706,560-byte shard passed provider checksum and local
SHA-256 validation. A cold readback followed by a full warm audit verified all
300 records with zero misses. The local cache ceiling is 5 GiB with LRU
eviction. The former Drive copy is retained as historical redundancy, not the
primary machine-training path.

## Training declaration

`configs/training/cultivation-pilot-v1.yaml` freezes the three seeds, exact
balanced family sampler, separate 512→256→256 actor and critic, H=400 physical
discount horizon, actor LR 3e-4, critic LR 1e-3, target KL 0.05, and the
Option-A-through-shared-compiler baseline. DEVELOPMENT selects checkpoints;
HELDOUT is evaluated only after selection.

The campaign is complete and the aggregate result is NO-GO. Per-seed learning
curves, checkpoint hashes, baseline comparisons, throughput, and rectification
assessment are in `evidence/cultivation-ppo-pilot-v1/`. Checkpoint and snapshot
bytes remain gitignored and are durable in GCS.
