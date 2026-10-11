# Cultivation open-economy pilot v1

This directory freezes the first 300-snapshot, fixed-location cultivation
campaign before PPO training. The Google Drive RL design document remains the
normative architecture; these files record the concrete campaign instance.

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

- Drive folder: `Anthropocene / RL Snapshot Banks / Elder v2 / Stage0 and Pilot Banks`
- Folder ID: `1BtKChDAhEpJOWdeEiJyfnowuuN7y_GSN`
- Object naming: `sha256/<first2>/<full-sha>.snapshot`
- Local cache ceiling: 5 GiB, LRU

The post-freeze sync passed: 288 objects deduplicated and 12 newly uploaded.
All 300 manifest records carry Drive identity, byte count, MD5, and provider
SHA-256 metadata without changing the canonical manifest identity. Local blobs
remain an evictable verified cache; pruning refuses to remove the sole copy.
Drive advanced the 12 new provider-version strings after upload finalization;
the strict first live audit caught them, a metadata-only sync reconciled them,
and the final live audit verified 300/300 with zero invalid or unrecoverable
objects.

## Training declaration

`configs/training/cultivation-pilot-v1.yaml` freezes the three seeds, exact
balanced family sampler, separate 512→256→256 actor and critic, H=400 physical
discount horizon, actor LR 3e-4, critic LR 1e-3, target KL 0.05, and the
Option-A-through-shared-compiler baseline. DEVELOPMENT selects checkpoints;
HELDOUT is evaluated only after selection.

The generated PPO results, per-seed learning curves, checkpoint hashes,
baseline comparisons, throughput, and go/no-go assessment are added at the
next clean evidence boundary. Checkpoint bytes and snapshot bytes remain
gitignored.
