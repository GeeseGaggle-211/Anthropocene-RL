# Cultivation pilot normalization-v3 scope-safe generation

Status: **BANK, NATIVE AUDIT, AND TRAIN-ONLY NORMALIZER PASS; PPO PENDING**

This is a new experiment generation. It replaces—but does not mutate—the
rejected `3.0.0-remediation` bank whose legal rollouts could activate husbandry.
The Python selector now rejects an entire causal group when any reset or legal
calibration observation activates husbandry, a domestic herd, herd action
masks, fishing, or movement. The ingest boundary independently enforces the
same invariant before normalization can be fitted.

## Frozen identities

- Native simulator: `GeeseGaggle-211/Anthropocene@5559961164d6f9da8e6d20c1dc563ecec3eb04b7`
- Generator configuration SHA-256: `f7eeaa5ce09cb2e2990fe5bb9a314cf480dbf37d54d29186e119816cc24a3ee2`
- Frozen manifest canonical SHA-256: `1cfabbc6057c063d2bca8d27c4bda861a090d5811e2b02f3fe51adc9fd2a6b83`
- Frozen manifest file SHA-256: `b4751fd965a4f73a43f6492e81f2bfa62e5ac71f24977a453554e2609b83168d`
- Native result SHA-256: `78aa2281067626feb11000846ee3d66e350457aac434f095e24b32387b6ba685`
- Snapshot-generator DLL SHA-256: `6823275d2634aaad8ea0bcdd0d9d1344bc2d56c0d066e1778c5337a912a6a9e9`
- RLBridge DLL SHA-256: `4d4136bda29a35ef4d3147cf63cf5e48b70b7ed814d189064152437d858cacc0`
- Final normalizer file SHA-256: `6563cea5d26a54f5914b666c28584066eb18a681e5fad0559a10207a78fa01c4`

## Bank and scope result

The generator evaluated 796 deterministic candidates twice and promoted 300:
210 TRAIN, 46 DEVELOPMENT, and 44 HELDOUT. Family counts remain F01 30, F02
35, F03 35, F13 40, F15 50, F16 40, F17 24, and F18 46. Native audit passed.

Twenty-eight candidate records were rejected under
`cultivation-scope-escape-during-calibration`; their complete causal groups
were excluded. Across every selected reset and five-step legal calibration
rollout, husbandry capability, domestic herd, herd masks, fishing capability,
fishing presence, and movement availability all have maximum zero in TRAIN,
DEVELOPMENT, and HELDOUT. Every frozen reset has structural signature `1010`.

## Normalization result

Normalization-v3 again uses 1,260 TRAIN observations only. All sparse causal
support gates pass. Maximum per-feature saturation at absolute output 0.95 is
0.1980% in TRAIN, 0.7246% in DEVELOPMENT, and 0.3788% in HELDOUT. The result
supports retaining the existing tanh actor/critic without LayerNorm for the
pilot; hidden-layer saturation is measured during PPO before any architecture
change is considered.

See `generation/audit.json`, `generation/coverage.md`,
`generation/generation-summary.json`, and `normalization-promotion-audit.json`.

## Durable storage

The scope-safe bank is published at
`gs://anthropocene-rl-datasets-508912/datasets/elder-v2-cultivation-pilot-normalization-v3-scope-safe/2026-10-12-manifest-1cfabbc6/manifest.json`.
Its 64,798,720-byte shard was independently re-fetched from GCS in 4 MiB range
requests, matched the content SHA-256, materialized a named snapshot, and passed
a 300-record cache audit. Provider MD5, CRC32C, generation, and immutable custom
SHA-256 metadata are recorded in `gcs-storage-audit.json`.
