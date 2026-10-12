# Elder 50-decision scale-out evidence

Status: frozen bank and TRAIN-only normalizer PASS; cloud training not started.

## Identities

- Native simulator: `5559961164d6f9da8e6d20c1dc563ecec3eb04b7`
- Generator config identity: `bf4517367870b3bd02a7ad7604aa0fe444b84671a883dbcc2ddefd927b4aa70e`
- Manifest identity: `6036d104dc655305f62bde2e7650cb4b1d20f36dcfdb8337f92c9dcbdbf7f06e`
- Native result identity: `f78310197bd8fbf64a666993783a1fee15de8eada28f98c7c0e52e767e0ef1bb`
- Normalizer file SHA-256: `9d0136764b213f9d3b1b6996b41adfe92d43573de8192b3c46c89e60ae345fd1`
- Published GCS manifest: `gs://anthropocene-rl-datasets-508912/datasets/elder-v2-broad-domain-50d-scaleout-v1/2026-10-12-manifest-6036d104/manifest.json`
- Published dataset-manifest SHA-256: `3d0c6483bb6c8082886f16a196d8d9c76775567e2ae66f0a32f6b936338cdba2`
- Frozen evidence bundle: `gs://anthropocene-rl-datasets-508912/run-objects/sha256/f6/f63b877a4b3beed9e388f6898e464093df3ae214b37cc78a8e27421289d737c0.tar`

## Result

- Accepted: 499 of 800 requested candidates.
- Split: 348 TRAIN / 75 DEVELOPMENT / 76 HELDOUT.
- Independent worlds/lineages: 382.
- Deterministic native regeneration: PASS.
- Broad-domain audit: PASS.
- Prior-manifest world/lineage/content overlap: zero.
- Normalizer fit: 348 TRAIN resets + 1,740 legal TRAIN rollout observations.
- DEV/HELDOUT observations used for fit: zero.
- Maximum normalization slice saturation: 0.444% (gate: 10%).
- Sparse delivered-trade and signed-migration support: PASS.
- Scope: cultivation, husbandry, and movement policy domains; live trade,
  migration, and fission; fishing excluded.

## Files

- `plan-summary.json`: requested design and exact accepted quotas.
- `bank/native-candidate-summary.json`: physical generation yield.
- `bank/generation-summary.json`: immutable generation result.
- `bank/audit.json`: hard-gate result.
- `bank/coverage.json` and `bank/coverage.md`: measured coverage.
- `bank/final-train-normalization.json`: frozen normalizer.
- `bank/final-normalization-diagnostics.json`: per-feature and slice diagnostics.
- `storage-audit.json`: GCS generations, provider checksums, full re-read, and disk cleanup.
- `astra-review.md`: added after independent launch review.

The source and cloud execution handoff is
`docs/ELDER_50D_SCALEOUT_GCS_HANDOFF_2026-10-12.md`.
