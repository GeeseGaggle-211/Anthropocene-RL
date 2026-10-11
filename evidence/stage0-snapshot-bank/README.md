# Elder v2 Stage-0 snapshot-bank evidence

This directory records the diagnostic-bank acceptance run against frozen native
commit `5559961164d6f9da8e6d20c1dc563ecec3eb04b7`. The authoritative SaveCodec
blobs are intentionally gitignored under `datasets/blobs/`; their paths and
SHA-256 digests are in the committed manifest.

## Result

- manifest: `datasets/manifests/elder-v2-stage0-v1.json`;
- manifest SHA-256: `b433f75837da942caf891406208c1416bca0c7e0fe8a13b7b7a98d4f598a112a`;
- accepted snapshots: 28;
- partitions: 11 TRAIN, 10 DEVELOPMENT, 7 HELDOUT;
- rejected candidates: 0;
- deterministic same-seed/config regeneration: PASS;
- independent manifest/blob/leakage audit: PASS;
- scenario families: F01-F13, F17, and F18;
- classifications: 24 controlled-valid fixtures and 4 synthetic-stress
  fixtures.

Measured structural coverage includes cultivation-only/no-herd,
husbandry-only, mixed cultivation/husbandry, legal no-Development,
knowledge-only Husbandry with no herd, one/two/three-ground fishing, young and
mature institutional memory, amenity bottlenecks, high-q states, overstocked
herds, matched primitive/Plough frontiers, and an endogenous Cultivation
capability transition.

## Artifacts

- `generation-summary.json`: config/result hashes and deterministic-run result;
- `audit.json`: duplicate, identity, blob, measured-stratum, and partition audit;
- `coverage.json` and `coverage.md`: family/partition/mask/stratum coverage;
- `provisional-train-normalization.json`: provisional Stage-0 scale manifest;
- `provisional-normalization-diagnostics.json`: all 512 named features, with
  presence, min, P5/P50/P95/P99/max, fitted `c_i`, and saturation for TRAIN,
  DEVELOPMENT, HELDOUT, high-q, fishing, no-herd, and young-memory slices.

The provisional fit used 44 TRAIN observations: 11 resets and 33 short legal
native calibration-rollout observations. DEVELOPMENT and HELDOUT observations
were transformed only after fitting and did not affect scales. Non-binary,
non-bounded weighted saturation was 0.097% on TRAIN, 0.471% on DEVELOPMENT,
and 1.949% on HELDOUT. This artifact is a pipeline test, not the final campaign
normalizer.

## Pilot ambiguity

The frozen native source recognizes an existing Plough capability in the local
production-frontier resolver but has no endogenous Plough acquisition path.
Stage-0 therefore uses a validated controlled Plough fixture and separately
includes an endogenous primary Cultivation transition. Before the 200-500-state
pilot, the design must settle whether controlled legal Plough counterfactuals
are sufficient or whether the pilot must wait for a future native contract (or
an authoritative historical SaveCodec state) with an endogenous acquisition
path. The desired natural-versus-controlled fixture mix for the larger pilot is
also not numerically specified.
