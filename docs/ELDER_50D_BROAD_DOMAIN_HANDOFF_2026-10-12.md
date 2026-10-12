# Elder 50-decision broad-domain campaign handoff

Date: 2026-10-12

## Purpose

This branch executes the first powered Elder PPO campaign over cultivation,
husbandry, movement, trade, migration, fission, institutional history, and
productive-frontier variation. Fishing, synthetic disasters, recurrence,
privileged critic inputs, and native semantic changes remain out of scope.

The five-decision value in the dataset generator is only the maximum search
window used to align natural F15 delivered-trade snapshots. Training and every
serious evaluation are exactly 50 decisions, or nominally 1,000 simulated years
at the native 20-year time step. The learner rejects a broad-domain run that is
configured for any other episode length.

## Frozen inputs

- Native simulator: `GeeseGaggle-211/Anthropocene@5559961164d6f9da8e6d20c1dc563ecec3eb04b7`
- Frozen bank identity: `10ab09c73942cc3e2bea0bef409c6682a909d07022b0fb1d529d3b566979e736`
- Accepted bank: 499 snapshots (347 TRAIN, 73 DEVELOPMENT, 79 HELDOUT)
- Generator configuration identity: `a9cb4620ecbff98053c3cda18b25bde91ed707735c7ef196ffc4c30aabbe623e`
- TRAIN-only normalizer file: `evidence/elder-broad-domain-50d-v2/bank/final-train-normalization.json`
- GCS dataset: `gs://anthropocene-rl-datasets-508912/datasets/elder-v2-broad-domain-50d-v2/2026-10-12-manifest-10ab09c7/manifest.json`
- Neural seeds: 1701, 1901, 2101

## Learner configuration

The exact declaration is `configs/training/elder-broad-domain-50d-v2.yaml`.

- Separate `512 -> 256 -> 256` Tanh actor and critic; no shared trunk.
- Actor LayerNorm off; critic LayerNorm on.
- Running PopArt-style value-target normalization.
- Actor learning rate `3e-4`; critic learning rate `1e-3`.
- Separate actor/critic gradient clipping; target KL `0.02`.
- Positive entropy coefficient `0.001`.
- One base TRAIN snapshot per family, four on-policy trajectories per snapshot,
  and a leave-one-out group-relative advantage at the first decision.
- 40 updates, up to 104,000 transitions per seed.
- Eight persistent native bridge workers; TRAIN replay is sampled every 20
  episodes and evaluation replay remains exhaustive.
- DEVELOPMENT selects checkpoints. HELDOUT is evaluated only on the final
  selected checkpoint and is not used for optimizer or configuration changes.

The frozen native bridge supports only `RandomStream=0`; keyed randomness is
owned by snapshot state. This was confirmed by an explicit clean interface
rejection during preflight, not a game crash or native defect.

Build the bridge with `scripts/bootstrap_simulator.py --root <pinned-native>`;
the bare C# project intentionally has no self-contained simulator reference.

## Preflight result

The eight-worker 50-decision smoke collected 2,577 transitions across all 13
families in about 48 seconds. It had zero illegal actions, requested-action
mismatches, non-finite values, replay failures, reward-denominator violations,
or inactive-head gradient leakage. PPO post-update KL was `0.0103`, clip
fraction was `0.149`, and the group-relative first-decision advantages were
centered to numerical zero.

## Storage caution

GCS upload and provider metadata verification passed for both immutable
dataset shards. Large-object downloads from this workstation currently stall
at zero bytes in both the Python client and `gcloud`, while the small manifest
downloads normally. Do not prune `datasets/blobs` until a real shard re-fetch,
SHA verification, and transparent record materialization have succeeded.

## Completion boundary

At campaign completion, preserve the exact learner SHA, all three checkpoint
hashes, learning/evaluation curves, family/world comparisons with Option A,
throughput metrics, cultivation-intensity trajectory, self-sufficiency signal,
and go/no-go assessment. Publish large run artifacts and checkpoints to GCS;
commit only compact evidence and provenance.
