# Astra-high independent launch review

Reviewed learner HEAD: `051887710c0c41266982b08d0fa0f313244d8275`

## Verdict

- **GO** for the fresh seed-1701 hero: at most 100 updates / 260,000
  transitions on Google AI Pro, full DEVELOPMENT every 10 updates, with
  HELDOUT sealed.
- **Conditional GO** for the fresh three-seed campaign: seeds 1701, 1901 and
  2101, at most 500 updates / 1.3 million transitions per seed, provided the
  hero passes every hard and health gate.
- This approves training, not final policy promotion.
- No architecture-level decision is required from the user.

## Basis

- The powered proof-of-readiness evidence shows no architecture collapse or
  frozen policy: actor updates and probe returns moved, log standard deviations
  remained near `-1`, actor means did not pin to endpoints, selected actor
  saturation stayed at or below 6.31%, critic saturation stayed at or below
  8.74%, and final rolling critic explained variance was approximately 0.93.
- The new 50-decision instrument uses 2,600 transitions per update,
  group-relative first-decision advantages, PopArt value targets, separate
  clipping, entropy support and parallel collection.
- The 499-state manifest contains 382 world/lineage groups and has no
  cross-partition world, lineage, counterfactual-group or content-hash overlap.
- F15 contains 25 complete pairs: five for each declared causal contrast, with
  measured native delivery/export outcomes and Player 1 control.
- Normalization uses 2,088 TRAIN-only observations, has 0.444% maximum slice
  saturation, and passes sparse trade and signed-migration support.
- Both GCS shards were re-downloaded and all 499 records were materialized and
  hash-verified. The hero command seals HELDOUT in code.

## Interpretation limit

The five F15 contrasts are not each represented in every evaluation partition.
DEVELOPMENT contains isolated/import and food/material contrasts; HELDOUT
contains migration and mild-import/export contrasts. This does not block the
launch, but final claims must not imply per-partition coverage that is absent.

## Conditions that reverse GO

- Any native, dataset or normalizer identity mismatch; remote materialization
  failure; partition leakage; or hero HELDOUT consumption.
- Any contract, replay, numerical, likelihood, inactive-head-gradient or
  checkpoint-parity failure; or post-update KL above 0.15.
- Two consecutive activation-health failures: saturation at least 10%, mean
  Tanh derivative at most 0.10, or any head log standard deviation at most
  `-4.5` on two consecutive evaluations.
- Failure of selected/final critic health: rolling eight-update median
  explained variance below 0.20, normalized MSE above 0.80, or fewer than six
  positive-fit updates in eight.
- Unresolved endpoint pinning, DEVELOPMENT critic collapse, widespread
  optimizer early stopping, or loss of durable checkpoint/evidence storage.

A welfare-parity miss or slow learning alone does not reverse architecture
readiness. Keep the frozen MLP and diagnose performance separately.
