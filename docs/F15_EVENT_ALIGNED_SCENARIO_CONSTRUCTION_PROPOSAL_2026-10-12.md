# F15 event-aligned scenario-construction proposal

Date: 2026-10-12

Status: proposal for review; no further generation is authorized by this document.

## Decision boundary

The broad-domain 50-decision bank is blocked before manifest promotion. The
frozen native simulator is stable and deterministic, but the current fixed-time
F15 constructor does not reliably produce the required paired mild-food-import
and food-surplus-export states. The bounded remediation panel produced one
complete independent TRAIN pair; its predeclared gate required at least two.

This is `NATIVE_SCENARIO_CONSTRUCTION_BLOCKED`, not a simulator crash, physics
defect, observation-schema defect, or feed-forward MLP failure. Normalization,
LayerNorm, activation choice, PPO, and policy capacity cannot cause a failure
that occurs before a valid snapshot bank exists.

## Existing native measurement contract

F15 promotion uses delivered trade from the most recently completed native
interval. For the controlled camp, the generator measures the delivered
fractions of `TradeRecord` quantities in `WorldState.History.Last()` and derives:

- `delivered_trade_total = delivered imports + delivered exports`;
- `delivered_food_import` from delivered food entering the controlled camp;
- `delivered_food_export` from delivered food leaving the controlled camp;
- `trade-active` when delivered trade is positive;
- `food-import` or `food-export` only when the corresponding delivered food is
  positive.

The actor receives only the Reset observation at the captured state. It does
not receive future deliveries or construction labels. Drive/GCS identifiers
remain storage metadata and never determine snapshot identity.

The current mild-import/export pair is accepted only when both counterfactual
siblings satisfy their exact delivered-direction predicate at their declared
snapshot decision. Contact presence or a hypothetical agreement is
insufficient.

## What the bounded evidence shows

- At decision 1, the focused 24-pair diagnostic yielded three complete pairs.
- Every one of the 27 rejected siblings had an observed contact but zero
  delivered trade in the measured interval.
- In legal calibration traces, 24 of those 27 siblings later showed positive
  contact trade flow. That demonstrates live trade and implicates interval
  timing, but it does not by itself prove the required import/export direction.
- Re-evaluating the same declared worlds at decision 2 yielded two complete
  pairs, different from the decision-1 successes. Decision 3 yielded none.
- The predeclared targeted panel used 24 new independent TRAIN worlds, three
  documented factor/timestamp configurations, strict unchanged physical
  predicates, and no outcome-based seed selection. It yielded seven accepted
  records but only one complete pair.
- An independent repeat reproduced the complete native result and all seven
  snapshot blobs byte-for-byte. The native-result SHA-256 is
  `522b6373acce9e5a2a8f3b2845de1390a0aaf7379f92563353a73a58183d548d`.

The missing determinant is therefore not basic contact construction. It is the
coincidence of the required delivered direction in both siblings at one fixed
measurement interval.

## Proposed v2 capture rule

Replace fixed per-trajectory timestamps for this one F15 contrast with a
versioned, group-level event-aligned rule:

1. Assign the complete world/counterfactual group to TRAIN, DEVELOPMENT, or
   HELDOUT from identity alone, before simulation or timestamp selection.
2. Build both siblings from the same world seed and declared factor row. Keep
   the existing causal variant as their only intentional construction
   difference.
3. Advance both siblings with the same predeclared legal directive schedule and
   native random-stream contract for at most five decisions (nominally 100
   physical years).
4. After each completed native interval, measure the existing delivered-trade
   fields. A shared decision index is eligible only when the mild-import sibling
   has positive delivered food import and the surplus-export sibling has
   positive delivered food export, with positive total delivered trade in each.
5. Capture both siblings at the earliest shared eligible decision index. Never
   capture each sibling at its own first event: unequal elapsed time would
   confound the counterfactual with history and memory maturity.
6. Reject the whole pair if no shared eligible decision exists within the fixed
   horizon. Do not relabel it, retain a lone sibling, extend the horizon after
   seeing outcomes, or substitute contact flow for delivered trade.

Five decisions is a construction horizon, not the eventual PPO episode
horizon. The proposed 50-decision training horizon remains unchanged.

## Implementation boundary

The change belongs in the learner-owned native snapshot generator. It requires
group-aware paired trajectory execution and selection metadata recording the
capture horizon, shared decision index, predicate values at every attempted
index, and whole-pair rejection reason.

It does not require or authorize changes to:

- native economics, ecology, trade matching, migration, fission, reward, or
  physics;
- observation or action schemas;
- requested/accepted/realized action semantics;
- SaveCodec identity or replay rules;
- fishing exclusion;
- the actor or critic architecture.

Use new generator/scenario/config versions and a new canonical configuration
SHA. The current failed artifacts remain immutable diagnostics and must not be
promoted into the new bank.

## Finite validation proposal

After owner approval, run one v2 validation panel with these frozen limits:

- at most 24 new independent causal pairs;
- no more than three predeclared factor configurations;
- identities and partitions fixed before native outcomes;
- exact shared event-alignment horizon of five decisions;
- two byte-identical native generations;
- all existing SaveCodec, Reset/Step, replay, finite-time, mask, fishing,
  configuration, schema, and identity gates unchanged.

Success requires at least two complete, unique, scope-valid TRAIN pairs and an
unchanged full-bank selector that can meet all F15 contrast and 17/4/4
TRAIN/DEVELOPMENT/HELDOUT pair quotas. Failure at the fixed cap returns to
`NATIVE_SCENARIO_CONSTRUCTION_BLOCKED`; it does not authorize a fifth search,
gate relaxation, or native behavior change.

## What follows only after a bank passes

1. Freeze and publish the complete manifest and blobs to GCS.
2. Fit a fresh normalization-v3 artifact from broad-domain TRAIN resets plus
   short legal calibration rollouts only. DEVELOPMENT and HELDOUT remain
   transform-only diagnostics.
3. Run the no-PPO headroom and supervised representation probes from
   `ELDER_MLP_CAMPAIGN_AUDIT_2026-10-12.md` before attributing any result to
   MLP capacity.
4. Use critic value-target normalization, separate actor/critic clipping,
   larger batches, and a powered transition budget before another PPO tuning
   comparison.
5. Train the proposed 50-decision cohort with movement and husbandry admitted
   through native masks and fishing excluded.

The starting hypothesis remains the separate `512 -> 256 -> 256` tanh actor
and critic, with critic-only LayerNorm. ReLU, actor LayerNorm, recurrence, a
Transformer, and a privileged critic remain ungated.
