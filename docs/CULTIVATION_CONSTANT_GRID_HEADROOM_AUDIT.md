# Cultivation constant-grid headroom audit

Date frozen: 2026-10-12

This is the pre-outcome design for Phase 0a of the Elder MLP remediation. It
does not change native physics, reward meaning, schemas, normalization, model
architecture, or optimizer behavior. It performs zero optimizer updates.

## Question

Does the fixed three-decision cultivation task contain measurable welfare
headroom over Option A within a coarse class of constant field-operation and
Development-budget actions? Five decisions are reported secondarily, but cannot
rescue a failed three-decision gate or authorize PPO.

The best grid result is a per-snapshot hindsight diagnostic within the tested
constant-action class. It is not an upper bound on feedback-policy performance.
A negative result does not prove that the broader task is unlearnable.

## Frozen panel and action grid

- All 210 TRAIN and 46 DEVELOPMENT snapshots.
- HELDOUT remains sealed.
- DEVELOPMENT has already been used in the campaign and is diagnostic, not a
  pristine final test.
- 25 constant requested actions: field operation × Development budget in
  `{0, 0.25, 0.5, 0.75, 1}` squared.
- Inactive herd retention remains zero, field Development share remains one,
  and movement remains explicit stay.
- Option A is evaluated through the same native shared compiler.
- Every episode has a common five-decision ceiling. Physical discounted returns
  through decisions three and five are extracted from the same trajectory.
- Budget: 6,656 episodes and at most 33,280 forward transitions, plus native
  replay verification.

At the frozen native SHA, `EpisodeDecisionLimit` is consulted only after the
physical turn resolves, reward is constructed, and the episode step increments.
Therefore the first three physical transitions of a five-decision trajectory
are the three-decision configuration prefix; only the truncation flag differs
at decision three. This avoids a second rollout matrix that would exceed the
frozen budget.

## Integrity gates

Every trajectory must preserve cultivation-only scope, requested action
correspondence, finite observations and rewards, the fixed reward denominator,
and native replay. A scope or contract failure stops escalation and invalidates
the affected complete-grid comparison. No offending cell may be discarded and
no horizon may be shortened per snapshot.

## Endpoints and uncertainty

The primary per-snapshot relative improvement is:

`(best constant-grid return - Option A return) / max(abs(Option A return), 1 raw reward unit)`

Absolute differences are reported alongside it. Scenario uncertainty uses
10,000 fixed-seed cluster-bootstrap replicates. Clusters are connected by any
shared world seed, trajectory-family plus lineage identity, or counterfactual
group, preserving causal pairs and shared lineages.

One global constant action is selected by mean three-decision TRAIN return only,
with lexicographic tie-breaking. Its DEVELOPMENT result is reported without
using DEVELOPMENT for selection.

## Frozen decision gates

`GO_PROPOSE_PHASE_0B` requires all of:

1. Complete three-decision integrity on TRAIN and DEVELOPMENT.
2. Mean best-grid improvement of at least 1.5 percentage points on both
   partitions, with one-sided 95% lower confidence bounds above zero.
3. At least two of the seven non-F15 families with mean improvement of at least
   2 points on both partitions and a positive DEVELOPMENT lower bound at
   one-sided 99.286% confidence (Bonferroni over seven families).
4. Each qualifying family has at least five independent DEVELOPMENT clusters.

`NO_GO_CONSTANT_GRID_ROUTE` requires at least 20 independent clusters in each
partition and a one-sided 95% upper bound below 1.5 points on either partition.
All other complete outcomes are `INCONCLUSIVE`.

F15 operational returns may be displayed, but its delivered-trade contrast
remains `UNEVALUABLE` and it cannot qualify a family gate. No outcome directly
authorizes architecture changes, a longer PPO run, or HELDOUT access.
