# Proposed 50-decision broad-domain Elder protocol

Status: proposal only; cohort generation and training are not authorized by this document.

Date: 2026-10-12

## Decision this protocol addresses

The cultivation-only, three-decision generation is closed as a non-promoted
diagnostic. Its fixed constant-action class has positive but limited headroom
over Option A, and its primary upper confidence bounds are below the frozen
useful-effect threshold. This does not show that the MLP failed. It shows that
the restricted 60-year task is a poor instrument for the intended Elder
question.

The proposed replacement is one new experiment generation whose primary
episode horizon is 50 native decisions. Husbandry and movement are allowed to
become active through the existing native structural masks. This is closer to
the stated Elder goal: choose among cultivation, livestock, location, trade,
migration, fission, amenities, memory, and productive-frontier opportunities
over long physical time.

## Frozen scope proposed for review

- Primary horizon: 50 decisions.
- A decision remains one native turn. Under the current 20-year timebase this is
  nominally a 1,000-year episode. Report actual elapsed physical time and flag
  any native variation rather than assuming every path is exactly 1,000 years.
- Cultivation controls: retained field operation and Development budget.
- Husbandry controls: existing herd retention and herd Development semantics,
  active only when native masks permit them.
- Movement: existing categorical stay-plus-destination head, active only when
  native masks expose legal candidates.
- Trade, immigration, emigration, fission, amenities, institutional memory,
  and productive-frontier variation remain live native dynamics.
- Fishing: no learned fishing head in this generation. Candidate worlds must
  remain free of fishing capability as a policy domain throughout the 50-turn
  stress audit. If that requirement materially destroys movement coverage,
  stop before bank freeze and request an explicit scope decision; do not shorten
  the horizon or silently admit fishing.
- Synthetic disaster curriculum: off.
- Privileged critic: off.
- Actor and critic remain separate feed-forward MLPs at the start.
- No native economics, ecology, reward, schema, or physics changes are allowed
  to make a gate pass.

## New experiment identity and cohort

Use a new bank, configuration hash, manifest identity, normalizer identity, and
evidence directory. Do not repurpose the 300-state cultivation bank or its
HELDOUT partition.

Proposed working identity:

`elder-v2-broad-domain-50d-pilot-v1`

The final bank size should be chosen from measured generation yield, with an
initial target of 400–600 accepted snapshots. Preserve complete causal groups
before timestamp selection and split approximately 70/15/15 into
TRAIN/DEVELOPMENT/HELDOUT. No world seed, lineage, ancestor/descendant state, or
counterfactual sibling may cross partitions.

Use constrained pairwise coverage rather than a Cartesian product. The bank
must include enough independent clusters for each promoted family; repeated
timestamps from one lineage do not substitute for independent worlds.

Required coverage families:

1. Cultivation q, maintenance burden, retained capital, labor, reserves, and
   tool/material constraints.
2. Initially absent, emerging, and established husbandry opportunity, including
   substitution and complementarity with cultivation.
3. Meaningful movement opportunities: stay, resource-seeking, trade access,
   exit from poor ecology, and movement with short-term cost but long-term gain.
4. Delivered trade and import dependence, not agreement values.
5. Signed migration and demographic pressure.
6. Pre- and post-fission states under `CurrentPlusExitOpportunity` semantics.
7. Young, improving, deteriorating, and mature causal memory histories.
8. Primitive and controlled productive-frontier contrasts without actor access
   to latent frontier identity.
9. Cross-domain regimes where the locally obvious cultivation choice is not the
   long-run welfare choice.

## Stage A — generation and 50-decision native audit

Before normalization or optimizer work:

- Save→Load fingerprint roundtrip, native Reset/Step, finite physical time,
  replay, schema/configuration identity, and reward-denominator checks pass.
- Movement and herd masks activate only when their native prerequisites hold.
- Every requested categorical action is legal under its mask.
- Inactive continuous and categorical heads are canonical and excluded from
  likelihoods; active heads retain requested/accepted/realized separation.
- Fishing never becomes a learned policy domain under the frozen proposal.
- Option A and a predeclared set of fixed/random legal policies are stressed for
  all 50 decisions. No scope failure is repaired by dropping a timestamp or
  shortening an episode after outcomes are seen.
- Coverage reports distinguish initial from emergent capability and report mask
  signatures over time, not only at Reset.

Failure stops at the bank boundary. It does not authorize native changes.

## Stage B — fresh TRAIN-only normalization

Freeze the bank manifest before fitting. Fit a fresh normalization-v2 instance
using only PILOT TRAIN reset observations plus short legal calibration paths
that exercise the broader masks. Use the same semantic normalization method as
the current v3 campaign; a new fit is generation setup, not evidence that the
old normalizer caused failure.

DEVELOPMENT and HELDOUT are transform-only diagnostics. Report presence, min,
P5, P50, P95, P99, max, fitted scale, and saturation for cultivation,
husbandry, movement-opportunity, trade, migration, fission, memory, and frontier
slices. Never refit on DEVELOPMENT or HELDOUT.

## Stage C — measurement before PPO

Define baselines and evaluation before optimizer changes:

- Option A through the same compiler, starting states, native randomness, and
  50-decision horizon.
- Whole-episode discounted physical welfare and raw prior-cohort welfare.
- Survival, population, QoL, needs coverage, reserves, fields, herds, movement,
  delivered trade, migration, fission, amenities, tools/materials, interventions,
  and end-of-horizon stocks.
- Per-scenario-cluster paired uncertainty; never treat transitions as
  independent samples.
- Descriptive prefixes at 3, 5, 20, and 32 decisions. The 50-decision endpoint
  remains primary and cannot be replaced after outcomes are seen.
- Explicit delayed-investment ranking fixtures and H=400/H=800 sensitivity
  before trusting critic-mediated long-horizon choices.

F15 remains `UNEVALUABLE` until the new bank contains measured delivered-trade
contrasts with enough independent clusters.

## Stage D — active-head and critic gates

Before a policy comparison:

1. Verify requested-action likelihoods for every mask signature.
2. Verify exactly zero gradient for every inactive head and nonzero finite
   gradient support for each active head on a balanced synthetic/on-policy batch.
3. Verify movement categorical probabilities renormalize only over legal rows.
4. Verify checkpoint reload and continuation parity.
5. Measure value ranking on paired delayed-investment paths against observed
   continuation returns.
6. Measure 1/2/4/8-worker throughput and set a fixed transition/time budget.

These gates may motivate separately reviewed learner ablations. Do not combine
value-target normalization, group-relative advantages, clipping changes, batch
changes, and a new task in one uninterpretable arm.

## Stage E — bounded PPO proposal

Only after Stages A–D pass, prepare a separately reviewed PPO run proposal. The
proposal must predeclare:

- three neural seeds;
- at least 100,000 transitions per seed unless a power analysis justifies more;
- family/mask-balanced reset sampling from TRAIN only;
- update batch and minibatch geometry large enough to avoid the former 24-sample
  full-batch updates;
- target KL, epochs, entropy/log-std bounds, clipping, and stopping rules;
- DEVELOPMENT checkpoint selection rules and a single sparse/final HELDOUT use;
- failure attribution order before any architecture change.

The starting architecture hypothesis remains separate
`512 → 256 → 256` actor and critic networks with existing hybrid heads and
critic-only LayerNorm. No ReLU, actor LayerNorm, GRU, Transformer, privileged
critic, or schema expansion is authorized by the current evidence.

## Promotion and stop rules

Promotion requires replicated welfare competitiveness or improvement over
Option A across heterogeneous DEVELOPMENT clusters, no strong HELDOUT
degradation, stable PPO diagnostics, valid active/inactive head behavior, and
qualitatively appropriate substitution among cultivation, husbandry, movement,
and external supply. Universal self-sufficiency, end-horizon liquidation, or a
single near-constant action regime is a failure even if pooled return rises.

Stop and preserve evidence if:

- the 50-decision bank cannot retain its declared scope;
- mask/likelihood/replay/reward contracts fail;
- independent family coverage is inadequate;
- critic ranking fails before PPO;
- measured local throughput makes the fixed budget impractical; or
- a proposed learner change lacks a single causal hypothesis and comparison.

## Authorization required

This protocol deliberately stops before generation. Starting the new bank is a
curriculum/scope decision—not an architecture decision. Authorization should
confirm the 50-decision primary horizon, native husbandry and movement
activation, and the explicit fishing treatment above.
