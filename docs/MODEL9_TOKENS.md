# model9 contract 12: tokenised Elder (design and plan)

Status: plan, 20 October 2026. The user asked to "tokenize what we have to ensure scalability, so the Elder is
future-proofed as complexity grows". This builds on Part 2 of the simulator's
`Docs/ElderCapitalCommitmentsDesign.md` (activity tokens), which it implements, and extends it to the input side.

## What does not scale today

1. **Inputs.** Each entity set (sites, fishing, herds, commodities, contacts, regions, trade) has its own set encoder,
   and its pooled block is concatenated into the trunk.
   - Every new set widens the trunk.
   - Contracts 9, 10 and 11 each needed a zero-column upgrade.
   - That works, but it grows the trunk linearly and gives no cross-set reasoning (a region row cannot attend to a
     trade row).
2. **Actions.** The economy mix has exactly three outputs (cultivation, herding, fishing), so every new activity
   needs a new head.

## Design

### Token encoder (inputs)

- Every row of every set becomes a **token**: a per-set linear projection to `d` = 128, plus a learned set-type
  embedding.
- The core observation becomes one more token, the **village token**.
- `L` = 2 self-attention layers (4 heads) run over the at most 70 present tokens, masked by row presence. Missing rows
  are excluded, never zeros.
- The village token's output feeds the existing trunk at fixed width. Pointer heads (movement over region tokens,
  later activity tokens) read their tokens' outputs directly.

Adding a set or a row kind is then one projection and one type embedding, with no trunk change. The upgrade between
layouts that only add sets becomes exact by construction: a new set's tokens are masked until trained.

Cost: attention over at most 70 tokens with d = 128 per village decision. This is small next to the simulator step,
and will be measured.

### Activity tokens (actions; Part 2, slice 1)

- **Simulator:** an `activities` set with one row per staffable activity (three kinds now). Each row carries:
  - the kind;
  - the marginal return per added worker-year (the planner's development bands);
  - capacity and current staffing;
  - capital at stake;
  - input readiness;
  - the services fed and their gaps;
  - legality.

  Later kinds (mines, smelters, trade routes) are rows, not contract changes.
- **Learner:** the village query attends over activity tokens, plus an "unassigned" token. The masked softmax is the
  discretionary-labour allocation, sampled as the additive logistic-normal the mix head uses today. The three-row
  case maps exactly to today's (cultivation, herding, fishing) mix, so teacher labels carry over.
- **Mix after the site** (contract 10): the region-row offset applies to the activity logits.
- **Trade posture** (contract 11): it moves onto the trade tokens. Each trade token's output gives its group's value
  and gate, so new tradeable groups are new tokens.

### Personality

The personality vector (`MODEL9_PERSONALITY.md`) enters as part of the village token, so preferences condition every
token's attention.

## Migration

This is a new network, so checkpoints cannot be upgraded exactly. It needs:
1. teacher data regenerated with contract-12 observations (TRAIN 1000-1599, DEVELOPMENT 2000-2059; on the 60-vCPU VM
   in about an hour);
2. stage-1 behaviour cloning of the token model, gated on fidelity against the current stage-1 clone (closed-loop
   welfare against the teacher within noise, on the decisive experiment's worlds);
3. stage 2, then the curriculum (mixed use, movement, fission, trade, self-play) on the token model, under the
   no-collapse guard.

## Order against the curriculum

- The combined stage (mixed use, movement, fission and trade) runs first on the current network. It answers whether
  these skills learn at all, which tokenisation does not change.
- The token model is built in parallel and replaces the current network from stage 1.
- The curriculum then reruns on it, with the personality vector.

Results from the current network remain the comparison the token model must match.

## Perturbations (optional; disasters later)

A training-only perturbation hook applies sudden shocks between turns, as surprises with no information in the
observation (per the disaster decision):
- tile food or biomass loss in a radius;
- herd loss;
- freshwater drop.

They are keyed to the episode seed, so they are deterministic and replayable. They give a first resilience curriculum
and a shock suite in the gates, before disasters exist as game features. It is optional; it is scheduled after the
token model's stage 2.
