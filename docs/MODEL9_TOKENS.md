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

## Cloning stability

Status: 22 October 2026, branch `feature/r3-clone-stability`.

### Symptom

Stage-1 clones of the token model across 10 seeds scored closed-loop welfare ratios of 0.40, 0.89, 0.12, 0.94, 0.92,
0.05, 0.95, 0.89, 0.88 and 0.07. The collapsed seeds lose clothing coverage (0.17-0.41 against the teacher's 0.99),
and seeds 2, 5 and 9 also lose housing. Their BC logs look like the healthy seeds' logs: every head converges, and
the final train and dev losses do not rank the seeds. Seed 0 has the best final train loss of seeds 0-2 and still
collapsed.

### Cause: a train/evaluation input shift, not optimisation

- **The shifted input.** Core feature `v2-480` is `Mask.Movement`, the flag that says whether movement is a legal
  decision.
  - Teacher shards (`--model9-teacher`) are written with `Reset(snapshot, movementEnabled: true)`, so the flag is 1 in
    every training row.
  - `evaluate.py` runs the bridge with `movement=False` (native movement), so the flag is 0 in every evaluated
    decision.
  - Its symlog std in TRAIN is 0, which the normalizer floors at 1e-3. The evaluated input is therefore
    z = -0.693 / 1e-3 = -693 on the village token, at every decision.
- **Why the result depends on the seed.** The normalized flag is about 0 in every training row, so its column of the
  village projection never receives a gradient and stays at its random initialisation. At evaluation, -693 times that
  random column dominates the village token, so the cloned policy is replaced by a direction set by the seed.
  The pooling encoder has the same exposure in principle. The pooling comparator (1.003) was a different, earlier
  checkpoint (`t1/inputs/stage1-seed2.pt`), so it is not evidence on the same data.

Evidence. All of it uses the local Model9 bridge, which reproduces the VM gate exactly: s0 on worlds 2000-2003 gives
0.367 locally and 0.367 from the VM's per-world numbers.

| Check | s0 | s1 | s2 |
| --- | --- | --- | --- |
| Deterministic service-level MAE vs teacher, in distribution (movement on, `--model9-teacher` rows) | 0.060 | 0.083 | 0.069 |
| Same, on the evaluation regime's teacher states (movement off) | 0.377 | 0.278 | 0.761 |
| VM closed-loop welfare ratio, 20 worlds | 0.40 | 0.89 | 0.12 |
| Same checkpoints, input guard added to the normalizer only (no retraining) | **1.085** | **1.154** | **1.070** |

In distribution, all three VM checkpoints clone equally well, with per-head losses matching their VM dev logs.
With the guard, clothing recovers to 0.985-0.991 and housing to 0.95. Dropping the flag is lossless for these
checkpoints because it was about 0 in all of their training inputs.

Other hypotheses were checked and rejected:
- **Services snapping to 0/1.** It happens, but identically across seeds in distribution.
- **Head divergence or a log-std floor.** None appears in the logs. The logistic-normal NLL swings by nats between
  epochs, from rows at the -5 log-std floor, while the deterministic mode barely moves. That makes it a poor fidelity
  proxy, but it is not the cause of the collapse.

### Fix (`train.py`, `model.py`)

Every flag defaults on for `--token` and to the original behaviour otherwise. Pooling runs reproduce bitwise; a test
checks this against the original loop.

| Flag | Token default | Otherwise | Effect |
| --- | --- | --- | --- |
| `--input-guard G` | 10 | 0 (off) | The normalizer drops inputs constant in TRAIN (symlog std < 1e-3) and clips the others to ±G. This is the fix. It is stored in the normalizer state (`guard`, `*_dead`), so evaluation, PPO and search apply it. Unguarded checkpoints load unchanged. |
| `--lr-warmup F` | 0.05 | 0 | Linear BC warm-up over the first F of the steps. |
| `--lr-final F` | 0.05 | flat | Cosine decay to F of the peak. |
| `--encoder-lr-scale` | 1.0 | 1.0 | Encoder peak learning rate relative to the heads' rate. |
| `--grad-clip` | 1.0 | 1.0 | Gradient-norm clip value. |

The BC log adds `dev_mode` each epoch: the deterministic action's per-service MAE, the strategic MAE and route
agreement against the teacher. It is a fast fidelity proxy.

The `.bc.pt` phase file now also stores the normalizer and `bc_config`. A resumed run reuses that normalizer.
Phase files written before this change carry no normalizer, so they refit as before.

### Seed-robustness check (local, reduced)

- **Data.** 80 TRAIN worlds (1000-1079) and 10 DEVELOPMENT worlds (2000-2009) from the local `--model9-teacher`
  (48.7k rows).
- **Training.** 6 BC epochs, seeds 0-3.
- **Evaluation.** The proxy is measured in distribution and on teacher states collected in the evaluation regime
  (bridge, movement off). Closed-loop fidelity uses worlds 2000-2005.

| Config | In-dist service MAE | Eval-regime service MAE (s0/s1/s2/s3) | Closed-loop ratio (s0/s1/s2/s3) |
| --- | --- | --- | --- |
| base (original) | 0.094-0.099 | 0.458 / 0.235 / 0.602 / 0.238 | 0.44 / 0.59 / 0.37 / 0.81 |
| schedule only | 0.099-0.100 | 0.414 / 0.235 / 0.603 / 0.246 | not run |
| guard only | 0.093-0.101 | 0.102 / 0.101 / 0.099 / 0.101 | not run |
| full (token defaults) | 0.099-0.101 | 0.103 / 0.103 / 0.101 / 0.106 | 0.95 / 0.95 / 0.85 / 0.97 |

The schedule neither causes nor cures the collapse, but it is kept as the token default; the guard alone removes the
seed spread. The clones are smaller than the VM's, so these ratios are relative, not gate values.

### Still open

- A full VM re-clone at the token defaults (10 seeds, 660 worlds, 12 epochs) and the 20-world gate.
- Generating teacher data in the evaluation regime, or evaluating with movement enabled, removes the shift at its
  source. The other movement inputs (`v2-481` to `v2-489`) still differ between the regimes; they vary in TRAIN, so
  the guard only clips them.
- PPO stages that run with movement off inherit the guarded normalizer from stage 1. The flag stays dropped, which is
  harmless while it is constant within a stage.

### New-run bounded physical-history contract

`train.py --normalization-contract bounded-v2` is an opt-in new-run contract; it never upgrades a historical
checkpoint. It stores top-level `feature_contract=model9-bounded-physical-history-v1` and the matching complete
contract inside the normalizer. Resume rejects a missing or different identifier.

Core 40:55 (previous-decision history) and core 144 (movement enabled) use fixed physical transforms instead of a
fitted near-zero standard deviation. Boolean and unit-fraction fields stay in their declared domains; compiler count
uses `log1p(x)/log1p(32)` and elapsed years uses `log1p(x)/log1p(20)`. Other standardized inputs use a 0.1 standard
deviation floor, and all normalized values retain the finite ±10 guard. The initial clone fit masks core 40:55 and
their presence bits. A later closed-loop refit may unmask only indices whose content-hashed corpus has at least 256
present finite rows and transformed range at least 1e-3. Support is computed from row values plus unique transition
IDs, policy versions and producer digest; callers cannot nominate supported indices. Constant legitimate flags may
therefore remain masked without blocking support collection or critic-only warmup. Actor-update gates request the
specific history indices they need rather than treating partial support as global readiness.

This is an engineering support floor, not calibration or readiness evidence. Requested/corrected/executed feedback
schema changes are deferred to the separately reviewed labor/retry contract; PPO continues to score the sampled
request and never substitutes compiler output as the action.
