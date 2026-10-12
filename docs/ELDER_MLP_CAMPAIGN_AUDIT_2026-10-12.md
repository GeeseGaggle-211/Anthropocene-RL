# Village Elder MLP campaign audit

Date: 2026-10-12
Branch audited: `feature/cultivation-pilot-tuning` @ `5cbdf16`
Native simulator: `GeeseGaggle-211/Anthropocene@5559961164d6f9da8e6d20c1dc563ecec3eb04b7` (unchanged)
Normative reference: *Anthropocene AI/RL Training Concept* §§33–43

## Scope and evidence limits

This audit compares the concept document, the three progress reports (pilot v1 conclusions,
normalization-v3 conclusions, tuning handoff) and the learner code on this branch. Numbers below
come from the in-repo evidence, mainly
`evidence/cultivation-ppo-pilot-normalization-v3-scope-safe/summary.json` and
`evidence/cultivation-ppo-pilot-v1/`. The evidence for the later tuning arms (optimizer control,
critic LayerNorm, reward scale 1e-4, actor LR 3e-4, 192 updates) lives only on GCS. This container
had no GCS credentials, so those arms are assessed from the handoff text. The v3 pilot that is in
the repo uses the same configuration family and shows the same failure pattern.

## Bottom line

1. **The infrastructure is on track; the learning experiments are not.** The contract,
   replay, partitioning, normalization, scope-safety and storage work is careful and
   matches the spec. The learning campaign, however, has been measuring with an
   instrument that cannot detect the effects it is tuning for.
2. **No architecture change is justified yet.** This agrees with the reports, but for a
   different reason. The reports say the MLP has not been shown to be inadequate. In fact
   the campaign has never given the MLP a clean enough learning signal to test it in either
   direction. LayerNorm, activation, reward-scale and clipping arms are architecture-adjacent
   knobs. They were tuned on noise.
3. **The binding problems are formulation and signal, not representation:**
   - 3-decision (60-year) episodes. The spec asks for 20–50.
   - About 2.3k–4.6k transitions per seed, in batches of 24.
   - No measurement of how much headroom exists over Option A.
   - A myopic 60-year evaluation metric.
4. **Stop the current tuning loop.** Run the cheap, decisive diagnostics in Phase 0 below
   before any new PPO arm. They will show whether there is anything to learn and whether
   the MLP can represent it, without PPO noise.

## What the evidence actually shows

### The learner received almost no data

| Quantity | Value | Source |
|---|---|---|
| Transitions per seed | 96 updates × 8 episodes × 3 decisions = **2,304** (4,608 for the 192-update arm) | summary.json `training.transitions` |
| Wall-clock training per seed | **~150 s** | `training.timing.end_to_end_seconds` |
| Native simulation share of wall-clock | 12 s of ~150 s (**≈8%**) | `native_step_seconds` |
| Rollout collection | 128–134 s, of which bridge stepping 48 s; the remaining ~75–80 s is reset, replay verification and diagnostics | `timing` |
| Transitions per PPO update | **24** (minibatch 64 > batch, so each of 8 epochs is one full-batch step) | config |

The spec's illustrative batch is 2,048 transitions per update (§21). The whole campaign for one
seed is about the size of one such batch. The reports call compute "not the bottleneck". That
is true, but it means the budget was the bottleneck: a one-hour run is about 25× the entire
current campaign even at today's throughput.

### The optimizer was fitting noise

- Mean post-update clip fraction was **0.43–0.49** in all three v3 seeds. Healthy PPO usually
  sits around 0.1–0.2.
- The target-KL early stop fired in **48, 59 and 54 of 96 updates**.
- Training return was flat. By quarter: 1701 = 11.75 / 12.08 / 11.55 / 11.70;
  1901 = 11.55 / 11.70 / 11.81 / 12.01; 2101 = 11.69 / 11.94 / 11.45 / 11.73. The
  per-update standard deviation was about 2.0.
- Critic post-update value MSE was 14–42 in scaled-return units (RMSE about 4–6.5), against a
  per-decision reward of about 4. The action-induced differences in 3-step return that we can
  observe are a few tenths of a unit. The **per-sample signal-to-noise ratio of the advantage
  is around 0.05–0.1**, so a 24-sample gradient is essentially random.

### The "seed divergence" is a random walk on a nearly flat objective

At evaluation, every seed converged to a **near-constant, state-independent** field-operation
level:

| Seed | Accepted field operation, mean (p50 / p95) | DEV gap vs Option A | F03 per-snapshot gap |
|---|---|---|---|
| 1701 | 0.127 (0.137 / 0.148) | −4.27% | −1.96 |
| 1901 | 0.581 (0.675 / 0.687) | −7.96% | −6.57 |
| 2101 | 0.680 (0.801 / 0.819) | −8.14% | −6.57 |
| Option A | 0.477 (0.491 / 1.0) | — | — |

- Almost the whole gap is **F03** (expensive cultivation with inadequate labor or tools), with
  a smaller contribution from F17. High constant field operation hurts F03; low constant
  field operation hurts it less.
- The HELDOUT **median per-snapshot difference from Option A is exactly 0.0 for all three
  seeds**. For seed 1701, the F02, F13, F17 and F18 HELDOUT differences are about 0. In most
  states, then, running fields at 13% versus about 50% does not change the 60-year return at
  all.
- Seed outcome is mostly decided by which constant the policy drifted to. "Seed 2101 failed
  KL" and "seed 1701 passed" are not stable properties of the configuration.

### Statistical power is below the decision thresholds

- The per-snapshot relative-difference standard deviation is 6–22 pp. On 46 DEV snapshots,
  that gives a standard error of the mean of about **0.9–3.3 pp per seed**.
- Seed means within one configuration span about 4 pp.
- The +1.5 pp duration gate and the differences between arms (for example −7.41% vs −7.39%
  for optimizer control vs critic LayerNorm) are **below the noise floor**. The arm-by-arm
  conclusions in the handoff ("critic conditioning repaired but policy unchanged", "reward
  scale made the actor worse", "LR 3e-4 helped") cannot be distinguished from chance.
- Several arms were also selected on the same 46 DEV snapshots, so DEV is starting to absorb
  selection bias.

## Issues, ranked

### P0: problem formulation

**1. The episode horizon (3 decisions = 60 years) deviates from the spec and removes the
delayed-investment problem.**
§20 and §30 specify an initial horizon of 20–50 decisions, with 32 as the smoke default. §36
gates B, C and E are about delayed investment and post-horizon liquidation. With 3 decisions
and γ ≈ 0.951 per decision, everything beyond 60 years is carried by a critic with an RMSE of
about 5. Development spends labor now and pays back later, so a noisy critic plus a 60-year
window systematically favors under-investment. The v1 report's "falling Development budget"
signal is consistent with this.

The 3-step cap was applied to **all 300 snapshots** because, under 8-step fixed-action
rollouts, husbandry capability appeared in 20/300 snapshots and herd masks in 16/300. In the
first such snapshot, husbandry appeared at step 4. That is about 5–7% of the bank
(`evidence/cultivation-ppo-pilot-v1/structural-horizon-audit.json`). In the 8-step run, the
"inactive heads inert" gate failed because herd heads correctly became active. That is the
designed masking behavior (§42.3), not a defect. The cultivation-only scope is a pilot choice,
not a spec requirement: §35 stage 1 includes the herd-retention and split controls.

**2. The promotion metric is myopic and does not match the acceptance principle.**
Evaluation computes the realized 3-step discounted return with no terminal value
(`pilot.py:_evaluate_episode`) and compares it with Option A. §35 and §36E require long-run
welfare and continuation beyond the training horizon. A 60-year metric rewards
non-investment and cannot detect the behavior the Elder exists to learn.

**3. Headroom over Option A has never been measured.**
The environment is deterministic per snapshot: `Seed = world_seed`, `RandomStream = 0`, and
replay verification passes. So Q(s, a) can be computed exactly by rollout. Nobody has checked
whether *any* policy beats Option A on this task by more than the noise floor. The flat HELDOUT
medians suggest the controllable headroom may be small and concentrated in F03 and F17. If the
best achievable gain is about 1–3%, the "competitive with Option A" gates are only measuring
noise, and further tuning has nothing to find.

### P1: the learning instrument

**4. Sample budget and batch size are one to two orders of magnitude too small** (see the
evidence above). Batch 24, 8 epochs, clip fraction ~0.45 and early stops in more than half of
updates are the expected signature.

**5. The advantage estimator is dominated by between-state variance.**
Resets span populations of 30–774 in DEV, and the reward N·Q·Δt is extensive. Value differences
between snapshots dwarf action effects within a snapshot. Batch-level advantage standardization
lets the largest villages dominate the gradient, because their advantages are up to about 25×
larger. The deterministic, resettable simulator allows a much better estimator that has not been
used: a per-snapshot group baseline (see Phase 1).

**6. Critic targets are not normalized.**
With γ(20 y) ≈ 0.951, an infinite-horizon value is roughly 20× the per-decision reward. This is
an estimate of O(10²) scaled units, produced by a unit-gain head on tanh features. That
explains the 90% layer-2 critic saturation in the v3 report. Critic LayerNorm and reward-scale
changes treat the symptom. The standard fix is value-target normalization (running return
statistics or PopArt), which is an optimizer change, not an architecture change.

**7. Joint gradient clipping couples the actor's step size to the critic's error.**
A single `clip_grad_norm_` over both networks (`ppo.py`, `gradient_clip_mode="joint"`) rescales
actor gradients by a factor set by the much larger critic gradient. This varies from step to
step. The current action-credit audit already tests separate clipping. It is a real but
second-order issue and should be fixed by default, not gated on an audit.

**8. Fixed exogenous randomness per snapshot during training.**
Every visit to a TRAIN snapshot replays the same exogenous future. With 210 snapshots each
visited about 11–22 times, the policy can fit trajectories rather than states. If the native
`RandomStream` drives exogenous stochasticity, vary it on TRAIN and keep it fixed on
DEV/HELDOUT for paired comparison. Confirm natively which stochastic processes it controls.

**9. Throughput overhead is in the learner and bridge, not the simulator.**
Replay verification runs after every training episode (`pilot.py:1130`). Reset captures a full
SaveCodec state for diagnostics (`BridgeProgram.cs`, reset handler). Training runs on a single
worker, although 8 workers measured 4.8× in pilot v1. The native step runs at about 190
transitions/s single-threaded. A 5–10× end-to-end gain looks attainable without touching the
simulator.

### P2: process

**10. The §36 escalation order was skipped in practice.** §36 orders diagnosis as:
observation → reward/return → discount/GAE → **rollout/critic accuracy** →
exploration/coverage → capacity. The campaign went from normalization straight to
activation, LayerNorm and clipping ablations, which sit at the capacity/conditioning end. It
never addressed rollout length or critic accuracy, and never ran §36 gate B/C fixtures, the
§41 regime sandbox, or the D34-DISCOUNT-01 H=400/H=800 comparison. None of these exist in
the repo.

**11. Gates are not power-analyzed.**
Promotion and arm-selection gates have no confidence intervals, no minimum detectable effect,
and no correction for repeated use of DEV.

## What the reports got right and wrong

Got right:

- Requested-action likelihoods, structural masking and inactive-head handling.
- Train-only normalization, partition isolation, consuming HELDOUT only once, and immutable
  provenance.
- Not escalating to a GRU, Transformer or privileged critic.
- ReLU is not a fix for credit assignment.
- Normalization-v3 is not the blocker.

Overstated or unsupported:

- "Field behavior became state-conditioned, so the MLP is representationally adequate." The
  v3 evidence shows near-constant policies. Whatever conditioning appeared in later arms was
  learned from a few thousand samples at SNR ≪ 1, and representational adequacy needs a
  direct test (Phase 0b).
- "Critic LayerNorm is justified and should remain." It is harmless, but the root cause is
  unnormalized value targets.
- Arm-to-arm conclusions on LR, reward scale and LayerNorm are within noise.
- "Seed 2101 is the failure." Seed identity mostly determines which constant action the
  policy drifts to.

## Remediation plan

Each phase has a go/no-go that decides whether the next phase runs. None of this changes native
physics, reward meaning, observation or action schemas.

### Phase 0: measure headroom and representability (no PPO; hours, CPU)

**0a. Oracle headroom.** On TRAIN and DEV, run open-loop constant-action grids per snapshot:
field operation × Development budget on a 5×5 grid. Use 3 decisions first, then the longest
scope-safe horizon per snapshot. Add a greedy per-decision search with 3 decisions × 25 actions.

- Report best-of-grid return vs Option A per family, with paired bootstrap confidence intervals.
- Cost: 25 × 256 snapshots × 3 decisions ≈ 19k transitions, about 20 minutes at today's
  throughput.
- **Go** if the headroom over Option A clearly exceeds the noise floor in multiple families.
- **No-go** if it does not. Then the cultivation-only, 3-decision task cannot support the
  promotion question, and you go straight to Phase 2.

**0b. Supervised representation probe.** Using the per-snapshot best grid action from 0a as a
diagnostic label, which §43.1 permits, fit:

- the existing actor MLP;
- a non-neural reference (gradient-boosted trees or kNN) on the same normalized 512 features.

Train on TRAIN and evaluate on DEV, scoring the regret of the predicted action under the true
simulator. This is the most direct test of the question "is the MLP adequate?" and it takes PPO
out of the loop.

- If the MLP fits TRAIN but generalizes worse than the tree model, that is an inductive-bias
  signal (see the architecture section).
- If neither generalizes, the observation lacks the information (§36 gate A). Fix the
  observation, not the network.

### Phase 1: repair the instrument (learner-only changes)

1. **Batch size and budget.** Use at least 512–2,048 transitions per update, 2–4 epochs,
   minibatch 128–256, and a target KL of about 0.02. Add a positive entropy coefficient or
   floor `log_std`. Freeze a budget of at least 10⁵ transitions per seed after measuring
   throughput.
2. **Group-relative advantages.** For each sampled TRAIN snapshot, run K = 4–8 on-policy
   rollouts with common random numbers. Use a leave-one-out group mean as the baseline for the
   first decision, and the critic for later decisions. This is still on-policy and still scores
   the requested action. It removes between-state variance and the population-scale imbalance
   without changing the reward.
3. **Value-target normalization** (running return statistics or PopArt) and **separate
   actor/critic gradient clipping** by default.
4. **Throughput.** Sample replay verification on TRAIN (for example 1 episode in 20, keeping
   100% on eval). Make reset diagnostics optional. Run 4–8 parallel bridge workers.
5. **Statistics.** Report paired bootstrap confidence intervals over snapshots and seeds.
   Predeclare a minimum detectable effect, and set gates above the measured noise floor.

**Go to Phase 2** when, on the 3-decision task, PPO approaches the Phase 0a oracle on TRAIN and
seeds agree within CI. Do not wait for it to "beat Option A".

### Phase 2: restore the spec's decision problem

1. **Horizon.** Move toward the spec's 20–50 decisions. Two options:
   - **(a) Recommended, and the spec-intended path:** admit husbandry. Herd heads activate
     through the existing structural masks (§42.3). Remove the fail-fast on herd activation
     and keep fail-fast only on movement and fishing, which remain out of scope.
   - **(b)** Create a new pinned configuration cohort in which husbandry cannot emerge during
     the horizon. This is a new experiment generation (§35 pinned-cohort rule), not a physics
     edit.
2. **Evaluation.** Score long-horizon welfare plus at least one horizon of continuation
   (gate E). Keep the 3-decision score as a secondary metric only.
3. **Spec gates that have not been run.** Build the §36 gate B/C nonlinear-investment fixtures
   and run H=400 vs H=800 (D34-DISCOUNT-01) before trusting critic-driven investment.
4. **Coverage.** Create a fresh HELDOUT generation, and densify achieved F03, F15
   (delivered-trade) and F17 contrasts, as the v3 report already recommends.

### Phase 3: the §41 regime sandbox, then movement

Run these unchanged from the spec once Phase 2 shows stable learning.

## Architecture decision tree

No architecture change now. A change becomes justified only through these specific outcomes,
in this order:

| Evidence | Implied change |
|---|---|
| 0b: MLP cannot fit TRAIN oracle actions (underfits) | Capacity grid from §36 (128², 256², 256³). Expected to be unlikely. |
| 0b: MLP fits TRAIN but generalizes clearly worse than a tree/kNN model on identical features | Inductive-bias issue in the flat 512-slot layout. Per §30, consider per-row set encoders for site, commodity and contact rows feeding the same trunk. This is the most plausible near-term architecture change. |
| 0b: no model generalizes | Observation defect (§36 gate A). Revise the schema; do not change the network. |
| Phase 2 / gate B: critic cannot rank known delayed-investment paths with correct horizon and coverage | Auxiliary fixed-horizon value heads (§36 Pivot B). |
| §41: paired-history aliasing persists after EMA/event-age features | Time-aware GRU (D36-GRU-01). |
| None of the above | Keep separate 512→256→256 actor and critic. |

Change actor activations or add actor LayerNorm only if a powered ablation run under Phase 1
conditions shows an effect above the confidence interval.

## Immediate next actions

1. Pause the action-credit-audit PPO gates, except as a cheap local-slope check. Its ±0.10
   perturbations around a near-constant policy test local sensitivity, not headroom.
2. Implement and run Phase 0a and 0b.
3. In parallel, implement Phase 1 items 1, 3 and 4. They are small learner-side changes.
4. Record a design decision in the concept doc on two points: the cultivation-only scope versus
   the spec's stage 1, and the evaluation horizon. §38 requires repo deviations from the doc to
   be made explicit.
