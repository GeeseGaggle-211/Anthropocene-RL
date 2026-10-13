# Autonomous Elder PPO gate v1 (environment v4)

Status: implementation and bounded CPU pilot; no scale-out authorization.
Supersedes the mechanics-only v3 gate, not the original five-head contract.

## Native boundary
Action 3 / observation 4 / environment 4 / replay 4. Preserve the v3 contract,
manifest and evidence as historical artifacts. Add movement ordinal 0 = stay,
1..7 = the existing knowledge-limited site row + 1. The native legal movement
mask and private tile tokens come from ElderObservationBuilder. Submit movement
and economic allocations together at WEGO; placement, abandonment, escrow,
production, demography, immigration and fission retain native chronology.
No economic rescue. Trade uses opening stock and opening contacts before movement.
Fission and migration remain native endogenous processes, not invented RL physics.
Daughter settlements use the background native controller in this pilot; the
controlled parent remains the reward/accounting unit. This bounds any readiness
claim to the parent-Elder plug-in, not learned dynastic self-play.
Step diagnostics are training telemetry, never actor inputs: per-controlled-camp
population/food/QoL/movement and world trade/migration/fission counts.

## Model and likelihood
Separate actor/critic 256x256 tanh MLPs, orthogonal initialization, following the
existing Elder MLP. Fixed sign-log feature encoding; no evaluation-time fitting.
Masked Dirichlet workforce and conditional activity shares; masked Beta capital
and livestock controls. Density is evaluated on requested shares, before rounding.
Movement is a masked categorical. Trade has conditional categorical heads:
quote enabled, counterparty, offered good, requested different good, stock lot
(5%,20%,50%), requested/offered unit ratio (0.5,1,2). Offered-good mask requires
positive own opening stock and native trade eligibility. No remote inventory.
Quote quantities are deterministic bounded transforms of sampled categorical
choices and own stock; store/score the choices, not their many-to-one wire image.
Inactive heads are canonical zero, contribute zero likelihood, consume no RNG.
Sum the conditional log probabilities. Entropy regularization applies only to
simplex/Beta, movement and enable distributions; do not use a biased downstream
conditional entropy gradient. Every PPO minibatch uses frozen rollout masks.

## Training gate
CPU only; three initialization seeds; at most 6,144 training transitions total
(initial budget 2,048 per seed), complete eight-decision episodes, four curriculum
families: mixed survival/resources, reciprocal stock trade, migration pressure,
fission/exit pressure. All families keep movement/trade/migration/fission enabled.
Use distinct native world seeds for training and development, never optimize on
development trajectories. Mix all families throughout training. Keep the frozen
old model/contracts/datasets/checkpoints untouched. Write new outputs exclusively.
PPO clipped ratio .2, separate actor/critic Adam groups, gradient norm .5,
physical-time GAE (100-year horizon, lambda .95 per 20 years), fixed reward scale
1/120 population units, discounted native population-QoL rate. No event bonuses,
no teacher action substitution, no reward from privileged event telemetry.
Checkpoint model, optimizer, sampler and shuffle RNG, iteration and provenance;
checkpoint only at complete-episode rollout boundaries. Validate exact next
rollout/update continuation in a small native test.

## Predeclared acceptance
All legal-mask/probability/gradient/chronology/conservation/replay and legacy tests
must pass. No illegal actions, NaN/Inf, invalid gradients or checkpoint divergence.
All three seeds must complete the budget with changed actor and critic parameters;
held-out-seed development survival must be >=75% and mean welfare >=80% of its
paired untrained policy across the complete curriculum (not cherry-picked episodes).
At least two seeds must improve paired development welfare; aggregate mean must
improve. Report per-family welfare/survival and physical population change as well.
Witness actual movement, matching trade, migration and fission in curriculum
rollouts (scripted witness episodes may supplement rare endogenous events, clearly
separated from learned-policy outcomes). Independently verify replay for each
family and seed and test exact checkpoint continuation. A reviewer may tighten
these requirements or identify blockers; iterate rather than waive failures.
A passing pilot supports only the next capped scale-out canary with monitoring;
it does not establish broad policy competence or authorize an expensive campaign.

## Frozen protocol and proposed canary
The executable protocol is configs/training/autonomous-ppo-gate-v1.json and the
native generation plan is configs/curriculum/autonomous-ppo-gate-v1.json in the
learner repository. Train world seeds 831001..831003; development 832001..832004,
each with four families. Evaluation uses sampler seeds 9101 and 9102 per world,
32 episodes per initialization, paired before/after, final checkpoint only.
Welfare is cumulative native welfare divided by opening focal population, then
equal-family averaged; extinction contributes zero for missing intervals.
Survival means focal alive at decision eight. Report food coverage and retention
alongside it. Terminal values are zero; truncation bootstraps final value and
cuts GAE carry into reset. Budgets count focal samples, never scripted peers,
replay verification, or duplicate exact-continuation validation.

Proposed subsequent canary (not launched): three seeds, at most 32,768 focal
transitions each (98,304 total), four CPU native workers, one Torch thread, four
hours maximum. Evaluate every 4,096 transitions against immutable pilot finals.
Abort on any illegal/nonfinite/replay error, survival below 75%, paired welfare
below 80% of pilot final, or unexplained >2x memory/throughput deterioration.
A separate promotion decision is required before any larger/GPU campaign.
