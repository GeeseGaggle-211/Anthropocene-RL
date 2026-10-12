# Cultivation PPO tuning handoff

Updated: 2026-10-12

## Repository identities

- Repository: `GeeseGaggle-211/Anthropocene-RL`
- Working branch: `feature/cultivation-pilot-tuning`
- Upstream learner state at the start of this tuning sequence: `0cd174fde4c77f0eb03b5541c1729e461b175677`
- Last committed learner state before the action-credit audit implementation: `0853ac2f6e8a239199352911c60f2249e3014e7d`
- Authoritative native simulator: `GeeseGaggle-211/Anthropocene@5559961164d6f9da8e6d20c1dc563ecec3eb04b7`
- The native simulator remains frozen. No economics, ecology, reward, observation, action, or physics changes were made to obtain these results.

The branch head containing this document is the current source handoff. Large evidence, checkpoints, and snapshot bytes remain outside Git.

## Frozen dataset and normalization identities

- Accepted snapshots: 300
- Family counts: F01 30, F02 35, F03 35, F13 40, F15 50, F16 40, F17 24, F18 46
- Partition counts: TRAIN 210, DEVELOPMENT 46, HELDOUT 44
- Frozen manifest SHA-256: `1cfabbc6057c063d2bca8d27c4bda861a090d5811e2b02f3fe51adc9fd2a6b83`
- Final TRAIN-only normalizer SHA-256: `6563cea5d26a54f5914b666c28584066eb18a681e5fad0559a10207a78fa01c4`
- Dataset manifest: `gs://anthropocene-rl-datasets-508912/datasets/elder-v2-cultivation-pilot-normalization-v3-scope-safe/2026-10-12-manifest-1cfabbc6/manifest.json`

The v3 normalizer remains healthy. Its observed saturation is low, and neither DEVELOPMENT nor HELDOUT was used for fitting. There is no current evidence for another normalizer, actor LayerNorm, or an activation change.

## Completed tuning sequence

All arms retained the separate feed-forward Elder actor and critic. The accepted architecture remains:

- actor: `512 -> 256 -> 256 -> active hybrid heads`
- critic: `512 -> 256 -> 256 -> scalar V`
- critic-only LayerNorm enabled after the control arm demonstrated critic saturation
- actor activation: tanh
- movement disabled; herd and fishing domains structurally inactive
- requested bounded actions used for PPO likelihoods; accepted and realized actions kept as environment outcomes

### Optimizer control

Evidence: `evidence/cultivation-ppo-tuning-v3-optimizer-control`

- DEVELOPMENT relative to Option A: -7.41%, -4.44%, -6.26%
- HELDOUT relative to Option A: -6.68%, -3.23%, -5.83%
- Diagnosis: critic saturation; no steady increase in cultivation intensity

### Critic LayerNorm

Evidence: `evidence/cultivation-ppo-tuning-v3-critic-layernorm`

- DEVELOPMENT: -7.39%, -4.18%, -6.94%
- HELDOUT: -6.74%, -3.16%, -6.26%
- Diagnosis: critic conditioning repaired, but policy performance remained below baseline

### Critic LayerNorm plus reward scale `1e-4`

Evidence: `evidence/cultivation-ppo-tuning-v3-critic-layernorm-reward-scale-1e-4`

- DEVELOPMENT: -8.98%, -5.84%, -7.92%
- HELDOUT: -9.26%, -5.00%, -8.59%
- Diagnosis: critic remained healthy; actor optimization was poorly conditioned

### Actor learning rate `3e-4`, 96 updates

Evidence: `evidence/ppo-tune-v3-actor-lr-3e-4`

- DEVELOPMENT: -4.94%, -3.21%, -3.87%
- F03: -12.36%, -16.91%, -14.39%
- Field behavior became state-conditioned, so the feed-forward MLP had not failed representationally.
- The seed-1901 tanh diagnostic crossed its historical trigger, but its derivative remained healthy. This was not evidence for ReLU, SiLU, or actor LayerNorm.

### Duration extension, 192 updates

Evidence: `evidence/ppo-tune-v3-actor-lr-3e-4-updates-192`

- All three runs exactly reproduced their update 1-96 prefixes.
- All selected checkpoints were at update 168.
- Selected DEVELOPMENT: -4.12%, -2.25%, -3.12%
- Terminal DEVELOPMENT: -3.89%, -2.16%, -2.86%
- Gain over update 96 was approximately +0.75 to +1.05 percentage points, below the predeclared +1.5 point gate in every seed.
- Selected critic explained variance: 0.924, 0.896, 0.962
- Maximum PPO KL: 0.144, 0.111, 0.112; all remained below the 0.15 campaign ceiling.
- Requested field operation changed from 0.4865 to 0.3782 across the campaign.
- Requested Development budget changed from 0.5071 to 0.2993.
- Neither action showed a steady increase. A rising cultivation level was not itself the success criterion; the desired behavior is context-sensitive use of cultivation within the open economy.
- The duration gate failed, so the campaign remains `NO_GO`.
- HELDOUT was deliberately not read for this tuning arm.
- F15 trade contrast remains `UNEVALUABLE`; the evaluation set did not produce enough contrasting delivered-trade cases to support a self-sufficiency claim.

Published run objects:

- Manifest: `gs://anthropocene-rl-datasets-508912/runs/ppo-tune-actor-lr3e4-u192/0853ac2-actor-lr3e4-u192-20261012/manifest.json`
- Manifest SHA-256: `18a1117a1c27175f994926a9ca62cc48ae2fbc868e0c8028ba9736ea2073487b`
- Immutable bundle: `gs://anthropocene-rl-datasets-508912/run-objects/sha256/f1/f16c16194e896308c2ca5d7d727c02379750d742c54936dfd3c44a2d99b49145.tar`
- Bundle SHA-256: `f16c16194e896308c2ca5d7d727c02379750d742c54936dfd3c44a2d99b49145`
- Bundle size: 44,083,200 bytes

## Current diagnostic boundary

An Astra high review recommended one bounded TRAIN-only action-credit audit before any further tuning. The branch therefore adds:

- paired +/-0.10 interventions for field operation and Development at every decision;
- exact requested, accepted, realized, compiler-intervention, physical-return, and bootstrap evidence;
- a fixed balanced batch of 32 TRAIN episodes per seed, four from each family and three decisions per episode;
- a discarded one-update shadow comparison of the existing joint actor/critic gradient clipping against separate actor and critic clipping;
- replicated, predeclared campaign thresholds across neural seeds 1701, 1901, and 2101.

The audit never reads DEVELOPMENT or HELDOUT, never changes the simulator, never substitutes accepted actions into PPO likelihoods, and never saves either shadow policy.

The interpretation gates are:

- `ACTION_CREDIT_FAILURE` if acceptance or physical credit is demonstrably unresponsive in at least two seeds;
- `JOINT_CLIPPING_ABLATION_JUSTIFIED` only if separate clipping produces at least 1.5x actor displacement, a better clipped surrogate, KL at or below 0.05, and no more than 10% critic-normalized-MSE deterioration in at least two seeds;
- `CLEAR_DIAGNOSTIC_PASS` only if acceptance responsiveness, directional agreement, and clipping invariance all replicate in at least two seeds;
- otherwise `INCONCLUSIVE`, with no new PPO arm justified.

## Architectural position

No architecture-level decision is required from the project owner at this boundary.

- The MLP has learned state-conditioned field control, so a feed-forward representational failure has not been demonstrated.
- Critic-only LayerNorm is supported by evidence and should remain.
- Actor LayerNorm is not supported by current diagnostics.
- ReLU is not preferred for sparse causal events: it does not solve temporal credit assignment and introduces dead-unit risk. Current tanh derivatives remain usable.
- GRU, Transformer, privileged critic, reward change, schema change, and native-physics changes remain out of scope.

## Continuation rule

Finish the three-seed TRAIN-only action-credit audit, preserve and publish its evidence, and obtain another Astra high review. Start another PPO arm only if a predeclared audit gate identifies a causal remediation. Otherwise stop at the clean audit boundary and report the result as inconclusive rather than searching hyperparameters.

The overall goal terminates only when one of these conditions is met:

1. the current MLP is shown unsustainable and the RL technology document plus escalation gates are updated;
2. a bounded, replicated tuning remediation succeeds; or
3. local disk capacity prevents safe continuation.

