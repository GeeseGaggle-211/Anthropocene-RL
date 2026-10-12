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
- Apparent field variation was not a powered representation test. It did not
  demonstrate MLP adequacy, and it also did not demonstrate MLP failure.
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

## Completed diagnostic boundary

### TRAIN-only action-credit audit

- Status: `ACTION_CREDIT_FAILURE`
- Evidence SHA-256: `c81dbea2490e1efff0267b5eacde1ef1796489c879d296ea1a8cab3997d9c3be`
- Native replay and contract checks passed; DEVELOPMENT and HELDOUT were unread.
- Acceptance responsiveness failed its replicated gate in seeds 1701 and 2101.
- Physical directional credit failed in seed 1701.
- Separate clipping did not meet the replicated escalation gate.

### Five-decision structural and delayed-credit audit

- Status: `DELAY_HYPOTHESIS_NOT_SUPPORTED`
- Evidence SHA-256: `f492ae60f3fe081a11f4e1a9075698cc6c6e6b20ebe2f68d931943993b3ac1c7`
- All 210 TRAIN snapshots passed 14 profiles for five decisions: 2,940 episodes
  and 14,700 transitions with zero scope, replay, request, numerical, or reward
  failures.
- The predeclared delayed Development effect did not replicate in any seed.
- A simple three-to-five-decision PPO extension was not justified.

Published diagnostic objects:

- Manifest: `gs://anthropocene-rl-datasets-508912/runs/ppo-tune-credit-diagnostics-v3/4141b9a-credit-audits-20261012/manifest.json`
- Manifest SHA-256: `08ea35ea4f8086aec66844725b62f9db00f4d18afb35b01e33d012a0024beb92`
- Bundle: `gs://anthropocene-rl-datasets-508912/run-objects/sha256/c5/c58d2d43662758b9a366274bde8669c3fa49a081f21b9b0f9addc4ab7b53e04c.tar`
- Bundle SHA-256: `c58d2d43662758b9a366274bde8669c3fa49a081f21b9b0f9addc4ab7b53e04c`

### Constant-grid headroom audit

- Status: `NO_GO_CONSTANT_GRID_ROUTE`
- Evidence SHA-256: `92e9ee135b651d7328eed27ce42413a6d434e5bdab84e3fc3d8c39dca75bd5f8`
- All 210 TRAIN and 46 DEVELOPMENT snapshots completed the fixed 5×5 grid and
  Option A: 6,656 episodes, zero optimizer updates, no integrity failures.
- Primary three-decision hindsight headroom over Option A:
  - TRAIN mean +0.8913 points, one-sided 95% upper +1.3166 points;
  - DEVELOPMENT mean +0.8118 points, upper +1.2846 points.
- Both upper bounds were below the predeclared +1.5-point useful-effect floor.
- Secondary five-decision headroom was larger (+1.6514 TRAIN, +1.5704
  DEVELOPMENT) but could not change the frozen decision.
- No family qualified. F16 was descriptive only because DEVELOPMENT contained
  one independent cluster.
- The TRAIN-selected global zero-field/zero-Development action was worse than
  Option A on both partitions. Limited headroom was state-dependent.
- HELDOUT remained sealed and F15 remained `UNEVALUABLE`.

Published headroom objects:

- Manifest: `gs://anthropocene-rl-datasets-508912/runs/ppo-tune-headroom-v3/b86740d-constant-grid-headroom-20261012/manifest.json`
- Manifest SHA-256: `b040de8f96782aabc0634fabbdbabcddc0f0c9b0677f667ee16694568a45803a`
- Bundle: `gs://anthropocene-rl-datasets-508912/run-objects/sha256/ff/ff40efb2319cb3bf7df9bde4aa78272c847984758ef1a4fccaef7025be4c4f68.tar`
- Bundle SHA-256: `ff40efb2319cb3bf7df9bde4aa78272c847984758ef1a4fccaef7025be4c4f68`

## Architectural position

No architecture-level decision is required from the project owner at this boundary.

- Feed-forward representational failure has not been demonstrated, but neither
  has adequacy. The predeclared entry gate for a supervised representation probe
  failed, so no architecture conclusion should be manufactured from it.
- Critic-only LayerNorm is supported by evidence and should remain.
- Actor LayerNorm is not supported by current diagnostics.
- ReLU is not preferred for sparse causal events: it does not solve temporal credit assignment and introduces dead-unit risk. Current tanh derivatives remain usable.
- GRU, Transformer, privileged critic, reward change, schema change, and native-physics changes remain out of scope.

## Campaign disposition and proposed continuation

Close the narrow cultivation-only generation as a non-promoted diagnostic. Do
not run another PPO, clipping, value-target, batch-size, activation, or
representation arm on this generation. Preserve its checkpoints, evidence, and
HELDOUT quarantine.

The next proposed action is documented in
`docs/ELDER_50_DECISION_BROAD_DOMAIN_PROTOCOL.md`: create a new experiment
generation with a primary 50-decision horizon and native husbandry and movement
availability. This is a proposal only. Generation requires explicit approval of
the curriculum scope, including fishing treatment; it is not an architecture
decision.

The narrow generation has reached a clean artifact boundary. Its continuation
is stopped because the predeclared remediation and headroom gates did not
promote another run, not because the MLP was shown unsustainable. The broader
50-decision generation begins only after its curriculum scope is approved.

