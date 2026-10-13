# Autonomous PPO actor LayerNorm ablation v3

Status: experimental comparison, not a training promotion. Native action3/observation4/environment4/replay4 and the original five-head contract remain unchanged.

The user's instruction permits trying actor LayerNorm if evidence warrants it. Retained v1 checkpoints have actor tanh saturation up to 83% and multiple heads fail to adapt in a 64-update retained-Adam diagnostic. LayerNorm may preserve state-dependent gradients; it cannot alone prevent large output-logit gaps or guarantee economic competence.

Candidate model `autonomous-ppo-model-v3` inserts LayerNorm(width=256, epsilon=1e-5, affine=true) between each actor linear layer and tanh. The critic is unchanged. Existing linear and head names/weights and every populated Adam moment/step are preserved by name and exact shape; only four LayerNorm affine tensors start fresh. This is explicitly a behavior-changing conversion, never an exact policy continuation. New rollouts always score actions under the converted model. Checkpoint and inference version are v3; v1 and v2 remain readable only by their explicit loaders.

Other candidate settings match v2: legal categorical mixture 0.95 softmax + 0.05 uniform; exact conditional joint entropy; 0.01 prospective legal reverse-KL coefficient; actor and critic learning rates 0.0003. No stronger regularizer, continuous-output penalty, critic LayerNorm, or changed native economics is included.

## Frozen comparison protocol

Use the retained stopped checkpoints, the same train-only 192-row probe, and independent disposable copies. `probe_autonomous_state_baseline.py` defines `retained-ppo-state-baseline-v2`. For each task, select the same largest legal face, feature-ranked 32 observations, first two legal choices, and balanced labels oriented against the original v1 model. All models receive identical labels. Report initial and final separation: converted v3 need not initially oppose those labels.

Each comparison uses 64 updates of 32 newly sampled requested actions and actual PPO/Adam state. Advantage subtracts detached exact expected task reward conditional on the observation. The expectation includes endogenous quote enable and offered-good gates, so downstream samples are never forced. This removes state-only label variance without changing the expected policy gradient. The historical global-baseline results remain frozen. Critic targets are original values plus balanced labels. Thresholds remain gain >=0.01, final signed separation >=0.01, critic MSE <0.9; branch exposure >=128 total and >=16 per label. Timeouts or insufficient legal faces are not passes.

Start with movement, quote enable, and control_0 at seed17 (three variants, 18,432 synthetic draws). Count all subsequent discarded comparisons toward a separate 200,000-draw diagnostic budget including the previous 79,872 v1 and 6,144 v2 tests. No native training transitions are consumed by this diagnostic. Native canary remains paused; remaining authorized native budget is 28,672 each for seeds29/43 and zero for seed17.

Measure exact entry KL (v2 to v3) across the full conditional action distribution, categorical entropy, continuous means, and activation statistics before learning. Evaluate pre-learning conversion in the frozen native bank, and verify fresh likelihoods and populated-state replay. Adoption requires improved response to learning targets and retained categorical recovery, not merely reduced saturation. Independent review is required before native continuation or scale-out.
