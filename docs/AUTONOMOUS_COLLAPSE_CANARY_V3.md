# Autonomous collapse remedial canary v3

Prospective protocol; launch requires independent preflight approval. Earlier v2 canary failures remain failures. The native simulator contract is unchanged.

## Evidence and candidate

The v2 canary consumed 40,960 additional training transitions: 32,768 at seed17 and 4,096 each at 29/43. All three stopped on the fresh critic probe. Preserved trajectories attribute that failure to finite-step Adam overshoot rather than a dead critic. Retained-Adam probes additionally found genuine categorical adaptation failures. Exact detached state baselines remove a substantial source of unrelated score-function variance; they are diagnostic only, not native PPO advantages.

Use model/checkpoint v3, actor-only LayerNorm, epsilon=.05 legal categorical mixture, prospective reverse KL(Ulegal||raw-policy) coefficient .05, actor/critic LR .0003, unchanged continuous heads. Original categorical probabilities, checkpoint versions and five-head legacy policy are preserved. LayerNorm is a selected candidate, not proven superior to v2. The paired .01-to-.05 comparison establishes that output regularization fixes quote-enable adaptation; LayerNorm alone does not.

## Fixed gate

- Frozen bank `autonomous-ppo-gate-v1`: equal survival/trade/migration/fission families; disjoint training/development world seeds; eight native decisions per episode. Movement/trade/migration/fission stay enabled. Native reward remains QoL × population × elapsed time, scaled only for PPO. Welfare and survival are reported without economic vetoes.
- Parent seed29 checkpoint SHA256 `6d32cc2ca4a4ef9abb97f225e58512e08f73fe506f951bfe14ff4f7f13b6a867`; seed43 `2bda685a7f743dfb96ef4a622eeeb776c17f002424a9fdafc6e7a31dfd0eb679`. Both are stopped v2-canary v1-model checkpoints. Preserve original Adam state by exact parameter name/shape, RNG and episode cursors. Recollect all actions under the converted v3 actor.
- Hard cap 4,096 additional native transitions per seed29/43, total8,192; seed17 receives zero further native training. Cumulative campaign use <=49,152 of original98,304. Four native CPU workers, one torch thread, 1,800 seconds maximum and60-second native request deadline. No GPU or scale-out run.
- Preregistered fresh diagnostic:16 steps, actor LR .001, critic LR .0001, original actor loss/separation and critic MSE thresholds. Lower critic probe LR addresses documented overshoot; it does not turn historical failed probes into passes. Actual native critic LR is .0003.
- Preflight: all13 covered seed17 retained-Adam tasks pass at kappa=.05; paired v2/v3 critical tasks retain recovery; both new entry models pass full covered fresh probes; exact1-versus4 native collection and nonempty-Adam resume/inference checks pass. Hash source/config/parent/probe/bank/bridge before launch.
- Report initial/final development32 episodes per seed with replay verification, saturation/derivative/rank, actual mixed-policy probabilities, concentrations/clamps, first-layer gradients, actual optimizer steps, KL/clip, wall time and process RSS. Pause on nonfinite behavior, replay failure, actual optimizer stall, failed fresh recovery, or doubling steady-state memory/latency under the existing monitor. Poor economic choices are not a collapse gate.
- Final13-head retained-Adam tests at both new checkpoints:64 updates ×32 fresh requested actions per task; fixed exact-state-baseline protocol; opposite own initial conditional preference. Require gain>=.01, final signed separation>=.01, critic MSE<.9 and adequate endogenous branch exposure. Uncovered tasks cannot count as passes. No threshold changes after results.
- Separate discarded synthetic diagnostic budget: <=200,000 draws including historical79,872 +6,144, paired ablations, one instrumented replay and final tests. Planned total192,512. These diagnostic updates never enter retained/native policies.

Implementation: `configs/training/autonomous-collapse-canary-v3.json`, `autonomous-ppo-layernorm-v3-recovery.json`, `scripts/run_autonomous_collapse_canary_v3.py`. Exact hashes are emitted before collection. Final independent review determines the next permitted gate; passing this bounded gate is not evidence of economic competence or unrestricted training readiness.
