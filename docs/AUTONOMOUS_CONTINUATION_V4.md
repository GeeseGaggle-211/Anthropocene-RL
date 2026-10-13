# Autonomous Elder capped CPU continuation gate v4

User authorized proceeding after the independent model-v3 gate. This gate keeps model/checkpoint v3 and native action 3/observation 4/environment 4/replay 4 unchanged. It continues existing v3 weights, LayerNorm parameters, every Adam moment, RNG states and episode cursors; it performs no warm-start conversion and no observation-normalizer refit.

The fixed training bank remains `autonomous-ppo-gate-v1`, with survival, trade, migration and fission families and all those mechanisms active. GCS curriculum metadata is audited separately. Unmaterialized snapshots and old five-head observations/actions/normalizers are never mixed into this run.

## Frozen envelope and provenance

`configs/training/autonomous-collapse-continuation-v4.json` pins the actual v3 parent checkpoints, both historical campaign summaries, bank, bridge, model configuration and train-only diagnostic probe. `scripts/run_autonomous_continuation_v4.py` verifies those pins before opening native workers. Historical accounting must equal seed 17=32768, seed 29=8192 and seed 43=8192 additional transitions. Historical native wall time is 1906.765997923 seconds.

Continue seeds 29/43 for at most 24,576 additional transitions each, 49,152 total. Seed17 receives none. Use four native CPU workers, one Torch thread, at most 12,493 seconds including native evaluations, and 60-second native request deadlines. Cumulative native use must remain <=98304 additional transitions and <=14400 seconds. Neither a new run directory nor exact checkpoint restoration resets those limits. Preserve original files and write a new run directory. No GPU, ecology work, main merge or unrestricted training is included.

Capture entry and every 4,096-transition boundary, with 32 fixed development episodes and replay verification. Keep the approved fresh diagnostic: 16 steps, actor LR .001, critic-probe LR .0001, original thresholds. Native PPO retains the exact prior configuration: actor/critic LR .0003, legal mixture .05, reverse-KL coefficient .05. Stop for numeric/replay errors, failed reproducible recovery, actual retained-optimizer stalls, or resource growth under the existing monitor. QoL×population×time, survival and trade success remain visible economic outcomes without economic vetoes.

## Final technical selection and diagnosis

For each seed select the union of: final/stopped checkpoint; checkpoint with maximum saturation of any single actor or critic layer; checkpoint with maximum legal nondegenerate categorical confidence on the fixed probe. Consider entry and all saved boundaries. Break ties by earliest transition count and deduplicate. Selection is independent of economic performance.

Each selected checkpoint undergoes all 13 currently covered retained-Adam reversal tasks, 64 PPO updates × 32 freshly sampled requested actions per task. Preserve the exact-state diagnostic baseline and thresholds: gain>=.01, final signed separation>=.01, critic MSE<.9, endogenous active exposure>=128 and>=16 per label. Insufficient coverage/timeouts are not passes. Cap six checkpoints total, 159,744 discarded draws and 900 seconds per checkpoint. Synthetic diagnostic updates never modify retained policies. Recheck exact one-versus-four native replay/next-update/Adam/RNG/inference equivalence on selected checkpoints.

Independent review is required before native launch and for the final gate verdict. A technical failure is preserved and attributed; criteria are not retroactively changed. Subsequent GCS-derived curriculum expansion requires its own native compatibility, partition and action-mask coverage gate.
