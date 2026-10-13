# Independent preflight review — collapse canary v2

**GO for the bounded CPU canary using the explicit v2 protocol.** The earlier NO-GO findings are resolved. This is prelaunch approval of the experiment and monitoring mechanics, not a finding that the eventual training run is healthy or economically successful. A separate review of campaign results is required.

## Independently verified

- Native-enabled collapse tests: six passed, covering one-versus-four-worker exact actions, likelihoods, fingerprints, PPO updates and populated-Adam checkpoint continuation; IPC deadline termination; deterministic episode keys; dead/zero-learning-rate controls; partial recipe/control collapse; and residual-qualified critic stall detection. The subsequently added real PPO populated-Adam critic-zero-learning-rate fault-injection test also passed independently.
- Recomputed frozen 192-row probe SHA-256: `1eae1380d17c1f884bc221bf858ce4c2c2f1419ea6d5f56d66fac081ac913308`.
- Independently ran full v2 recovery probes on pilot finals 17, 29 and 43. All passed; critic residual MSE was respectively **0.5104036, 0.3099223, 0.3204957**. Model hashes were unchanged by the probes.
- Repeated the earlier partial-dead counterexample on pilot 29: zeroing recipe/control output weights and setting their biases to 100 still left workforce and critic probes passing, but the revised full gate correctly failed covered activity rows 0/4/5/9 and control 0. Healthy workforce/categorical paths can no longer conceal those dead outputs.
- Reviewed the immutable-policy collector, per-episode keyed RNG, canonical result ordering, independent native processes, bounded IPC deadlines, global wall-clock guard, frozen observation/mask handling, sample caps, recovery repetition, and actual actor/critic update monitoring. No remaining prelaunch blocker found.

The critic probe correction is supported by preserved attribution: the old probe learning rate caused a transient overshoot, while longer runs recovered. V2 freezes a smaller disposable critic-probe learning rate before campaign outcomes; it does not change PPO training rates or lower the pass threshold. Requested-good probe conditioning is independent of model weights, preserving probe faces and targets across checkpoints.

## Mandatory bounds and interpretation

Use `configs/training/autonomous-collapse-canary-v2.json`: at most **32,768 additional focal samples per seed, 98,304 total**, seeds 17/29/43, at most four native CPU processes, one Torch thread, and **14,400 seconds maximum**. Preserve pilot finals, both preflight probe histories, new checkpoints, and failure evidence separately. Evaluate at entry, every 4,096 samples, final boundary, and a diagnostic stop.

Per the user's revised instruction, poor welfare, survival, food coverage or population retention are monitored outcomes, not automatic abort criteria. Do not rescue economically poor legal choices. Stop on illegal/nonfinite/replay failures and enforce the wall/sample bounds. Reproducible recovery failure, actual optimizer stalls, or unexplained resource deterioration must pause collection for attribution. Do not silently relax thresholds or copy recovery-clone changes into the training model.

Recovery probes are bounded operational tests, not proofs of global learnability. Faces with insufficient observations remain explicitly uncovered; in this cultivation-heavy bank some recipe and livestock controls are uncovered. Stable or concentrated behavior alone is not collapse. Future probe failures must distinguish optimizer/probe-budget limitations from dead computation using retained evidence.

This GO excludes GPU or larger training campaigns, dynastic control, curriculum expansion without a new version, and claims of learned trading competence. No canary was launched by the reviewer.
