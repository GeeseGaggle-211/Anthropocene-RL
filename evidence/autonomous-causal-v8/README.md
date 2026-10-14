# Reproducing the bounded causal gate

Canonical contracts/results are in `docs/AUTONOMOUS_*V8*.md`; the final simulator
and checkpoint contract is `configs/simulator/elder-autonomous-model-v6.json`.
The native branch's manifest describes the same code by source hash; the learner
manifest additionally pins the actual simulator commit, avoiding a self-referential
commit hash in the native manifest.

Raw source snapshots, checkpoints, binary builds and native trajectories remain
under ignored `datasets/`, `runs/` and `integration/history-v*/` paths. They were
not overwritten. `artifact-index.json` anchors raw result/config/archive bytes;
compact summaries omit full trajectories. Checkpoint hashes are in `matrix.csv`.
Generate data/binaries from their versioned source contracts if an artifact is
absent; an absolute path in a historical plan is a recorded workspace location,
not a request to substitute a new binary or source silently.

Failed calibration, fixture and initial-matrix artifacts remain alongside their
successful iterations. `ledgers.json` includes both failed and completed phases;
postflight conservatively reserved full tool invocations before launch. Total
resolves include replay and failed attempted calls. Active elapsed wall time sums
active stages; it is not summed CPU-core time across four workers.

Source pins are frozen at actual launch. The pre-resume GAE driver and runner
are preserved in `source-at-launch/causal.py` and
`source-at-launch/run_autonomous_causal_gate_v7.py`. The original matrix's
native6 contract/adapter sources are preserved in
`source-at-launch/contracts_v6.py` and `source-at-launch/env_v6.py`; current
versions add exact name/control/save schema guards without changing outcomes.
Use those originals to verify historical pins, not the subsequently hardened
files at their former source paths.

`matrix.csv` labels integrated welfare as the sum of two500-year DEV episodes
and separately reports the mean per episode. Annual welfare divides the sum by
1000years. The report table averages across three continuation sampling seeds;
those seeds are not independent model initializations. Cold-head control ranges
describe T(E[latent]), not E[T(latent)] after endpoint clipping. Mature-head
inputs are frozen pre-PPO TRAIN states, not new on-policy evaluations.

The v9 plan is an unexecuted proposal. Its inputs, critic targets, operational
checks and total resource envelope still require a concrete frozen preflight
and independent Astra/high GO. This evidence authorizes no automatic scale-out.
