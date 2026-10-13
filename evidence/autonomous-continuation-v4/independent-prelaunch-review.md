# Independent v4 continuation prelaunch review

Reviewer: `/root/independent_review`. **GO for the frozen capped CPU continuation described below.** This follows the user's authorization and the completed model-v3 readiness gate; it is not approval for a broader curriculum or unrestricted training.

Independently inspected the v4 runner against the approved v3 runner, cumulative budget validator, deterministic checkpoint-selection helper, selected-checkpoint resume checker, pinned configuration and shared protocol. Independently reran `tests/test_autonomous_continuation.py`: **2 passed**. Tests exercise cumulative time/sample overages, missing history, seed/worker limits, critic extremes, ineligible categories, earliest ties and deduplication.

Verified both historical summary hashes and both parent checkpoint hashes against disk. Historical accounting is **49,152 transitions** (seed 17:32,768; seeds 29/43:8,192 each) and **1,906.765997923 seconds**, leaving **12,493.234002077 seconds**. The runner enforces at most 12,493 further seconds, including its evaluations. It restores model-v3 weights, existing LayerNorm parameters, populated Adam state, sampler/shuffle and episode cursors without conversion; it checks the restored model fingerprint. Pins are checked before native workers start, and output requires a new directory.

The completed parent-native preflight reports exact one-versus-four-worker actions, likelihoods, replay fingerprints, next PPO update, model/Adam/RNG and inference equivalence for both pinned checkpoints, with originals unchanged. These are the same final checkpoint hashes independently rerun and verified during the preceding final review. Model code and native bank are unchanged.

Approved scope: **24,576 additional native transitions each for seeds 29 and 43, 49,152 total; no seed-17 training; four native CPU workers; one Torch thread; 12,493 seconds maximum**. Preserve the fixed bank, source/model/input pins, 4,096-transition evaluations, recovery/replay/numeric/stall/resource pauses, and economic outcomes as diagnostic rather than collapse vetoes. GCS remains a metadata audit; no GCS snapshots enter this run.

Final selection includes entry and saved boundaries: union of final/stopped checkpoint, maximum single actor-or-critic-layer saturation, and maximum eligible categorical confidence; earliest ties, deduplicated. Require all 13 covered retained-Adam tasks and selected-checkpoint native continuation checks on that set. The separate diagnostic ceiling is **six checkpoints, 159,744 discarded draws, 900 seconds per checkpoint**. Failures/timeouts/uncovered tasks are not passes. Preserve historical failures and request a separate final independent readiness verdict after the run.

Reviewed SHA256 values:

- Config `configs/training/autonomous-collapse-continuation-v4.json`: `b84b0026f793370c0cfb2e6be4c6b1fbd72cccf9d2fd1b3759adc8dbd485799f`.
- Runner `scripts/run_autonomous_continuation_v4.py`: `9d3a339fd82c29455170d08b602450c62cd4134623bdae571584f4604fd90714`.
- Budget/selection helper `src/anthropocene_rl/autonomous_v4/continuation.py`: `6912e9db17a56bcae125710f2d24ea83033a7f86ce1e289869aeaf6865b56f31`.

No GPU, main merge, increased worker count, learned-peer/dynasty control, new-world curriculum or unrestricted scale-out is covered by this verdict.
