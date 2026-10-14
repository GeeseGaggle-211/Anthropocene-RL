# Fixed model7 postflight v9.7

This final verification follows the six durable 400-transition v9.6 cells.
It does not collect critic targets or update retained weights. Preserve all
failures, checkpoints, original source pins and the 448 failed v9.5 transitions
(three updates). The archived launch contract corrects a reporting typo only.

For each low-LR checkpoint (101, 211, 307), use the pinned fitting-TRAIN-only
probe and exactly 64 forward plus 64 reversed workforce-density updates on a
disposable clone, inheriting its populated Adam state and actor LR .00003.
Require loss improvement, directional improvement and a strictly positive
signed end gap in both phases. An improved negative reverse gap is not a full
switch. No extension until pass and no retained optimization are authorized.
Hash retained model/optimizer/RNG/counters before and after every audit.

Report health on cached fitting states. Evaluate cached pre-PPO frozen-actor
targets only as distribution-labeled drift: these are not current on-policy
post-PPO critic readiness. The previous calibration validation was reused, and
the seed307 critic derives from donor101; do not claim independent validation
or three independent critic initializations.

Run two separate learned model7 Elders (101 and211) on the pinned fitting TRAIN
trade fixture at four common WEGO boundaries, then repeat with a save/load after
boundary2 and exact restoration of both independent sampler RNG streams. Each
MLP receives only its owner's observation. Compare actions, realized staffing,
local rewards, receipts, observations and fingerprints; verify both full and
split replay. Charge exactly8 attempted native and8 replay resolves.

Chain the actual27935 resolves /906.3314627960208 active elapsed seconds. The
prospective result reaches at most27951, leaves40 under the existing27991 ceiling,
and uses the same2369.4072460189927-second cumulative cap and one Torch thread.
Freeze source/data/checkpoint/DLL pins and obtain Astra/high prospective review.
No new retained PPO campaign, GRU, reward/physics changes, main merge, HELDOUT
evaluation or GPU ecology work occurs. Final broad promotion remains conditional
on economic evidence; passing mechanics alone clears a bounded learning gate.

The final learner regression found an IPC teardown race: the timer killed a
native process but could return EOF before its exit was reaped. The original
canary source is archived. Timeout/close cleanup now kills and waits at most
five seconds for exit. This changes failure cleanup, not policy distributions
or successful native requests.
