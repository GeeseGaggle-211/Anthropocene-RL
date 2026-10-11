# Section 34 / 42 learner bring-up gates

The Google Drive design document is normative. This file records the current
implementation status in Anthropocene-RL.

## Frozen native boundary

The learner pins `GeeseGaggle-211/Anthropocene` at
`5559961164d6f9da8e6d20c1dc563ecec3eb04b7`:

- observation schema 2;
- action schema 1;
- environment/replay schema 2;
- critic-sidecar schema 1;
- tensor width 512.

Native v2 already implements the observation/time/memory/reward fixes. Do not
recreate them in Python.

## Settled v1 learner contract

- separate 512 -> 256 -> 256 actor and critic MLPs;
- active continuous heads use `a=(tanh(z)+1)/2` with transformed-density
  log-probability;
- native structural bits determine field retention, herd retention, field
  Development and herd Development availability;
- Development budget is active when either Development channel exists;
- field Development share is active only when both channels exist;
- inactive continuous heads contribute zero policy log-probability and entropy
  and use the native canonical value;
- movement is a masked categorical over stay plus seven candidate rows;
- PPO likelihood is attached to the bounded requested action; native
  accepted/realized actions are outcomes;
- movement is disabled in the first optimizer smoke;
- the training-only native frontier sidecar is excluded from the baseline
  actor/critic and is an optional later asymmetric-critic ablation.

## Normalization-v2

Normalization is fit only from the frozen training partition and serialized with
the checkpoint. The implementation performs semantic reference scaling, then
log1p/asinh/bounded transforms, then frozen per-feature tanh scales for
unbounded features. Binary masks remain binary. There is no running inference
normalizer.

## Physical-time credit

The native simulator emits welfare rate, undiscounted interval welfare and
elapsed years. The learner owns discounting and GAE. H=400 is a provisional
mechanics-smoke setting only. H=400 versus H=800, lambda_ref, rollout length and
bootstrap conventions remain explicit experiments after the loop is healthy.

## Completed native handoff validation

The exact-SHA schema/reset/step/replay commands, 143 focused Elder assertions,
and a persistent Gym/bridge Reset/Step run against actual native `SaveCodec`
snapshot bytes pass. Their hashes, fingerprints and selected learner commit are
archived under `evidence/elder-v2-handoff/`.

## Completed diagnostic snapshot-bank gate

The reproducible Stage-0 bank contains 28 native SaveCodec states across TRAIN,
DEVELOPMENT, and HELDOUT. Its committed manifest is
`datasets/manifests/elder-v2-stage0-v1.json`; generation, coverage, audit, and
provisional TRAIN-only normalization evidence is under
`evidence/stage0-snapshot-bank/`. Snapshot blobs remain gitignored and
content-addressed. This is an integration diagnostic, not the final campaign
bank or final normalizer.

## Implemented one-snapshot mechanics gate

`anthropocene-rl overfit-one-snapshot` now runs the existing masked hybrid actor,
critic, physical-time reward/GAE, and PPO optimizer against one real
cultivation-only TRAIN snapshot through a persistent native bridge. Movement is
disabled. The gate checks critic return fit, deterministic actor movement,
inactive-head gradient inertness, legal likelihood support, separate
requested/accepted/realized channels, intervention trends, finite normalized
observations, checkpoint/reload parity, and native replay. It is explicitly not
a gameplay-performance test.

The checkpoint includes model and optimizer state, all RNG states, the frozen
normalizer, run configuration, snapshot/manifest identities, simulator and
schema identities, and the runtime bridge digest. Native Step, policy inference,
PPO update, and end-to-end timing are reported separately. A companion
`benchmark-native-throughput` command compares 1, 2, 4, and 8 persistent native
environments without changing the action or reset semantics.

The hardened gate samples only structurally active Gaussian heads, aggregates
PPO diagnostics over every minibatch, and recomputes KL, clip fraction, entropy,
and value MSE over the full rollout after each update. A target-KL stop freezes
further actor changes for that rollout while allowing the separate critic to
finish its scheduled value-fitting epochs. The actor and critic use distinct
optimizer parameter groups so the actor can take the conservative 3e-4 step
while the one-snapshot critic uses 1e-3. Checkpoint verification restores RNG
and optimizer state and proves the next stochastic policy sample and next PPO
update are bit-exact, rather than checking deterministic inference alone.

## Cultivation-only readiness matrix

`anthropocene-rl cultivation-readiness` repeats the real native gate across F01
(young/low-q), F03 (expensive/high-q), and F13 (amenity-constrained)
cultivation-only TRAIN snapshots and neural seeds 1701, 1901, and 2101. This is
a 3 x 3 independent-replica matrix; it does not pool partitions or alter the
learner objective. Movement, herding, and fishing remain disabled. The matrix
requires every contract, replay, numerical, checkpoint, critic-fit, and PPO
stability gate to pass; actor movement and positive fixed-policy return signal
must replicate across seeds. Flat-return snapshots may demonstrate a lower
compiler-repair burden, but that is reported separately and is not called
gameplay improvement.

Passing this matrix supports retaining the separate 512 -> 256 -> 256 actor and
critic MLPs for the bounded multi-snapshot pilot. GRU, Transformer, and
privileged-critic experiments remain evidence-triggered follow-ups rather than
preconditions for adding the next action capabilities.

The preserved 3 x 3 artifacts have been independently re-audited: all nine
diagnostic hashes, checkpoint byte hashes, checkpoint identities, native SHA,
and per-run/aggregate gates match. The result remains a cultivation mechanics
and optimizer-readiness PASS, not a gameplay-performance claim.

## GCS-backed dataset and run cache

GCS is the durable backend for new snapshot banks, diagnostics, and checkpoints.
Content-addressed immutable shards use create-only object generations; provider
CRC32C/MD5 and application SHA-256 metadata are verified, and download admission
recomputes SHA-256. The learner uses a 5 GiB LRU cache and downloads a shard once
on the first miss rather than streaming reset bytes per episode. The prior Drive
store remains a historical compatibility backend.

## Cultivation-only pilot result

The frozen v1 bank contains 300 snapshots across F01/F02/F03/F13/F15/F16/F17/F18
and is stored in GCS. The corrected three-decision, three-seed PPO campaign is a
NO-GO: seed 1701 passed, seed 1901 breached the KL gate, and seed 2101 regressed
against Option A on DEVELOPMENT and HELDOUT. Contracts and inactive heads stayed
clean. No feed-forward representational failure was demonstrated.

## Still required before promotion

- create a new bank generation with pristine held-out groups;
- increase sparse trade/migration event coverage in TRAIN calibration;
- fit and archive a new TRAIN-only robust sparse-event normalizer;
- rerun three seeds with a more conservative actor update and an explicit
  compiler-intervention-distance gate.

GRU, Transformer, online gameplay SGD, the sparse event scheduler, later-era
goods encoders and multi-agent RLlib scale-out remain deliberately deferred.
