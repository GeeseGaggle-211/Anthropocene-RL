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

## Still required before a serious campaign

- run and archive native schema/reset/step/replay and focused Elder regressions
  on the exact simulator SHA;
- build/freeze the training/development/held-out snapshot bank;
- fit and archive normalization-v2 on the training partition;
- run the persistent bridge and Gym adapter against real snapshot bytes;
- pass the tiny-snapshot PPO mechanics gate;
- benchmark environment throughput and decide worker/batching counts;
- freeze checkpoint contents and experiment manifests.

GRU, Transformer, online gameplay SGD, the sparse event scheduler, later-era
goods encoders and multi-agent RLlib scale-out remain deliberately deferred.
