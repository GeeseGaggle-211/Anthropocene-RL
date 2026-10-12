# Elder 50-decision broad-domain campaign handoff

Date: 2026-10-12

## Purpose

This branch executes the first powered Elder PPO campaign over cultivation,
husbandry, movement, trade, migration, fission, institutional history, and
productive-frontier variation. Fishing, synthetic disasters, recurrence,
privileged critic inputs, and native semantic changes remain out of scope.

The five-decision value in the dataset generator is only the maximum search
window used to align natural F15 delivered-trade snapshots. Training and every
serious evaluation are exactly 50 decisions, or nominally 1,000 simulated years
at the native 20-year time step. The learner rejects a broad-domain run that is
configured for any other episode length.

## Frozen inputs

- Native simulator: `GeeseGaggle-211/Anthropocene@5559961164d6f9da8e6d20c1dc563ecec3eb04b7`
- Frozen bank identity: `10ab09c73942cc3e2bea0bef409c6682a909d07022b0fb1d529d3b566979e736`
- Accepted bank: 499 snapshots (347 TRAIN, 73 DEVELOPMENT, 79 HELDOUT)
- Generator configuration identity: `a9cb4620ecbff98053c3cda18b25bde91ed707735c7ef196ffc4c30aabbe623e`
- TRAIN-only normalizer file: `evidence/elder-broad-domain-50d-v2/bank/final-train-normalization.json`
- GCS dataset: `gs://anthropocene-rl-datasets-508912/datasets/elder-v2-broad-domain-50d-v2/2026-10-12-manifest-10ab09c7/manifest.json`
- Neural seeds: 1701, 1901, 2101

## Learner configuration

The exact declaration is `configs/training/elder-broad-domain-50d-v2.yaml`.

- Separate `512 -> 256 -> 256` Tanh actor and critic; no shared trunk.
- Actor LayerNorm off; critic LayerNorm on.
- Running PopArt-style value-target normalization.
- Actor learning rate `3e-4`; critic learning rate `1e-3`.
- Separate actor/critic gradient clipping; target KL `0.02`.
- Positive entropy coefficient `0.001`.
- First-layer actor and critic columns with zero nonzero support in PILOT TRAIN
  are initialized to zero. They remain trainable if a longer trajectory later
  activates a husbandry, movement, contact, or other sparse input.
- One base TRAIN snapshot per family, four on-policy trajectories per snapshot,
  and a leave-one-out group-relative advantage at the first decision.
- 40 updates, up to 104,000 transitions per seed.
- Eight persistent native bridge workers for TRAIN collection and up to eight
  workers for evaluation; TRAIN replay is sampled every 20 episodes and
  evaluation replay remains exhaustive.
- DEVELOPMENT selects checkpoints. HELDOUT is evaluated only on the final
  selected checkpoint and is not used for optimizer or configuration changes.

The executable records all four continuous-head log standard deviations,
actor-mean endpoint rates, critic value-head norms, clip fraction, KL and
target-KL early stops on every update. DEVELOPMENT evaluations additionally
measure reset-value explained variance against the realized 50-decision
discounted return. Presence-bit or absent-value encoding changes remain a
future schema generation and are not mixed into this frozen cohort.

The command-line forwarding of `rollouts_per_snapshot` and
`group_relative_first_decision` has a regression test. A superseded K=1
preflight seed exposed the missing forwarding before the powered campaign and
is retained only as a diagnostic, never as campaign evidence.

The frozen native bridge supports only `RandomStream=0`; keyed randomness is
owned by snapshot state. This was confirmed by an explicit clean interface
rejection during preflight, not a game crash or native defect.

Build the bridge with `scripts/bootstrap_simulator.py --root <pinned-native>`;
the bare C# project intentionally has no self-contained simulator reference.

## Preflight result

The eight-worker 50-decision smoke collected 2,577 transitions across all 13
families in about 48 seconds. It had zero illegal actions, requested-action
mismatches, non-finite values, replay failures, reward-denominator violations,
or inactive-head gradient leakage. PPO post-update KL was `0.0103`, clip
fraction was `0.149`, and the group-relative first-decision advantages were
centered to numerical zero.

## Storage caution

GCS upload and provider metadata verification passed for both immutable
dataset shards. Large-object downloads from this workstation currently stall
at zero bytes in both the Python client and `gcloud`, while the small manifest
downloads normally. Do not prune `datasets/blobs` until a real shard re-fetch,
SHA verification, and transparent record materialization have succeeded.

## Completion boundary

At campaign completion, preserve the exact learner SHA, all three checkpoint
hashes, learning/evaluation curves, family/world comparisons with Option A,
throughput metrics, cultivation-intensity trajectory, self-sufficiency signal,
and go/no-go assessment. Publish large run artifacts and checkpoints to GCS;
commit only compact evidence and provenance.

## Powered PoR completion and training disposition

The 40-update, K=4 campaign completed on all three neural seeds at learner
commit `91c5ccee0a35a03c1df60ecdd79c43c833593a6f`. It collected approximately
103,600 transitions per seed. Every contract, replay, requested-action,
inactive-head, numerical, checkpoint, scope, and PPO stability gate passed.
The immutable run bundle is
`gs://anthropocene-rl-datasets-508912/run-objects/sha256/7e/7e1baeefa245b17119bb91971e89ff6e5a7d82ce62fb901b2e5a271707fe240e.tar`.

This result is proof of readiness, not a promoted welfare result. Full
DEVELOPMENT comparison with Option A was:

| Seed | QoL difference | Population difference | Disposition |
|---|---:|---:|---|
| 1701 | -2.97% | -13.59% | population outside the +/-10% target |
| 1901 | -7.70% | -13.23% | population outside the +/-10% target |
| 2101 | -4.59% | -0.50% | inside the target on both measures |

There is no observed architecture-collapse signal. Actor and critic hidden
probes remained below the declared Tanh saturation trigger, continuous-head
log standard deviations remained near `-1`, actor means were not stuck at
endpoints, and critic DEVELOPMENT explained variance was positive. The policy
also moved during training; it was not excessively stable. The unresolved
issue is therefore duration and cross-world generalization, not a demonstrated
failure of the separate feed-forward MLP.

No further disposable short PPO pilots are planned. The declaration in
`configs/training/elder-broad-domain-50d-full-v3.yaml` resumes the exact PoR
model, optimizer, PopArt, and RNG states and treats 500 as the total update
budget. Full DEVELOPMENT is evaluated every 20 updates. Checkpoint selection
minimizes the worse of the absolute QoL and population gaps relative to Option
A, and success requires both gaps to remain within 10% on two consecutive new
checkpoints for every seed. HELDOUT remains sealed because v2 already consumed
it.

Additional scenario curricula belong to a new frozen manifest generation with
a group-safe split, native audit, and TRAIN-only normalizer. They do not require
a separate short PPO pilot per scenario: the first checkpoints of the same
full training run serve as retained safety gates. A hidden-activation collapse,
continuous-variance collapse, contract failure, KL hard breach, or declared
plateau is the boundary for review rather than an arbitrary pilot-length stop.
