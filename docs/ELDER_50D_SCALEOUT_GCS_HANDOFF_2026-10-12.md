# Elder 50-decision scale-out handoff

Date: 2026-10-12

## Decision boundary

The fresh scale-out curriculum is complete and locally validated. Training has
not been launched on this workstation. The next execution target is Google AI
Pro, using GCS for the immutable dataset and all large run/checkpoint objects.

An independent Astra-high review must issue an explicit GO before the hero
command below is run. A single-seed hero can produce a provisional result only;
promotion requires the subsequent three-seed campaign.

## Frozen identities

- Native simulator: `GeeseGaggle-211/Anthropocene@5559961164d6f9da8e6d20c1dc563ecec3eb04b7`
- Bank: `elder-v2-broad-domain-50d-scaleout-v1`
- Manifest identity: `6036d104dc655305f62bde2e7650cb4b1d20f36dcfdb8337f92c9dcbdbf7f06e`
- Manifest file SHA-256: `0a5fa3d3067211616b7c93a4229a5420fb8647251122b2f50d0ae5f3a7b97b4d`
- Generator configuration identity: `bf4517367870b3bd02a7ad7604aa0fe444b84671a883dbcc2ddefd927b4aa70e`
- Native result SHA-256: `f78310197bd8fbf64a666993783a1fee15de8eada28f98c7c0e52e767e0ef1bb`
- TRAIN-only normalizer SHA-256: `9d0136764b213f9d3b1b6996b41adfe92d43573de8192b3c46c89e60ae345fd1`
- Dataset URI: `gs://anthropocene-rl-datasets-508912/datasets/elder-v2-broad-domain-50d-scaleout-v1/2026-10-12-manifest-6036d104/manifest.json`
- Published dataset-manifest SHA-256: `3d0c6483bb6c8082886f16a196d8d9c76775567e2ae66f0a32f6b936338cdba2`
- Dataset packaging learner commit: `bebd5dbca52e0b25a225bba81d691e089590571e`
- Frozen evidence bundle: `gs://anthropocene-rl-datasets-508912/run-objects/sha256/f6/f63b877a4b3beed9e388f6898e464093df3ae214b37cc78a8e27421289d737c0.tar`
- Declarative training configuration: `configs/training/elder-broad-domain-50d-scaleout-v1.yaml`

The source commit used to pack the GCS dataset is recorded in the remote
dataset manifest and storage audit. No snapshot or checkpoint bytes belong in
Git.

Both immutable shards were re-downloaded into a fresh cache and all 499 records
were materialized with SHA-256 verification. The read produced two cache misses
(one per shard) followed by 497 cache hits. The full normalizer diagnostics,
configs, manifests, and this handoff are separately preserved in the frozen
evidence bundle above. Generated local staging and diagnostic caches were then
removed, reclaiming 712,200,192 bytes; local source blobs remain retained.

## Frozen curriculum

The accepted bank contains 499 snapshots: 348 TRAIN, 75 DEVELOPMENT, and 76
HELDOUT. It contains 382 independent world/lineage groups and has no world,
lineage, or content-hash overlap with any earlier committed bank.

| Family | TRAIN | DEV | HELDOUT | Total |
|---|---:|---:|---:|---:|
| F01 | 17 | 4 | 4 | 25 |
| F02 | 30 | 4 | 4 | 38 |
| F03 | 38 | 13 | 9 | 60 |
| F04 | 28 | 6 | 6 | 40 |
| F05 | 22 | 4 | 4 | 30 |
| F06 | 25 | 5 | 5 | 35 |
| F07 | 25 | 5 | 5 | 35 |
| F13 | 22 | 4 | 4 | 30 |
| F14 | 38 | 8 | 8 | 54 |
| F15 | 34 | 8 | 8 | 50 |
| F16 | 20 | 4 | 4 | 28 |
| F17 | 27 | 6 | 9 | 42 |
| F18 | 22 | 4 | 6 | 32 |

Movement and husbandry are admitted through their native structural masks.
Cultivation, trade, migration, fission, amenities, institutional memory, and
productive-frontier variation remain live. Fishing and the synthetic-disaster
curriculum remain excluded.

F15 has five complete counterfactual pairs for each declared causal contrast:

- isolated versus strong food import;
- mild import versus surplus export;
- food shortfall with contacts versus material/tool import;
- immigration-secure versus emigration-distress;
- immigration-labor versus immigration-stress.

The generator requested 800 candidates. It rejected or declined 301: 104 at
native physical-stratum generation and 197 at complete-group, identity, and
quota selection. Sixteen otherwise valid F16 records were rejected because
nominal variants duplicated native identities; they were not admitted merely
to reach a planned family count.

## Audit and normalization

Deterministic native regeneration, SaveCodec ingestion, broad-domain scope,
structural masks, replay, finite-time, schema, partition, and blob-integrity
audits pass.

Normalization-v3 was fit from 348 TRAIN resets plus 1,740 bounded legal TRAIN
rollout observations. DEVELOPMENT and HELDOUT contributed no fit data. Sparse
causal support passes for delivered trade and signed migration. The maximum
slice saturation is 0.444%, against the 10% gate.

The local proof-of-readiness experiments did not show hidden-layer collapse,
log-standard-deviation collapse, endpoint pinning, or excessive policy
stability. They also did not show a steady, replicated monotonic increase in
cultivation intensity: earlier short runs drifted toward different near-
constant operating levels, while the powered 50-decision PoR demonstrated
parameter movement and critic fit without yet producing a promoted welfare
policy. The scale-out campaign is therefore a generalization/performance test,
not an architecture rescue.

Keep the separate Tanh `512 -> 256 -> 256` actor and critic, critic-only
LayerNorm, running PopArt value-target normalization, separate actor/critic
gradient clipping, and entropy floor. Do not introduce ReLU, actor LayerNorm,
recurrence, a privileged critic, or a schema change without new evidence.

## Google AI Pro sequence

Build the bridge against the exact pinned native checkout first. Use the GCS
dataset manifest directly; the dataset layer verifies shard and record hashes
and caches materialized records locally.

### 1. Single-seed hero (HELDOUT sealed)

Run seed 1701 for 100 updates, or at most 260,000 transitions. Evaluate full
DEVELOPMENT every 10 updates. This is a scaled safety/performance gate, not a
promotion result.

```powershell
python -m anthropocene_rl.cli cultivation-pilot `
  --dataset-manifest gs://anthropocene-rl-datasets-508912/datasets/elder-v2-broad-domain-50d-scaleout-v1/2026-10-12-manifest-6036d104/manifest.json `
  --dataset-store-config configs/storage/gcs-v1.yaml `
  --normalizer evidence/elder-broad-domain-50d-scaleout-v1/bank/final-train-normalization.json `
  --bridge-dll integration/bin/Release/net8.0/Anthropocene.RLBridge.dll `
  --campaign-scope broad-domain --seeds 1701 --updates 100 `
  --episodes-per-update 13 --rollouts-per-snapshot 4 `
  --group-relative-first-decision --episode-steps 50 `
  --evaluation-interval 10 --evaluation-steps 50 `
  --horizon-years 400 --lambda-ref 0.95 --lambda-reference-years 20 `
  --reward-scale 0.001 --replay-verification-interval 20 `
  --collector-workers 8 --evaluation-workers 8 `
  --checkpoint-selection development-heuristic-parity `
  --heuristic-parity-tolerance 0.10 --heuristic-parity-patience 2 `
  --parity-plateau-evaluations 5 --parity-plateau-min-added-updates 100 `
  --parity-plateau-min-improvement 0.01 `
  --learning-rate 0.001 --actor-learning-rate 0.0003 `
  --ppo-epochs 3 --minibatch-size 256 --target-kl 0.02 `
  --gradient-clip-mode separate --entropy-coefficient 0.001 `
  --value-target-normalization --critic-layer-norm `
  --skip-heldout-evaluation `
  --evidence-directory evidence/elder-broad-domain-50d-scaleout-hero-seed1701 `
  --checkpoint-directory checkpoints/elder-broad-domain-50d-scaleout-hero-seed1701
```

Do not continue if contracts, replay, inactive-head gradients, numerical
guards, post-update KL, activation health, exploration variance, or checkpoint
parity fail. Also stop if the policy pins continuous means to endpoints, the
critic's DEV explained variance collapses, or clip/KL early-stop rates show
that most updates are being discarded.

### 2. Fresh replicated campaign

If the hero passes, start all three declared seeds from fresh initial states
under the unchanged configuration. Run 500 updates, or at most 1.3 million
transitions per seed, with full DEVELOPMENT every 20 updates. Do not cherry-pick
the best seed. HELDOUT is used once after DEVELOPMENT checkpoint selection.

Use the same command with these replacements:

```text
--seeds 1701,1901,2101
--updates 500
--evaluation-interval 20
--evidence-directory evidence/elder-broad-domain-50d-scaleout-v1
--checkpoint-directory checkpoints/elder-broad-domain-50d-scaleout-v1
```

Remove `--skip-heldout-evaluation` only for this predeclared final campaign.
The hero checkpoint is not resumed; restarting seed 1701 keeps the replicated
comparison symmetric.

## Required run evidence and GCS outputs

For the hero and every replicated seed, preserve:

- contract/replay/inactive-head/likelihood/numerical results;
- per-head log standard deviation and actor-mean endpoint rates;
- actor/critic hidden saturation and critic value-head norms;
- TRAIN and DEV critic explained variance;
- clip fraction, observed KL, and target-KL early-stop rate;
- family/mask visitation and per-world evaluation results;
- requested, accepted, realized, and intervention action channels;
- cultivation-intensity trajectory, delivered trade, signed migration,
  fission, population, QoL, needs, reserves, tools, and terminal stocks;
- Option-A comparisons on identical frozen evaluation states;
- checkpoint hashes and exact learner/native/dataset identities.

Upload checkpoints and complete run bundles to content-addressed GCS objects.
Commit only compact summaries, identities, and the final decision record.

## Interpretation and escalation

A hero GO means the training instrument remains healthy at scale. It is not a
claim that the model is promoted. Final advancement requires three-seed
replication, competitive heterogeneous DEVELOPMENT welfare, no strong sealed
HELDOUT degradation, context-sensitive substitution among local production,
movement, husbandry, and external supply, and no universal self-sufficiency or
end-of-horizon liquidation signal.

If the campaign fails, attribute it in this order: native defect, interface or
observation defect, optimizer instability, reward/credit horizon, scenario
coverage, then feed-forward representational limitation. Architecture changes
require direct evidence at the final step, not merely slow learning.
