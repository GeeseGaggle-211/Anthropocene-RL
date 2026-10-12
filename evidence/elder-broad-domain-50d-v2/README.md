# Elder broad-domain 50-decision v2 evidence

Date: 2026-10-12

## Frozen identities

- Learner starting point: `c0628fcd2d348b709ce7abca6972390c6a130f84`
- Native simulator: `5559961164d6f9da8e6d20c1dc563ecec3eb04b7`
- Bank: `elder-v2-broad-domain-50d-pilot-v2`
- Generator config SHA-256:
  `a9cb4620ecbff98053c3cda18b25bde91ed707735c7ef196ffc4c30aabbe623e`
- Frozen manifest SHA-256:
  `10ab09c73942cc3e2bea0bef409c6682a909d07022b0fb1d529d3b566979e736`
- Native result SHA-256:
  `397f5bfa1954bf33c3f9288033e1fcf17c52fe4b634deb98cfefd8fd8878b6e7`

## Result

The native generator requested 800 candidates and produced 686 valid
candidates. Deterministic regeneration passed. Group-aware selection promoted
exactly 499 snapshots: 347 TRAIN, 73 DEVELOPMENT, and 79 HELDOUT. Every family
met its frozen family/partition quota; the bank audit reports no errors or
warnings.

The F15 shared-event constructor uses the existing deterministic heuristic
settlements and ordinary native trade ledger. It promoted five complete
mild-import/surplus-export pairs: three aligned at decision 1 and two at
decision 2. No direct inventory injection, privileged counterparty, or future
trade observation is present.

The five-decision F15 capture ceiling is only a scenario-construction rule. The
learning/evaluation episode is separately frozen at 50 decisions (nominally
1,000 simulated years at 20 years per native turn).

## Normalization

The final `elder-normalization-v3` artifact was fitted from 2,082 TRAIN
observations only: 347 resets plus 1,735 legal bounded-rollout observations.
DEVELOPMENT and HELDOUT were transform-only diagnostics. The sparse causal
support gate passed. Maximum saturation was 1.14% on DEVELOPMENT, 0.21% on
HELDOUT, and 0.19% on TRAIN, below the frozen 10% limit.

The full per-feature diagnostics, raw native results, SaveCodec blobs, and run
artifacts are durable GCS artifacts. Git retains this hash-anchored summary,
the frozen manifest/configuration, coverage/audit reports, and fitted normalizer.

## Training execution contract

The executable campaign configuration is
`configs/training/elder-broad-domain-50d-v2.yaml`. Broad-domain training and
evaluation are code-gated to exactly 50 decisions; a three- or five-decision
configuration is rejected. Cultivation, husbandry, and movement use their
native structural masks. Fishing and synthetic shocks remain excluded.

Each update samples one TRAIN snapshot from each of the 13 families and runs
four on-policy common-random-number trajectories per snapshot. The first
decision uses a leave-one-out within-snapshot advantage; later decisions use
the critic. Forty updates provide up to 104,000 transitions per neural seed.
The critic uses running PopArt-style value-target normalization, critic-only
LayerNorm, and separate actor/critic clipping. The actor remains the existing
Tanh MLP without LayerNorm.

The powered runner also neutralizes random first-layer influence from tensor
columns with zero nonzero support in PILOT TRAIN while leaving those columns
trainable when a sparse event appears. It records per-head log standard
deviation and mean-endpoint rates, critic value-head size, clip/KL early-stop
statistics, and DEVELOPMENT reset-value explained variance. Presence-bit
changes are explicitly deferred because they would create a new observation
schema generation.

An eight-worker 50-decision preflight collected 2,577 transitions in 48 seconds
and passed all contract, replay, finite-value, reward-denominator, and
per-state inactive-head gradient gates. The corresponding single-worker
collection took about 137 seconds. Serious evaluation also uses up to eight
workers while retaining exhaustive replay. The frozen native interface accepts only
`RandomStream=0`; snapshot state owns keyed randomness, so repeated rollouts
share the native future while policy actions remain independently sampled.

## Powered training result

The corrected K=4 campaign completed at learner commit
`91c5ccee0a35a03c1df60ecdd79c43c833593a6f`. All three seeds passed the hard
contract, replay, numerical, structural-mask, inactive-gradient, checkpoint,
critic-finiteness, activation-health, and PPO-stability gates. Each seed saw
approximately 103,600 transitions over 40 updates.

The campaign remains `NO_GO` for welfare promotion. Against Option A on full
DEVELOPMENT, seed 1701 was -2.97% on mean QoL and -13.59% on mean population;
seed 1901 was -7.70% and -13.23%; seed 2101 was -4.59% and -0.50%. Thus only
seed 2101 independently met the declared +/-10% joint QoL/population target.
The campaign must not be described as passing by pooling the three seeds.

The hidden-layer and exploration diagnostics do not show architecture
collapse: selected actor/critic probes stayed below the 10% Tanh saturation
trigger, log standard deviations stayed near `-1`, and actor means did not
stick at endpoints. This makes v2 the final proof-of-readiness run. The next
stage is the durable full-duration continuation declared in
`configs/training/elder-broad-domain-50d-full-v3.yaml`, not another short pilot.

Cultivation intensity did not rise monotonically. Requested field operation
rose on average, but requested Development budget fell, so the result is a
shift in the cultivation mix rather than a steady universal increase. No seed
showed the declared universal-self-sufficiency signal.

## Storage

The frozen bank is published at
`gs://anthropocene-rl-datasets-508912/datasets/elder-v2-broad-domain-50d-v2/2026-10-12-manifest-10ab09c7/manifest.json`.
Both shards passed upload byte-count, CRC32C, MD5, and custom SHA-256 metadata
checks. Small-object reads succeed, but a real large-shard re-fetch stalled at
zero bytes in both the Python client and Google Cloud CLI on this machine.
Local blobs therefore remain protected from pruning until an actual shard
download and record re-materialization pass.
