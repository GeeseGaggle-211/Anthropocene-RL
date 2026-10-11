# Anthropocene-RL

Training and evaluation tooling for learned institutional agents in
[Anthropocene](https://github.com/GeeseGaggle-211/Anthropocene).

The Google Drive document **Anthropocene AI/RL Training Concept — Neolithic
Curriculum and Scalable Agent Architecture** is normative. This repository is a
client of the authoritative C# simulation and never reimplements its economics,
ecology, trade, migration, fission or demography.

## Frozen Elder v2 handoff

The first PPO bring-up pins:

- simulator commit `5559961164d6f9da8e6d20c1dc563ecec3eb04b7`;
- observation schema 2;
- action schema 1;
- environment/replay schema 2;
- critic-sidecar schema 1;
- tensor width 512.

See `configs/simulator/elder-v2.yaml`.

## Implemented bring-up surface

- frozen schema-v2 observation/presence model;
- semantic normalization-v2 with fit/freeze/serialize manifests;
- persistent JSON-lines process transport;
- learner-owned C# bridge compiled from the pinned native simulation sources;
- Gymnasium single-Elder adapter;
- separate 512 -> 256 -> 256 PyTorch actor and critic;
- tanh-squashed Gaussian active continuous heads;
- native structural masking with zero likelihood/entropy for inactive heads;
- masked categorical movement support;
- requested-action PPO likelihood under native repairs;
- physical-time interval discount and GAE utilities;
- a minimal PPO optimizer and CI mechanics smoke;
- a real movement-disabled one-snapshot PPO overfit gate with checkpoint parity,
  replay validation, and native/policy/update timing;
- a three-snapshot by three-seed cultivation-readiness matrix with aggregate
  convergence and stability gates;
- a fixed-semantics 1/2/4/8 native-environment throughput benchmark.
- a legacy pluggable Google Drive snapshot store for compatibility;
- a sharded GCS dataset and run-artifact store with bounded resumable uploads,
  authoritative manifests, immutable generations, and a 5 GiB checksum-keyed
  LRU cache.

The initial optimizer smoke keeps movement disabled. H=400 and lambda_ref=0.95
are provisional debug settings in config, not final design decisions.

## Setup

Python 3.11-3.13 and `uv`:

```bash
uv sync --extra dev --extra adapter --extra training
uv run anthropocene-rl check-manifest
uv run anthropocene-rl ppo-smoke
```

PyTorch is pinned to 2.14.1 for the first reproducible bring-up.

## Native checkout and bridge

Use an exact sibling/local checkout:

```bash
export ANTHROPOCENE_SIM_ROOT=../Anthropocene
uv run python scripts/bootstrap_simulator.py --root "$ANTHROPOCENE_SIM_ROOT"
uv run anthropocene-rl check-simulator --root "$ANTHROPOCENE_SIM_ROOT"
uv run anthropocene-rl adapter-smoke
```

If `ANTHROPOCENE_SIM_ROOT` is omitted, bootstrap creates a detached checkout
under `.simulator/Anthropocene`.

The bridge protocol is learner-owned but compiles the pinned C# simulation
source files directly. It exposes only `schema/reset/step/verify_replay/close`;
all physical and legal behavior remains in `NativeElderEnvironment`.

Exact-SHA schema/reset/step/replay results, 143 focused Elder assertions, and a
persistent bridge Reset/Step run against real `SaveCodec` bytes are archived in
`evidence/elder-v2-handoff/`.

## Stage-0 snapshot bank

The learner-owned snapshot-bank tool compiles the pinned simulator sources and
stores native `SaveCodec` payloads under gitignored, content-addressed storage.
Python does not construct worlds or reproduce simulation formulas. The checked
in Stage-0 manifest and reports are:

- `datasets/manifests/elder-v2-stage0-v1.json`;
- `evidence/stage0-snapshot-bank/`.

Generate and independently audit the bank with:

```bash
uv run anthropocene-rl generate-snapshot-bank \
  --config configs/datasets/stage0-v1.yaml \
  --manifest datasets/manifests/elder-v2-stage0-v1.json \
  --blob-root datasets/blobs \
  --evidence-dir evidence/stage0-snapshot-bank \
  --simulator-root "$ANTHROPOCENE_SIM_ROOT"

uv run anthropocene-rl audit-snapshot-bank \
  --manifest datasets/manifests/elder-v2-stage0-v1.json \
  --blob-root datasets/blobs
```

Partition assignment is deterministic and happens before trajectory timestamps
are selected. Complete world seeds are isolated, which is stricter than the
required lineage/counterfactual grouping. Historical or already-consumed
diagnostics are forced to DEVELOPMENT. Requested bands, scenario labels, and
source-policy selection metadata live only in the manifest and are never added
to actor observations.

The Stage-0 normalization artifact is explicitly provisional. It is fitted from
TRAIN reset observations and short legal TRAIN calibration rollouts only; the
DEVELOPMENT and HELDOUT samples appear solely in post-fit diagnostics.

## Durable snapshot storage

Google Cloud Storage is the durable store for machine-consumed snapshots,
training outputs, diagnostics, and checkpoints; see
[`docs/GCS_DATASET_STORE.md`](docs/GCS_DATASET_STORE.md). Git remains
authoritative for manifests, configuration, and compact provenance. Dataset
shards use global content-addressed object names, run bundles are immutable,
and each authoritative manifest is published last.

Install the optional client and configure local Application Default
Credentials. Credentials and refresh tokens stay outside the repository:

```bash
uv sync --extra gcs
gcloud auth application-default login

uv run anthropocene-rl dataset publish --manifest <staged-manifest>
uv run anthropocene-rl dataset fetch-record <gs-manifest-uri> <snapshot-id>
uv run anthropocene-rl dataset verify <gs-manifest-uri>
```

Uploads use create-only generations, CRC32C transport validation, provider MD5,
and application SHA-256 metadata. Downloads recompute SHA-256 before admitting
bytes to the 5 GiB cache. Training loaders prefer a valid local blob, then a
materialized cache record, and otherwise download and verify the containing
shard once; they never stream reset bytes from GCS each episode. The older Drive
commands remain available only for historical manifests and migration fallback.

## Real one-snapshot mechanics gate

After restoring the gitignored Stage-0 blobs and building the bridge against the
frozen native SHA, run:

```bash
uv run anthropocene-rl overfit-one-snapshot
uv run anthropocene-rl cultivation-readiness
uv run anthropocene-rl benchmark-native-throughput \
  --checkpoint checkpoints/stage0-one-snapshot.pt
```

The overfit command chooses a simple cultivation-only F01/F02 TRAIN snapshot by
default, freezes its matching provisional normalizer, keeps movement disabled,
and writes deterministic gate diagnostics plus a complete reloadable checkpoint.
`--device` accepts `cpu` or `cuda`; CPU remains the default. The throughput
command measures the same snapshot class with 1, 2, 4, and 8 persistent native
processes and does not alter training semantics.

`cultivation-readiness` is the promotion gate before enabling movement, herding,
or fishing. It runs independent PPO replicas on the F01, F03, and F13
cultivation-only TRAIN states using three neural seeds each. Every run keeps the
other capabilities disabled and checks real native replay, action-channel
separation, inactive-head/RNG inertness, deterministic fixed-policy outcomes,
compiler-repair magnitude, full-update KL/clipping, on-policy critic fit, and
bit-exact checkpoint continuation. Its aggregate result is a mechanics and
optimization-readiness claim, not a gameplay-performance claim.

## Cultivation-only pilot v1

The first serious pilot is complete. Its frozen bank contains 300 accepted
snapshots across F01/F02/F03/F13/F15/F16/F17/F18, split 210 TRAIN / 45
DEVELOPMENT / 45 HELDOUT before timestamp selection. Trade, migration, fission,
memory, and frontier variation remain live while movement, husbandry, and
fishing are structurally excluded for the three-decision campaign horizon.

Regenerate the plan deterministically with:

```bash
uv run python scripts/build_cultivation_pilot_plan.py
```

Audit the accepted physical states rather than trusting requested bands:

```bash
uv run anthropocene-rl audit-snapshot-bank \
  --manifest datasets/manifests/elder-v2-cultivation-pilot-v1.json \
  --blob-root datasets/blobs \
  --no-stage0-coverage \
  --cultivation-pilot-coverage
```

The final v1 normalization was fitted from PILOT TRAIN resets and short legal
calibration rollouts only. The three-seed PPO result is NO-GO: all contracts
passed, but one seed breached the KL gate and another materially regressed on
DEVELOPMENT and HELDOUT. Sparse trade/migration slices also revealed inadequate
TRAIN-only normalization coverage. See `evidence/cultivation-ppo-pilot-v1/`.

## What is still experimental

Before any gameplay-performance claim:

1. create a fresh v2 generation with new held-out world/lineage groups;
2. increase rare trade/migration exposure in TRAIN calibration and fit a new
   TRAIN-only robust sparse-event normalizer;
3. rerun three feed-forward seeds with a more conservative actor update;
4. predeclare an intervention-distance promotion gate;
5. compare H=400 versus H=800 only after the fixed-location loop is robust;
6. enable movement or husbandry only in a separately declared curriculum.

GRU, Transformer, online gameplay SGD, later-era goods encoders, sparse
event-time multi-agent inference and RLlib scale-out remain deferred.
