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

## What is still experimental

Before any gameplay-performance claim:

1. expand the diagnostic Stage-0 bank into the 200-500-state pilot bank;
2. fit and freeze the final campaign normalizer on the pilot TRAIN partition;
3. promote the real tiny-snapshot PPO gate from diagnostic to campaign settings;
4. use the measured native-worker scaling to select pilot hardware and batching;
5. compare H=400 versus H=800;
6. run the q_k/technology-regime MLP adaptation sandbox;
7. optionally test the privileged critic;
8. enable movement only after the fixed-location loop is healthy.

GRU, Transformer, online gameplay SGD, later-era goods encoders, sparse
event-time multi-agent inference and RLlib scale-out remain deferred.
