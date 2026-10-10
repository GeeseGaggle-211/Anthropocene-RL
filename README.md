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
- a minimal PPO optimizer and CI mechanics smoke.

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
source files directly. It exposes only `schema/reset/step/close`; all physical
and legal behavior remains in `NativeElderEnvironment`.

## What is still experimental

Before any gameplay-performance claim:

1. run/archive native regressions on the exact SHA;
2. construct and freeze the training/development/held-out snapshot bank;
3. fit normalization on the train partition;
4. run the real tiny-snapshot PPO overfit gate;
5. benchmark native workers and batching;
6. compare H=400 versus H=800;
7. run the q_k/technology-regime MLP adaptation sandbox;
8. optionally test the privileged critic;
9. enable movement only after the fixed-location loop is healthy.

GRU, Transformer, online gameplay SGD, later-era goods encoders, sparse
event-time multi-agent inference and RLlib scale-out remain deferred.
