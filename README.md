# Anthropocene-RL

Training, evaluation, dataset, and model-export tooling for the learned agents in
[Anthropocene](https://github.com/GeeseGaggle-211/Anthropocene).

This repository is a **client of the authoritative C# simulation**. It must not
reimplement Anthropocene economics, ecology, trade, immigration, fission, or
resolution logic in Python.

## Normative design authority

The Google Drive document **Anthropocene AI/RL Training Concept — Neolithic
Curriculum and Scalable Agent Architecture** is the normative specification for
RL architecture, observation semantics, reward semantics, curriculum, evaluation
gates, and future scale-out.

Repository code, YAML, manifests, README files, and tests are implementations or
reproducibility records of that specification. They do **not** override it. If
the repository and the design document disagree, treat the document as correct
unless the document is explicitly revised; change the implementation to conform.

Canonical design document:
https://docs.google.com/document/d/1rZjIo7NMrxn3OQzGFHJI2QAZN7rauHgSCm-nbMNUCT0/edit

## Native simulator contract

The initial Elder-v1 scaffold handoff is pinned to:

- repository: `GeeseGaggle-211/Anthropocene`
- branch: `feature/elder-rl-scaffold`
- commit: `81bae49d71b23c32d956991601546d3cb2064ad7`
- observation schema: `1`
- action schema: `1`
- environment schema: `1`
- tensor width: `512`

`configs/simulator/elder-v1.yaml` is the reproducibility pin for this native
handoff. It is not the design source of truth.

The native scaffold is implementation-ready. The normalization-v1 **design
policy is approved in the design document**, but serious training remains blocked
until the native observation-semantic plumbing is updated to match that policy
and the still-open section-34 adapter decisions are locked.

## Scope of this skeleton

Included now:

- a framework-neutral Python core plus optional Gymnasium v1 adapter,
  PyTorch training, baseline, distributed/MARL, and export dependencies;
- a versioned simulator manifest and repository boundary;
- package/module structure for transport, environment, normalization, policy,
  training, evaluation, datasets, and export;
- bootstrap and contract-check command stubs;
- initial curriculum/evaluation manifests from the design document;
- CI for Python structure and contract-manifest checks.

Not implemented yet:

- C#↔Python transport;
- the Gymnasium `reset`/`step` adapter;
- normalization-v1 transforms and frozen normalization manifest;
- the native C# observation-semantic fixes specified in design section 34;
- PPO action distributions, GAE/discount semantics, or training loop;
- snapshot-bank generation;
- runtime export/inference integration.

See `docs/SECTION_34_GATES.md` for the implementation blockers mirrored from
the design document.

## Setup

Use `uv` and Python 3.12:

```bash
uv sync --extra dev --extra adapter --extra training
uv run anthropocene-rl check-manifest
```

Distributed/MARL packages are deliberately optional:

```bash
uv sync --extra dev --extra distributed
```

The first cloud bootstrap should generate and commit `uv.lock` after resolving
against the selected platform. Do not hand-edit a lockfile.

## Simulator checkout

The harness supports either:

1. a sibling/local checkout supplied through `ANTHROPOCENE_SIM_ROOT`; or
2. a private clone at the exact commit in the simulator manifest.

No Git submodule is used. Every training manifest records an exact simulator
commit and schema set.

```bash
export ANTHROPOCENE_SIM_ROOT=../Anthropocene
uv run anthropocene-rl check-simulator
```

## Repository boundary

`Anthropocene` owns:

- world truth and deterministic physics;
- native Elder observation/action/environment contracts;
- Reset/Step, replay, masks, interventions, and reward calculation;
- eventually, the promoted runtime actor and Unity inference integration.

`Anthropocene-RL` owns:

- learner-side transforms and normalization;
- framework-neutral transport clients and learner adapters;
- the Gymnasium v1 adapter without making Gymnasium the authoritative contract;
- training algorithms and experiment configs;
- snapshot-bank manifests and partitions;
- evaluation, checkpoints, and candidate model exports.

See `docs/REPOSITORY_BOUNDARY.md`.
