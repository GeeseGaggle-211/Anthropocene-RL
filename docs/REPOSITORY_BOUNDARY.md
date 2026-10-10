# Repository boundary

## Normative specification

The Google Drive document **Anthropocene AI/RL Training Concept — Neolithic
Curriculum and Scalable Agent Architecture** is normative.

Canonical design document:
https://docs.google.com/document/d/1rZjIo7NMrxn3OQzGFHJI2QAZN7rauHgSCm-nbMNUCT0/edit

## Invariant

Anthropocene-RL is a client of the authoritative C# simulation. Python may
normalize observations, choose bounded directives, train policies, evaluate
checkpoints and export candidate actors. It may not reproduce ecology,
economics, trade, immigration, fission, demography or resolution.

The frozen native-v2 pin is recorded in `configs/simulator/elder-v2.yaml`.
The C# stdio bridge in this repository compiles the simulation source files
from that exact checkout; it does not fork them.

## Observation versus step context

The schema-v2 neural input is 512 floats plus presence bits. Structural action
masks are encoded at fixed native offsets and are also preserved as adapter
context. Requested, accepted and realized directives, elapsed physical time,
interventions, reward records and the optional critic sidecar remain transition
metadata rather than normalized actor features.

## No submodule

The repositories are joined by exact commit manifests, not a Git submodule.
`ANTHROPOCENE_SIM_ROOT` may point at a local sibling checkout only when its
HEAD equals the frozen simulator SHA. Managed bootstrap uses a detached checkout
of that SHA.

## Promotion boundary

A promoted model must include:

- exported actor parameters/runtime artifact;
- frozen normalization manifest;
- simulator/config/observation/action/environment schema identities;
- training/development/held-out partition identities;
- action-distribution and physical-time-credit configuration;
- numerical parity evidence for runtime inference.

The native training-only critic sidecar is never part of the exported actor.
