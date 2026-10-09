# Repository boundary

## Normative specification

The Google Drive document **Anthropocene AI/RL Training Concept — Neolithic
Curriculum and Scalable Agent Architecture** is normative.

Canonical design document:
https://docs.google.com/document/d/1rZjIo7NMrxn3OQzGFHJI2QAZN7rauHgSCm-nbMNUCT0/edit

This repository and the `Anthropocene` game repository implement that document.
A repository manifest records what code was run; it does not define what the RL
system ought to mean. When code/configuration conflicts with the design document,
the default resolution is to change the code/configuration to match the document,
not to rewrite the document to match the implementation.

## Invariant

The Python repository is a client of Anthropocene's native simulation. It may
transform observations, select actions, train policies, evaluate checkpoints,
and export candidate actors. It may not reproduce or supersede authoritative
game logic.

The native C# environment contract is framework-neutral. Gymnasium, PettingZoo,
RLlib, JAX/Ray, or another learner frontend must adapt to that contract rather
than redefine it.

## No submodule

The relationship is an exact simulator manifest, not a Git submodule. Cloud
bootstrap clones the pinned commit. Local development may use a sibling checkout
through `ANTHROPOCENE_SIM_ROOT`, but experiment manifests still record an exact
commit.

## Observation versus step context

The 512-float `EncodedObservation` is the neural input tensor, not the complete
environment-step envelope. Action masks, agent-local physical delta-t, trusted
action metadata, schema identities, and replay/intervention data remain
environment/adapter context and must not be reconstructed from normalized neural
features.

The concrete Python step-envelope type is intentionally deferred until the
remaining section-34 transport/action contract is locked.

## Promotion boundary

Training artifacts remain outside the game repository until a model passes the
held-out gates. Promotion produces:

- an exported actor model in a reviewed runtime format;
- a versioned normalization manifest;
- a model manifest containing simulator/config/schema/training identities;
- numerical parity evidence between the training implementation and runtime
  inference path.

ONNX is one candidate export/runtime format, not a settled architecture decision.
The game repo owns the promoted runtime artifact and compatibility checks.
