# Repository boundary

## Invariant

The Python repository is a client of Anthropocene's native simulation. It may
transform observations, select actions, train policies, evaluate checkpoints,
and export actors. It may not reproduce or supersede authoritative game logic.

## No submodule

The relationship is an exact simulator manifest, not a Git submodule. Cloud
bootstrap clones the pinned commit. Local development may use a sibling checkout
through `ANTHROPOCENE_SIM_ROOT`, but experiment manifests still record an exact
commit.

## Promotion boundary

Training artifacts remain outside the game repository until a model passes the
held-out gates. Promotion produces:

- an exported actor model;
- a versioned normalization manifest;
- a model manifest containing simulator/config/schema/training identities;
- numerical parity evidence between PyTorch and the runtime inference path.

The game repo owns the promoted runtime artifact and compatibility checks.
