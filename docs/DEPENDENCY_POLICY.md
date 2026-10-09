# Dependency policy

- `core` is framework-neutral: schemas, configuration, serialization, and
  process-boundary contracts must not depend on Gymnasium, RLlib, PettingZoo, or
  another learner framework.
- `adapter` adds Gymnasium for the first single-Elder learner facade. Gymnasium
  is a v1 adapter, not the authoritative environment contract.
- `training` adds PyTorch and TensorBoard for the first actor/critic experiment.
- `baseline` adds Stable-Baselines3 only for controlled baseline experiments;
  it is not assumed to support the final hybrid action contract.
- `distributed` adds Ray RLlib and PettingZoo as candidate scale-out/MARL
  backends. They are optional and do not define the long-term architecture.
- `export` adds ONNX tooling as one candidate promoted-actor backend. The final
  runtime actor format remains undecided.

Gymnasium is pinned to `1.2.2` in the adapter extra because the pinned Ray
`2.59.0` RLlib extra requires that exact version. Stable-Baselines3 `2.9.0`
supports the same Gymnasium major range. Upgrade these as one tested
compatibility set rather than advancing Gymnasium independently.

Ray/RLlib is not presumed to be the eventual TPU/distributed learner. Ray Core,
Ray Train with JAX/Optax, or a custom actor-learner implementation remain valid
future paths if measured throughput or accelerator requirements justify them.

The first cloud setup must resolve and commit `uv.lock`. Accelerator-specific
PyTorch/JAX installation remains environment-specific and must not be encoded as
a universal local dependency.
