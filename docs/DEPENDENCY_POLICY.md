# Dependency policy

- `core` contains the Gymnasium-facing contract, schemas, configuration, and
  serialization utilities.
- `training` adds PyTorch and TensorBoard.
- `baseline` adds Stable-Baselines3 only for controlled baseline experiments;
  it is not assumed to support the final hybrid action contract.
- `distributed` adds Ray RLlib and PettingZoo for later scale-out/multi-agent
  work; it is not installed for the first adapter smoke by default.
- `export` adds ONNX tooling for eventual promoted runtime actors.

Gymnasium is pinned to `1.2.2` because the pinned Ray `2.59.0` RLlib extra
requires that exact version. Stable-Baselines3 `2.9.0` supports the same
Gymnasium major range. Upgrade these as one tested compatibility set rather than
advancing Gymnasium independently.

The first cloud setup must resolve and commit `uv.lock`. Accelerator-specific
PyTorch/JAX installation remains environment-specific and must not be encoded as
a universal local dependency.
