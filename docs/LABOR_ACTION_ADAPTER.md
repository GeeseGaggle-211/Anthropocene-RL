# Canonical actor action inputs

This standalone adapter is based on `90eee94bcc124ba49482cd75fb57235fc3afa7cc`.
It uses the existing `labor-action-semantic-vector-v1` encoder: 349 float32
features with fixed specialty, activity, movement and trade-group ordering.
It does not change the model, action schema, feature layout, critic normalizers,
targets, optimizer or PPO estimator. It has no dependency on the separate
privileged PPO/warm-up commits.

## Units and transforms

The input is the submitted schema-12 request. Identical requests from actor,
heuristic and search producers yield identical feature vectors.

* The production heuristic converter divides resident counts by population and
  worker-years by assigned resident time to obtain labor shares. Three of four
  residents become a specialty share of `0.75`.
* Actor labor samples already contain masked simplex shares. The existing wire
  serializer corrects one coordinate for floating-point roundoff. Before that
  correction, the adapter uses the actor likelihood's `1e-6` simplex tolerance;
  the resulting request must pass the stricter wire validator.
* Actor trade-value log latents pass through the existing exponential and
  `[0.25, 4]` clamp. The critic encoder uses
  `log(clamp(value, 0.25, 4)) / log(4)`. Actor latent `log(3)` and heuristic
  trade request `3` therefore agree.
* Actor trade-share logits pass through the existing sigmoid. Movement, fission
  and trade gates retain their native integer categories and one-hot encoding.

Wire trade values `1.5` and `3` are different requests. There is no empirical
multiplier or matching of producer distributions. Different requested shares
can remain different features even when native integer allocation happens to
produce the same result. Compiled counts, worker-years and realized outputs are
not substitutes for the request-based input contract.

## Integration

Use the retained samples that produced the submitted action, rather than
resampling or feeding actor head means, standard deviations or logits directly:

```python
from anthropocene_rl.autonomous_v7.labor_action_adapter import (
    action_adapter_contract,
    action_adapter_digest,
    actor_action_features,
    canonical_action_features,
)

features = actor_action_features(
    retained_labor,
    retained_initial_nonlabor,
    catalog,
    index=row_index,
    expected_layout=critic_artifact["action_layout"],
)
```

For a retry, replace only `retained_labor` with that retry's labor sample; retain
the initial nonlabor sample for the same parent. Select matching rows, or slice
both samples to that parent and use index zero. The selected catalog's labor
masks must match the retained sample. Output is a detached CPU float32 tensor of
shape `(349,)`; the adapter does not mutate samples or advance either RNG.

For an already canonical heuristic/search request, use
`canonical_action_features(action, catalog, expected_layout=...)`. Requests
must use canonical registry order. Search artifacts with explicit reordered
stable IDs retain the existing `action_features(..., specialty_ids=...,
activity_ids=..., trade_groups=...)` binding before comparison.

Although `expected_layout` is optional for standalone use, checkpoint consumers
must supply the intended artifact's complete action layout to fail closed on a
different transform, ordering or width. Persist `action_adapter_contract()` and
`action_adapter_digest()` alongside the feature data and check them on replay;
this standalone module cannot establish checkpoint provenance for its caller.
The strengthened input-validation record changes the adapter digest relative
to commit `f49e7d6`; the underlying feature layout and normalization are unchanged.

Malformed shapes, nonfinite values, fractional/boolean categories, invalid
movement bindings, masked labor allocations and non-simplex counts are rejected
before decoding can hide them. Unused labor latents are checked for shape and
finiteness, but are not reconstructed from rounded heuristic requests.

These are fixed semantic conversions available before the first decision.
Nine decisions do not fit new normalization statistics or establish critic
readiness. Critic construction, fitting and PPO integration remain separate work.
