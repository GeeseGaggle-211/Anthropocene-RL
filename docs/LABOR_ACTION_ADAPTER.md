# Canonical actor action inputs

The population-aware adapter requires the encoder implementation on
`feature/r5-a-fit` at `a8d6bbb8abd322ff8f159478d76d3705b10b13e5` (or a descendant).
That dependency must land before this PR is merged. The PR's original base
`90eee94bcc124ba49482cd75fb57235fc3afa7cc` does not support the required
`action_features(..., population=)` argument; there is no raw-share fallback.

The encoder still has 349 float32 features and the same
`labor-action-semantic-vector-v1` layout. Population changes the labor features'
semantics, so the adapter contract is now v2. This change does not alter the
model, action schema, critic normalizers, targets, optimizer or PPO estimator.
It has no dependency on the separate privileged PPO/warm-up commits.

## Units and transforms

The inputs are the submitted schema-12 request and its decision-boundary catalog.
Both adapter entry points always forward `catalog.population` to the existing
encoder. Population must be a nonnegative integer; missing values, booleans,
floats (including whole-valued floats), strings and negative values are rejected
before coercion. The adapter does not duplicate the apportionment algorithm.

* The encoder canonicalizes specialty shares and applies native stable
  largest-remainder apportionment, then encodes residents divided by population.
  For example, requested shares `[0.25, 0.75]` at population 23 encode as
  `[6/23, 17/23]`, matching the search payload reader.
* Activity rows of specialties receiving zero residents become all idle. At
  population zero, all specialty features are zero and every activity row is
  idle. Zero is passed explicitly, never interpreted as missing population.
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

At the same population, requests with identical apportioned residents,
staffed-role activity and nonlabor posture encode identically, even when their
raw shares or unstaffed-role activity differ. Wire trade values `1.5` and `3`
remain different requests. No empirical multiplier or producer-distribution
matching is introduced. Worker-years and realized outputs do not replace the
declared input contract.

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
    expected_adapter_digest=critic_artifact["action_adapter_digest"],
)
```

For a retry, replace only `retained_labor` with that retry's labor sample; retain
the initial nonlabor sample and pre-action catalog for the same parent. Select
matching rows, or slice both samples to that parent and use index zero. For a
batch with different populations, pass the catalog belonging to `row_index` on
each call; there is no batch-wide population or inferred population. The selected
catalog's labor masks must match the retained sample. Output is a detached CPU float32 tensor of
shape `(349,)`; the adapter does not mutate samples or advance either RNG.

For an already canonical heuristic/search request, use
`canonical_action_features(action, catalog, expected_layout=...,
expected_adapter_digest=...)`. Requests
must use canonical registry order. Search artifacts with explicit reordered
stable IDs retain the existing `action_features(..., specialty_ids=...,
activity_ids=..., trade_groups=..., population=root.population)` binding before
comparison. The actual search payload reader resolves population from the root
or its compiled candidates; it must agree with the adapter's root catalog.

Checkpoint consumers must supply the intended artifact's action layout **and**
its recorded adapter digest. Supplying a layout without a digest is rejected;
the layout alone cannot distinguish raw from post-apportionment features.
The v2 contract explicitly records population, post-native apportionment and idle
unstaffed activity. An old v1 digest is rejected even when its layout matches.
Missing or old bindings require explicit provenance review/migration; do not
stamp the current digest onto an unverified old checkpoint. Persist
`action_adapter_contract()` and `action_adapter_digest()` with newly verified
artifacts and feature data. Standalone calls without a checkpoint may omit both
expected bindings. The adapter cannot verify which root catalog its caller chose.

Malformed shapes, nonfinite values, fractional/boolean categories, invalid
movement bindings, masked labor allocations and non-simplex counts are rejected
before decoding can hide them. Unused labor latents are checked for shape and
finiteness, but are not reconstructed from rounded heuristic requests.

Regression tests include H12 shares that are not multiples of `1/population`,
bitwise agreement with the actual search payload reader through both population
sources, zero-staffed role neutrality, per-row actor populations and retry posture,
invalid population rejection, and rejection of old or missing checkpoint bindings.

These are fixed semantic conversions available before the first decision.
Nine decisions do not fit new normalization statistics or establish critic
readiness. Critic construction, fitting and PPO integration remain separate work.
