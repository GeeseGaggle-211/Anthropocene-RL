# Canonical actor action inputs

The population-aware adapter requires the encoder implementation on
`feature/r5-a-fit` at `ac5c280` (or a descendant), which builds on `a8d6bbb`
(`action_features(..., population=)`) and bumps the critic layout to
`labor-action-semantic-vector-v2`. That dependency must land before this PR is
merged. The PR's original base `90eee94bcc124ba49482cd75fb57235fc3afa7cc` does
not support the required `population=` argument; there is no raw-share fallback.

The encoder still has 349 float32 features. The v2 layout itself records the
post-apportionment specialty encoding (`specialtyEncoding`) and idle unstaffed
activity (`unstaffedActivity`). The adapter contract is now
`labor-canonical-actor-action-adapter-v3`: it embeds that layout and no longer
repeats those two semantics as its own keys, because the layout now identifies
them. It keeps what the layout cannot state: population comes from the
selected root catalog and must be a nonnegative integer, and retained actor
samples decode to wire units through fixed transforms (labor ALR sample to
canonical simplex, exp/clamp trade values, sigmoid trade share, native
categorical codes, no fitted moments). This change does not alter the model,
action schema, critic normalizers, targets, optimizer or PPO estimator.

## Units and transforms

The inputs are the submitted schema-12 request and its decision-boundary catalog.
Every entry point forwards `catalog.population` to the existing encoder.
Population must be a nonnegative integer; missing values, booleans, floats
(including whole-valued floats), strings and negative values are rejected before
coercion. The adapter does not duplicate the apportionment algorithm.

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
matching is introduced.

## Mandatory binding

Every public entry point (`canonical_action_features`, `actor_action_features`,
`actor_action_features_batch`, `categorical_counterfactuals`) takes
`expected_layout` and `expected_adapter_digest` as **required** keyword
arguments. Omitting either is a `TypeError`; `None`, a layout other than
`action_layout_state()`, or a digest other than `action_adapter_digest()` is a
`ValueError`. There is no unbound standalone path: a caller without a checkpoint
passes the current `action_layout_state()` and `action_adapter_digest()`
explicitly. Persist `action_adapter_contract()` and `action_adapter_digest()`
with newly verified artifacts and feature data.

The digest changed twice in this PR: v1 lacked population; the pre-merge v2
embedded the v1 layout and repeated the specialty/unstaffed keys. Both are
rejected even where their critic layout matched; a v1 critic layout is rejected
before the digest is compared. Missing or old bindings require explicit
provenance review/migration; do not stamp the current digest onto an unverified
old checkpoint. The adapter cannot verify which root catalog its caller chose.

## Integration

Use the retained samples that produced the submitted action, rather than
resampling or feeding actor head means, standard deviations or logits directly:

```python
from anthropocene_rl.autonomous_v7.labor_action_adapter import (
    action_adapter_digest,
    actor_action_features_batch,
    categorical_counterfactuals,
    movement_union_legal,
)

binding = dict(
    expected_layout=critic_artifact["action_layout"],
    expected_adapter_digest=critic_artifact["action_adapter_digest"],
)
features = actor_action_features_batch(
    retained_labor, retained_initial_nonlabor, row_catalogs, **binding
)  # [B, 349]
movement = categorical_counterfactuals(features, "movement", **binding)  # [B, 17, 349]
legal = movement_union_legal(retained_initial_nonlabor)  # [B, 17], union order
fission = categorical_counterfactuals(features, "fission", **binding)  # [B, 3, 349]
gate_g = categorical_counterfactuals(features, "trade_gate", group=g, **binding)  # [B, 3, 349]
```

`row_catalogs[i]` is row `i`'s own decision-boundary catalog; there is no
batch-wide or inferred population. Each row's labor masks must match its own
catalog. `actor_action_features(..., catalog, index=i, **binding)` encodes a
single row and is bitwise equal to row `i` of the batch. For a retry, replace only
the labor sample with the retry's; retain the initial nonlabor sample and
pre-action catalog for the same parent. Output is a detached CPU float32 tensor;
the adapter does not mutate samples or advance either RNG.

For an already canonical heuristic/search request, use
`canonical_action_features(action, catalog, **binding)`. Requests must use
canonical registry order. Search artifacts with explicit reordered stable IDs
retain the existing `action_features(..., specialty_ids=..., activity_ids=...,
trade_groups=..., population=root.population)` binding before comparison.

Malformed shapes, nonfinite values, fractional/boolean categories, invalid
movement bindings, masked labor allocations and non-simplex counts are rejected
before decoding can hide them. Unused labor latents are checked for shape and
finiteness, but are not reconstructed from rounded heuristic requests.

## Batch cost

The per-row entry point used to re-validate the whole batch on every call
(4,096 rows: 45 s, 11 ms/row). Validation is now split: batch structure
(fields, shapes, dtypes, movement registry) is O(1) per call, and value checks
(finiteness, simplex, catalog masks, categorical ranges) run only on the rows
being encoded. The batch API validates all rows once, converts the labor
allocations with one `tolist()`, and decodes each row with `_decode_row`, which
is tested equal to `schema12_initial_action` on every row. The trade transforms
still run on each row's own tensor (a 0-d `torch.sigmoid` differs from the
batched kernel in about 4% of float32 elements on this CPU build, so batching
them would change the executed wire value).

Measured on this workstation (CPU, torch 2.14.1, real `LaborActor` samples,
population 23, after the `ac5c280` merge):

| rows | batch API | per row | per-row API loop | per row |
|---:|---:|---:|---:|---:|
| 256 | 0.23 s | 0.91 ms | 0.37 s | 1.43 ms |
| 4,096 | 3.16 s | 0.77 ms | 6.16 s | 1.50 ms |

The per-row cost is flat in batch size. At 4,096 rows the remaining time splits
as: batch validation 0.09 s; decode 0.55 s (0.13 ms/row); the strict wire
validator `validate_schema12_action` 1.06 s (0.26 ms/row, pure-Python simplex
checks); `action_features` 0.83 s (0.20 ms/row, mostly Python
largest-remainder apportionment). About 60% of the residual is therefore the
wire validator plus `action_features`, neither of which this adapter vectorizes.
Categorical counterfactuals are tensor writes into already-encoded vectors and do
not re-encode.

## Categorical counterfactuals

`categorical_counterfactuals(features, head, group=None, **binding)` returns
`[N, K, 349]` copies of encoded executed actions with one head set to each
category: movement over `MOVEMENT_UNION_CODES` (K=17), fission policy (K=3), or
the gate of one canonical trade group (K=3). Only the head's one-hot columns
change. This is valid because `action_features` writes each categorical field
into its own block and nowhere else; in particular trade values are encoded
independently of the gate. That property is checked on the live encoder before
use (fixed probe actions re-encoded at every category; result cached per encoder
function object), and the head is refused if any other column moves. The input
block must be an exact one-hot, so relaxed (Gumbel-softmax) vectors are refused.
Illegal movement categories are included; mask them with `movement_union_legal`.
Tests compare every code of every head on six fixtures (five real actor samples
at populations 0 to 64, plus an H12 request with mixed gates and values) with
full re-encoding of the modified wire action through
`canonical_action_features`, bitwise. A monkeypatched encoder in which a closed
gate suppresses its group's trade value is refused for that head.

Semantically, the native may ignore a group's trade value when its gate is
closed; the encoder (and therefore the counterfactual) still carries the value.
That is an encoder design choice, not a counterfactual inconsistency.

## Parity with the reparameterized path

Since `feature/r5-a-fit` 6780203, `labor_advantage.reparameterized_action_features`
forwards the executed action: every block is `action_features()` of the actor's
serialized wire action (`schema12_initial_action`), bitwise, with
straight-through soft gradients. The categoricals execute
`argmax(logits + g)` (movement mask applied first). It accepts injected noises
(`draws=`, keys `REPARAMETERIZED_DRAW_KEYS`) and returns a draw record
(`return_draw=True`: noises, retained samples, wire actions).

Three tests pin the parity:

* Block by block, on a shared draw. 96 rows, populations 0 to 100, with trade
  values pushed into both clamps. The test replays the generator to build the
  retained sample of the same draw and confirms alignment: the forward
  categorical blocks are the one-hots of the replayed perturbed argmax. The
  whole forward vector then equals `actor_action_features_batch` of the
  executed sample bitwise: specialty, activity, movement, fission, gates, trade
  value and trade share.
* Categorical columns equal the executed one-hots.
* The draw record round-trips. The adapter's encoding of the recorded samples
  equals the forward vector, each recorded wire action encodes to the same row,
  and re-injecting the recorded noises reproduces the vector without the
  generator.

The specialty near-tie case (float32 shares about 1e-7 off the simplex, where
renormalizing every coordinate and correcting only the last one can apportion
differently) is tested against the wire in `test_labor_advantage.py`. Before
6780203 the forward value was the Gumbel-softmax relaxation for categoricals
and a renormalized share for specialty. This PR recorded both as strict
`xfail` tests, and both are now resolved.

## Regression coverage

Tests cover H12 shares that are not multiples of `1/population`, bitwise
agreement with the actual search payload reader through both population sources,
zero-staffed role neutrality, per-row actor populations and retry posture,
invalid population rejection, rejection of old or missing bindings, omission of
the binding on every entry point, batch versus per-row bitwise equality,
per-row catalog masks in a batch, counterfactual bitwise equality and refusal,
and the reparameterized parity above.

These are fixed semantic conversions available before the first decision.
They do not fit normalization statistics or establish critic readiness. Critic
construction, fitting and PPO integration remain separate work.
