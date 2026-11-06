"""Pinned actor/heuristic/search action semantics for the existing critic encoder.

Decode the retained schema-12 request, then apply the encoder's population-aware
native apportionment. No producer-statistic matching or duplicate allocation
algorithm is used here.

Every public entry point requires ``expected_layout`` and
``expected_adapter_digest`` as keyword arguments; there is no unbound path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from . import contracts
from . import labor_advantage as _advantage
from .labor_advantage import MOVEMENT_UNION_CODES, action_layout_state, block_slices
from .labor_bridge import (
    ACTIVITY_ROWS,
    ROLE_ROWS,
    SPECIALTY_ROWS,
    LaborCatalog,
    LaborPolicySample,
)
from .labor_ppo import validate_schema12_action
from .provenance import digest

# Categorical heads whose counterfactuals can be written into an encoded vector:
# head -> (action block, number of categories). ``trade_gate`` takes a group.
CATEGORICAL_HEADS: dict[str, tuple[str, int]] = {
    "movement": ("movement_onehot", len(MOVEMENT_UNION_CODES)),
    "fission": ("fission_onehot", contracts.FISSION_POLICIES),
    "trade_gate": ("trade_gate_onehot", contracts.TRADE_GATES),
}

_LABOR_FIELDS = frozenset(
    f"{kind}_{field}"
    for kind in ("specialty", "activity")
    for field in ("allocation", "latent", "mask")
)
_NONLABOR_FIELDS = frozenset(
    {
        "movement",
        "movement_index",
        "movement_codes",
        "movement_mask",
        "fission",
        "trade_gate",
        "trade_value_latent",
        "trade_share_latent",
    }
)


def action_adapter_contract() -> dict[str, Any]:
    # The critic layout (v2) itself records the post-apportionment specialty
    # encoding and idle unstaffed activity, so the adapter no longer repeats
    # them. It records only what the layout cannot: where population comes
    # from and how retained actor samples decode to wire units.
    return {
        "kind": "labor-canonical-actor-action-adapter-v3",
        "actionSchema": 12,
        "criticLayout": action_layout_state(),
        "labor": "retained-masked-ALR-sample-to-canonical-simplex-request",
        "population": "selected-root-catalog-nonnegative-integer-required",
        "tradeValue": "retained-log-latent-to-clamped-exp-request",
        "tradeShare": "retained-logit-to-sigmoid-request",
        "categories": "retained-native-action-codes",
        "normalization": "fixed-semantic-transforms-no-fitted-moments",
        "inputValidation": "retained-shapes-finite-categories-catalog-masks-v1",
    }


def action_adapter_digest() -> str:
    return digest(action_adapter_contract())


# ---------------------------------------------------------------------------
# Binding and population
# ---------------------------------------------------------------------------


def _check_binding(expected_layout: Any, expected_adapter_digest: Any) -> None:
    """Mandatory checkpoint binding: the caller's layout and adapter digest must match."""
    if not isinstance(expected_layout, Mapping):
        raise ValueError("pretrained action layout is required to bind the canonical adapter")
    if dict(expected_layout) != action_layout_state():
        raise ValueError("pretrained action layout differs from the canonical actor adapter")
    if expected_adapter_digest is None:
        raise ValueError(
            "checkpoint action layout requires a population-aware action adapter digest"
        )
    if expected_adapter_digest != action_adapter_digest():
        raise ValueError(
            "pretrained action adapter digest differs from the population-aware contract"
        )


def _population(catalog: LaborCatalog) -> int:
    population = getattr(catalog, "population", None)
    if (
        isinstance(population, (bool, np.bool_))
        or not isinstance(population, (int, np.integer))
        or population < 0
    ):
        raise ValueError("catalog population must be a nonnegative integer")
    return int(population)


def _encode(action: Mapping[str, Any], catalog: LaborCatalog) -> torch.Tensor:
    population = _population(catalog)
    validate_schema12_action(action, catalog, retry=False)
    vector = _advantage.action_features(action, population=population)
    if not np.isfinite(vector).all():
        raise ValueError("canonical action features must be finite")
    return torch.from_numpy(vector).detach()


# ---------------------------------------------------------------------------
# Public encoders
# ---------------------------------------------------------------------------


def canonical_action_features(
    action: Mapping[str, Any],
    catalog: LaborCatalog,
    *,
    expected_layout: Mapping[str, Any],
    expected_adapter_digest: str,
) -> torch.Tensor:
    """Encode a submitted request using exactly the search-training layout.

    Inputs must already use wire units; resident counts, worker-years, logits,
    or arbitrary scores are rejected where shares are required. The catalog
    validates legal specialty/activity masks. Nonlabor categories and the
    frozen trade transform are checked by the existing feature encoder. The
    selected root catalog supplies population, including zero; omission never
    selects the encoder's legacy raw-share path. The consuming checkpoint's
    layout and adapter digest are required.
    """
    _check_binding(expected_layout, expected_adapter_digest)
    return _encode(action, catalog)


def actor_action_features(
    labor_sample: Mapping[str, torch.Tensor],
    nonlabor_sample: Mapping[str, torch.Tensor],
    catalog: LaborCatalog,
    *,
    index: int = 0,
    expected_layout: Mapping[str, Any],
    expected_adapter_digest: str,
) -> torch.Tensor:
    """Decode one retained actor sample row before applying the critic encoding.

    For a retry, pass the retry labor sample and the retained initial nonlabor
    sample and decision-boundary catalog for that parent. In a batch, supply
    the catalog belonging to ``index``; its population is not a batch aggregate.
    Do not draw a new action or feed head means/logits.

    Validation covers the whole batch's structure (fields, shapes, dtypes,
    movement registry) and the values of the selected row only, so the cost
    does not grow with the batch. :func:`actor_action_features_batch` encodes
    every row and is bitwise equal to this function row by row.
    """
    _check_binding(expected_layout, expected_adapter_digest)
    batch = _validate_structure(labor_sample, nonlabor_sample)
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < batch:
        raise ValueError("actor sample index must select an existing row")
    _population(catalog)
    rows = slice(index, index + 1)
    _validate_values(labor_sample, nonlabor_sample, rows, [catalog])
    labor_rows = _labor_lists(labor_sample, rows)
    return _encode(_decode_row(labor_rows, 0, nonlabor_sample, index), catalog)


def actor_action_features_batch(
    labor_sample: Mapping[str, torch.Tensor],
    nonlabor_sample: Mapping[str, torch.Tensor],
    catalogs: Sequence[LaborCatalog],
    *,
    expected_layout: Mapping[str, Any],
    expected_adapter_digest: str,
) -> torch.Tensor:
    """Encode every row of a retained actor batch, ``[B, 349]`` float32 CPU.

    ``catalogs[i]`` is row ``i``'s own decision-boundary catalog (its masks
    and population); there is no batch-wide population. The batch is validated
    once (each row's labor masks against its own catalog), then each row is
    decoded and encoded through the same internals as
    :func:`actor_action_features`, so row ``i`` is bitwise equal to
    ``actor_action_features(..., catalogs[i], index=i, ...)``.
    """
    _check_binding(expected_layout, expected_adapter_digest)
    batch = _validate_structure(labor_sample, nonlabor_sample)
    if isinstance(catalogs, (str, bytes, Mapping)) or not isinstance(catalogs, Sequence):
        raise ValueError("actor batch needs one catalog per row")
    catalogs = list(catalogs)
    if len(catalogs) != batch:
        raise ValueError("actor batch needs exactly one catalog per row")
    for catalog in catalogs:
        _population(catalog)
    if batch == 0:
        return torch.zeros((0, _advantage.ACTION_FEATURE_WIDTH), dtype=torch.float32)
    rows = slice(0, batch)
    _validate_values(labor_sample, nonlabor_sample, rows, catalogs)
    labor_rows = _labor_lists(labor_sample, rows)
    return torch.stack(
        [
            _encode(_decode_row(labor_rows, index, nonlabor_sample, index), catalog)
            for index, catalog in enumerate(catalogs)
        ]
    )


# ---------------------------------------------------------------------------
# Categorical counterfactuals
# ---------------------------------------------------------------------------


def movement_union_legal(nonlabor_sample: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Retained movement legality in movement-union order, ``[B, 17]`` bool CPU.

    Pair with ``categorical_counterfactuals(..., "movement")``, whose category
    axis follows ``MOVEMENT_UNION_CODES``. Union codes absent from the retained
    registry are illegal. Fission and trade gates have no legality masks.
    """
    if not isinstance(nonlabor_sample, Mapping):
        raise ValueError("actor nonlabor sample must be a mapping")
    codes, mask = nonlabor_sample.get("movement_codes"), nonlabor_sample.get("movement_mask")
    if (
        not isinstance(codes, torch.Tensor)
        or codes.dtype != torch.long
        or codes.ndim != 1
        or not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or mask.ndim != 2
        or mask.shape[1] != len(codes)
    ):
        raise ValueError("actor movement codes/mask are malformed")
    code_list = codes.cpu().tolist()
    if len(set(code_list)) != len(code_list) or any(
        code not in MOVEMENT_UNION_CODES for code in code_list
    ):
        raise ValueError("actor movement codes differ from the native action union")
    out = torch.zeros((mask.shape[0], len(MOVEMENT_UNION_CODES)), dtype=torch.bool)
    out[:, [MOVEMENT_UNION_CODES.index(code) for code in code_list]] = mask.cpu()
    return out


def categorical_counterfactuals(
    features: torch.Tensor,
    head: str,
    *,
    group: int | None = None,
    expected_layout: Mapping[str, Any],
    expected_adapter_digest: str,
) -> torch.Tensor:
    """Encoded actions with one categorical head set to each of its categories.

    ``features`` are already-encoded executed actions ``[N, 349]`` (from this
    adapter). The result is ``[N, K, 349]`` float32 CPU: row ``n`` with
    ``head`` (and, for ``trade_gate``, the trade ``group`` in canonical
    ``contracts.TRADE_GROUPS`` order) replaced by category ``k``. Category
    order: ``MOVEMENT_UNION_CODES`` for movement, the native policy ordinal for
    fission, the native gate ordinal for trade gates. No other column is
    touched. That is valid only because the encoder writes each categorical
    field into its own block alone; :func:`_require_isolated` verifies this on
    the live encoder and refuses the head otherwise. Illegal movement categories
    are included; mask them with :func:`movement_union_legal`.
    """
    _check_binding(expected_layout, expected_adapter_digest)
    if head not in CATEGORICAL_HEADS:
        raise ValueError(
            f"counterfactuals exist only for categorical heads {sorted(CATEGORICAL_HEADS)}"
        )
    if head == "trade_gate":
        if (
            isinstance(group, bool)
            or not isinstance(group, int)
            or not 0 <= group < len(contracts.TRADE_GROUPS)
        ):
            raise ValueError("trade-gate counterfactuals need a canonical trade group index")
    elif group is not None:
        raise ValueError(f"{head} counterfactuals take no trade group")
    if (
        not isinstance(features, torch.Tensor)
        or features.dtype != torch.float32
        or features.ndim != 2
        or features.shape[1] != _advantage.ACTION_FEATURE_WIDTH
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError(
            f"features must be finite float32 [N, {_advantage.ACTION_FEATURE_WIDTH}] vectors"
        )
    categories = CATEGORICAL_HEADS[head][1]
    columns = _head_columns(head, group)
    current = features[:, columns]
    # Only executed (hard) encodings: a relaxed or perturbed block is refused.
    if not bool(((current == 0) | (current == 1)).all()) or not bool(
        (current.sum(dim=1) == 1).all()
    ):
        raise ValueError(f"features do not carry an exact {head} one-hot")
    _require_isolated(head, group)
    out = features.detach().cpu().unsqueeze(1).repeat(1, categories, 1)
    out[:, :, columns] = torch.eye(categories, dtype=torch.float32)
    return out


def _head_columns(head: str, group: int | None) -> slice:
    block, categories = CATEGORICAL_HEADS[head]
    span = block_slices()[block]
    if head == "trade_gate":
        start = span.start + int(group) * categories
        if span.stop - span.start != len(contracts.TRADE_GROUPS) * categories:
            raise ValueError("trade gate block width differs from groups x gates")
        return slice(start, start + categories)
    if span.stop - span.start != categories:
        raise ValueError(f"{head} block width differs from its category count")
    return span


_ISOLATION_CACHE: dict[tuple[str, int | None, Any], bool] = {}


def _isolation_probes() -> list[tuple[dict[str, Any], int]]:
    """Fixed wire actions with nonneutral trade/labor posture, and their populations."""
    groups = len(contracts.TRADE_GROUPS)
    specialty = [0.0] * SPECIALTY_ROWS
    specialty[0], specialty[1], specialty[2] = 0.25, 0.5, 0.25
    activity = [[1.0] + [0.0] * (ACTIVITY_ROWS - 1) for _ in range(ROLE_ROWS)]
    activity[0] = [0.5, 0.5] + [0.0] * (ACTIVITY_ROWS - 2)
    base = {
        "schemaVersion": 12,
        "laborSpecialtyAllocation": specialty,
        "laborActivityAllocation": activity,
        "movement": -1,
        "fission": 0,
        "tradeGate": [0] * groups,
        "tradeValue": [0.5 + 0.5 * g for g in range(groups)],
        "tradeShare": 0.3,
    }
    other = dict(
        base,
        movement=MOVEMENT_UNION_CODES[-1],
        fission=contracts.FISSION_POLICIES - 1,
        tradeGate=[g % contracts.TRADE_GATES for g in range(groups)],
        tradeValue=[3.0] * groups,
        tradeShare=0.9,
    )
    return [(base, 7), (other, 7), (base, 0)]


def _with_category(action: Mapping[str, Any], head: str, group: int | None, k: int) -> dict:
    out = dict(action)
    if head == "movement":
        out["movement"] = MOVEMENT_UNION_CODES[k]
    elif head == "fission":
        out["fission"] = k
    else:
        gates = list(out["tradeGate"])
        gates[int(group)] = k
        out["tradeGate"] = gates
    return out


def _require_isolated(head: str, group: int | None) -> None:
    """Verify on the live encoder that ``head`` writes only its own one-hot columns.

    Each fixed probe action is re-encoded at every category of the head; any
    change outside the head's columns, or a non-one-hot value inside them,
    refuses the head (for example, a gate that suppressed trade-value
    features). The verdict is cached per encoder function object.
    """
    encoder = _advantage.action_features
    key = (head, group, encoder)
    if key not in _ISOLATION_CACHE:
        columns = _head_columns(head, group)
        categories = CATEGORICAL_HEADS[head][1]
        isolated = True
        for action, population in _isolation_probes():
            reference = encoder(action, population=population)
            for k in range(categories):
                varied = encoder(_with_category(action, head, group, k), population=population)
                expected = reference.copy()
                expected[columns] = 0.0
                expected[columns.start + k] = 1.0
                if varied.dtype != expected.dtype or not np.array_equal(varied, expected):
                    isolated = False
                    break
            if not isolated:
                break
        _ISOLATION_CACHE[key] = isolated
    if not _ISOLATION_CACHE[key]:
        raise ValueError(
            f"the action encoder couples the {head} head to other features; "
            "counterfactuals require full re-encoding"
        )


# ---------------------------------------------------------------------------
# Retained-sample validation and decoding
# ---------------------------------------------------------------------------


def _tensor(values, name, shape, dtype=None):
    value = values[name]
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"actor {name} shape differs from the retained sample contract")
    if dtype is None:
        if not value.is_floating_point():
            raise ValueError(f"actor {name} must contain finite floating values")
    elif value.dtype != dtype:
        raise ValueError(f"actor {name} dtype differs from the retained sample contract")
    return value


def _validate_structure(labor, nonlabor) -> int:
    """Fields, shapes, dtypes and the movement registry; independent of row values."""
    if not isinstance(labor, Mapping) or set(labor) != _LABOR_FIELDS:
        raise ValueError("actor labor sample fields are incomplete or unexpected")
    if not isinstance(nonlabor, Mapping) or set(nonlabor) != _NONLABOR_FIELDS:
        raise ValueError("actor nonlabor sample fields are incomplete or unexpected")
    specialty = labor["specialty_allocation"]
    if not isinstance(specialty, torch.Tensor) or specialty.ndim != 2:
        raise ValueError("actor specialty allocation must be a batched tensor")
    batch = specialty.shape[0]
    for name, shape in (
        ("specialty", (batch, SPECIALTY_ROWS)),
        ("activity", (batch, ROLE_ROWS, ACTIVITY_ROWS)),
    ):
        _tensor(labor, name + "_allocation", shape)
        _tensor(labor, name + "_latent", shape)
        _tensor(labor, name + "_mask", shape, torch.bool)
    codes = nonlabor["movement_codes"]
    if not isinstance(codes, torch.Tensor) or codes.ndim != 1 or not len(codes):
        raise ValueError("actor movement codes must be a nonempty vector")
    codes = _tensor(nonlabor, "movement_codes", (len(codes),), torch.long)
    code_list = codes.cpu().tolist()
    if len(set(code_list)) != len(code_list) or any(
        code not in MOVEMENT_UNION_CODES for code in code_list
    ):
        raise ValueError("actor movement codes differ from the native action union")
    _tensor(nonlabor, "movement_index", (batch,), torch.long)
    _tensor(nonlabor, "movement", (batch,), torch.long)
    _tensor(nonlabor, "movement_mask", (batch, len(codes)), torch.bool)
    _tensor(nonlabor, "fission", (batch,), torch.long)
    _tensor(nonlabor, "trade_gate", (batch, len(contracts.TRADE_GROUPS)), torch.long)
    _tensor(nonlabor, "trade_value_latent", (batch, len(contracts.TRADE_GROUPS)))
    _tensor(nonlabor, "trade_share_latent", (batch,))
    return batch


_MASK_CACHE: dict[int, tuple[Any, torch.Tensor, torch.Tensor]] = {}


def _catalog_masks(catalog: LaborCatalog) -> tuple[torch.Tensor, torch.Tensor]:
    masks = catalog.masks
    cached = _MASK_CACHE.get(id(masks))
    if cached is None or cached[0] is not masks:
        cached = (
            masks,
            torch.tensor(masks.specialty, dtype=torch.bool),
            torch.tensor(masks.activity, dtype=torch.bool),
        )
        if len(_MASK_CACHE) >= 4096:
            _MASK_CACHE.clear()
        _MASK_CACHE[id(masks)] = cached
    return cached[1], cached[2]


def _validate_values(labor, nonlabor, rows: slice, catalogs: Sequence[LaborCatalog]) -> None:
    """Reject malformed row values before int/exp/sigmoid or simplex serialization.

    ``rows`` is a contiguous row range and ``catalogs`` its catalogs, one per
    row. Latents remain likelihood bookkeeping: this adapter neither refits nor
    reconstructs them from rounded requests.
    """
    expected = [_catalog_masks(catalog) for catalog in catalogs]
    for name, position in (("specialty", 0), ("activity", 1)):
        allocation = labor[name + "_allocation"].detach()[rows]
        latent = labor[name + "_latent"].detach()[rows]
        retained_mask = labor[name + "_mask"].detach()[rows]
        if not bool(torch.isfinite(allocation).all()):
            raise ValueError(f"actor {name}_allocation must contain finite floating values")
        if not bool(torch.isfinite(latent).all()):
            raise ValueError(f"actor {name}_latent must contain finite floating values")
        catalog_mask = torch.stack([masks[position] for masks in expected])
        if not torch.equal(retained_mask.cpu(), catalog_mask):
            raise ValueError(f"actor {name} mask differs from the selected catalog")
        # Match the actor likelihood's float32 tolerance before the serializer
        # corrects one coordinate for the stricter wire contract. Do not repair
        # arbitrary counts or non-simplex input by passing it to that serializer.
        if (
            bool(((allocation < 0) | (allocation > 1)).any())
            or bool((allocation.masked_select(~retained_mask) != 0).any())
            or not torch.allclose(
                allocation.sum(dim=-1),
                retained_mask.any(dim=-1).to(allocation.dtype),
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise ValueError(f"actor {name} allocation must be a masked simplex")
    codes = nonlabor["movement_codes"].detach().cpu()
    indices = nonlabor["movement_index"].detach()[rows].cpu()
    movement = nonlabor["movement"].detach()[rows].cpu()
    mask = nonlabor["movement_mask"].detach()[rows].cpu()
    if bool(((indices < 0) | (indices >= len(codes))).any()):
        raise ValueError("actor movement index is out of range")
    selected = torch.arange(len(indices))
    if not torch.equal(movement, codes[indices]) or not bool(mask[selected, indices].all()):
        raise ValueError("actor movement does not match its retained legal sample")
    fission = nonlabor["fission"].detach()[rows]
    gates = nonlabor["trade_gate"].detach()[rows]
    if bool(((fission < 0) | (fission >= contracts.FISSION_POLICIES)).any()):
        raise ValueError("actor fission category is out of range")
    if bool(((gates < 0) | (gates >= contracts.TRADE_GATES)).any()):
        raise ValueError("actor trade gate category is out of range")
    for name in ("trade_value_latent", "trade_share_latent"):
        if not bool(torch.isfinite(nonlabor[name].detach()[rows]).all()):
            raise ValueError(f"actor {name} must contain finite floating values")


def _labor_lists(labor, rows: slice) -> tuple[list, list]:
    """Row-range labor allocations as Python floats (exact float32 -> float64)."""
    return (
        labor["specialty_allocation"].detach()[rows].cpu().tolist(),
        labor["activity_allocation"].detach()[rows].cpu().tolist(),
    )


def _decode_row(
    labor_rows: tuple[list, list], local: int, nonlabor: Mapping[str, torch.Tensor], index: int
) -> dict[str, Any]:
    """The wire action of ``schema12_initial_action(labor, nonlabor, index)``, bitwise.

    Labor rows come pre-converted (``labor_rows[.][local]``); the trade
    transforms run on the row's own tensors and device exactly as the
    serializer does. Tested equal to ``schema12_initial_action`` on every row.
    """
    specialty, activity = labor_rows
    labor = LaborPolicySample(
        tuple(specialty[local]), tuple(tuple(row) for row in activity[local])
    ).payload()
    value = nonlabor["trade_value_latent"][index]
    trade_share = torch.sigmoid(nonlabor["trade_share_latent"][index])
    low, high = contracts.TRADE_VALUE_RANGE
    return {
        "schemaVersion": 12,
        "movement": int(nonlabor["movement"][index]),
        "fission": int(nonlabor["fission"][index]),
        "tradeValue": torch.exp(value).clamp(low, high).detach().cpu().tolist(),
        "tradeGate": [int(item) for item in nonlabor["trade_gate"][index].detach().cpu().tolist()],
        "tradeShare": float(trade_share.detach().cpu()),
        **labor,
    }
