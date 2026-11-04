"""Pinned actor/heuristic/search action semantics for the existing critic encoder.

Decode the retained schema-12 request, then apply the encoder's population-aware
native apportionment. No producer-statistic matching or duplicate allocation
algorithm is used here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from . import contracts
from .labor_advantage import MOVEMENT_UNION_CODES, action_features, action_layout_state
from .labor_bridge import ACTIVITY_ROWS, ROLE_ROWS, SPECIALTY_ROWS, LaborCatalog
from .labor_model import schema12_initial_action
from .labor_ppo import validate_schema12_action
from .provenance import digest


def action_adapter_contract() -> dict[str, Any]:
    return {
        "kind": "labor-canonical-actor-action-adapter-v2",
        "actionSchema": 12,
        "criticLayout": action_layout_state(),
        "labor": "retained-masked-ALR-sample-to-canonical-simplex-request",
        "population": "selected-root-catalog-nonnegative-integer-required",
        "specialty": "post-native-largest-remainder-apportionment",
        "unstaffedActivity": "idle",
        "tradeValue": "retained-log-latent-to-clamped-exp-request",
        "tradeShare": "retained-logit-to-sigmoid-request",
        "categories": "retained-native-action-codes",
        "normalization": "fixed-semantic-transforms-no-fitted-moments",
        "inputValidation": "retained-shapes-finite-categories-catalog-masks-v1",
    }


def action_adapter_digest() -> str:
    return digest(action_adapter_contract())


def canonical_action_features(
    action: Mapping[str, Any],
    catalog: LaborCatalog,
    *,
    expected_layout: Mapping[str, Any] | None = None,
    expected_adapter_digest: str | None = None,
) -> torch.Tensor:
    """Encode a submitted request using exactly the search-training layout.

    Inputs must already use wire units; resident counts, worker-years, logits,
    or arbitrary scores are rejected where shares are required. The catalog
    validates legal specialty/activity masks. Nonlabor categories and the
    frozen trade transform are checked by the existing feature encoder. The
    selected root catalog supplies population, including zero; omission never
    selects the encoder's legacy raw-share path. Checkpoint consumers must bind
    the adapter digest as well as layout, which alone does not identify these
    population semantics.
    """
    if expected_layout is not None and dict(expected_layout) != action_layout_state():
        raise ValueError("pretrained action layout differs from the canonical actor adapter")
    if expected_layout is not None and expected_adapter_digest is None:
        raise ValueError(
            "checkpoint action layout requires a population-aware action adapter digest"
        )
    if expected_adapter_digest is not None and expected_adapter_digest != action_adapter_digest():
        raise ValueError(
            "pretrained action adapter digest differs from the population-aware contract"
        )
    population = getattr(catalog, "population", None)
    if (
        isinstance(population, (bool, np.bool_))
        or not isinstance(population, (int, np.integer))
        or population < 0
    ):
        raise ValueError("catalog population must be a nonnegative integer")
    validate_schema12_action(action, catalog, retry=False)
    vector = action_features(action, population=int(population))
    if not np.isfinite(vector).all():
        raise ValueError("canonical action features must be finite")
    return torch.from_numpy(vector).detach()


def actor_action_features(
    labor_sample: Mapping[str, torch.Tensor],
    nonlabor_sample: Mapping[str, torch.Tensor],
    catalog: LaborCatalog,
    *,
    index: int = 0,
    expected_layout: Mapping[str, Any] | None = None,
    expected_adapter_digest: str | None = None,
) -> torch.Tensor:
    """Decode retained actor samples before applying pretrained input transforms.

    For a retry, pass the retry labor sample and the retained initial nonlabor
    sample and decision-boundary catalog for that parent. In a batch, supply
    the catalog belonging to ``index``; its population is not a batch aggregate.
    Do not draw a new action or feed head means/logits.
    """
    _validate_retained_samples(labor_sample, nonlabor_sample, catalog, index)
    action = schema12_initial_action(labor_sample, nonlabor_sample, index)
    return canonical_action_features(
        action,
        catalog,
        expected_layout=expected_layout,
        expected_adapter_digest=expected_adapter_digest,
    )


def _validate_retained_samples(labor, nonlabor, catalog, index):
    """Reject malformed samples before int/exp/sigmoid or simplex serialization.

    Latents remain likelihood bookkeeping: this adapter neither refits nor
    reconstructs them from rounded requests. Only the selected row's catalog
    is supplied; the other rows may belong to different catalogs.
    """
    labor_fields = {
        f"{kind}_{field}"
        for kind in ("specialty", "activity")
        for field in ("allocation", "latent", "mask")
    }
    nonlabor_fields = {
        "movement",
        "movement_index",
        "movement_codes",
        "movement_mask",
        "fission",
        "trade_gate",
        "trade_value_latent",
        "trade_share_latent",
    }
    if not isinstance(labor, Mapping) or set(labor) != labor_fields:
        raise ValueError("actor labor sample fields are incomplete or unexpected")
    if not isinstance(nonlabor, Mapping) or set(nonlabor) != nonlabor_fields:
        raise ValueError("actor nonlabor sample fields are incomplete or unexpected")
    specialty = labor["specialty_allocation"]
    if not isinstance(specialty, torch.Tensor) or specialty.ndim != 2:
        raise ValueError("actor specialty allocation must be a batched tensor")
    batch = specialty.shape[0]
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < batch:
        raise ValueError("actor sample index must select an existing row")

    def tensor(values, name, shape, dtype=None):
        value = values[name]
        if not isinstance(value, torch.Tensor) or value.shape != shape:
            raise ValueError(f"actor {name} shape differs from the retained sample contract")
        if dtype is None:
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"actor {name} must contain finite floating values")
        elif value.dtype != dtype:
            raise ValueError(f"actor {name} dtype differs from the retained sample contract")
        return value.detach()

    for name, shape, mask in (
        ("specialty", (batch, SPECIALTY_ROWS), catalog.masks.specialty),
        ("activity", (batch, ROLE_ROWS, ACTIVITY_ROWS), catalog.masks.activity),
    ):
        allocation = tensor(labor, name + "_allocation", shape)
        tensor(labor, name + "_latent", shape)
        retained_mask = tensor(labor, name + "_mask", shape, torch.bool)
        if not torch.equal(retained_mask[index].cpu(), torch.tensor(mask, dtype=torch.bool)):
            raise ValueError(f"actor {name} mask differs from the selected catalog")
        selected, selected_mask = allocation[index], retained_mask[index]
        # Match the actor likelihood's float32 tolerance before the serializer
        # corrects one coordinate for the stricter wire contract. Do not repair
        # arbitrary counts or non-simplex input by passing it to that serializer.
        if (
            bool(((selected < 0) | (selected > 1)).any())
            or bool((selected.masked_select(~selected_mask) != 0).any())
            or not torch.allclose(
                selected.sum(dim=-1),
                selected_mask.any(dim=-1).to(selected.dtype),
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise ValueError(f"actor {name} allocation must be a masked simplex")
    codes = nonlabor["movement_codes"]
    if not isinstance(codes, torch.Tensor) or codes.ndim != 1 or not len(codes):
        raise ValueError("actor movement codes must be a nonempty vector")
    codes = tensor(nonlabor, "movement_codes", (len(codes),), torch.long)
    if len(codes.unique()) != len(codes) or any(
        code not in MOVEMENT_UNION_CODES for code in codes.cpu().tolist()
    ):
        raise ValueError("actor movement codes differ from the native action union")
    movement_index = tensor(nonlabor, "movement_index", (batch,), torch.long)
    movement = tensor(nonlabor, "movement", (batch,), torch.long)
    mask = tensor(nonlabor, "movement_mask", (batch, len(codes)), torch.bool)
    # CPU validation also permits a CPU code registry alongside accelerator samples.
    rows = torch.arange(batch)
    indices = movement_index.cpu()
    if bool(((indices < 0) | (indices >= len(codes))).any()):
        raise ValueError("actor movement index is out of range")
    if not torch.equal(movement.cpu(), codes.cpu()[indices]) or not bool(
        mask.cpu()[rows, indices].all()
    ):
        raise ValueError("actor movement does not match its retained legal sample")
    fission = tensor(nonlabor, "fission", (batch,), torch.long)
    gates = tensor(nonlabor, "trade_gate", (batch, len(contracts.TRADE_GROUPS)), torch.long)
    if bool(((fission < 0) | (fission >= contracts.FISSION_POLICIES)).any()):
        raise ValueError("actor fission category is out of range")
    if bool(((gates < 0) | (gates >= contracts.TRADE_GATES)).any()):
        raise ValueError("actor trade gate category is out of range")
    tensor(nonlabor, "trade_value_latent", (batch, len(contracts.TRADE_GROUPS)))
    tensor(nonlabor, "trade_share_latent", (batch,))
