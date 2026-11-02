"""Pinned actor/heuristic/search action semantics for the existing critic encoder.

Normalize by decoding to the actual schema-12 request, never by matching head
statistics. The pretrained layout consumes shares, categories and a declared
trade-value transform; compiled resident/worker-year arrays are not its input.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from .labor_advantage import action_features, action_layout_state
from .labor_bridge import LaborCatalog
from .labor_model import schema12_initial_action
from .labor_ppo import validate_schema12_action
from .provenance import digest


def action_adapter_contract() -> dict[str, Any]:
    return {
        "kind": "labor-canonical-actor-action-adapter-v1",
        "actionSchema": 12,
        "criticLayout": action_layout_state(),
        "labor": "retained-masked-ALR-sample-to-canonical-simplex-request",
        "tradeValue": "retained-log-latent-to-clamped-exp-request",
        "tradeShare": "retained-logit-to-sigmoid-request",
        "categories": "retained-native-action-codes",
        "normalization": "fixed-semantic-transforms-no-fitted-moments",
    }


def action_adapter_digest() -> str:
    return digest(action_adapter_contract())


def canonical_action_features(
    action: Mapping[str, Any],
    catalog: LaborCatalog,
    *,
    expected_layout: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Encode a submitted request using exactly the search-training layout.

    Inputs must already use wire units; resident counts, worker-years, logits,
    or arbitrary scores are rejected where shares are required. The catalog
    validates legal specialty/activity masks. Nonlabor categories and the
    frozen trade transform are checked by the existing feature encoder.
    """
    if expected_layout is not None and dict(expected_layout) != action_layout_state():
        raise ValueError("pretrained action layout differs from the canonical actor adapter")
    validate_schema12_action(action, catalog, retry=False)
    vector = action_features(action)
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
) -> torch.Tensor:
    """Decode retained actor samples before applying pretrained input transforms.

    For a retry, pass the retry labor sample and the retained initial nonlabor
    sample for that parent. Do not draw a new action or feed head means/logits.
    """
    action = schema12_initial_action(labor_sample, nonlabor_sample, index)
    return canonical_action_features(action, catalog, expected_layout=expected_layout)
