"""Exact checkpoint upgrade from contract 8 to contract 9.

Contract 9 appends ``contracts.COMMITMENT_FEATURES`` features to the end of the core observation and adds the
commitment span head. The core encoder's input is ``[core, core_present]`` (plus the critic's privileged block), so
the upgrade inserts zero columns for the new features in both halves, extends the normalizer with identity
moments, and adds a span head that starts at about even odds for "this turn only". The upgraded policy's
contract-8 outputs are unchanged, and its deterministic span is "this turn only".
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from . import contracts
from .model import init_span_head

_CORE_KEY = "encoder.core.0.weight"


def _widen(state: dict[str, torch.Tensor], old: int, new: int) -> dict[str, torch.Tensor]:
    state = dict(state)
    w = state[_CORE_KEY]
    extra = w.shape[1] - 2 * old
    if extra < 0:
        raise ValueError(f"core encoder input {w.shape[1]} does not match core width {old}")
    widened = torch.zeros(w.shape[0], 2 * new + extra, dtype=w.dtype)
    widened[:, :old] = w[:, :old]
    widened[:, new:new + old] = w[:, old:2 * old]
    widened[:, 2 * new:] = w[:, 2 * old:]
    state[_CORE_KEY] = widened
    return state


def upgrade(state: dict[str, Any]) -> dict[str, Any]:
    """Return a contract-9 checkpoint state; contract-9 states pass through unchanged."""
    contract = int(state.get("contract", 8))
    if contract >= contracts.NATIVE_CONTRACT:
        return state
    if contract != 8:
        raise ValueError(f"no upgrade path from contract {contract}")
    state = dict(state)
    old = int(state["core_width"]); new = old + contracts.COMMITMENT_FEATURES
    for key in ("actor", "teacher_actor", "critic", "cost_critic"):
        if key in state:
            state[key] = _widen(state[key], old, new)
    for key in ("actor", "teacher_actor"):
        if key in state and "span.weight" not in state[key]:
            width = state[key]["stay.weight"].shape[1]
            layer = nn.Linear(width, len(contracts.SPANS)); init_span_head(layer)
            state[key] = dict(state[key])
            state[key]["span.weight"] = layer.weight.detach().clone()
            state[key]["span.bias"] = layer.bias.detach().clone()
    normalizer = dict(state["normalizer"])
    normalizer["core_mean"] = list(normalizer["core_mean"]) + [0.0] * contracts.COMMITMENT_FEATURES
    normalizer["core_std"] = list(normalizer["core_std"]) + [1.0] * contracts.COMMITMENT_FEATURES
    state["normalizer"] = normalizer
    state["core_width"] = new
    state["contract"] = contracts.NATIVE_CONTRACT
    state["upgraded_from"] = contract
    return state
