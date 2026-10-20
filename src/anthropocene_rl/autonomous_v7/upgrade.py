"""Exact checkpoint upgrades: contract 8 to 9, and contract 9 to 10.

Contract 9 appends ``contracts.COMMITMENT_FEATURES`` features to the end of the core observation and adds the
commitment span head. The core encoder's input is ``[core, core_present]`` (plus the critic's privileged block), so
the upgrade inserts zero columns for the new features in both halves, extends the normalizer with identity
moments, and adds a span head that starts at about even odds for "this turn only". The upgraded policy's
contract-8 outputs are unchanged, and its deterministic span is "this turn only".

Contract 10 appends the ``regions`` entity set. Its pooled block enters each encoder's trunk through new zero
columns, so every earlier output is unchanged; the set encoder itself gets a fixed-seed default initialisation, the
normalizer identity moments, and the actor the region movement heads (native-biased, zero mix offset).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from . import contracts
from .model import SetEncoder, init_span_head, NATIVE_START

_CORE_KEY = "encoder.core.0.weight"
_TRUNK_KEY = "encoder.trunk.0.weight"
_REGIONS = next(s for s in contracts.SETS if s[0] == "regions")


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


def _to_9(state: dict[str, Any]) -> dict[str, Any]:
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
    state["contract"] = 9
    return state


def _add_regions(net: dict[str, torch.Tensor], seed: int, actor: bool) -> dict[str, torch.Tensor]:
    net = dict(net)
    if "encoder.sets.regions.row.0.weight" in net:
        return net
    hidden = net["encoder.sets.sites.row.2.weight"].shape[0]
    w = net[_TRUNK_KEY]
    net[_TRUNK_KEY] = torch.cat([w, torch.zeros(w.shape[0], 3 * hidden + 1, dtype=w.dtype)], dim=1)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        encoder = SetEncoder(_REGIONS[2], hidden)
    for name, value in encoder.state_dict().items():
        net["encoder.sets.regions." + name] = value.detach().clone()
    if actor:
        width = net["stay.weight"].shape[1]
        others = contracts.REGION_MOVEMENT_SLOTS - 1
        for name, (rows, cols) in {"region_native": (1, width), "region_keep": (1, width),
                                   "region_query": (hidden, width), "mix_region": (len(contracts.ECONOMY_MIX), hidden)}.items():
            net[name + ".weight"] = torch.zeros(rows, cols)
            net[name + ".bias"] = torch.zeros(rows)
        net["region_native.bias"] = torch.full((1,), float(torch.log(torch.tensor(NATIVE_START / (1 - NATIVE_START) * others))))
    return net


def _to_10(state: dict[str, Any]) -> dict[str, Any]:
    state = dict(state)
    for seed, key in enumerate(("actor", "teacher_actor", "critic", "cost_critic")):
        if key in state:
            state[key] = _add_regions(state[key], 10_000 + seed, actor=key in ("actor", "teacher_actor"))
    normalizer = dict(state["normalizer"])
    normalizer.setdefault("regions_mean", [0.0] * _REGIONS[2])
    normalizer.setdefault("regions_std", [1.0] * _REGIONS[2])
    state["normalizer"] = normalizer
    state["contract"] = 10
    return state


def upgrade(state: dict[str, Any]) -> dict[str, Any]:
    """Return a current-contract checkpoint state; current states pass through unchanged."""
    contract = int(state.get("contract", 8))
    if contract >= contracts.NATIVE_CONTRACT:
        return state
    if contract not in (8, 9):
        raise ValueError(f"no upgrade path from contract {contract}")
    original = contract
    if contract == 8:
        state = _to_9(state); contract = 9
    state = _to_10(state)
    state.setdefault("upgraded_from", original)  # a contract-9 state upgraded from 8 keeps its first origin
    return state
