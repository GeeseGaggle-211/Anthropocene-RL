"""Stages 2-4 of the model9 curriculum (M7): PPO from the stage-1 checkpoint.

    python -m anthropocene_rl.autonomous_v7.ppo --init <stage1.pt> --stage capability --seeds 1000-1599 --out <dir>

Every village of a world acts with the same shared policy. Actions are sampled from the actor's own heads and their
latent draws are stored, so a minibatch recomputes exact log-probabilities:

* unit strategic controls and service levels: logistic-normal (Gaussian in logit space);
* economy mix (cultivation, herding, fishing): additive logistic-normal on the available coordinates;
* clothing and shelter routes, and (stage 4) movement: masked categorical;
* fission stays native until stages 5-6 and movement until stage 4.

Rewards are the stage-1 lineage rewards (decision D4) and returns use the physical discount exp(-years / 400) with
GAE(lambda). The critic keeps its value equation (u = G / P, two-hot) and its critic-only privileged inputs. A
frozen copy of the stage-1 actor anchors the strategic and service heads with a KL penalty; posture (economy mix)
and routes carry no teacher anchor. The guard on settlements below 30 residents (young daughters exempt) is a
constraint: a separate cost critic and a Lagrange multiplier, not a reward bonus, so being small is never rewarded or
punished beyond the guard.
"""  # noqa: E501 - docstring/prose line kept verbatim

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import contracts, data
from .bridge import Model9Bridge, observation_batch
from .daughter_bank import CAPTURES_PER_WORLD, DaughterBank, Reservoir
from .model import (
    MASKED,
    Actor,
    Critic,
    Normalizer,
    VectorCritic,
    condition_mix,
    mix_soft_cross_entropy,
    region_code,
    region_mode,
    to_tensors,
)

SMALL_SETTLEMENT = 30
YOUNG_DAUGHTER_TURNS = 3


@dataclass(frozen=True)
class Stage:
    name: str
    movement: bool  # the policy chooses movement (else native)
    routes: bool  # the policy chooses clothing/shelter routes (else native)
    description: str
    # Teacher KL on retained field/herd and development levels (service levels always). Freeing them
    # in stage 3 drove the
    # development budget up ~30% and closed-loop welfare down 12% within 20 iterations, so every
    # stage keeps the
    # anchor; the economy mix (posture) and routes stay unanchored.
    anchor_strategic: bool = True
    # Teacher KL on the clothing and shelter route categoricals (stage-3 regression isolation).
    anchor_routes: bool = False
    # Contract 9: the policy chooses a capital commitment span (--commitments); otherwise every turn
    # decides for
    # itself (span 1), exactly as contract 8.
    commitments: bool = False
    # Contract 10 (--learned-movement): the policy moves over region rows with destination intents
    # and chooses the
    # economy mix given the destination; the contract-8 site pointer is not used.
    region_movement: bool = False
    # Contract 10 (--observe-movement): the bridge enables movement, so the policy sees the same
    # movement masks as a
    # learned-movement arm, but every decision stays native (the untrained region head argmaxes to
    # native).
    observe_movement: bool = False
    # Learned fission timing (--learned-fission): the policy chooses the fission policy (Neutral,
    # Discourage,
    # Encourage: the split threshold and daughter share); the destination stays native. Anchored to
    # the teacher.
    fission: bool = False
    # Contract 11 (--learned-trade): the policy sets the trade posture (relative values, offer
    # gates, export share),
    # capital controls anchored to the teacher's native posture.
    trade: bool = False

    @property
    def bridge_movement(self) -> bool:
        return self.movement or self.observe_movement


STAGES = {
    "capability": Stage(
        "capability",
        movement=False,
        routes=False,
        description="stage 2: single-capability worlds; economy levels and mix, native routes and "
        "movement",
    ),
    "mixed": Stage(
        "mixed",
        movement=False,
        routes=True,
        anchor_strategic=True,
        description="stage 3: mixed-use worlds; routes and posture, no teacher anchor on either",
    ),
    # Stage-3 regression isolation: the mixed-use curriculum with native routes (route heads
    # untrained), and with
    # learned routes anchored to the teacher.
    "mixed-native-routes": Stage(
        "mixed-native-routes",
        movement=False,
        routes=False,
        description="stage 3 curriculum, native clothing/shelter routes",
    ),
    "mixed-anchored-routes": Stage(
        "mixed-anchored-routes",
        movement=False,
        routes=True,
        anchor_routes=True,
        description="stage 3 curriculum, learned routes with a teacher KL anchor",
    ),
    "movement": Stage(
        "movement",
        movement=True,
        routes=True,
        anchor_strategic=True,
        description="stage 4: the policy moves settlements over value-ordered sites",
    ),
}


# ---------------------------------------------------------------------------------------------
# distributions
def _normal_logp(z: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    return -0.5 * ((z - mean) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)


def _mix_reference(mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()
    return (torch.arange(mask.shape[1], device=mask.device) * m).argmax(dim=1, keepdim=True)


def _mix_parts(
    out: dict[str, torch.Tensor], mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Means of the free log-ratios, their mask, and the reference coordinate."""
    ref = _mix_reference(mask)
    mu = out["mix_logits"] - out["mix_logits"].gather(1, ref)
    free = mask.float().clone()
    free.scatter_(1, ref, 0.0)
    return mu, free, ref


def explore(out: dict[str, torch.Tensor], floor: float) -> dict[str, torch.Tensor]:
    """Behaviour cloning drives the logistic-normal scales toward the clamp (the teacher is near-deterministic).
    PPO samples, recomputes and anchors with the same floored scales, so ratios stay bounded."""  # noqa: E501 - docstring/prose line kept verbatim
    out = dict(out)
    for key in (
        "strategic_log_std",
        "service_log_std",
        "mix_log_std",
        "trade_value_log_std",
        "trade_share_log_std",
    ):
        if key in out:
            out[key] = out[key].clamp(min=floor)
    return out


def mix_from_ratios(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = torch.where(mask, z, torch.full_like(z, MASKED))
    return torch.softmax(logits, dim=1) * mask.float()


def sample(
    out: dict[str, torch.Tensor],
    t: dict[str, torch.Tensor],
    stage: Stage,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Draw one action per row; returns the latent draws needed to recompute log-probabilities."""

    def gauss(mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        return mean + log_std.exp() * torch.randn(
            mean.shape, generator=generator, device=mean.device
        )

    def categorical(logits: torch.Tensor) -> torch.Tensor:
        return torch.multinomial(torch.softmax(logits, dim=1), 1, generator=generator).squeeze(1)

    mask = t["economy_mix_mask"]
    # Contract 10: the region slot is drawn first and the mix given it.
    # Round 3: the region slot is drawn from the head's joint distribution (hierarchical by default:
    # group, then slot).
    region = (
        (
            torch.multinomial(out["region_probs"], 1, generator=generator).squeeze(1)
            if "region_probs" in out
            else categorical(out[region_key(out)])
        )
        if stage.region_movement
        else torch.full((len(mask),), contracts.REGION_NATIVE, dtype=torch.long)
    )
    if stage.region_movement:
        out = condition_mix(out, region)
    mu, free, _ = _mix_parts(out, mask)
    mix_z = torch.where(free > 0, gauss(mu, out["mix_log_std"]), torch.zeros_like(mu))
    draw = {
        "region": region,
        "strategic_z": gauss(out["strategic_mean"], out["strategic_log_std"]),
        "service_z": gauss(out["service_mean"], out["service_log_std"]),
        "mix_z": mix_z,
        "clothing": categorical(out["clothing"])
        if stage.routes
        else torch.full((len(mu),), -1, dtype=torch.long),
        "shelter": categorical(out["shelter"])
        if stage.routes
        else torch.full((len(mu),), -1, dtype=torch.long),
        "movement": categorical(out["movement"])
        if stage.movement and not stage.region_movement
        else torch.full((len(mu),), -1, dtype=torch.long),
        "span": categorical(out["span"])
        if stage.commitments
        else torch.full((len(mu),), contracts.SPAN_SINGLE, dtype=torch.long),
        "fission": categorical(out["fission"])
        if stage.fission
        else torch.full((len(mu),), -1, dtype=torch.long),
    }
    if stage.trade:
        gate = out["trade_gate"]
        draw["trade_value_z"] = gauss(out["trade_value_mean"], out["trade_value_log_std"])
        draw["trade_gate"] = categorical(gate.reshape(-1, gate.shape[2])).view(
            gate.shape[0], gate.shape[1]
        )
        draw["trade_share_z"] = gauss(out["trade_share_mean"], out["trade_share_log_std"])
    return draw


def _log_prob_with_components(
    out: dict[str, torch.Tensor],
    t: dict[str, torch.Tensor],
    draw: dict[str, torch.Tensor],
    stage: Stage,
    *,
    include_coordinates: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    dict[str, dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
]:
    """Joint log-probability of the policy-chosen parts of an action, and the joint entropy of those heads.

    Contract 9: on a hold row (span 0) the capital controls (strategic levels, economy mix and routes) are the
    committed ones, so they are not part of this turn's action: their draws contribute no log-probability or entropy.
    Service levels and movement are per-turn in every row; the span head counts whenever the stage learns it."""  # noqa: E501 - docstring/prose line kept verbatim
    half_log_2pie = 0.5 * math.log(2 * math.pi * math.e)
    if stage.region_movement:
        out = condition_mix(out, draw["region"])
    mu, free, _ = _mix_parts(out, t["economy_mix_mask"])
    strategic_by_coordinate = _normal_logp(
        draw["strategic_z"], out["strategic_mean"], out["strategic_log_std"]
    )
    strategic_entropy_by_coordinate = out["strategic_log_std"] + half_log_2pie
    strategic = strategic_by_coordinate.sum(dim=1)
    strategic_ent = strategic_entropy_by_coordinate.sum(dim=1)
    economy_mix = (_normal_logp(draw["mix_z"], mu, out["mix_log_std"]) * free).sum(dim=1)
    economy_mix_ent = ((out["mix_log_std"] + half_log_2pie) * free).sum(dim=1)
    service_by_coordinate = _normal_logp(
        draw["service_z"], out["service_mean"], out["service_log_std"]
    )
    service_entropy_by_coordinate = out["service_log_std"] + half_log_2pie
    service = service_by_coordinate.sum(dim=1)
    service_ent = service_entropy_by_coordinate.sum(dim=1)
    capital_lp = strategic + economy_mix
    capital_ent = strategic_ent + economy_mix_ent
    lp = service
    entropy = service_ent
    capital_components = {
        "strategic": (strategic, strategic_ent),
        "economyMix": (economy_mix, economy_mix_ent),
    }
    components = {"service": (service, service_ent)}
    coordinate_arrays = None
    if include_coordinates:
        coordinate_arrays = {
            "strategic": (
                contracts.STRATEGIC[: contracts.UNIT_STRATEGIC],
                strategic_by_coordinate,
                strategic_entropy_by_coordinate,
                True,
            ),
            "service": (
                contracts.SERVICES,
                service_by_coordinate,
                service_entropy_by_coordinate,
                False,
            ),
        }

    def categorical(head: str, key: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        logq = out[key] if key == "region_logp" else torch.log_softmax(out[key or head], dim=1)
        return logq.gather(1, draw[head].clamp(min=0).unsqueeze(1)).squeeze(1), -(
            logq.exp() * logq
        ).sum(dim=1)

    if stage.routes:
        for head, label in (("clothing", "clothingRoute"), ("shelter", "shelterRoute")):
            a, b = categorical(head)
            capital_lp = capital_lp + a
            capital_ent = capital_ent + b
            capital_components[label] = (a, b)
    if stage.trade:
        # The trade posture is a capital control: a hold row re-emits the committed posture.
        trade_value_by_coordinate = _normal_logp(
            draw["trade_value_z"], out["trade_value_mean"], out["trade_value_log_std"]
        )
        trade_value_entropy_by_coordinate = out["trade_value_log_std"] + half_log_2pie
        trade_value = trade_value_by_coordinate.sum(dim=1)
        trade_value_ent = trade_value_entropy_by_coordinate.sum(dim=1)
        trade_share = _normal_logp(
            draw["trade_share_z"], out["trade_share_mean"], out["trade_share_log_std"]
        )
        trade_share_ent = out["trade_share_log_std"] + half_log_2pie
        capital_lp = capital_lp + trade_value
        capital_lp = capital_lp + trade_share
        capital_ent = capital_ent + trade_value_ent + trade_share_ent
        logq = torch.log_softmax(out["trade_gate"], dim=2)
        trade_gate_by_coordinate = logq.gather(
            2, draw["trade_gate"].clamp(min=0).unsqueeze(2)
        ).squeeze(2)
        trade_gate_entropy_by_coordinate = -(logq.exp() * logq).sum(dim=2)
        trade_gate = trade_gate_by_coordinate.sum(dim=1)
        trade_gate_ent = trade_gate_entropy_by_coordinate.sum(dim=1)
        capital_lp = capital_lp + trade_gate
        capital_ent = capital_ent + trade_gate_ent
        capital_components.update(
            {
                "tradeValue": (trade_value, trade_value_ent),
                "tradeShare": (trade_share, trade_share_ent),
                "tradeGate": (trade_gate, trade_gate_ent),
            }
        )
        if coordinate_arrays is not None:
            coordinate_arrays.update(
                {
                    "tradeValue": (
                        contracts.TRADE_GROUPS,
                        trade_value_by_coordinate,
                        trade_value_entropy_by_coordinate,
                        True,
                    ),
                    "tradeGate": (
                        contracts.TRADE_GROUPS,
                        trade_gate_by_coordinate,
                        trade_gate_entropy_by_coordinate,
                        True,
                    ),
                }
            )
    if stage.region_movement:
        a, b = categorical("region", region_key(out))
        lp = lp + a
        entropy = entropy + b
        components["regionMovement"] = (a, b)
    elif stage.movement:
        a, b = categorical("movement")
        lp = lp + a
        entropy = entropy + b
        components["movement"] = (a, b)
    if stage.fission:
        a, b = categorical("fission")
        lp = lp + a
        entropy = entropy + b
        components["fission"] = (a, b)
    decided = torch.ones_like(lp)
    if stage.commitments:
        a, b = categorical("span")
        lp = lp + a
        entropy = entropy + b
        components["commitmentSpan"] = (a, b)
        decided = (draw["span"] != contracts.SPAN_HOLD).float()
    components.update(
        {
            name: (decided * value[0], decided * value[1])
            for name, value in capital_components.items()
        }
    )
    coordinates = {}
    if coordinate_arrays is not None:
        for group, (names, logp, coordinate_entropy, capital) in coordinate_arrays.items():
            active = decided.bool() if capital else torch.ones_like(decided, dtype=torch.bool)
            score = active.to(logp.dtype)
            coordinates[group] = {
                name: (score * logp[:, index], score * coordinate_entropy[:, index], active)
                for index, name in enumerate(names)
            }
    return lp + decided * capital_lp, entropy + decided * capital_ent, components, coordinates


def log_prob(
    out: dict[str, torch.Tensor],
    t: dict[str, torch.Tensor],
    draw: dict[str, torch.Tensor],
    stage: Stage,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint log-probability and entropy, preserving the historical update expression."""
    lp, entropy, _, _ = _log_prob_with_components(out, t, draw, stage)
    return lp, entropy


def log_prob_components(
    out: dict[str, torch.Tensor],
    t: dict[str, torch.Tensor],
    draw: dict[str, torch.Tensor],
    stage: Stage,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Per-head log-probability and entropy terms whose sums reconstruct ``log_prob``."""
    _, _, components, _ = _log_prob_with_components(out, t, draw, stage)
    return components


def log_prob_coordinate_components(
    out: dict[str, torch.Tensor],
    t: dict[str, torch.Tensor],
    draw: dict[str, torch.Tensor],
    stage: Stage,
) -> dict[str, dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Native-labelled coordinate terms for bounded optimizer diagnostics.

    The third tensor is whether this policy coordinate was scored on each row.
    It is deliberately distinct from the simulator's native active-control mask.
    """
    _, _, _, coordinates = _log_prob_with_components(out, t, draw, stage, include_coordinates=True)
    return coordinates


def region_key(out: dict[str, torch.Tensor]) -> str:
    """The region head's distribution: the joint log-probabilities (hierarchical or flat, model.Actor) when present,
    else the raw flat logits (outputs built before round 3)."""  # noqa: E501 - docstring/prose line kept verbatim
    return "region_logp" if "region_logp" in out else "region"


def teacher_kl(
    out: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    strategic: bool = True,
    routes: bool = False,
    fission: bool = False,
    trade: bool = False,
) -> torch.Tensor:
    """KL(policy || teacher) on the anchored heads: service levels, and (stage 2 only) the unit strategic controls.
    The economy mix and routes are never anchored; from stage 3 the retained field/herd levels are free too, so
    pastoral and mixed postures can depart from the field-leaning teacher."""  # noqa: E501 - docstring/prose line kept verbatim

    def gauss_kl(m1, s1, m2, s2):
        return (s2 - s1) + (torch.exp(2 * s1) + (m1 - m2) ** 2) / (2 * torch.exp(2 * s2)) - 0.5

    kl = gauss_kl(
        out["service_mean"],
        out["service_log_std"],
        teacher["service_mean"],
        teacher["service_log_std"],
    ).sum(dim=1)
    if strategic:
        kl = kl + gauss_kl(
            out["strategic_mean"],
            out["strategic_log_std"],
            teacher["strategic_mean"],
            teacher["strategic_log_std"],
        ).sum(dim=1)
    heads = (("clothing", "shelter") if routes else ()) + (("fission",) if fission else ())
    for head in heads:
        logp, logq = torch.log_softmax(out[head], dim=1), torch.log_softmax(teacher[head], dim=1)
        kl = kl + (logp.exp() * (logp - logq)).sum(dim=1)
    if trade:
        kl = kl + gauss_kl(
            out["trade_value_mean"],
            out["trade_value_log_std"],
            teacher["trade_value_mean"],
            teacher["trade_value_log_std"],
        ).sum(dim=1)
        kl = kl + gauss_kl(
            out["trade_share_mean"],
            out["trade_share_log_std"],
            teacher["trade_share_mean"],
            teacher["trade_share_log_std"],
        )
        logp, logq = (
            torch.log_softmax(out["trade_gate"], dim=2),
            torch.log_softmax(teacher["trade_gate"], dim=2),
        )
        kl = kl + (logp.exp() * (logp - logq)).sum(dim=(1, 2))
    return kl


def to_bridge_actions(
    draw: dict[str, torch.Tensor], t: dict[str, torch.Tensor]
) -> list[dict[str, Any]]:
    strategic = torch.sigmoid(draw["strategic_z"]).numpy()
    service = torch.sigmoid(draw["service_z"]).numpy()
    mix = mix_from_ratios(draw["mix_z"], t["economy_mix_mask"]).numpy()
    rest = mix[:, 0] + mix[:, 1]
    field_share = np.where(rest > 1e-9, mix[:, 0] / np.maximum(rest, 1e-12), 0.5)
    actions = []
    for i in range(len(strategic)):
        movement = int(draw["movement"][i])
        # A contract-8 site draw (stage 4) wins; otherwise the contract-10 region slot (native
        # unless learned).
        code = (
            movement - 1
            if movement >= 0
            else region_code(int(draw["region"][i]))
            if "region" in draw
            else contracts.NATIVE_MOVEMENT
        )
        actions.append(
            {
                "strategic": [*strategic[i].tolist(), float(field_share[i]), float(mix[i, 2])],
                "service": service[i].tolist(),
                "clothingRoute": int(draw["clothing"][i]),
                "shelterRoute": int(draw["shelter"][i]),
                "movement": code,
                "fission": int(draw["fission"][i]) if "fission" in draw else -1,
                "commit": contracts.SPANS[int(draw["span"][i])],
            }
        )
        if "trade_value_z" in draw:
            lo, hi = contracts.TRADE_VALUE_RANGE
            actions[-1]["tradeValue"] = np.clip(
                np.exp(draw["trade_value_z"][i].numpy()), lo, hi
            ).tolist()
            actions[-1]["tradeGate"] = [int(g) for g in draw["trade_gate"][i]]
            actions[-1]["tradeShare"] = float(torch.sigmoid(draw["trade_share_z"][i]))
    return actions


def actor_policy_active_mask(action: dict[str, Any], stage: Stage) -> dict[str, Any]:
    """Heads that contributed to this row's policy decision.

    Simulator eligibility is recorded independently by the bridge.  In particular,
    ``observe_movement`` enables the simulator contract without making native
    movement an actor choice, and commitment holds re-emit capital controls without
    scoring their freshly sampled values in the actor loss.
    """
    capital_decision = (
        not stage.commitments
        or int(action.get("commit", contracts.SPAN_SINGLE)) != contracts.SPAN_HOLD
    )
    movement_decision = stage.movement or stage.region_movement
    return {
        "strategic": [capital_decision] * 5,
        "service": [True] * 6,
        "clothingRoute": bool(stage.routes and capital_decision),
        "shelterRoute": bool(stage.routes and capital_decision),
        "movement": bool(movement_decision),
        "intentTile": bool(stage.region_movement),
        "fission": bool(stage.fission),
        "commit": bool(stage.commitments),
        "tradeValue": [bool(stage.trade and capital_decision)] * len(contracts.TRADE_GROUPS),
        "tradeGate": [bool(stage.trade and capital_decision)] * len(contracts.TRADE_GROUPS),
        "tradeShare": bool(stage.trade and capital_decision),
    }


# ---------------------------------------------------------------------------------------------
# rollouts
@dataclass
class Rollout:
    """One decision per row, in the shard layout data.py reads, plus the sampled draw and behaviour log-prob."""  # noqa: E501 - docstring/prose line kept verbatim

    obs: list[dict[str, np.ndarray]] = field(default_factory=list)
    draws: list[dict[str, torch.Tensor]] = field(default_factory=list)
    logp: list[np.ndarray] = field(default_factory=list)
    outcome: dict[str, list] = field(
        default_factory=lambda: {
            k: []
            for k in data.OUTCOME_KEYS  # noqa: RUF005 - concatenation kept (operand type not guaranteed list)
            + ["world", "seed", "mask", "cost", "bootstrap_index", "trade_utility", "bank"]
        }
    )
    bootstrap_obs: list[dict[str, np.ndarray]] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)
    worker_contracts: list[dict] = field(default_factory=list)
    world_contracts: list[dict] = field(default_factory=list)
    economic_rows: list[dict] = field(default_factory=list)
    economic_bytes: int = 0


def village_batch(villages: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Observation arrays plus the critic-only privileged block (never an actor input)."""
    batch = observation_batch(villages)
    if villages and villages[0].get("privileged") is not None:
        batch["privileged"] = np.asarray([v["privileged"] for v in villages], dtype=np.float32)
    return batch


def _cost(v: dict[str, Any], founded_turn: int, turn: int) -> float:
    young = v["parentSociety"] > 0 and turn - founded_turn <= YOUNG_DAUGHTER_TURNS
    return float(not v["terminated"] and v["population"] < SMALL_SETTLEMENT and not young)


def validate_reset_producer(actor, bridge):
    expected = getattr(actor, "run_producer", None)
    if expected is not None:
        from .evaluate import producer_identity
        from .provenance import ProvenanceError

        if producer_identity(bridge) != expected:
            raise ProvenanceError(
                "rollout reset actual bridge/flags/raw-material producer differs from frozen run"
            )


def collect(
    actor: Actor,
    normalizer: Normalizer,
    seeds: list,
    turns: int,
    stage: Stage,
    generator: torch.Generator,
    workers: int = 6,
    floor: float = -2.0,
    personality=None,
    inherit=None,
    bank: DaughterBank | None = None,
    economic_trace: Any | None = None,
) -> Rollout:
    """Run every world for ``turns`` steps with the sampled shared policy (worlds step in lockstep). A world is a
    seed, a (seed, withheld-technology mask) pair, (seed, mask, heuristic warm-up turns) or (seed, mask, 0, bank entry
    id): a daughter-bank world, reset from that entry's snapshot. ``personality``: a sampler of Elder personalities,
    drawn per (world, lineage) and inherited by daughters; None keeps the default personality (no input). ``inherit``:
    a daughter's personality from its parent's (None: an exact copy). ``bank``: capture the world when a daughter is
    first seen founded (at most daughter_bank.CAPTURES_PER_WORLD per world) and resolve bank worlds."""  # noqa: E501 - docstring/prose line kept verbatim
    full = [
        (s, 0, 0, -1)
        if isinstance(s, int)
        else (int(s[0]), int(s[1]), int(s[2]) if len(s) > 2 else 0, int(s[3]) if len(s) > 3 else -1)
        for s in seeds
    ]
    warmups = [w for _, _, w, _ in full]
    worlds = [(s, m) for s, m, _, _ in full]
    banked = [b for _, _, _, b in full]
    if any(b >= 0 for b in banked) and bank is None:
        raise ValueError("bank worlds need the daughter bank")
    held_manifest = getattr(actor, "calibration_manifest", None)
    if held_manifest:
        import hashlib

        from .calibration import assert_training

        assert_training(
            held_manifest,
            [s for s, _, _, _ in full],
            [hashlib.sha256(bank.snapshot(bank.get(b))).hexdigest() for b in banked if b >= 0],
        )
    seeds = [s for s, _ in worlds]
    bridges = [
        Model9Bridge(economic_diagnostics=True) if economic_trace else Model9Bridge() for _ in seeds
    ]
    roll = Rollout()
    traits: list[dict[int, np.ndarray]] = [{} for _ in seeds]
    # Bank captures: per world, a reservoir sample of CAPTURES_PER_WORLD over all founded daughters
    # (keyed by the
    # world), so captures spread over the episode. Kept daughters are captured on founding; evicted
    # ones are deleted.
    reservoirs = [
        Reservoir(CAPTURES_PER_WORLD, np.random.default_rng([s, m, w, b + 1, 0xDA]))
        for s, m, w, b in full
    ]

    def with_personality(
        batch: dict[str, np.ndarray], villages: list[tuple[int, dict]]
    ) -> dict[str, np.ndarray]:
        if personality is None:
            return batch
        # Founders first (societies whose parent is neither known nor present), so a daughter seen
        # together with its
        # parent (the first step of a bank world) inherits with drift whatever the village order.
        present = [{int(v["society"]) for w2, v in villages if w2 == w} for w in range(len(seeds))]
        for w, v in villages:
            society, parent = int(v["society"]), int(v.get("parentSociety", 0))
            if (
                society not in traits[w]
                and parent not in traits[w]
                and (parent == society or parent not in present[w])
            ):
                traits[w][society] = personality()
        rows = []
        for w, v in villages:
            society, parent = int(v["society"]), int(v.get("parentSociety", 0))
            if society not in traits[w]:
                if parent in traits[w]:
                    traits[w][society] = (
                        traits[w][parent] if inherit is None else inherit(traits[w][parent])
                    )
                else:
                    traits[w][society] = personality()
            rows.append(traits[w][society])
        batch["personality"] = np.asarray(rows, dtype=np.float32)
        return batch

    try:
        expected_simulator = getattr(actor, "run_simulator", None)
        if expected_simulator is not None:
            from .provenance import ProvenanceError, simulator_identity

            if any(
                simulator_identity(b.schema, production=True) != expected_simulator for b in bridges
            ):
                raise ProvenanceError("rollout bridge differs from the frozen pre-run simulator")
        from .worker_contract import validate, world

        prepared = []
        for bridge, (seed, mask), warm, entry in zip(
            bridges, worlds, warmups, banked, strict=False
        ):
            record = bank.get(entry) if entry >= 0 else None
            snapshot = bank.snapshot(record) if record is not None else None
            descriptor = world(
                seed,
                mask,
                snapshot_sha256=record["snapshot_sha256"] if record else "",
                turn=record["turn"] if record else None,
                movement=stage.bridge_movement,
                head=getattr(actor, "region_head", None),
            )
            observed = validate(
                bridge,
                getattr(actor, "worker_contract", None),
                descriptor,
                snapshot=snapshot,
                head=getattr(actor, "region_head", None),
            )
            roll.worker_contracts.append(observed)
            roll.world_contracts.append(descriptor)
            prepared.append((bridge, (seed, mask), warm, descriptor, snapshot))
        with ThreadPoolExecutor(max_workers=workers) as pool:

            def start(pair):
                bridge, (seed, mask), warm, descriptor, snapshot = pair
                validate(
                    bridge,
                    getattr(actor, "worker_contract", None),
                    descriptor,
                    snapshot=snapshot,
                    head=getattr(actor, "region_head", None),
                )
                step = bridge.reset(
                    seed, movement=stage.bridge_movement, withheld=mask, snapshot=snapshot
                )
                validate_reset_producer(actor, bridge)
                for _ in range(warm):  # fixture states: the heuristic plays the warm-up turns
                    step = bridge.step({})
                if economic_trace:
                    import base64

                    bridge.economic_reset_snapshot_sha256 = economic_trace.store_snapshot(
                        base64.b64decode(bridge.capture())
                    )
                return step

            steps = list(pool.map(start, prepared))
            founded = [{v["camp"]: s["turn"] for v in s["villages"]} for s in steps]
            for _ in range(turns):
                live = [
                    (w, v)
                    for w, s in enumerate(steps)
                    for v in s["villages"]
                    if v.get("observation") is not None and not v.get("terminated")
                ]
                if not live:
                    break
                batch = with_personality(village_batch([v for _, v in live]), live)
                t = to_tensors(batch, normalizer)
                policy_input_health = None
                if economic_trace:
                    from .economic_trace import capture_policy_input_health

                    # Read only the exact tensors supplied to the actor.  This is
                    # deliberately inside the existing opt-in trace branch.
                    policy_input_health = capture_policy_input_health(batch, t, normalizer)
                with torch.no_grad():
                    out = explore(actor(t), floor)
                    draw = sample(out, t, stage, generator)
                    lp, _ = log_prob(out, t, draw, stage)
                actions = to_bridge_actions(draw, t)
                policy_active_masks = [
                    actor_policy_active_mask(action, stage) for action in actions
                ]
                per_world: list[dict[int, dict]] = [{} for _ in seeds]
                for (w, v), a in zip(live, actions, strict=False):
                    per_world[w][int(v["camp"])] = a
                turn_before = [s["turn"] for s in steps]
                fingerprints_before = [s["fingerprint"] for s in steps] if economic_trace else None
                steps = list(
                    pool.map(
                        lambda pair: pair[0].step(pair[1]), zip(bridges, per_world, strict=False)
                    )
                )
                for bridge in bridges:
                    roll.diagnostics.extend(getattr(bridge, "decision_diagnostics", []))
                action_diagnostics = [
                    {int(record["campId"]): record for record in bridge.decision_diagnostics}
                    for bridge in bridges
                ]
                row_of = {}
                for k, (w, v) in enumerate(live):
                    row_of[(w, int(v["camp"]))] = k
                offset = sum(len(o["core"]) for o in roll.obs)  # noqa: F841 - assignment kept verbatim
                kept = []
                for w, s in enumerate(steps):
                    for v in s["villages"]:
                        new = v["camp"] not in founded[w]
                        founded[w].setdefault(v["camp"], s["turn"])
                        if (
                            bank is not None
                            and new
                            and v.get("founded")
                            and not v.get("terminated")
                        ):
                            # A daughter's first sight: if the reservoir keeps it, its world (before
                            # it decides)
                            # joins the bank, and the daughter it displaces leaves.
                            slot, evicted = reservoirs[w].offer(None)
                            if slot is not None:
                                entry = bank.add(
                                    bridges[w].capture(),
                                    seeds[w],
                                    worlds[w][1],
                                    s["turn"],
                                    int(v["society"]),
                                    int(v["camp"]),
                                    int(v["parentSociety"]),
                                    movement=stage.bridge_movement,
                                )
                                reservoirs[w].items[slot] = int(entry["id"])
                                if evicted is not None:
                                    bank.remove(evicted)
                        k = row_of.get((w, int(v["camp"])))
                        if k is None:
                            continue
                        kept.append(k)
                        if economic_trace:
                            import copy

                            economic = bridges[w].economic_diagnostics.get(int(v["camp"]))
                            if economic is None:
                                raise ValueError("missing prospective economic decision row")
                            action_diagnostic = action_diagnostics[w].get(int(v["camp"]))
                            if action_diagnostic is None:
                                raise ValueError("missing prospective action decision row")
                            roll.economic_rows.append(
                                {
                                    "economicDiagnostics": copy.deepcopy(economic),
                                    "actionDiagnostics": copy.deepcopy(action_diagnostic),
                                    "actorPolicyActiveMask": copy.deepcopy(policy_active_masks[k]),
                                    "policyInputHealth": copy.deepcopy(policy_input_health[k]),
                                    "actorDistribution": {
                                        name: values[k].detach().cpu().tolist()
                                        for name, values in out.items()
                                        if isinstance(values, torch.Tensor)
                                        and values.ndim > 0
                                        and len(values) == len(live)
                                    },
                                    "sampledAction": actions[k],
                                    "provenance": {
                                        "preFingerprint": fingerprints_before[w],
                                        "postFingerprint": s["fingerprint"],
                                        "resetSnapshotHash": bridges[
                                            w
                                        ].economic_reset_snapshot_sha256,
                                        "simulatorCommit": bridges[w].schema.simulator_commit,
                                        "bridgeSha256": bridges[w].schema.bridge_sha256,
                                        **bridges[w].last_reset_identity,
                                    },
                                }
                            )
                            from .economic_trace import canonical_bytes

                            roll.economic_bytes += len(canonical_bytes(roll.economic_rows[-1]))
                            if roll.economic_bytes > 256 * 1024 * 1024:
                                raise ValueError("economic rollout capture exceeds 256 MiB bound")
                        o = roll.outcome
                        # "world" is this iteration's world index: a seed drawn twice (say as a
                        # default world and a
                        # fixture) must not merge sequences.
                        o["world"].append(w)
                        o["seed"].append(seeds[w])
                        o["mask"].append(worlds[w][1])
                        o["camp"].append(v["camp"])
                        o["turn"].append(turn_before[w])
                        o["society"].append(v["society"])
                        o["parent_society"].append(v["parentSociety"])
                        o["prior_population"].append(v["priorPopulation"])
                        o["population"].append(v["population"])
                        o["qol"].append(v["qol"])
                        o["nutrition"].append(v["nutrition"])
                        o["years"].append(v["years"])
                        o["welfare"].append(v["welfare"])
                        o["terminated"].append(v["terminated"])
                        o["trade_utility"].append(float(v.get("tradeUtility", 0.0)))
                        o["cost"].append(_cost(v, founded[w][v["camp"]], s["turn"]))
                        o["bootstrap_index"].append(-1)
                        o["bank"].append(banked[w])
                order = np.asarray(kept, dtype=np.int64)
                roll.obs.append({key: value[order] for key, value in batch.items()})
                roll.draws.append({key: value[order] for key, value in draw.items()})
                roll.logp.append(lp.numpy()[order])
            # Bootstrap observations for sequences truncated at the rollout horizon.
            tail = [
                (w, v)
                for w, s in enumerate(steps)
                for v in s["villages"]
                if v.get("observation") is not None and not v.get("terminated")
            ]
            if tail:
                roll.bootstrap_obs.append(
                    with_personality(village_batch([v for _, v in tail]), tail)
                )
                index = {(w, int(v["camp"])): i for i, (w, v) in enumerate(tail)}
                o = roll.outcome
                last: dict[tuple[int, int], int] = {}
                for row, key in enumerate(zip(o["world"], o["camp"], strict=False)):
                    last[key] = row
                for key, row in last.items():
                    if key in index and not o["terminated"][row]:
                        o["bootstrap_index"][row] = index[key]
    finally:
        for b in bridges:
            b.close()
    return roll


def flatten(roll: Rollout) -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor], np.ndarray]:
    obs = {key: np.concatenate([o[key] for o in roll.obs]) for key in roll.obs[0]}
    draws = {key: torch.cat([d[key] for d in roll.draws]) for key in roll.draws[0]}
    outcome = {key: np.asarray(value) for key, value in roll.outcome.items()}
    obs.update(outcome)
    return obs, draws, np.concatenate(roll.logp)


# ---------------------------------------------------------------------------------------------
# advantages
def values(critic: Critic, batch: dict[str, np.ndarray], normalizer: Normalizer) -> np.ndarray:
    """Raw-unit state values V = u x P, P the residents at the decision."""
    with torch.no_grad():
        u = critic.expectation(critic(to_tensors(batch, normalizer))).numpy().astype(np.float64)
    return u * np.maximum(1.0, batch["prior_population"].astype(np.float64))


# Objective components that scale with the settlement's residents: their value heads learn a
# per-resident return
# (u = G / P). qol_years is QoL x years regardless of headcount, so its head learns the raw return;
# dividing it by P
# would make its target fall as 1/P and blur it across village sizes.
PER_RESIDENT = frozenset({"welfare", "population_years", "trade_utility"})


def head_scale(name: str, population: np.ndarray) -> np.ndarray:
    return (
        np.maximum(1.0, population.astype(np.float64))
        if name in PER_RESIDENT
        else np.ones(len(population))
    )


def vector_values(
    critic: VectorCritic, batch: dict[str, np.ndarray], normalizer: Normalizer
) -> dict[str, np.ndarray]:
    """Raw-unit state values per objective component."""
    with torch.no_grad():
        logits = critic(to_tensors(batch, normalizer))
    return {
        name: critic.expectation(value).numpy().astype(np.float64)
        * head_scale(name, batch["prior_population"])
        for name, value in logits.items()
    }


def region_groups(draws: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    slot = draws["region"].numpy() if "region" in draws else np.zeros(0, dtype=np.int64)
    return {
        "native": slot == contracts.REGION_NATIVE,
        "keep": slot == contracts.REGION_KEEP,
        "stay": slot == 2,
        "move": slot > 2,
    }


def combine(advantages: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """The policy's advantage: the weighted sum of component advantages (Part 4 of the commitments design)."""  # noqa: E501 - docstring/prose line kept verbatim
    return sum(float(weights.get(name, 0.0)) * value for name, value in advantages.items())


def shape_components(components: dict[str, np.ndarray], share: np.ndarray) -> dict[str, np.ndarray]:
    """Personality runs charge the shaping costs (``share`` of each decision's reward) on every component in its own
    units, so an Elder with no welfare weight still pays for churn (docs/MODEL9_PERSONALITY.md, welfare leakage).
    ``welfare`` is returned unchanged: its costs are already charged. Default-personality runs keep the costs on
    welfare only, which keeps earlier runs reproducible."""  # noqa: E501 - docstring/prose line kept verbatim
    return {
        name: value if name == "welfare" else value - share * np.maximum(0.0, value)
        for name, value in components.items()
    }


def combine_personal(
    advantages: dict[str, np.ndarray],
    personality: np.ndarray,
    scales: str = "batch",
    clip: float = 0.0,
) -> np.ndarray:
    """Per-row personality weights over contracts.OBJECTIVES on scale-normalised component advantages: components
    differ by orders of magnitude (population-years against QoL-years), so a weight means a share of each component's
    spread. ``scales`` "reference" (round 3; the training default, through Hyper.personality_scales) divides by the
    fixed contracts.OBJECTIVE_REFERENCE_SCALES, A = sum_k p_k A_k / s_k_ref, so the exchange rate between objectives
    never moves with batch content; "batch" (the earlier rule, and this function's default) divides by each
    component's batch std and rescales to the welfare component's spread. ``update`` standardises the result, so the
    overall scale does not matter and a welfare-only personality updates identically under either rule (exactly with
    ``clip`` 0). ``clip`` > 0 (reference rule only) clips each normalised component A_k / s_k_ref to +/- clip before
    weighting, so one component's critic noise (an untrained head's) cannot dominate the sum."""  # noqa: E501 - docstring/prose line kept verbatim
    if scales == "reference":
        scale = {name: float(contracts.OBJECTIVE_REFERENCE_SCALES[name]) for name in advantages}
    elif scales == "batch":
        scale = {name: float(np.std(a)) + 1e-9 for name, a in advantages.items()}
    else:
        raise ValueError(f"unknown personality scales {scales!r}")
    total = np.zeros(len(personality))
    for k, name in enumerate(contracts.OBJECTIVES):
        if name in advantages:
            z = advantages[name] / scale[name]
            if scales == "reference" and clip > 0:
                z = np.clip(z, -clip, clip)
            total = total + personality[:, k] * z
    return total * scale.get("welfare", 1.0) if scales == "batch" else total


def clip_fractions(advantages: dict[str, np.ndarray], clip: float) -> dict[str, float]:
    """Each objective's share of rows whose normalised advantage A_k / s_k_ref lies beyond +/- clip (0 when off)."""  # noqa: E501 - docstring/prose line kept verbatim
    return {
        name: float(np.mean(np.abs(a / float(contracts.OBJECTIVE_REFERENCE_SCALES[name])) > clip))
        if clip > 0
        else 0.0
        for name, a in advantages.items()
        if name in contracts.OBJECTIVE_REFERENCE_SCALES
    }


def scale_drift(advantages: dict[str, np.ndarray]) -> dict[str, float]:
    """Each objective's batch advantage spread over its reference spread: how far the batch rule's exchange rates
    would have moved this iteration (1 everywhere is no drift)."""  # noqa: E501 - docstring/prose line kept verbatim
    return {
        name: float(np.std(a)) / float(contracts.OBJECTIVE_REFERENCE_SCALES[name])
        for name, a in advantages.items()
        if name in contracts.OBJECTIVE_REFERENCE_SCALES
    }


def gae(
    batch: dict[str, np.ndarray],
    reward: np.ndarray,
    value: np.ndarray,
    bootstrap: np.ndarray,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE(lambda) per (world, camp) sequence with the physical per-turn discount. ``bootstrap`` is the value of
    the observation after each row's decision when a sequence is truncated (0 elsewhere)."""  # noqa: E501 - docstring/prose line kept verbatim
    advantage = np.zeros(len(reward))
    target = np.zeros(len(reward))
    for rows in data.sequences(batch):
        a_next = 0.0
        for position in range(len(rows) - 1, -1, -1):
            i = rows[position]
            gamma = float(data.discount(batch["years"][i]))
            if batch["terminated"][i]:
                v_next, a_next = 0.0, 0.0
            elif position == len(rows) - 1:
                v_next, a_next = bootstrap[i], 0.0
            else:
                v_next = value[rows[position + 1]]
            delta = reward[i] + gamma * v_next - value[i]
            advantage[i] = delta + gamma * lam * a_next
            target[i] = advantage[i] + value[i]
            a_next = advantage[i]
    return advantage, target


class CostCritic(nn.Module):
    """Discounted guard cost (settlement-turns below 30 residents) per decision; plain regression."""  # noqa: E501 - docstring/prose line kept verbatim

    def __init__(self, core_width: int, width: int = 128, token: bool = False) -> None:
        super().__init__()
        from .model import Encoder, TokenEncoder

        self.encoder = TokenEncoder(core_width, width) if token else Encoder(core_width, width)
        self.head = nn.Linear(width, 1)

    def forward(self, t: dict[str, torch.Tensor]) -> torch.Tensor:
        h, _ = self.encoder(t)
        return self.head(h).squeeze(1)


def cost_bootstrap(
    cost_critic: CostCritic,
    roll: Rollout,
    batch: dict[str, np.ndarray],
    normalizer: Normalizer,
    rows: np.ndarray,
) -> np.ndarray:
    """The guard cost critic's value of each truncated living village's next observation (0 elsewhere): the GAE
    bootstrap of the cost sequence. Guard cost is a per-settlement indicator, so the value is not scaled by residents."""  # noqa: E501 - docstring/prose line kept verbatim
    boot = np.zeros(len(batch["cost"]))
    if roll.bootstrap_obs and len(rows):
        with torch.no_grad():
            tail = (
                cost_critic(to_tensors(roll.bootstrap_obs[0], normalizer))
                .numpy()
                .astype(np.float64)
            )
        boot[rows] = tail[batch["bootstrap_index"][rows]]
    return boot


# ---------------------------------------------------------------------------------------------
# update
@dataclass
class Hyper:
    clip: float = 0.2
    epochs: int = 4
    batch: int = 1024
    lr: float = 1e-5  # measured: approximate KL 0.004-0.015 per update on the stage-1 checkpoint
    critic_lr: float = 1e-4
    entropy: float = 1e-3
    teacher_kl: float = 0.05
    cost_budget: float = 0.01  # tolerated share of guarded settlement-turns below 30
    lagrange_lr: float = 0.5
    target_kl: float = 0.03  # early stop on approximate KL to the behaviour policy
    log_std_floor: float = -2.0  # exploration floor on logit-space scales (std 0.135)
    balance_regimes: bool = (
        True  # standardize advantages per world regime and give each regime equal weight
    )
    regime_weight_cap: float = 4.0  # per-decision regime weights are clipped to [1 / cap, cap]
    mix_change_cost: float = (
        0.02  # reward cost per unit of turn-to-turn total-variation change in the economy mix,
    )
    # as a share of the village's welfare that turn (a slight switching penalty)
    break_cost: float = (
        0.01  # contract 9: reward cost of a voluntary break (re-deciding while a hold was legal),
    )
    # as a share of the village's welfare that turn
    objective_weights: dict = field(default_factory=lambda: dict(contracts.OBJECTIVE_WEIGHTS))
    # Elder personality distribution (docs/MODEL9_PERSONALITY.md): "default", "fixed:w,w,w,w" or
    # "dirichlet:alpha";
    # drawn per (world, lineage), inherited by daughters. Non-default personalities weight
    # scale-normalised component
    # advantages per row.
    personality: str = "default"
    # Daughters inherit the parent's personality with a small Dirichlet shift (personality_drift); 0
    # copies exactly.
    personality_drift: float = 200.0
    # How personality weights exchange objectives (combine_personal): "reference" divides each
    # component advantage by
    # its fixed contracts.OBJECTIVE_REFERENCE_SCALES entry (round 3); "batch" by its batch std
    # (earlier runs).
    personality_scales: str = "reference"
    # Reference rule: clip each normalised component advantage A_k / s_k_ref to +/- this before
    # weighting; 0 off.
    personality_clip: float = 5.0
    # Search teacher (docs/MODEL9_SEARCH_TEACHER.md): weight of the imitation loss on search-winner
    # labels, the
    # relative gain a label must clear, and label rows per minibatch. 0 turns it off.
    search_weight: float = 0.0
    # search utility v2: U_p gain in reference spreads (about the old 0.5% at a mean village)
    search_margin: float = 0.004
    search_batch: int = 256
    # Critic warm-up: when a run starts by converting a scalar critic into the vector critic, its
    # population, QoL and
    # trade heads are new (only welfare is copied). This many extra iterations first train only the
    # critics, so
    # personality advantages are not built on untrained heads. 0 turns it off.
    critic_warmup: int = 5
    # Round 3 (docs/MODEL9_M7.md, *Round 3 build*): critic-only iterations at the start of every run
    # (a resume past them
    # does not repeat them; with a fresh-heads warm-up the larger count applies, once).
    round_start_warmup: int = 0
    # Share of each critic minibatch drawn from search branch shards (ppo --branch-shards); their
    # targets are
    # lambda-returns over each branch sequence, recomputed every iteration with the current critic.
    branch_share: float = 0.25
    # One extra critic-only epoch on an iteration's batch when its approximate KL exceeded target_kl
    # (the actor stopped
    # early, so the critic would otherwise see fewer epochs). Off here so programmatic runs
    # reproduce; the CLI turns it
    # on for new runs (--kl-refit, default on; --no-kl-refit).
    kl_refit: bool = False
    # Round 3 (review 5.5(1)): the guard cost critic's GAE bootstraps every living village truncated
    # at the rollout
    # horizon from the cost critic's value of its next observation, as the value critics do. Off is
    # the earlier
    # return-only target (truncated villages counted as if the guard cost stopped at the horizon).
    # The CLI keeps a resumed
    # run's own setting (off for runs saved before the flag).
    cost_bootstrap: bool = True
    # Opt-in optimizer trace: audit at most this many rows from only the first actor minibatch of an
    # iteration.
    # Zero is the production/default path. The CLI requires --economic-trace when this is nonzero.
    optimizer_audit_rows: int = 0
    # Reviewed Stage-1 learner fixes are opt-in. ``legacy`` preserves the historical batching,
    # warm-up and
    # pre-step KL behavior exactly; observed-anchor-v1 uses independent returns and post-step
    # rollback.
    learner_contract: str = "legacy"
    observed_anchor_weight: float = 1.0
    post_update_kl_rows: int = 128


def behavior_policy_configuration(stage: Stage, hyper: Hyper, actor: Actor) -> dict[str, Any]:
    """Settings that change the sampled rollout policy without changing tensors."""
    return {
        "schemaVersion": 1,
        "samplingSemantics": "autonomous-v7-explore-and-sample-v1",
        "stage": dataclasses.asdict(stage),
        "logStdFloor": float(hyper.log_std_floor),
        "personality": hyper.personality,
        "personalityDrift": float(hyper.personality_drift),
        "personalityScales": hyper.personality_scales,
        "personalityClip": float(hyper.personality_clip),
        "actorRegionHead": getattr(actor, "region_head", None),
    }


def learner_feature_contract(learner_contract: str) -> str:
    """Exact input contract accepted by each versioned learner."""
    from .model import BOUNDED_FEATURE_CONTRACT, LEGACY_FEATURE_CONTRACT

    if learner_contract == "observed-anchor-v1":
        return BOUNDED_FEATURE_CONTRACT
    if learner_contract == "legacy":
        return LEGACY_FEATURE_CONTRACT
    raise ValueError(f"unknown learner contract {learner_contract!r}")


def versioned_learner_checkpoint_metadata(
    learner_contract: str,
    feature_contract: str | None,
    *,
    normalizer: dict[str, Any] | None = None,
    observed_train=None,
    observed_gate=None,
) -> dict[str, Any]:
    """Return metadata carried only by the opt-in learner contract.

    Legacy checkpoints predate these keys.  Omitting them, rather than writing
    ``None`` or a default label, is part of exact legacy artifact compatibility.
    """

    if learner_contract == "legacy":
        return {}
    if learner_contract != "observed-anchor-v1" or not feature_contract:
        raise ValueError("modern learner checkpoint lacks its feature contract")
    result: dict[str, Any] = {
        "learner_contract": learner_contract,
        "feature_contract": feature_contract,
    }
    if normalizer is not None:
        result["normalizer"] = normalizer
    if observed_train is not None or observed_gate is not None:
        if observed_train is None or observed_gate is None:
            raise ValueError("modern learner checkpoint needs both observed-return roles")
        result["observed_anchor_sources"] = {
            "train": observed_train.metadata,
            "gate": observed_gate.metadata,
        }
    return result


def warmup_iterations(
    fresh_heads: bool, critic_warmup: int, round_start_warmup: int, *, promoted_init=False
) -> int:
    """Critic-only iterations at the head of a run: the fresh-heads warm-up and the round-start warm-up overlap (the
    larger count, once). The loop runs iterations ``[0, warmup)`` critic-only, so a resume past them repeats none."""  # noqa: E501 - docstring/prose line kept verbatim
    return max(critic_warmup if fresh_heads or promoted_init else 0, round_start_warmup)


def resolve_configuration(args, state, resume_state=None):
    """Pure, shared CLI/default expansion for training and read-only launch qualification; consumes no RNG."""  # noqa: E501 - docstring/prose line kept verbatim
    from .checkpoint import resume_setting

    hyper = Hyper()
    for key in (
        "lr",
        "teacher_kl",
        "lagrange_lr",
        "cost_budget",
        "log_std_floor",
        "mix_change_cost",
        "break_cost",
        "critic_warmup",
        "round_start_warmup",
        "branch_share",
    ):
        value = getattr(args, key)
        if value is not None:
            setattr(hyper, key, value)
    hyper.balance_regimes = not args.no_balance
    stage = STAGES[args.stage]
    for argument, values in (
        ("commitments", {"commitments": True}),
        ("learned_movement", {"movement": True, "region_movement": True}),
        ("observe_movement", {"observe_movement": True}),
        ("learned_fission", {"fission": True}),
        ("learned_trade", {"trade": True}),
    ):
        if getattr(args, argument):
            stage = dataclasses.replace(stage, **values)
    if args.learned_trade:
        hyper.objective_weights["trade_utility"] = contracts.TRADE_UTILITY_WEIGHT
    hyper.personality = args.personality
    hyper.personality_scales = args.personality_scales
    hyper.personality_clip = args.personality_clip
    if args.search_labels:
        hyper.search_weight = args.search_weight
        if args.search_margin is not None:
            hyper.search_margin = args.search_margin
    if args.objective_weights:
        for item in args.objective_weights.split(","):
            key, value = item.split("=")
            if key not in contracts.OBJECTIVES:
                raise ValueError(f"unknown objective {key}")
            hyper.objective_weights[key] = float(value)
    hyper.kl_refit = (
        args.kl_refit
        if args.kl_refit is not None
        else bool((resume_state or {}).get("kl_refit", resume_state is None))
    )
    hyper.cost_bootstrap = resume_setting("cost_bootstrap", resume_state, args.cost_bootstrap, True)
    from .optimizer_update_debug import checked_audit_rows

    hyper.optimizer_audit_rows = checked_audit_rows(getattr(args, "optimizer_audit_rows", 0))
    hyper.learner_contract = getattr(args, "learner_contract", "legacy")
    hyper.observed_anchor_weight = float(getattr(args, "observed_anchor_weight", 1.0))
    hyper.post_update_kl_rows = int(getattr(args, "post_update_kl_rows", 128))
    if not math.isfinite(hyper.observed_anchor_weight) or hyper.observed_anchor_weight <= 0:
        raise ValueError("observed anchor weight must be positive")
    if hyper.post_update_kl_rows <= 0 or isinstance(hyper.post_update_kl_rows, bool):
        raise ValueError("post-update KL audit rows must be positive")
    if not math.isfinite(hyper.target_kl) or hyper.target_kl <= 0:
        raise ValueError("target KL must be finite and positive")
    fresh = not state.get("vector_critic") and (
        args.vector_critic or args.learned_trade or args.personality != "default"
    )
    warmup = (
        0
        if hyper.learner_contract != "legacy"
        else warmup_iterations(
            bool(fresh),
            hyper.critic_warmup,
            hyper.round_start_warmup,
            promoted_init=state.get("artifact_kind") == "promotion",
        )
    )
    return (
        hyper,
        stage,
        {"critic_only": warmup, "actor": args.iterations, "total": warmup + args.iterations},
    )


def default_personality() -> np.ndarray:
    return np.asarray([contracts.OBJECTIVE_WEIGHTS[k] for k in contracts.OBJECTIVES], np.float32)


def personality_drift(
    parent: np.ndarray, rng: np.random.Generator, concentration: float
) -> np.ndarray:
    """A daughter's personality: a small random shift of its parent's, Dirichlet(concentration x parent) (mean the
    parent; spread about sqrt(p (1 - p) / concentration)). Concentration 0 copies the parent exactly. Drift makes
    lineages diverge, so a population of Elders is not a set of clones (resilience under self-play)."""  # noqa: E501 - docstring/prose line kept verbatim
    if concentration <= 0:
        return parent
    return rng.dirichlet(np.maximum(np.asarray(parent, np.float64), 1e-4) * concentration).astype(
        np.float32
    )


def personality_sampler(spec: str, rng: np.random.Generator):
    """A draw of one personality (weights over contracts.OBJECTIVES, summing to 1)."""
    if spec == "default":
        return lambda: default_personality()
    kind, _, arg = spec.partition(":")
    if kind == "fixed":
        w = np.maximum(0.0, np.asarray([float(x) for x in arg.split(",")], np.float32))
        if len(w) != contracts.PERSONALITY_WIDTH or not w.sum() > 0:
            raise ValueError(
                f"a fixed personality needs {contracts.PERSONALITY_WIDTH} non-negative weights"
            )
        return lambda: (w / w.sum()).astype(np.float32)
    if kind == "dirichlet":
        alpha = float(arg)
        return lambda: rng.dirichlet(np.full(contracts.PERSONALITY_WIDTH, alpha)).astype(np.float32)
    raise ValueError(f"unknown personality spec {spec!r}")


def load_search_labels(
    paths: list[Path],
    margin: float,
    legacy_margin: float = 0.005,
    provenance: str = "flag",
    report: dict | None = None,
    declared_sources: set[str] | None = None,
    calibration_manifest: dict | None = None,
) -> dict[str, np.ndarray] | None:
    """Search-teacher label rows (search_teacher.py) whose best alternative beats the policy by more than ``margin``
    and is not the policy itself: the rows PPO imitates. Gains are on search utility version 2 (U_p in reference
    spreads, the units of PPO's advantage); version-1 files are rescored from their return vectors, and rows whose
    label is no longer the best are dropped. Files too old to rescore keep the relative rule against
    ``legacy_margin``. ``provenance`` (provenance.check: reject, flag or allow) applies to the rows' (source checkpoint,
    code commit), files written before provenance counting as unknown; ``report`` receives the provenance report."""  # noqa: E501 - docstring/prose line kept verbatim
    from . import provenance as prov
    from .search_teacher import POLICY, label_gain, rescore

    parts = []
    decisions = set()
    identities_complete = True
    for path in paths:
        part = dict(np.load(path))
        if not part:
            continue
        ids, complete = prov.normalize_label_identity(
            part,
            prov.checkpoint_hash(path),
            provenance == "reject" or calibration_manifest is not None,
        )
        identities_complete &= complete
        for identity in ids:
            if identity in decisions:
                raise prov.ProvenanceError(f"duplicate search decision: {identity}")
            decisions.add(identity)
        if calibration_manifest is not None:
            from .calibration import assert_training_rows

            assert_training_rows(calibration_manifest, part)
        parts.append(data.normalize_optional(rescore(part)))
    if not parts:
        return None
    sources: list[tuple[str, ...]] = []
    counts: list[int] = []
    loaded_records = []
    for p in parts:
        n = len(p["label_kind"])
        if np.asarray(p.get("heldout_calibration", np.zeros(n)), bool).any():
            raise prov.ProvenanceError("held-out calibration labels cannot train the actor")
        if "source_record" in p:
            loaded_records.extend(json.loads(str(x)) for x in np.unique(p["source_record"]))
            for i, record in enumerate(p["source_record"]):
                source_record = json.loads(str(record))
                prov.verify_link(
                    source_record,
                    {
                        k: p[k][i] if k in p else prov.UNKNOWN
                        for k in ("source_checkpoint", "code_commit", "source_run")
                    },
                    provenance,
                )
                exclusion = prov.source_exclusion(source_record)
                saved_digest = str(p.get("calibration_manifest_digest", np.full(n, ""))[i])
                if saved_digest != exclusion.get("manifest_digest", ""):
                    raise prov.ProvenanceError(
                        "search label exclusion manifest differs from its source record"
                    )
                if exclusion:
                    from .calibration import assert_training_rows

                    assert_training_rows(exclusion, {k: v[i : i + 1] for k, v in p.items()})
        elif provenance == "reject":
            raise prov.ProvenanceError("search labels lack full production provenance")
        if "source_checkpoint" in p and "code_commit" in p:
            pairs = list(
                zip(
                    np.asarray(p["source_checkpoint"]).astype(str).tolist(),
                    np.asarray(p["code_commit"]).astype(str).tolist(),
                    np.asarray(p.get("source_run", np.full(n, prov.UNKNOWN))).astype(str).tolist(),
                    strict=False,
                )
            )
            for pair in sorted(set(pairs)):
                sources.append(pair)
                counts.append(pairs.count(pair))
        else:
            sources.append((prov.UNKNOWN, prov.UNKNOWN, prov.UNKNOWN))
            counts.append(n)
    checked = {
        **prov.check(sources, provenance, "search labels", counts, declared_sources),
        **prov.validate_records(loaded_records, provenance),
        "records": loaded_records,
        "decision_identities_complete": identities_complete,
        "calibration_manifest_digest": (calibration_manifest or {}).get("manifest_digest"),
    }
    from .lineage import families

    checked["families"] = families(
        [s for p in parts for s in p.get("source_seed", [])],
        [s for p in parts for s in p.get("source_snapshot", [])],
    )
    if report is not None:
        report.update(checked)
    groups = len(contracts.TRADE_GROUPS)
    for (
        p
    ) in parts:  # round-3 label arrays, absent from earlier label files: no fission or trade label
        n = len(p["label_kind"])
        p.setdefault("label_fission", np.full(n, -1))
        p.setdefault("label_trade_value", np.full((n, groups), np.nan))
        p.setdefault("label_trade_gate", np.full((n, groups), -1))
        p.setdefault("label_trade_mask", np.zeros((n, groups), np.int8))
    keys = set.union(*(set(p) for p in parts))
    for k in keys:
        template = next(p[k] for p in parts if k in p)
        for p in parts:
            if k in p:
                continue
            fill = (
                prov.UNKNOWN
                if template.dtype.kind in "US"
                else -1
                if k in ("alt_kind", "alt_slot", "alt_calibration")
                else False
                if template.dtype.kind == "b"
                else np.nan
            )
            dtype = template.dtype if isinstance(fill, (str, bool)) or fill == -1 else np.float64
            p[k] = np.full((len(p["label_kind"]),) + template.shape[1:], fill, dtype=dtype)  # noqa: RUF005 - concatenation kept (operand type not guaranteed list)

    def stack(k: str) -> np.ndarray:
        # Per-alternative arrays are padded to their file's MAX_ALTERNATIVES; pad to the widest
        # before stacking.
        arrays = [p[k] for p in parts]
        if k.startswith("alt_") and len({a.shape[1] for a in arrays}) > 1:
            width = max(a.shape[1] for a in arrays)
            fill = -1 if k in ("alt_kind", "alt_slot", "alt_calibration") else np.nan
            arrays = [
                np.concatenate(
                    [
                        a,
                        np.full(
                            (a.shape[0], width - a.shape[1]) + a.shape[2:],  # noqa: RUF005 - concatenation kept (operand type not guaranteed list)
                            fill,
                            dtype=a.dtype if fill == -1 else np.float64,
                        ),
                    ],
                    axis=1,
                )
                for a in arrays
            ]
        return np.concatenate(arrays)

    rows = {k: stack(k) for k in keys}
    _, kept = label_gain(rows, margin, legacy_margin)
    keep = kept & (rows["label_kind"] != POLICY)
    return {k: v[keep] for k, v in rows.items()} if keep.any() else None


def search_loss(
    out: dict[str, torch.Tensor],
    t: dict[str, torch.Tensor],
    kind: torch.Tensor,
    slot: torch.Tensor,
    mix: torch.Tensor,
    fission: torch.Tensor | None = None,
    trade_value: torch.Tensor | None = None,
    trade_gate: torch.Tensor | None = None,
    trade_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Imitation of search winners: cross-entropy of the region head on region and native labels (native is slot 0),
    and soft cross-entropy of the economy mix on mix labels and on native labels that carry the heuristic's mix. The mix
    is conditioned on the label's slot for native labels and on the policy's own most likely slot otherwise.

    Round 3: cross-entropy of the fission head on fission labels (``fission`` >= 0), and on trade labels, for exactly
    the groups and components the winning probe changed (``trade_mask`` bits: 1 value, 2 gate), a Gaussian negative
    log-likelihood pulling the trade value mean toward log(label) (its scale detached, so imitation never narrows the
    head's exploration) and cross-entropy of that group's gate. Every other trade component is untouched: probes inform
    the heads and never restrict them. Each label row counts once in the average."""  # noqa: E501 - docstring/prose line kept verbatim
    from .search_teacher import NATIVE, REGION, TRADE_GATE_BIT, TRADE_VALUE_BIT

    zero = out["mix_logits"].sum() * 0.0
    loss, count = zero, 0
    on_slot = (kind == REGION) | (kind == NATIVE)
    if on_slot.any() and "region" in out:
        # Round 3: the hierarchical head's joint log-probability is log p(group) + log p(slot |
        # group), so one
        # cross-entropy supervises both levels (a native label teaches the group only).
        logq = (
            out["region_logp"] if "region_logp" in out else torch.log_softmax(out["region"], dim=1)
        )
        loss = loss - logq.gather(1, slot.clamp(min=0).unsqueeze(1)).squeeze(1)[on_slot].sum()
        count += int(on_slot.sum())
    has_mix = ~torch.isnan(mix).any(dim=1)
    if has_mix.any():
        own = region_mode(out) if "region" in out else torch.zeros_like(slot)
        conditioned = condition_mix(out, torch.where(kind == NATIVE, slot.clamp(min=0), own))
        ce, valid = mix_soft_cross_entropy(
            conditioned["mix_logits"], torch.nan_to_num(mix), t["economy_mix_mask"]
        )
        rows = has_mix & valid
        loss = loss + ce[rows].sum()
        count += int(rows.sum())
    if fission is not None and "fission" in out:
        on_fission = fission >= 0
        if on_fission.any():
            logq = torch.log_softmax(out["fission"], dim=1)
            loss = (
                loss
                - logq.gather(1, fission.clamp(min=0).unsqueeze(1)).squeeze(1)[on_fission].sum()
            )
            count += int(on_fission.sum())
    if trade_mask is not None and "trade_value_mean" in out and bool((trade_mask > 0).any()):
        value_on = (trade_mask & TRADE_VALUE_BIT) > 0
        gate_on = (trade_mask & TRADE_GATE_BIT) > 0
        target = torch.log(torch.nan_to_num(trade_value, nan=1.0).clamp(min=1e-6))
        log_std = out["trade_value_log_std"].detach()
        nll = 0.5 * ((target - out["trade_value_mean"]) / log_std.exp()) ** 2 + log_std
        gate_lp = (
            torch.log_softmax(out["trade_gate"], dim=2)
            .gather(2, trade_gate.clamp(min=0).unsqueeze(2))
            .squeeze(2)
        )
        terms = torch.where(value_on, nll, torch.zeros_like(nll)).sum(dim=1) - torch.where(
            gate_on, gate_lp, torch.zeros_like(gate_lp)
        ).sum(dim=1)
        components = (value_on.sum(dim=1) + gate_on.sum(dim=1)).float()
        rows = components > 0
        loss = loss + (terms[rows] / components[rows]).sum()
        count += int(rows.sum())
    return loss / max(1, count)


def update(
    actor: Actor,
    critic: Critic,
    cost_critic: CostCritic,
    teacher: Actor,
    normalizer: Normalizer,
    batch: dict[str, np.ndarray],
    draws: dict[str, torch.Tensor],
    old_logp: np.ndarray,
    advantage: np.ndarray,
    target_u: np.ndarray,
    cost_adv: np.ndarray,
    cost_target: np.ndarray,
    lagrange: float,
    stage: Stage,
    hyper: Hyper,
    optimizers: tuple,
    rng: np.random.Generator,
    regimes: np.ndarray | None = None,
    search: dict[str, np.ndarray] | None = None,
    train_actor: bool = True,
    branch: tuple[dict[str, np.ndarray], Any] | None = None,
    economic_trace=None,
    trace_context=None,
    observed_anchor=None,
) -> dict[str, float]:
    """One PPO update. ``train_actor`` False updates only the critics (critic warm-up). ``branch``: search branch rows
    and their critic targets (branch_shards.targets); each critic minibatch then draws ``hyper.branch_share`` of its
    rows from them. With ``hyper.kl_refit``, an update stopped early on approximate KL gets one extra critic-only epoch."""  # noqa: E501 - docstring/prose line kept verbatim
    actor_opt, critic_opt, cost_opt = optimizers
    n = len(advantage)
    trace_metrics = {}
    optimizer_debug = None
    audit_rows = 0
    if economic_trace is not None:
        from .optimizer_update_debug import checked_audit_rows

        audit_rows = checked_audit_rows(hyper.optimizer_audit_rows)
    debug_enabled = audit_rows > 0
    modern = hyper.learner_contract == "observed-anchor-v1"
    if modern and observed_anchor is None:
        raise ValueError("observed-anchor-v1 update lacks independent training anchors")
    actor_halted = False
    rollback_count = 0
    accepted_actor_steps = 0
    fixed_post_step_kl: list[float] = []
    epoch_coverage_records = []
    anchor_value_losses: list[float] = []
    anchor_cost_losses: list[float] = []

    def native_recorded_coordinate_activity(coordinates, rows: np.ndarray):
        """Read native execution masks already captured for these exact rollout rows.

        These masks are descriptive, not likelihood masks: in particular the service mask records a positive
        post-request allocation and is therefore action-dependent.
        """  # noqa: E501 - docstring/prose line kept verbatim
        roll = trace_context[0] if trace_context else None
        captured_rows = getattr(roll, "economic_rows", None)
        output = {group: {name: [] for name in values} for group, values in coordinates.items()}
        for source_row in rows:
            captured = (
                captured_rows[int(source_row)]
                if captured_rows is not None and int(source_row) < len(captured_rows)
                else {}
            )
            active_mask = (captured.get("actionDiagnostics") or {}).get("activeMask") or {}
            for group, values in coordinates.items():
                recorded = active_mask.get(group)
                for coordinate, name in enumerate(values):
                    value = (
                        recorded[coordinate]
                        if isinstance(recorded, list) and coordinate < len(recorded)
                        else None
                    )
                    output[group][name].append(value if isinstance(value, bool) else None)
        return output

    def value_step(
        t: dict[str, torch.Tensor], index: np.ndarray, *, audit: bool = False
    ) -> tuple[torch.Tensor, dict | None]:
        """One value-critic step on this minibatch (plus its share of branch rows)."""

        def cross_entropy(logits, target, rows) -> torch.Tensor:
            return (
                -(
                    critic.two_hot(torch.as_tensor(target[rows], dtype=torch.float32))
                    * torch.log_softmax(logits, dim=-1)
                )
                .sum(-1)
                .mean()
            )

        audit_count_value = min(audit_rows, len(index)) if audit else 0
        if audit and isinstance(target_u, dict):
            # Build and restore each probe before the real critic graph exists: restoring a module
            # after a graph has
            # been built would invalidate autograd version counters. Branch rows are excluded from
            # this bounded,
            # explicitly on-policy row sample.
            from .optimizer_direction_audit import capture_component_gradients
            from .optimizer_update_debug import geometry_record

            count = audit_count_value
            audit_t = {key: value[:count] for key, value in t.items()}
            audit_index = index[:count]
            encoder_parameters = [
                (f"encoder.{name}", parameter)
                for name, parameter in critic.encoder.named_parameters()
            ]

            def head_loss(name):
                return lambda: cross_entropy(critic(audit_t)[name], target_u[name], audit_index)

            captured = capture_component_gradients(
                {name: head_loss(name) for name in target_u},
                encoder_parameters,
                optimizers=(critic_opt,),
                state_modules=(critic,),
                numpy_generators={"minibatch": rng},
            )
            optimizer_debug["vectorCriticSharedEncoderGradientGeometry"] = geometry_record(captured)
            optimizer_debug["vectorCriticSharedEncoderGradientGeometry"]["scope"] = {
                "rowCount": count,
                "rows": "first-rows-of-actual-minibatch",
                "branchRowsIncluded": False,
            }
        elif audit:
            optimizer_debug.setdefault("unavailable", {})[
                "vectorCriticSharedEncoderGradientGeometry"
            ] = "scalar-critic-no-per-head-shared-encoder-conflict"
        logits = critic(t)
        if audit:
            from .optimizer_update_debug import critic_distribution_record

            if isinstance(target_u, dict):
                optimizer_debug["criticHeadDiagnostics"] = {
                    name: critic_distribution_record(
                        logits[name][:audit_count_value],
                        torch.as_tensor(
                            target_u[name][index[:audit_count_value]], dtype=torch.float32
                        ),
                        critic.bins,
                    )
                    for name in target_u
                }
            else:
                optimizer_debug["criticHeadDiagnostics"] = {
                    "value": critic_distribution_record(
                        logits[:audit_count_value],
                        torch.as_tensor(target_u[index[:audit_count_value]], dtype=torch.float32),
                        critic.bins,
                    )
                }
            optimizer_debug["criticHeadDiagnosticsScope"] = {
                "rowCount": audit_count_value,
                "rows": "first-rows-of-actual-minibatch",
                "branchRowsIncluded": False,
                "target": "internal-lambda-return",
            }
        value_by = None
        if isinstance(target_u, dict):
            value_by = {
                name: cross_entropy(logits[name], target_u[name], index) for name in target_u
            }
        else:
            value_loss = cross_entropy(logits, target_u, index)
        if branch is not None and hyper.branch_share > 0:
            rows_b, target_b = branch
            m = len(rows_b["camp"])
            k = max(1, round(len(index) * hyper.branch_share / max(1e-9, 1 - hyper.branch_share)))
            pick = rng.permutation(m)[:k] if m > k else np.arange(m)
            logits_b = critic(to_tensors({key: v[pick] for key, v in rows_b.items()}, normalizer))
            w_on, w_b = len(index) / (len(index) + len(pick)), len(pick) / (len(index) + len(pick))
            if value_by is not None:
                value_by = {
                    name: w_on * value_by[name]
                    + w_b * cross_entropy(logits_b[name], target_b[name], pick)
                    for name in value_by
                    if name in target_b
                }
            else:
                value_loss = w_on * value_loss + w_b * cross_entropy(logits_b, target_b, pick)
        if value_by is not None:
            value_loss = sum(value_by.values())
        critic_opt.zero_grad()
        value_loss.backward()
        critic_norm = nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_opt.step()
        if economic_trace is not None:
            trace_metrics["criticGradientNormBeforeClipping"] = float(critic_norm)
            trace_metrics["criticClippingScale"] = min(1.0, 1.0 / (float(critic_norm) + 1e-6))
        return value_loss, value_by

    def cost_step(t: dict[str, torch.Tensor], index: np.ndarray) -> torch.Tensor:
        cost_loss = (
            (cost_critic(t) - torch.as_tensor(cost_target[index], dtype=torch.float32)) ** 2
        ).mean()
        cost_opt.zero_grad()
        cost_loss.backward()
        cost_norm = nn.utils.clip_grad_norm_(cost_critic.parameters(), 1.0)
        cost_opt.step()
        if economic_trace is not None:
            trace_metrics["costGradientNormBeforeClipping"] = float(cost_norm)
            trace_metrics["costClippingScale"] = min(1.0, 1.0 / (float(cost_norm) + 1e-6))
        return cost_loss

    objective_scale = max(1e-8, advantage.std())
    constraint_scale = max(1e-8, cost_adv.std() + 1e-8)
    combined = advantage / objective_scale - lagrange * cost_adv / constraint_scale
    combined, weight = balance(
        combined, regimes if hyper.balance_regimes else None, hyper.regime_weight_cap
    )
    objective_balanced = constraint_balanced = None
    if debug_enabled:
        from .optimizer_update_debug import balanced_advantage_contributions

        objective_raw = advantage / objective_scale
        constraint_raw = -lagrange * cost_adv / constraint_scale
        objective_balanced, constraint_balanced = balanced_advantage_contributions(
            objective_raw, constraint_raw, combined, regimes if hyper.balance_regimes else None
        )
    if economic_trace is not None:
        from .economic_trace import emit_rollout

        emit_rollout(
            economic_trace,
            *trace_context,
            combined,
            weight,
            regimes,
            hyper,
            lagrange,
            actor,
            critic,
            cost_critic,
            normalizer,
        )
    fixed_audit_t = fixed_audit_d = fixed_audit_old = None
    fixed_audit_identity = []
    fixed_audit_digest = None
    if modern and train_actor:
        from .ppo_correctness import fixed_audit_indices
        from .provenance import digest

        fixed_rows = fixed_audit_indices(n, hyper.post_update_kl_rows)
        fixed_audit_t = to_tensors(
            {key: value[fixed_rows] for key, value in batch.items()}, normalizer
        )
        fixed_audit_d = {key: value[torch.as_tensor(fixed_rows)] for key, value in draws.items()}
        fixed_audit_old = torch.as_tensor(old_logp[fixed_rows], dtype=torch.float32)
        fixed_audit_identity = [
            {
                "world": int(batch["world"][row]),
                "camp": int(batch["camp"][row]),
                "turn": int(batch["turn"][row]),
                "rolloutRow": int(row),
            }
            for row in fixed_rows
        ]
        fixed_audit_digest = digest(fixed_audit_identity)
    stats: dict[str, float] = {}
    for epoch in range(hyper.epochs):
        approx = []
        epoch_batches = []
        epoch_order = rng.permutation(n) if modern else None
        for start in range(0, n, hyper.batch):
            if modern:
                index = epoch_order[start : start + hyper.batch]
                epoch_batches.append(index.copy())
            else:
                index = rng.permutation(n)[: hyper.batch] if n > hyper.batch else np.arange(n)
            actor_step_enabled = bool(train_actor and not actor_halted)
            expensive_critic_audit = bool(debug_enabled and epoch == 0 and start == 0)
            expensive_actor_audit = bool(expensive_critic_audit and actor_step_enabled)
            optimizer_debug = None
            if debug_enabled:
                optimizer_debug = {
                    "schemaVersion": 1,
                    "enabled": True,
                    "scope": {
                        "epoch": epoch,
                        "minibatch": start // hyper.batch,
                        "minibatchRowCount": len(index),
                        "expensiveAudit": expensive_critic_audit,
                        "actorAudit": expensive_actor_audit,
                        "criticAudit": expensive_critic_audit,
                    },
                    "attributionScope": (
                        "component gradients are local sampled-row geometry; they are not "
                        "additive attribution of "
                        "gradient clipping or the nonlinear Adam update"
                    ),
                }
            if economic_trace is not None:
                trace_metrics = {}
                import hashlib

                from .economic_trace import canonical_bytes, state_digest

                trace_before = {
                    "actor": state_digest(actor),
                    "critic": state_digest(critic),
                    "cost": state_digest(cost_critic),
                }
                trace_parameters = {
                    name: torch.cat([p.detach().view(-1).cpu().clone() for p in net.parameters()])
                    for name, net in (("actor", actor), ("critic", critic), ("cost", cost_critic))
                }
                economic_trace.write(
                    "updates",
                    {
                        "event": "mutation-start",
                        "policyVersionBefore": economic_trace.policy_version,
                        "batchTransitionIds": [economic_trace.transition_ids[i] for i in index],
                        "batchRowDigest": hashlib.sha256(
                            canonical_bytes([economic_trace.transition_ids[i] for i in index])
                        ).hexdigest(),
                        "epoch": epoch,
                        "minibatch": start // hyper.batch,
                        "beforeHashes": trace_before,
                    },
                )
            sub = {k: v[index] for k, v in batch.items()}
            t = to_tensors(sub, normalizer)
            d = {k: v[torch.as_tensor(index)] for k, v in draws.items()}
            audit_count = min(audit_rows, len(index)) if expensive_actor_audit else 0
            audit_t = (
                {key: value[:audit_count] for key, value in t.items()}
                if expensive_actor_audit
                else None
            )
            audit_d = (
                {key: value[:audit_count] for key, value in d.items()}
                if expensive_actor_audit
                else None
            )
            if expensive_actor_audit:
                # The component probes finish and restore state before constructing the real actor
                # graph below.
                from .optimizer_direction_audit import capture_component_gradients
                from .optimizer_update_debug import (
                    frozen_branch_policy_terms,
                    geometry_record,
                    preserved_observer_state,
                )

                audit_old = torch.as_tensor(old_logp[index[:audit_count]], dtype=torch.float32)
                audit_objective = torch.as_tensor(
                    objective_balanced[index[:audit_count]], dtype=torch.float32
                )
                audit_constraint = torch.as_tensor(
                    constraint_balanced[index[:audit_count]], dtype=torch.float32
                )
                audit_weight = torch.as_tensor(weight[index[:audit_count]], dtype=torch.float32)
                with (
                    preserved_observer_state(actor, numpy_generators={"minibatch": rng}),
                    torch.no_grad(),
                ):
                    probe_out = explore(actor(audit_t), hyper.log_std_floor)
                    probe_lp, _ = log_prob(probe_out, audit_t, audit_d, stage)
                    probe_ratio = torch.exp(probe_lp - audit_old)
                    _, frozen_branch = frozen_branch_policy_terms(
                        probe_ratio, audit_objective, audit_constraint, audit_weight, hyper.clip
                    )
                    audit_anchor = explore(teacher(audit_t), hyper.log_std_floor)

                def actor_probe_losses():
                    probe = explore(actor(audit_t), hyper.log_std_floor)  # noqa: B023 - late-binding closure kept as written (called within the iteration)
                    probe_logp, probe_entropy = log_prob(probe, audit_t, audit_d, stage)  # noqa: B023 - late-binding closure kept as written (called within the iteration)
                    terms, _ = frozen_branch_policy_terms(
                        torch.exp(probe_logp - audit_old),  # noqa: B023 - late-binding closure kept as written (called within the iteration)
                        audit_objective,  # noqa: B023 - late-binding closure kept as written (called within the iteration)
                        audit_constraint,  # noqa: B023 - late-binding closure kept as written (called within the iteration)
                        audit_weight,  # noqa: B023 - late-binding closure kept as written (called within the iteration)
                        hyper.clip,
                        use_unclipped=frozen_branch,  # noqa: B023 - late-binding closure kept as written (called within the iteration)
                    )
                    terms["entropyRegularizer"] = -hyper.entropy * probe_entropy.mean()
                    terms["teacherAnchor"] = (
                        hyper.teacher_kl
                        * teacher_kl(
                            probe,
                            audit_anchor,  # noqa: B023 - late-binding closure kept as written (called within the iteration)
                            stage.anchor_strategic,
                            stage.anchor_routes,
                            stage.fission,
                            stage.trade,
                        ).mean()
                    )
                    return terms

                actor_parameter_ids = {
                    id(parameter)
                    for group in actor_opt.param_groups
                    for parameter in group["params"]
                }
                actor_parameters = [
                    (name, parameter)
                    for name, parameter in actor.named_parameters()
                    if id(parameter) in actor_parameter_ids
                ]
                component_names = (
                    "objectivePolicy",
                    "constraintPolicy",
                    "entropyRegularizer",
                    "teacherAnchor",
                )
                captured = capture_component_gradients(
                    {
                        name: (lambda name=name: actor_probe_losses()[name])
                        for name in component_names
                    },
                    actor_parameters,
                    optimizers=(actor_opt,),
                    state_modules=(actor,),
                    numpy_generators={"minibatch": rng},
                )
                optimizer_debug["actorGradientGeometry"] = geometry_record(captured)
                optimizer_debug["scope"].update(
                    {
                        "selection": "first-rows-of-actual-minibatch",
                        "requestedRowCap": audit_rows,
                        "sampledRowCount": audit_count,
                        "auditTransitionIds": [
                            economic_trace.transition_ids[i] for i in index[:audit_count]
                        ],
                    }
                )
            out = explore(actor(t), hyper.log_std_floor)
            lp, ent = log_prob(out, t, d, stage)
            old = torch.as_tensor(old_logp[index], dtype=torch.float32)
            ratio = torch.exp(lp - old)
            adv = torch.as_tensor(combined[index], dtype=torch.float32)
            w = torch.as_tensor(weight[index], dtype=torch.float32)
            policy = (
                -(
                    w * torch.min(ratio * adv, ratio.clamp(1 - hyper.clip, 1 + hyper.clip) * adv)
                ).sum()
                / w.sum()
            )
            policy_terms = None
            if debug_enabled:
                from .optimizer_update_debug import frozen_branch_policy_terms

                policy_terms, _ = frozen_branch_policy_terms(
                    ratio.detach(),
                    torch.as_tensor(objective_balanced[index], dtype=torch.float32),
                    torch.as_tensor(constraint_balanced[index], dtype=torch.float32),
                    w,
                    hyper.clip,
                )
            with torch.no_grad():
                anchor = explore(teacher(t), hyper.log_std_floor)
            kl = teacher_kl(
                out, anchor, stage.anchor_strategic, stage.anchor_routes, stage.fission, stage.trade
            ).mean()
            loss = policy - hyper.entropy * ent.mean() + hyper.teacher_kl * kl
            imitation = None
            if actor_step_enabled and search is not None and hyper.search_weight > 0:
                m = len(search["label_kind"])
                pick = (
                    rng.permutation(m)[: hyper.search_batch]
                    if m > hyper.search_batch
                    else np.arange(m)
                )
                s_obs = {k: v[pick] for k, v in search.items()}
                ts = to_tensors(s_obs, normalizer)
                round3 = {}
                if "label_fission" in s_obs:
                    round3 = {
                        "fission": torch.as_tensor(s_obs["label_fission"], dtype=torch.long),
                        "trade_value": torch.as_tensor(
                            s_obs["label_trade_value"], dtype=torch.float32
                        ),
                        "trade_gate": torch.as_tensor(s_obs["label_trade_gate"], dtype=torch.long),
                        "trade_mask": torch.as_tensor(s_obs["label_trade_mask"].astype(np.int64)),
                    }
                imitation = search_loss(
                    actor(ts),
                    ts,
                    torch.as_tensor(s_obs["label_kind"], dtype=torch.long),
                    torch.as_tensor(s_obs["label_slot"], dtype=torch.long),
                    torch.as_tensor(s_obs["label_mix"], dtype=torch.float32),
                    **round3,
                )
                loss = loss + hyper.search_weight * imitation
            actor_step_performed = False
            actor_step_rolled_back = False
            if actor_step_enabled:
                if expensive_actor_audit:
                    before_joint = lp[:audit_count].detach().clone()
                    before_entropy = ent[:audit_count].detach().clone()
                    before_heads = {
                        name: (
                            values[0][:audit_count].detach().clone(),
                            values[1][:audit_count].detach().clone(),
                        )
                        for name, values in log_prob_components(out, t, d, stage).items()
                    }
                    before_coordinates = {
                        group: {
                            name: (
                                values[0][:audit_count].detach().clone(),
                                values[1][:audit_count].detach().clone(),
                                values[2][:audit_count].detach().clone(),
                            )
                            for name, values in coordinates.items()
                        }
                        for group, coordinates in log_prob_coordinate_components(
                            out, t, d, stage
                        ).items()
                    }
                step_snapshot = None
                if modern:
                    from .ppo_correctness import capture_step_state

                    step_snapshot = capture_step_state(actor, actor_opt)
                actor_opt.zero_grad()
                loss.backward()
                actor_norm = nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_opt.step()
                actor_step_performed = True
                if modern:
                    from .ppo_correctness import restore_step_state, sampled_nonnegative_kl

                    try:
                        with torch.no_grad():
                            audit_out = explore(actor(fixed_audit_t), hyper.log_std_floor)
                            audit_logp, _ = log_prob(audit_out, fixed_audit_t, fixed_audit_d, stage)
                            post_step_kl = float(
                                sampled_nonnegative_kl(audit_logp, fixed_audit_old)
                            )
                    except BaseException:
                        restore_step_state(actor, actor_opt, step_snapshot)
                        raise
                    fixed_post_step_kl.append(post_step_kl)
                    if post_step_kl > hyper.target_kl:
                        restore_step_state(actor, actor_opt, step_snapshot)
                        actor_step_performed = False
                        actor_step_rolled_back = True
                        actor_halted = True
                        rollback_count += 1
                    else:
                        accepted_actor_steps += 1
                if economic_trace is not None:
                    trace_metrics["actorGradientNormBeforeClipping"] = float(actor_norm)
                    trace_metrics["actorClippingScale"] = min(1.0, 1.0 / (float(actor_norm) + 1e-6))
                if expensive_actor_audit:
                    from .optimizer_update_debug import (
                        preserved_observer_state,
                        same_sampled_action_record,
                    )

                    with (
                        preserved_observer_state(actor, numpy_generators={"minibatch": rng}),
                        torch.no_grad(),
                    ):
                        after_out = explore(actor(audit_t), hyper.log_std_floor)
                        after_joint, after_entropy = log_prob(after_out, audit_t, audit_d, stage)
                        after_heads = {
                            name: (values[0].detach().clone(), values[1].detach().clone())
                            for name, values in log_prob_components(
                                after_out, audit_t, audit_d, stage
                            ).items()
                        }
                        after_coordinates = {
                            group: {
                                name: (
                                    values[0].detach().clone(),
                                    values[1].detach().clone(),
                                    values[2].detach().clone(),
                                )
                                for name, values in coordinates.items()
                            }
                            for group, coordinates in log_prob_coordinate_components(
                                after_out, audit_t, audit_d, stage
                            ).items()
                        }
                    native_activity = native_recorded_coordinate_activity(
                        before_coordinates, index[:audit_count]
                    )
                    optimizer_debug["sameSampledAction"] = same_sampled_action_record(
                        optimizer_debug["scope"]["auditTransitionIds"],
                        before_joint,
                        after_joint.detach(),
                        before_entropy,
                        after_entropy.detach(),
                        before_heads,
                        after_heads,
                        before_coordinates,
                        after_coordinates,
                        native_activity,
                    )
            elif expensive_critic_audit:
                optimizer_debug.setdefault("unavailable", {}).update(
                    {
                        "actorGradientGeometry": "critic-only-warmup-no-actor-mutation",
                        "sameSampledAction": "critic-only-warmup-no-actor-mutation",
                    }
                )
                optimizer_debug["scope"].update(
                    {
                        "selection": "first-rows-of-actual-minibatch",
                        "requestedRowCap": audit_rows,
                        "sampledRowCount": min(audit_rows, len(index)),
                        "auditTransitionIds": [
                            economic_trace.transition_ids[i]
                            for i in index[: min(audit_rows, len(index))]
                        ],
                    }
                )
            value_loss, value_by = value_step(t, index, audit=expensive_critic_audit)
            cost_loss = cost_step(t, index)
            approx.append(float((old - lp).mean().detach()))
            stats = {
                "policy": float(policy.detach()),
                "teacher_kl": float(kl.detach()),
                "entropy": float(ent.mean().detach()),
                "value": float(value_loss.detach()),
                "cost_value": float(cost_loss.detach()),
            }
            if imitation is not None:
                stats["search"] = float(imitation.detach())
            if (
                value_by is not None
            ):  # per-objective value loss: are the personality heads learning?
                stats["value_by"] = {name: float(v.detach()) for name, v in value_by.items()}
            if debug_enabled:
                objective_loss = float(policy_terms["objectivePolicy"].detach())
                constraint_loss = float(policy_terms["constraintPolicy"].detach())
                log_ratio = (lp - old).detach()
                nonnegative_kl = torch.expm1(log_ratio) - log_ratio
                quantiles = torch.quantile(
                    log_ratio.to(torch.float64),
                    torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64, device=log_ratio.device),
                )
                optimizer_debug["sampledPolicyShift"] = {
                    "measurementPhase": "pre-actor-step",
                    "rowSelection": "entire-current-minibatch",
                    "rowCount": len(log_ratio),
                    "nonnegativeKlEstimateMean": float(nonnegative_kl.mean()),
                    "logRatioMean": float(log_ratio.mean()),
                    "logRatioMin": float(log_ratio.min()),
                    "logRatioP10": float(quantiles[0]),
                    "logRatioMedian": float(quantiles[1]),
                    "logRatioP90": float(quantiles[2]),
                    "logRatioMax": float(log_ratio.max()),
                    "interpretation": (
                        "diagnostic sampled-action policy shift only; the historical "
                        "signed approximate-KL "
                        "early-stop rule and optimizer update are unchanged"
                    ),
                }
                optimizer_debug["lossContributions"] = {
                    "policyClippedCombined": float(policy.detach()),
                    "objectivePolicyFrozenBranch": objective_loss,
                    "constraintPolicyFrozenBranch": constraint_loss,
                    "policyDecompositionResidual": float(policy.detach())
                    - objective_loss
                    - constraint_loss,
                    "entropyRegularizerWeighted": float((-hyper.entropy * ent.mean()).detach()),
                    "teacherAnchorWeighted": float((hyper.teacher_kl * kl).detach()),
                    "searchImitationWeighted": (
                        None
                        if imitation is None
                        else float((hyper.search_weight * imitation).detach())
                    ),
                    "totalActorLoss": float(loss.detach()),
                }
                optimizer_debug["constraintContext"] = {
                    "lagrange": float(lagrange),
                    "objectiveAdvantageScale": float(objective_scale),
                    "costAdvantageScale": float(constraint_scale),
                    "clipBranch": (
                        "selected-by-actual-combined-advantage-and-frozen-for-component-probes"
                    ),
                    "interpretation": (
                        "objective and guard-cost terms partition the actual combined PPO "
                        "branch; a positive GAE "
                        "is training context, not an independently established beneficial effect"
                    ),
                }
                actual_clip = {
                    "actorStepPerformed": actor_step_performed,
                    "actorPreClipNorm": trace_metrics.get("actorGradientNormBeforeClipping"),
                    "actorClipScale": trace_metrics.get("actorClippingScale"),
                    "criticPreClipNorm": trace_metrics.get("criticGradientNormBeforeClipping"),
                    "criticClipScale": trace_metrics.get("criticClippingScale"),
                    "costPreClipNorm": trace_metrics.get("costGradientNormBeforeClipping"),
                    "costClipScale": trace_metrics.get("costClippingScale"),
                }
                if modern:
                    actual_clip["actorStepRolledBack"] = actor_step_rolled_back
                optimizer_debug["actualClip"] = actual_clip
            if economic_trace is not None:
                economic_trace.policy_version += int(actor_step_performed)
                changes = {
                    name: float(
                        torch.linalg.vector_norm(
                            torch.cat([p.detach().view(-1).cpu() for p in net.parameters()])
                            - trace_parameters[name]
                        )
                    )
                    for name, net in (("actor", actor), ("critic", critic), ("cost", cost_critic))
                }
                fixed_changes = (
                    optimizer_debug.get("sameSampledAction")
                    if optimizer_debug is not None
                    else None
                )
                unavailable = {}
                if fixed_changes is None:
                    if not debug_enabled:
                        unavailable_reason = "optimizer-audit-disabled"
                    elif expensive_critic_audit and not actor_step_enabled:
                        unavailable_reason = "critic-only-warmup-no-actor-mutation"
                    else:
                        unavailable_reason = "not-first-actor-minibatch-of-iteration"
                    unavailable["fixedAuditChanges"] = unavailable_reason
                    unavailable["perHeadEntropy"] = unavailable_reason
                economic_trace.write(
                    "updates",
                    {
                        "event": "mutation-complete",
                        "epoch": epoch,
                        "minibatch": start // hyper.batch,
                        "batchTransitionIds": [economic_trace.transition_ids[i] for i in index],
                        "policyVersionAfter": economic_trace.policy_version,
                        "batchRowDigest": hashlib.sha256(
                            canonical_bytes([economic_trace.transition_ids[i] for i in index])
                        ).hexdigest(),
                        "afterHashes": {
                            "actor": state_digest(actor),
                            "critic": state_digest(critic),
                            "cost": state_digest(cost_critic),
                        },
                        "ppoRatioMean": float(ratio.detach().mean()),
                        "ppoClipFraction": float(
                            ((ratio - 1).abs() > hyper.clip).float().mean().detach()
                        ),
                        "approxKl": float((old - lp).mean().detach()),
                        "approxKlMeasurementPhase": "pre-actor-step",
                        "losses": stats,
                        "gradientMetrics": dict(trace_metrics),
                        "parameterUpdateNorms": changes,
                        "fixedAuditChanges": fixed_changes,
                        "optimizerDebug": optimizer_debug,
                        "unavailable": unavailable,
                    },
                )
            if n <= hyper.batch:
                break
        if modern:
            from .ppo_correctness import epoch_coverage, epoch_minibatches
            from .value_calibration import GUARD_COST, anchor_loss, cost_anchor_loss

            coverage = epoch_coverage(epoch_batches, n)
            if not coverage["exactlyOnce"]:
                raise RuntimeError(
                    "observed-anchor-v1 epoch did not cover each rollout row exactly once"
                )
            epoch_coverage_records.append(coverage)
            vector_rows = np.flatnonzero(
                np.logical_or.reduce(
                    [observed_anchor.target_masks[name] for name in contracts.OBJECTIVES]
                )
            )
            for local in epoch_minibatches(len(vector_rows), hyper.batch, rng):
                anchor_total, _ = anchor_loss(
                    critic, observed_anchor, normalizer, vector_rows[local]
                )
                weighted = hyper.observed_anchor_weight * anchor_total
                critic_opt.zero_grad()
                weighted.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_opt.step()
                anchor_value_losses.append(float(anchor_total.detach()))
            cost_rows = np.flatnonzero(observed_anchor.target_masks[GUARD_COST])
            for local in epoch_minibatches(len(cost_rows), hyper.batch, rng):
                anchor_cost = cost_anchor_loss(
                    cost_critic, observed_anchor, normalizer, cost_rows[local]
                )
                weighted_cost = hyper.observed_anchor_weight * anchor_cost
                cost_opt.zero_grad()
                weighted_cost.backward()
                nn.utils.clip_grad_norm_(cost_critic.parameters(), 1.0)
                cost_opt.step()
                anchor_cost_losses.append(float(anchor_cost.detach()))
        stats["approx_kl"] = float(np.mean(approx))
        stats["epochs"] = epoch + 1
        if not modern and stats["approx_kl"] > hyper.target_kl:
            break
    if modern:
        stats["epoch_coverage"] = epoch_coverage_records
        stats["post_update_fixed_audit"] = {
            "rows": min(n, hyper.post_update_kl_rows) if train_actor else 0,
            "selection": "deterministic-evenly-spaced-across-full-rollout-order",
            "rowIdentities": fixed_audit_identity,
            "rowIdentityDigest": fixed_audit_digest,
            "sampledNonnegativeKl": fixed_post_step_kl,
            "target": hyper.target_kl,
            "acceptedActorSteps": accepted_actor_steps,
            "rolledBackActorSteps": rollback_count,
            "actorHaltedAfterRollback": rollback_count > 0,
            "gateConsumedForNextRollout": accepted_actor_steps > 0,
            "rollbackScope": "actor-parameters-buffers-and-Adam-state",
            "interpretation": "fixed sampled-action KL control; not full-policy KL",
        }
        stats["independent_anchor"] = {
            "weight": hyper.observed_anchor_weight,
            "vectorLossMean": float(np.mean(anchor_value_losses)),
            "guardCostLossMean": float(np.mean(anchor_cost_losses)),
            "trainingRole": "frozen-behaviour-independent-return-regularizer",
            "calibrationRole": False,
        }
    if hyper.kl_refit and not modern:
        refit = kl_refit_due(stats, hyper)
        if refit:  # one more critic-only epoch on this iteration's batch
            for start in range(0, n, hyper.batch):  # noqa: B007 - loop variable name kept (renaming skipped)
                index = rng.permutation(n)[: hyper.batch] if n > hyper.batch else np.arange(n)
                t = to_tensors({k: v[index] for k, v in batch.items()}, normalizer)
                if economic_trace is not None:
                    trace_metrics = {}
                    economic_trace.write(
                        "updates",
                        {
                            "event": "critic-refit-start",
                            "epoch": epoch + 1,
                            "batchTransitionIds": [economic_trace.transition_ids[i] for i in index],
                            "beforeHashes": {
                                "critic": state_digest(critic),
                                "cost": state_digest(cost_critic),
                            },
                            "batchRowDigest": hashlib.sha256(
                                canonical_bytes([economic_trace.transition_ids[i] for i in index])
                            ).hexdigest(),
                        },
                    )
                value_step(t, index)
                cost_step(t, index)
                if economic_trace is not None:
                    economic_trace.write(
                        "updates",
                        {
                            "event": "critic-refit-complete",
                            "epoch": epoch + 1,
                            "batchTransitionIds": [economic_trace.transition_ids[i] for i in index],
                            "afterHashes": {
                                "critic": state_digest(critic),
                                "cost": state_digest(cost_critic),
                            },
                            "batchRowDigest": hashlib.sha256(
                                canonical_bytes([economic_trace.transition_ids[i] for i in index])
                            ).hexdigest(),
                            "gradientMetrics": dict(trace_metrics),
                            "actorMetrics": None,
                            "unavailable": {"actorMetrics": "critic-only-refit-no-actor-mutation"},
                        },
                    )
                if n <= hyper.batch:
                    break
        stats["kl_refit"] = refit
    if economic_trace is not None:
        economic_trace.write(
            "updates",
            {
                "event": "iteration-summary",
                "policyVersion": economic_trace.policy_version,
                "metrics": stats,
            },
        )
    return stats


def kl_refit_due(stats: dict[str, float], hyper: Hyper) -> bool:
    """The KL refit trigger: the update's approximate KL exceeded target_kl (it stopped early)."""
    return bool(hyper.kl_refit and stats.get("approx_kl", 0.0) > hyper.target_kl)


def calibration_pause_after_update(
    actor: Actor,
    behavior_configuration: dict[str, Any],
    actor_gate_report: dict[str, Any],
    gate_source_policy_digest: str,
    latest_checkpoint: Path,
    request_path: Path,
    normalization_binding: dict[str, str],
    enabled_value_heads: tuple[str, ...],
) -> dict[str, Any] | None:
    """Return the required next-rollout pause after one frozen-behaviour PPO update.

    The gate remains valid for every minibatch/epoch in the current update.  Any
    accepted actor step makes it stale only for the next rollout.  A current
    policy whose held-out gate remains closed may perform one critic iteration,
    then pauses visibly instead of spinning.
    """
    from .value_calibration import behavior_policy_digest, write_refresh_request

    current_policy = behavior_policy_digest(actor, behavior_configuration)
    if current_policy != gate_source_policy_digest:
        return write_refresh_request(
            request_path,
            latest_checkpoint,
            actor,
            behavior_configuration,
            normalization_binding,
            enabled_value_heads,
            "accepted-actor-step-made-gate-policy-stale",
        )
    if not actor_gate_report.get("actorGateOpen", False):
        return {
            "status": "current-policy-gate-closed-after-one-critic-iteration",
            "action": (
                "inspect metrics; resume critic-only with the same gate data "
                "or collect more evidence"
            ),
            "localReadinessClaim": False,
        }
    return None


def iteration(
    actor: Actor,
    critic: Critic,
    cost_critic: CostCritic,
    teacher: Actor,
    normalizer: Normalizer,
    seeds: list[int],
    turns: int,
    stage: Stage,
    hyper: Hyper,
    optimizers: tuple,
    lagrange: float,
    generator: torch.Generator,
    rng: np.random.Generator,
    grassland: frozenset = frozenset(),
    search: dict[str, np.ndarray] | None = None,
    train_actor: bool = True,
    branch: tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None] | None = None,
    bank: DaughterBank | None = None,
    economic_trace=None,
    observed_train=None,
    observed_gate=None,
    enabled_value_heads: tuple[str, ...] = (),
) -> tuple[float, dict[str, float]]:
    value_calibration = None
    actor_requested = bool(train_actor)
    if hyper.learner_contract == "observed-anchor-v1":
        if observed_train is None or observed_gate is None or not enabled_value_heads:
            raise ValueError("observed-anchor-v1 iteration lacks its train/gate data")
        from .value_calibration import behavior_policy_digest, calibration_report

        behaviour = behavior_policy_configuration(stage, hyper, actor)
        value_calibration = calibration_report(
            critic,
            cost_critic,
            observed_gate,
            normalizer,
            enabled_heads=enabled_value_heads,
            current_policy_digest=behavior_policy_digest(actor, behaviour),
        )
        train_actor = bool(train_actor and value_calibration["actorGateOpen"])
    sampler = (
        None if hyper.personality == "default" else personality_sampler(hyper.personality, rng)
    )
    inherit = (
        None
        if sampler is None
        else (lambda parent: personality_drift(parent, rng, hyper.personality_drift))
    )
    expected_worker = getattr(actor, "worker_contract", None)
    expected_worlds = []
    if expected_worker is not None:
        from .worker_contract import world

        for item in seeds:
            seed, mask = (item, 0) if isinstance(item, int) else (item[0], item[1])
            record = (
                bank.get(item[3])
                if not isinstance(item, int) and len(item) > 3 and item[3] >= 0
                else None
            )
            expected_worlds.append(
                world(
                    seed,
                    mask,
                    snapshot_sha256=record["snapshot_sha256"] if record else "",
                    turn=record["turn"] if record else None,
                    movement=stage.bridge_movement,
                    head=getattr(actor, "region_head", None),
                )
            )
    collect_options = {"economic_trace": economic_trace} if economic_trace is not None else {}
    roll = collect(
        actor,
        normalizer,
        seeds,
        turns,
        stage,
        generator,
        floor=hyper.log_std_floor,
        personality=sampler,
        inherit=inherit,
        bank=bank,
        **collect_options,
    )
    if expected_worker is not None:
        from .provenance import ProvenanceError
        from .worker_contract import checked

        if (
            len(roll.worker_contracts) != len(seeds)
            or len(roll.world_contracts) != len(seeds)
            or any(checked(row) != expected_worker for row in roll.worker_contracts)
            or roll.world_contracts != expected_worlds
        ):
            raise ProvenanceError("collection producer/world proof differs before optimization")
    batch, draws, old_logp = flatten(roll)
    reward = data.lineage_rewards(batch)
    regimes = np.asarray(
        [
            regime(int(s), int(m), grassland)
            for s, m in zip(batch["seed"], batch["mask"], strict=False)
        ]
    )
    span = draws["span"].numpy()
    hold_legal = batch.get("commit_hold_mask", np.zeros(len(reward), bool)).astype(bool)
    voluntary_break = (
        hold_legal & (span != contracts.SPAN_HOLD)
        if stage.commitments
        else np.zeros(len(reward), bool)
    )
    # Shaping costs as a share of the decision's reward that turn: the mix-change and
    # voluntary-break costs.
    change = mix_changes(batch, draws)
    share = hyper.mix_change_cost * change + hyper.break_cost * voluntary_break
    welfare_now = np.maximum(0.0, batch["welfare"].astype(np.float64))
    reward = (
        reward
        - hyper.mix_change_cost * change * welfare_now
        - hyper.break_cost * voluntary_break * welfare_now
    )
    rows = np.flatnonzero(batch["bootstrap_index"] >= 0)
    drift = clipped = saturated = None
    if isinstance(critic, VectorCritic):
        # Each objective component: its own reward, value, bootstrap and GAE; the policy follows
        # their weighted sum.
        components = data.reward_components(batch)
        components["welfare"] = reward
        if "personality" in batch:
            components = shape_components(components, share)
        value_by = vector_values(critic, batch, normalizer)
        tails = {}
        if roll.bootstrap_obs:
            with torch.no_grad():
                logits = critic(to_tensors(roll.bootstrap_obs[0], normalizer))
            tails = {name: critic.expectation(v).numpy() for name, v in logits.items()}
        advantages, target_by = {}, {}
        objective_trace = {}
        raw_components = data.reward_components(batch) if economic_trace is not None else None
        for name in critic.heads:
            boot = np.zeros(len(reward))
            if name in tails:
                boot[rows] = tails[name][batch["bootstrap_index"][rows]] * head_scale(
                    name, batch["population"][rows]
                )
            advantages[name], target_by[name] = gae(batch, components[name], value_by[name], boot)
            if economic_trace is not None:
                objective_trace[name] = (
                    components[name],
                    raw_components[name],
                    value_by[name],
                    boot,
                    advantages[name],
                    target_by[name],
                )
        advantage = (
            combine(advantages, hyper.objective_weights)
            if "personality" not in batch
            else combine_personal(
                advantages, batch["personality"], hyper.personality_scales, hyper.personality_clip
            )
        )
        drift = scale_drift(advantages)
        # Value-bin saturation per objective head (review 5.5(3)): the share of this batch's values
        # at the bin range's ends.
        from .branch_bias import saturation

        saturated = saturation(
            {
                name: value_by[name] / head_scale(name, batch["prior_population"])
                for name in value_by
            }
        )
        clipped = clip_fractions(
            advantages, hyper.personality_clip if hyper.personality_scales == "reference" else 0.0
        )
        target = {
            name: value / head_scale(name, batch["prior_population"])
            for name, value in target_by.items()
        }
    else:
        value = values(critic, batch, normalizer)
        bootstrap = np.zeros(len(reward))
        if roll.bootstrap_obs:
            with torch.no_grad():
                u_tail = critic.expectation(
                    critic(to_tensors(roll.bootstrap_obs[0], normalizer))
                ).numpy()
            # The next observation's value per resident times the residents it was taken at (this
            # row's end population).
            bootstrap[rows] = u_tail[batch["bootstrap_index"][rows]] * np.maximum(
                1.0, batch["population"][rows]
            )
        advantage, target = gae(batch, reward, value, bootstrap)
        objective_trace = {
            "welfare": (
                reward,
                data.lineage_rewards(batch),
                value,
                bootstrap,
                advantage,
                target.copy(),
            )
        }
        target = data.scale_free(target, batch)
    with torch.no_grad():
        cost_value = cost_critic(to_tensors(batch, normalizer)).numpy().astype(np.float64)
    cost_boot = (
        cost_bootstrap(cost_critic, roll, batch, normalizer, rows)
        if hyper.cost_bootstrap
        else np.zeros(len(reward))
    )
    cost_adv, cost_target = gae(batch, batch["cost"].astype(np.float64), cost_value, cost_boot)
    branch_targets = None
    if branch is not None:
        # Search branch rows (critic only): lambda-returns over each branch sequence, bootstrapped
        # by the current
        # critic after the last tail turn, recomputed every iteration.
        from . import branch_shards

        branch_targets = (
            branch[0],
            branch_shards.targets(critic, branch[0], branch[1], normalizer),
        )
    trace_options = (
        {
            "economic_trace": economic_trace,
            "trace_context": (
                roll,
                batch,
                draws,
                old_logp,
                objective_trace,
                (cost_value, cost_boot, cost_adv, cost_target),
            ),
        }
        if economic_trace is not None
        else {}
    )
    stats = update(
        actor,
        critic,
        cost_critic,
        teacher,
        normalizer,
        batch,
        draws,
        old_logp,
        advantage,
        target,
        cost_adv,
        cost_target,
        lagrange,
        stage,
        hyper,
        optimizers,
        rng,
        regimes,
        search,
        train_actor,
        branch_targets,
        observed_anchor=observed_train,
        **trace_options,
    )
    if value_calibration is not None:
        stats["value_calibration"] = value_calibration
        stats["actor_gate_requested"] = actor_requested
        stats["actor_gate_open"] = bool(value_calibration["actorGateOpen"])
    if "personality" in batch and hasattr(actor.encoder, "film_extent"):  # FiLM instability monitor
        sample = {k: v[:512] for k, v in batch.items()}
        stats["film_max"] = actor.encoder.film_extent(to_tensors(sample, normalizer))
    if drift is not None:
        stats["scale_drift"] = drift
        stats["scale_clipped"] = clipped
        stats["value_saturation"] = saturated
    stats["bank_decisions"] = int((batch["bank"] >= 0).sum()) if "bank" in batch else 0
    from .diagnostics import aggregate

    stats["decision_diagnostics"] = aggregate(roll.diagnostics)
    stats["worker_contracts"] = roll.worker_contracts
    stats["world_contracts"] = roll.world_contracts
    # Guard cost bootstrap (round 3): whether it is on and the mean bootstrapped cost value of
    # truncated villages.
    stats["cost_bootstrap"] = bool(hyper.cost_bootstrap)
    stats["cost_bootstrap_mean"] = float(cost_boot[rows].mean()) if len(rows) else 0.0
    mean_cost = float(batch["cost"].mean())
    lagrange = max(0.0, lagrange + hyper.lagrange_lr * (mean_cost - hyper.cost_budget))
    stats.update(
        {
            "decisions": float(len(reward)),
            "welfare": float(batch["welfare"].sum()),
            "mean_cost": mean_cost,
            "lagrange": lagrange,
            "mean_qol": float(batch["qol"].mean()),
            "mix_change": float(change.mean()),
            "span_shares": {
                str(s): float((span == i).mean()) for i, s in enumerate(contracts.SPANS)
            },
            # Contract 10 movement diagnostics: sampled slot shares (native, keep, row 0 stay, rows
            # 1-7 move) and
            # each group's mean advantage, so "never moves" can be told apart from "moved and it did
            # not pay".
            "region_shares": {name: float(m.mean()) for name, m in region_groups(draws).items()},
            "region_advantage": {
                name: float(advantage[m].mean())
                for name, m in region_groups(draws).items()
                if m.any()
            },
            "hold_legal": float(hold_legal.mean()),
            "voluntary_breaks": float(voluntary_break.mean()),
            "regime_decisions": {
                r: int((regimes == r).sum()) for r in REGIMES if (regimes == r).any()
            },
        }
    )
    return lagrange, stats


REGIMES = ("default", "grassland", "herding", "farming", "fishing")


def regime(seed: int, mask: int, grassland: frozenset) -> str:
    """The world regime of a decision: a capability fixture by its withheld mask, else a grassland-rich opening or an
    ordinary world. These are the stage-gate categories."""  # noqa: E501 - docstring/prose line kept verbatim
    for name, m in contracts.CAPABILITY_MASKS.items():
        if mask == m:
            return name
    return "grassland" if seed in grassland else "default"


def balance(
    combined: np.ndarray, regimes: np.ndarray | None, cap: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-regime advantage balancing: standardize within each regime and weight every present regime equally (each
    decision's weight n / (K n_r), clipped to [1 / cap, cap] and renormalized to mean 1). Without regimes, the
    previous global standardization with unit weights."""  # noqa: E501 - docstring/prose line kept verbatim
    if regimes is None or len(set(regimes.tolist())) <= 1:
        out = (combined - combined.mean()) / max(1e-8, combined.std())
        return out, np.ones(len(combined))
    out = np.empty_like(combined)
    weight = np.empty(len(combined))
    names = sorted(set(regimes.tolist()))
    n = len(combined)
    for r in names:
        rows = regimes == r
        x = combined[rows]
        out[rows] = (x - x.mean()) / max(1e-8, x.std()) if rows.sum() > 1 else 0.0
        weight[rows] = np.clip(n / (len(names) * rows.sum()), 1.0 / cap, cap)
    return out, weight / weight.mean()


def mix_changes(batch: dict[str, np.ndarray], draws: dict[str, torch.Tensor]) -> np.ndarray:
    """Total-variation distance between each decision's sampled economy mix and the same village's previous one
    (0 at a village's first decision)."""  # noqa: E501 - docstring/prose line kept verbatim
    mix = mix_from_ratios(
        draws["mix_z"], torch.as_tensor(batch["economy_mix_mask"], dtype=torch.bool)
    ).numpy()
    change = np.zeros(len(mix))
    for rows in data.sequences(batch):
        for a, b in zip(rows[:-1], rows[1:], strict=False):  # noqa: RUF007 - zip over slices kept (not plain pairwise)
            change[b] = 0.5 * float(np.abs(mix[b] - mix[a]).sum())
    return change


def world_sampler(
    stage: Stage,
    train: np.ndarray,
    grassland: np.ndarray,
    rng: np.random.Generator,
    valley: list[tuple[int, int]] | None = None,
    bank: DaughterBank | None = None,
    schedule="legacy",
):
    """Curriculum worlds as (seed, withheld-technology mask[, heuristic warm-up turns[, bank entry id]]). Stage 2 draws
    the three single-capability fixtures equally; stages 3-4 draw half ordinary TRAIN worlds, a quarter grassland-rich
    openings (where herding competes with farming) and a quarter rehearsed capability fixtures. With learned movement
    and a valley bank, a quarter of worlds (any stage) are valley fixture states, where moving can pay. With a
    non-empty daughter bank, slot i % 4 == 0 is a banked daughter start (stage 2: only capability-fixture entries);
    an empty bank (or no eligible entry) leaves that slot as before."""  # noqa: E501 - docstring/prose line kept verbatim
    if schedule == "r3-balanced-v1":
        from .curriculum import BalancedSampler

        return BalancedSampler(stage, train, grassland, rng, valley, bank)
    if schedule != "legacy":
        raise ValueError("unknown frozen curriculum")
    from .protocol import seeds

    seeds(train, "legacy TRAIN")
    seeds(grassland, "legacy grassland")
    seeds([s for s, _ in valley or []], "legacy valley")
    masks = list(contracts.CAPABILITY_MASKS.values())

    def draw(n: int) -> list[tuple]:
        worlds = []
        for i in range(n):
            entry = None
            if bank is not None and i % 4 == 0:
                entry = bank.sample(rng, set(masks) if stage.name == "capability" else None)
            if entry is not None:
                worlds.append((int(entry["seed"]), int(entry["mask"]), 0, int(entry["id"])))
            elif valley and stage.region_movement and i % 4 == 2:
                seed, warm = valley[int(rng.integers(len(valley)))]
                worlds.append((int(seed), 0, int(warm)))
            elif stage.name == "capability":
                worlds.append((int(rng.choice(train)), int(masks[i % len(masks)])))
            elif i % 4 == 1 and len(grassland) > 0:
                worlds.append((int(rng.choice(grassland)), 0))
            elif i % 4 == 3:
                # Rehearsal: a quarter of stage 3-4 worlds are the stage-2 capability fixtures, so
                # single-capability
                # skills (the fishing regression seen after stage 2) are not forgotten.
                worlds.append((int(rng.choice(train)), int(masks[(i // 4) % len(masks)])))
            else:
                worlds.append((int(rng.choice(train)), 0))
        return worlds

    return draw


def evaluate_closed_loop(
    checkpoint: Path | None,
    worlds: list[tuple[int, int]],
    turns: int,
    workers: int = 6,
    movement: bool = False,
    commitments: bool | None = None,
    region: bool | None = None,
    fission: bool | None = None,
    trade: bool | None = None,
    personality: list[float] | None = None,
    grouped: bool = False,
    expected_producer=None,
) -> dict[str, float]:
    """Deterministic closed-loop rollouts (evaluate.rollout) summed over held-out worlds. ``commitments`` overrides
    the checkpoint's own setting (the baseline of a single-turn run evaluates its init single-turn). A world is
    (seed, withheld mask) or (seed, withheld mask, heuristic warm-up turns). ``personality`` evaluates every village
    at that Elder personality (None: the default)."""  # noqa: E501 - docstring/prose line kept verbatim
    from concurrent.futures import ProcessPoolExecutor

    from .evaluate import rollout
    from .worker_contract import from_state, world

    if expected_producer is None and checkpoint is not None:
        expected_producer = from_state(
            torch.load(checkpoint, map_location="cpu", weights_only=False)
        )
    descriptors = [world(w[0], w[1], movement=movement) for w in worlds]
    with ProcessPoolExecutor(max_workers=min(workers, len(worlds))) as pool:
        rows = list(
            pool.map(
                rollout,
                [w[0] for w in worlds],
                [turns] * len(worlds),
                [None if checkpoint is None else str(checkpoint)] * len(worlds),
                [w[1] for w in worlds],
                [movement] * len(worlds),
                [commitments] * len(worlds),
                [region] * len(worlds),
                [w[2] if len(w) > 2 else 0 for w in worlds],
                [fission] * len(worlds),
                [trade] * len(worlds),
                [personality] * len(worlds),
                [grouped] * len(worlds),
                [False] * len(worlds),
                [expected_producer] * len(worlds),
                descriptors,
            )
        )
    if expected_producer is not None:
        from .provenance import ProvenanceError

        expected_old = {
            "simulator": expected_producer["simulator"],
            "bridge_sha256": expected_producer["bridge_sha256"],
            "flags": expected_producer["flags"],
            "raw_materials": False,
        }
        if len(rows) != len(descriptors) or any(
            row.get("worker_contract") != expected_producer
            or row.get("producer") != expected_old
            or row.get("world_contract") != descriptor
            for row, descriptor in zip(rows, descriptors, strict=False)
        ):
            raise ProvenanceError(
                "selection worker producer differs from frozen run; cannot select checkpoint"
            )
    population = sum(r["population"] for r in rows)
    return {
        "worker_contract": expected_producer,
        "producer": rows[0].get("producer") if rows else None,
        "world_contracts": descriptors,
        "welfare": float(sum(r["welfare"] for r in rows)),
        "population": float(population),
        "qol": float(sum(r["qol"] * r["population"] for r in rows) / max(1, population)),
        "specialist_share": float(
            sum(r.get("specialist_share", 0.0) * r["population"] for r in rows) / max(1, population)
        ),
        "below_30": float(np.mean([r["below_30"] for r in rows])),
        "extinctions": float(sum(r["extinctions"] for r in rows)),
        "settlements": float(np.mean([r["settlements"] for r in rows])),
        "interventions_per_decision": float(
            np.mean([r["interventions_per_decision"] for r in rows])
        ),
        "mix_change": float(np.mean([r.get("mix_change", 0.0) for r in rows])),
        "mix_switches": float(np.mean([r.get("mix_switches", 0.0) for r in rows])),
        "hold_share": float(np.mean([r.get("hold_share", 0.0) for r in rows])),
        "commit_share": float(np.mean([r.get("commit_share", 0.0) for r in rows])),
        "region_move_share": float(np.mean([r.get("region_move_share", 0.0) for r in rows])),
        "fission_encourage_share": float(
            np.mean([r.get("fission_encourage_share", 0.0) for r in rows])
        ),
        "fission_discourage_share": float(
            np.mean([r.get("fission_discourage_share", 0.0) for r in rows])
        ),
        "trade_utility": float(sum(r.get("trade_utility", 0.0) for r in rows)),
        "trades": float(sum(r.get("trades", 0) for r in rows)),
        # Per-world outcomes, in world order, for paired comparisons across policies.
        "worlds": [
            [int(w[0]), int(w[1]), float(r["welfare"]), int(r["extinctions"])]
            for w, r in zip(worlds, rows, strict=False)
        ],
    }


# Spread selection (docs/MODEL9_PERSONALITY.md, welfare leakage): the personalities a personality
# run's checkpoints are
# judged at, over contracts.OBJECTIVES (welfare, population_years, trade_utility, qol_years).
SELECTION_POINTS: dict[str, list[float]] = {
    "welfare": [1, 0, 0, 0],
    "population": [0, 1, 0, 0],
    "qol": [0, 0, 0, 1],
}
SELECTION_REGIME_FLOOR = 0.97


def spread_selection(points: dict[str, dict], baseline: dict) -> dict[str, Any]:
    """Score one personality-run checkpoint from its evaluations at SELECTION_POINTS (each an ``evaluate`` total with
    per-regime results). Eligible only when every point keeps extinctions at or below the init's and the welfare
    point keeps every regime's welfare ratio at or above SELECTION_REGIME_FLOOR. The score is the smaller of two log
    separations, so both directions must move: QoL at the QoL point over QoL at the population point, and population
    at the population point over population at the QoL point. A score of 0 is no separation."""  # noqa: E501 - docstring/prose line kept verbatim
    eps = 1e-9
    qol_gap = float(np.log(max(eps, points["qol"]["qol"]) / max(eps, points["population"]["qol"])))
    pop_gap = float(
        np.log(max(eps, points["population"]["population"]) / max(eps, points["qol"]["population"]))
    )
    ratios = {
        name: r["welfare"] / max(eps, baseline["regimes"][name]["welfare"])
        for name, r in points["welfare"]["regimes"].items()
    }
    extinctions = {name: p["extinctions"] for name, p in points.items()}
    eligible = (
        all(x <= baseline["extinctions"] for x in extinctions.values())
        and min(ratios.values()) >= SELECTION_REGIME_FLOOR
    )
    return {
        "score": min(qol_gap, pop_gap),
        "qol_gap": qol_gap,
        "population_gap": pop_gap,
        "welfare_regime_ratios": ratios,
        "extinctions": extinctions,
        "eligible": bool(eligible),
    }


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learner-contract",
        choices=("legacy", "observed-anchor-v1"),
        default="legacy",
        help="opt in to independently anchored values, exact epoch coverage and post-step KL "
        "rollback",
    )
    parser.add_argument(
        "--observed-train-anchor",
        type=Path,
        default=None,
        help="independent observed-return training anchors (required by observed-anchor-v1)",
    )
    parser.add_argument(
        "--observed-actor-gate",
        type=Path,
        default=None,
        help="disjoint current-policy observed returns controlling actor updates",
    )
    parser.add_argument(
        "--observed-anchor-weight",
        type=float,
        default=1.0,
        help="critic-only weight of independent observed-return anchors",
    )
    parser.add_argument(
        "--post-update-kl-rows",
        type=int,
        default=128,
        help="fixed rollout rows used to accept or roll back each new-mode actor step",
    )
    parser.add_argument(
        "--economic-trace",
        type=Path,
        default=None,
        help="opt-in prospective pre-update economic/critic trace directory (must be new)",
    )
    parser.add_argument(
        "--optimizer-audit-rows",
        type=int,
        default=0,
        help="with --economic-trace, audit first N rows of only the first PPO minibatch (0 off, "
        "max 64)",
    )
    parser.add_argument(
        "--init", type=Path, required=True, help="stage-1 or earlier-stage checkpoint"
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        default=None,
        help="anchor actor (default: the stage-1 actor in --init)",
    )
    parser.add_argument("--stage", choices=sorted(STAGES), default="capability")
    parser.add_argument("--seeds", default="1000-1599", help="TRAIN seed range")
    parser.add_argument(
        "--grassland-seeds", type=Path, default=None, help="comma-separated grassland-rich seeds"
    )
    parser.add_argument("--worlds", type=int, default=12, help="worlds per iteration")
    parser.add_argument("--turns", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument(
        "--eval-seeds",
        default="2020-2025",
        help="DEVELOPMENT selection seeds for periodic evaluation (disjoint from the stage "
        "gate's 2000-2019)",
    )
    parser.add_argument(
        "--no-balance", action="store_true", help="global advantage standardization (pre-pilot)"
    )
    parser.add_argument("--mix-change-cost", type=float, default=None)
    parser.add_argument(
        "--commitments", action="store_true", help="contract 9: learn the capital commitment span"
    )
    parser.add_argument(
        "--learned-movement",
        action="store_true",
        help="contract 10: the policy moves over region rows with destination intents (mix "
        "given the destination)",
    )
    parser.add_argument(
        "--valley-seeds",
        type=Path,
        default=None,
        help="valley fixture bank (lines 'seed,warmup'); with --learned-movement a quarter of "
        "training worlds",
    )
    parser.add_argument(
        "--native-start",
        type=float,
        default=None,
        help="re-seat the region head's native probability (all slots legal), e.g. 0.5",
    )
    parser.add_argument(
        "--personality",
        default="default",
        help="Elder personalities for training: default | fixed:w,w,w,w | dirichlet:alpha "
        "(token model only)",
    )
    parser.add_argument(
        "--personality-scales",
        choices=("reference", "batch"),
        default="reference",
        help="personality exchange rates: fixed contracts.OBJECTIVE_REFERENCE_SCALES (round 3) "
        "or each "
        "component's batch std (earlier runs)",
    )
    parser.add_argument(
        "--personality-clip",
        type=float,
        default=5.0,
        help="clip each normalised component advantage A_k / s_k_ref to +/- this (reference "
        "scales; 0 off)",
    )
    parser.add_argument(
        "--daughter-bank",
        type=Path,
        default=None,
        help="daughter-start bank directory: capture founding daughters and draw a quarter of "
        "worlds from it",
    )
    parser.add_argument(
        "--search-labels",
        default=None,
        help="comma-separated search_teacher labels.npz files: imitate search winners (expert "
        "iteration)",
    )
    parser.add_argument("--search-weight", type=float, default=0.5)
    parser.add_argument(
        "--critic-warmup",
        type=int,
        default=None,
        help="critic-only iterations before PPO when the vector critic's heads are new (default "
        "5; 0 off)",
    )
    parser.add_argument("--search-margin", type=float, default=None)
    parser.add_argument(
        "--branch-shards",
        default=None,
        help="comma-separated search branch-shard directories (search_teacher "
        "<out>/branch_shards): "
        "critic-only rows mixed into every critic minibatch",
    )
    parser.add_argument(
        "--branch-share",
        type=float,
        default=None,
        help="share of each critic minibatch from branch rows (default 0.25)",
    )
    parser.add_argument(
        "--round-start-warmup",
        type=int,
        default=None,
        help="critic-only iterations at the start of every run (not repeated on --resume past "
        "them; "
        "overlaps the fresh-heads warm-up: the larger count, once)",
    )
    parser.add_argument(
        "--kl-refit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="one extra critic-only epoch after an update stopped on target_kl (default on for "
        "new runs; "
        "a resumed run keeps its own setting, off for runs saved before the flag)",
    )
    parser.add_argument(
        "--provenance",
        choices=("reject", "flag", "allow"),
        default="reject",
        help="search labels and branch shards that mix source checkpoints or code commits, or "
        "hold rows "
        "of unknown source: reject (default), flag (warn and log) or allow",
    )
    parser.add_argument(
        "--cost-bootstrap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="bootstrap truncated living villages in the guard cost critic from its own value "
        "(round 3; "
        "default on for new runs, a resumed run keeps its own setting, off for runs saved before)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from <out>/resume.pt when present (preemptible VMs); otherwise start fresh",
    )
    parser.add_argument(
        "--learned-trade",
        action="store_true",
        help="contract 11: the policy sets its trade posture; adds the trade_utility critic "
        "component "
        "(weight contracts.TRADE_UTILITY_WEIGHT unless --objective-weights sets it)",
    )
    parser.add_argument(
        "--learned-fission",
        action="store_true",
        help="the policy chooses the fission policy (split timing and daughter share), anchored "
        "to the teacher",
    )
    parser.add_argument(
        "--observe-movement",
        action="store_true",
        help="contract 10 control: movement is enabled in the bridge (same observations) but "
        "every decision is native",
    )
    parser.add_argument(
        "--vector-critic",
        action="store_true",
        help="one value head per objective component; the advantage is their weighted sum",
    )
    parser.add_argument(
        "--objective-weights",
        default=None,
        help="name=weight[,name=weight] over contracts.OBJECTIVES",
    )
    parser.add_argument("--break-cost", type=float, default=None)
    parser.add_argument(
        "--experts",
        action="store_true",
        help="add capability-conditioned expert adapters (warm start: identical to --init)",
    )
    parser.add_argument("--eval-turns", type=int, default=25)
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=6,
        help="processes for periodic selection evaluation (raise with a wider --eval-seeds bank)",
    )
    parser.add_argument(
        "--film-bound",
        type=float,
        default=None,
        help="bound the FiLM scale and shift to +/- this with tanh (off by default: it also "
        "dampens FiLM; "
        "use only if film_max shows instability)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--teacher-kl",
        type=float,
        default=None,
        help="weight of the teacher KL anchor (default 0.05)",
    )
    parser.add_argument("--log-std-floor", type=float, default=None)
    parser.add_argument("--lagrange-lr", type=float, default=None)
    parser.add_argument("--cost-budget", type=float, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, default=None)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--prior-use-inventory", type=Path, default=None)
    parser.add_argument("--curriculum", choices=("legacy", "r3-balanced-v1"), default=None)
    parser.add_argument("--bank-image", type=Path, default=None)
    parser.add_argument(
        "--arm",
        choices=("R3def", "R3per", "R3lo"),
        default=None,
        help="frozen preregistered treatment identity for formal reports",
    )
    parser.add_argument("--execution-manifest", type=Path, default=None)
    parser.add_argument(
        "--nonqualifying",
        action="store_true",
        help="isolated local diagnostic only; no best/qualification artifacts",
    )
    parser.add_argument("--local-execution-design", type=Path, default=None)
    return parser


def main() -> int:
    parser = argument_parser()
    from .checkpoint import (
        actor_parameters,
        parse,
        resolve_head,
        restore_actor,
        resume_setting,
        run_manifest,
    )

    args = parse(parser)
    if args.optimizer_audit_rows and args.economic_trace is None:
        raise ValueError("--optimizer-audit-rows requires --economic-trace")
    from .protocol import enforce_execution

    qualified = enforce_execution(args)
    if qualified and "live_probe" in qualified:
        args._qualified_probe = qualified["live_probe"]
    if args.provenance != "reject" and (args.search_labels or args.branch_shards):
        raise ValueError(
            "flag/allow provenance is for analysis and migrations; training requires reject"
        )
    from .calibration import assert_training, manifest

    held_manifest = manifest(args.calibration_manifest) if args.calibration_manifest else None
    from .provenance import declared_sources

    declared = declared_sources(args.source_manifest)
    resume_path = args.out / "resume.pt"
    resume_state = (
        torch.load(resume_path, map_location="cpu", weights_only=False)
        if args.resume and resume_path.exists()
        else None
    )
    saved_curriculum = (resume_state or {}).get("curriculum", "legacy")
    if resume_state is not None and args.curriculum not in (None, saved_curriculum):
        raise ValueError("curriculum differs from frozen resume; start a new run")
    args.curriculum = (
        saved_curriculum if resume_state is not None else args.curriculum or "r3-balanced-v1"
    )
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    from .upgrade import upgrade

    state = upgrade(torch.load(args.init, map_location="cpu", weights_only=False))
    if resume_state is not None:
        saved_learner_contract = resume_state.get("learner_contract", "legacy")
        if saved_learner_contract != args.learner_contract:
            raise ValueError("resume learner contract differs from --learner-contract")
        if args.learner_contract == "observed-anchor-v1":
            from .provenance import digest

            if resume_state.get("feature_contract") != state.get("feature_contract") or digest(
                resume_state.get("normalizer")
            ) != digest(state.get("normalizer")):
                raise ValueError("resume input-normalization contract differs from --init")
    core_width = int(state["core_width"])
    privileged = int(state.get("privileged_width", 0))
    from .model import load_checkpoint_normalizer

    normalizer = load_checkpoint_normalizer(
        state,
        expected_feature_contract=learner_feature_contract(args.learner_contract),
        required_history_indices=[],
    )
    experts = bool(state.get("experts", False)) or args.experts
    token = bool(
        state.get("token", False)
    )  # contract 12 token model (from stage-1 cloning; never an upgrade)
    actor, teacher = Actor(core_width, experts=experts, token=token), Actor(core_width, token=token)
    critic = Critic(core_width, privileged_width=privileged, token=token)
    missing, unexpected = restore_actor(actor, state, args.region_head, strict=False)
    if unexpected or any(not k.startswith("experts.") for k in missing):
        raise ValueError(
            f"checkpoint does not match the actor: missing {missing}, unexpected {unexpected}"
        )
    # Contract 11: learned trade needs the trade_utility value head (approved credit: own realised
    # trade utility).
    fresh_heads = False
    if (
        args.vector_critic
        or state.get("vector_critic")
        or args.learned_trade
        or args.personality != "default"
    ):
        if state.get("vector_critic"):
            vector = VectorCritic(core_width, privileged_width=privileged, token=token)
            vector.load_state_dict(state["critic"])
            critic = vector
        else:
            critic.load_state_dict(state["critic"])
            critic = VectorCritic.from_scalar(critic, core_width, privileged)
            fresh_heads = True
    else:
        critic.load_state_dict(state["critic"])
    if args.learner_contract == "observed-anchor-v1" and not isinstance(critic, VectorCritic):
        raise ValueError("observed-anchor-v1 requires --vector-critic (one head per objective)")
    anchor = (
        upgrade(torch.load(args.teacher, map_location="cpu", weights_only=False))
        if args.teacher
        else state
    )
    if args.learner_contract == "observed-anchor-v1" and args.teacher:
        from .provenance import digest

        if anchor.get("feature_contract") != state.get("feature_contract") or digest(
            anchor.get("normalizer")
        ) != digest(state.get("normalizer")):
            raise ValueError(
                "modern explicit teacher uses a different input-normalization contract"
            )
    teacher.load_state_dict(anchor.get("teacher_actor", anchor["actor"]))
    teacher.eval()
    if resume_state is not None:
        actor.set_region_head(resolve_head(resume_state, args.region_head, resume=True))
    teacher.set_region_head(actor.region_head)
    frozen_run = run_manifest(args, actor, resume_state)
    actor.run_simulator = frozen_run["simulator"]
    actor.worker_contract = frozen_run["worker_contract"]
    actor.run_producer = {
        "simulator": frozen_run["simulator"],
        "bridge_sha256": frozen_run["bridge_sha256"],
        "flags": {key: frozen_run[key] for key in ("sim_rules", "sim_statics")},
        "raw_materials": False,
    }
    cost_critic = CostCritic(core_width, token=token)
    if "cost_critic" in state:
        cost_critic.load_state_dict(state["cost_critic"])
    hyper, _, _ = resolve_configuration(args, state, resume_state)
    observed_train = observed_gate = None
    observed_binding = None
    enabled_value_heads = ()
    if hyper.learner_contract == "observed-anchor-v1":
        if args.observed_train_anchor is None or args.observed_actor_gate is None:
            raise ValueError(
                "observed-anchor-v1 requires --observed-train-anchor and --observed-actor-gate"
            )
        from .value_calibration import (
            GUARD_COST,
            load_dataset,
            normalization_binding,
            validate_disjoint,
        )

        observed_binding = normalization_binding(
            state["normalizer"], state["normalizer"]["normalization_contract"]
        )
        observed_train = load_dataset(
            args.observed_train_anchor, "train-anchor", expected_binding=observed_binding
        )
        observed_gate = load_dataset(
            args.observed_actor_gate, "actor-gate", expected_binding=observed_binding
        )
        validate_disjoint(observed_train, observed_gate)
        enabled_objectives = (
            contracts.OBJECTIVES
            if hyper.personality != "default"
            else tuple(
                name for name in contracts.OBJECTIVES if hyper.objective_weights.get(name, 0.0) != 0
            )
        )
        enabled_value_heads = (*enabled_objectives, GUARD_COST)
        missing_train = [
            name
            for name in enabled_value_heads
            if not bool(observed_train.target_masks[name].any())
        ]
        if missing_train:
            raise ValueError(
                f"train-anchor lacks independently identified enabled heads: {missing_train}"
            )
    if args.lr is not None:
        hyper.lr = args.lr
    if args.teacher_kl is not None:
        hyper.teacher_kl = args.teacher_kl
    if args.lagrange_lr is not None:
        hyper.lagrange_lr = args.lagrange_lr
    if args.cost_budget is not None:
        hyper.cost_budget = args.cost_budget
    if args.log_std_floor is not None:
        hyper.log_std_floor = args.log_std_floor
    if args.no_balance:
        hyper.balance_regimes = False
    if args.mix_change_cost is not None:
        hyper.mix_change_cost = args.mix_change_cost
    optimizers = (
        torch.optim.Adam(actor_parameters(actor, resume_state), lr=hyper.lr),
        torch.optim.Adam(critic.parameters(), lr=hyper.critic_lr),
        torch.optim.Adam(cost_critic.parameters(), lr=hyper.critic_lr),
    )
    a, b = (int(x) for x in args.seeds.split("-"))
    train = np.arange(a, b + 1)
    grassland = (
        np.array([int(x) for x in args.grassland_seeds.read_text().split(",") if x.strip()])
        if args.grassland_seeds
        else np.array([], dtype=int)
    )
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    stage = STAGES[args.stage]
    lagrange = float(state.get("lagrange", 0.0))
    if args.commitments:
        stage = dataclasses.replace(stage, commitments=True)
    if args.learned_movement:
        stage = dataclasses.replace(stage, movement=True, region_movement=True)
    if args.observe_movement:
        stage = dataclasses.replace(stage, observe_movement=True)
    if args.learned_fission:
        stage = dataclasses.replace(stage, fission=True)
    if args.learned_trade:
        stage = dataclasses.replace(stage, trade=True)
        hyper.objective_weights["trade_utility"] = contracts.TRADE_UTILITY_WEIGHT
    if args.personality != "default":
        if not token:
            raise ValueError("personalities need the token model (its village token reads them)")
        hyper.personality = args.personality
        personality_sampler(args.personality, rng)  # validate the spec
    hyper.personality_scales = args.personality_scales
    hyper.personality_clip = args.personality_clip
    if args.break_cost is not None:
        hyper.break_cost = args.break_cost
    search = None
    label_report: dict = {}
    if args.search_labels:
        if args.search_margin is not None:
            hyper.search_margin = args.search_margin
        hyper.search_weight = args.search_weight
        search = load_search_labels(
            [Path(p) for p in args.search_labels.split(",")],
            hyper.search_margin,
            provenance=args.provenance,
            report=label_report,
            declared_sources=declared,
            calibration_manifest=held_manifest,
        )
        print(
            json.dumps(
                {
                    "search_rows": 0 if search is None else len(search["label_kind"]),
                    "search_provenance": label_report,
                }
            ),
            flush=True,
        )
    if args.objective_weights:
        for item in args.objective_weights.split(","):
            name, weight = item.split("=")
            if name not in contracts.OBJECTIVES:
                raise ValueError(f"unknown objective {name}")
            hyper.objective_weights[name] = float(weight)
    valley = (
        [
            tuple(int(x) for x in line.split(","))
            for line in args.valley_seeds.read_text().split()
            if line.strip()
        ]
        if args.valley_seeds
        else None
    )
    if args.native_start is not None:
        # Re-seat the native prior of a loaded region head (the learned weights are kept).
        with torch.no_grad():
            others = contracts.REGION_MOVEMENT_SLOTS - 1
            actor.region_native.bias.fill_(
                float(np.log(args.native_start / (1 - args.native_start) * others))
            )
    bank = DaughterBank(args.daughter_bank) if args.daughter_bank else None
    if bank is not None and args.curriculum == "r3-balanced-v1":
        from .curriculum import bank_state, private_bank

        if args.bank_image is None:
            raise ValueError("balanced curriculum needs a frozen private bank image")
        private_bank(bank, args.bank_image, args.out, resume=resume_state is not None)
    draw = world_sampler(stage, train, grassland, rng, valley, bank, args.curriculum)
    ea, eb = (int(x) for x in args.eval_seeds.split("-"))
    # Selection: default worlds plus two of each capability fixture in every stage; best.pt
    # maximizes the geometric
    # mean of the per-regime welfare ratios, so no regime can be traded away for the majority.
    eval_worlds = [(s, 0) for s in range(ea, eb + 1)]
    eval_worlds += [
        (s, m) for s in range(ea, min(eb, ea + 1) + 1) for m in contracts.CAPABILITY_MASKS.values()
    ]
    if held_manifest:
        assert_training(
            held_manifest,
            list(train)
            + list(grassland)
            + [s for s, _ in eval_worlds]
            + ([] if valley is None else [s for s, _ in valley]),
        )
        if bank is not None:
            import hashlib

            entries = bank.training_entries()
            assert_training(
                held_manifest,
                [int(e["seed"]) for e in entries],
                [hashlib.sha256(bank.snapshot(e)).hexdigest() for e in entries],
            )
    grassland_set = frozenset(int(x) for x in grassland)
    branch = None
    branch_report: dict = {}
    if args.branch_shards:
        from . import branch_shards

        branch = branch_shards.load(
            [Path(p) for p in args.branch_shards.split(",")],
            args.provenance,
            branch_report,
            declared_sources=declared,
            calibration_manifest=held_manifest,
        )
        print(
            json.dumps(
                {
                    "branch_provenance": branch_report,
                    "branch_rows": len(branch[0]["camp"]),
                    "branch_bootstraps": 0 if branch[1] is None else len(branch[1]["core"]),
                }
            ),
            flush=True,
        )
    from .provenance import compatible_sources

    offline_compatibility = compatible_sources(
        label_report.get("records", []) + branch_report.get("records", []),
        frozen_run["simulator"],
        frozen_run["simulator_compatibility"],
        {k: frozen_run[k] for k in ("sim_rules", "sim_statics")},
    )
    hyper.critic_warmup = (
        args.critic_warmup if args.critic_warmup is not None else hyper.critic_warmup
    )
    hyper.round_start_warmup = (
        args.round_start_warmup if args.round_start_warmup is not None else hyper.round_start_warmup
    )
    hyper.branch_share = args.branch_share if args.branch_share is not None else hyper.branch_share
    hyper.kl_refit = (
        args.kl_refit
        if args.kl_refit is not None
        else bool((resume_state or {}).get("kl_refit", resume_state is None))
    )
    hyper.cost_bootstrap = resume_setting("cost_bootstrap", resume_state, args.cost_bootstrap, True)
    frozen_run["resolved_hyper"] = dataclasses.asdict(hyper)
    frozen_run["resolved_stage"] = dataclasses.asdict(stage)
    effective_warmup = (
        0
        if hyper.learner_contract != "legacy"
        else warmup_iterations(
            fresh_heads,
            hyper.critic_warmup,
            hyper.round_start_warmup,
            promoted_init=state.get("artifact_kind") == "promotion",
        )
    )
    frozen_run["effective_iterations"] = {
        "critic_only": effective_warmup,
        "actor": args.iterations,
        "total": effective_warmup + args.iterations,
    }
    resolved_hyper, resolved_stage, effective = resolve_configuration(args, state, resume_state)
    if (
        dataclasses.asdict(resolved_hyper) != frozen_run["resolved_hyper"]
        or dataclasses.asdict(resolved_stage) != frozen_run["resolved_stage"]
        or effective != frozen_run["effective_iterations"]
    ):
        raise ValueError("actual PPO setup drifted from the shared frozen configuration resolver")
    if (
        resume_state is not None
        and "resolved_hyper" in resume_state.get("run_manifest", {})
        and resume_state["run_manifest"] != frozen_run
    ):
        raise ValueError("resolved hyperparameters/stage differ from frozen resume")
    from .lineage import build as build_lineage
    from .lineage import families, merge, source_families
    from .provenance import checkpoint_hash

    entries = bank.training_entries() if bank is not None else []
    import hashlib

    current_training = merge(
        families(
            list(train)
            + list(grassland)
            + ([] if valley is None else [s for s, _ in valley])
            + [int(e["seed"]) for e in entries],
            [hashlib.sha256(bank.snapshot(e)).hexdigest() for e in entries],
        ),
        source_families(label_report),
        source_families(branch_report),
    )
    current_selection = families([s for s, _ in eval_worlds])
    if resume_state is not None:
        lineage = resume_state.get("family_lineage")
        if held_manifest and lineage is None:
            raise ValueError("resume lacks frozen initialization/training family lineage")
    else:
        parents = [
            (
                checkpoint_hash(args.init),
                state,
                {"path": str(args.init.resolve()), "sha256": checkpoint_hash(args.init)},
            )
        ]
        if args.teacher:
            parents.append(
                (
                    checkpoint_hash(args.teacher),
                    anchor,
                    {"path": str(args.teacher.resolve()), "sha256": checkpoint_hash(args.teacher)},
                )
            )
        lineage = build_lineage(held_manifest, current_training, current_selection, parents)
    actor.calibration_manifest = held_manifest
    if held_manifest and args.curriculum == "r3-balanced-v1":
        from .lineage import checkpoint_evidence

        if (
            state.get("artifact_kind") != "promotion"
            or not checkpoint_evidence(state, held_manifest)["verified"]
        ):
            raise ValueError(
                "Round-3 balanced training requires the verified promoted guarded clone"
            )
    args.out.mkdir(parents=True, exist_ok=True)
    trajectory = args.out / "trajectory.jsonl"
    log: list = []
    baseline = None
    best = -np.inf

    def atomic(obj: Any, path: Path) -> None:
        # Write then rename, so a preempted VM never leaves a half-written checkpoint.
        tmp = path.with_suffix(path.suffix + ".tmp")
        from .promotion import seal

        torch.save(seal(obj), tmp)
        os.replace(tmp, path)

    def save(path: Path) -> None:
        checkpoint = {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "cost_critic": cost_critic.state_dict(),
            "teacher_actor": teacher.state_dict(),
            "normalizer": state["normalizer"],
            "core_width": core_width,
            "privileged_width": privileged,
            "contract": contracts.NATIVE_CONTRACT,
            "stage": stage.name,
            "commitments": stage.commitments,
            "region_movement": stage.region_movement,
            "observe_movement": stage.observe_movement,
            "learned_fission": stage.fission,
            "learned_trade": stage.trade,
            "experts": experts,
            "token": token,
            "vector_critic": isinstance(critic, VectorCritic),
            "region_head": actor.region_head,
            "nonqualifying": args.nonqualifying,
            "iteration": it,
            "calibration_manifest": held_manifest,
            "run_manifest": frozen_run,
            "family_lineage": lineage,
            "offline_compatibility": offline_compatibility,
            "cost_bootstrap": hyper.cost_bootstrap,
            "objective_weights": hyper.objective_weights,
            "hyper": dataclasses.asdict(hyper),
            "personality_scales": hyper.personality_scales,
            "personality_clip": hyper.personality_clip,
            "daughter_bank": None if bank is None else str(bank.directory),
            "lagrange": lagrange,
            "log": log,
        }
        if hyper.learner_contract != "legacy":
            checkpoint.update(
                versioned_learner_checkpoint_metadata(
                    hyper.learner_contract,
                    state.get("feature_contract"),
                    observed_train=observed_train,
                    observed_gate=observed_gate,
                )
            )
        atomic(checkpoint, path)

    def evaluate(checkpoint: Path, personality: list[float] | None = None) -> dict:
        groups: dict[str, list] = {}
        for seed, mask in eval_worlds:
            groups.setdefault(regime(seed, mask, frozenset()), []).append((seed, mask))
        result = {
            name: evaluate_closed_loop(
                checkpoint,
                worlds,
                args.eval_turns,
                workers=args.eval_workers,
                movement=stage.bridge_movement,
                commitments=stage.commitments,
                region=stage.region_movement or stage.observe_movement,
                fission=stage.fission,
                trade=stage.trade,
                personality=personality,
                expected_producer=actor.worker_contract,
            )
            for name, worlds in groups.items()
        }
        total = {
            key: float(sum(r[key] for r in result.values()))
            for key in ("welfare", "population", "extinctions")
        }
        total["below_30"] = float(np.mean([r["below_30"] for r in result.values()]))
        total["qol"] = float(
            sum(r["qol"] * r["population"] for r in result.values()) / max(1.0, total["population"])
        )
        total["specialist_share"] = float(
            sum(r["specialist_share"] * r["population"] for r in result.values())
            / max(1.0, total["population"])
        )
        total["regimes"] = result
        total["worker_contract"] = actor.worker_contract
        total["producer"] = actor.run_producer
        return total

    # Resume (--resume): everything an iteration depends on, written atomically after each
    # iteration's trajectory line.
    resume_path = args.out / "resume.pt"
    start = 0
    if args.resume and resume_path.exists():
        r = torch.load(resume_path, map_location="cpu", weights_only=False)
        restore_actor(actor, r, args.region_head, resume=True)
        teacher.set_region_head(actor.region_head)
        critic.load_state_dict(r["critic"])
        cost_critic.load_state_dict(r["cost_critic"])
        for optimizer, saved in zip(optimizers, r["optimizers"], strict=False):
            optimizer.load_state_dict(saved)
        lagrange, best, baseline, log, start = (
            r["lagrange"],
            r["best"],
            r["baseline"],
            r["log"],
            r["iteration"] + 1,
        )
        rng.bit_generator.state = r["numpy_rng"]
        generator.set_state(r["generator"])
        torch.set_rng_state(r["torch_rng"])
        if args.curriculum == "r3-balanced-v1":
            draw.load_state_dict(r["curriculum_state"])
            if bank is not None:
                from .curriculum import restore_bank

                restore_bank(bank, r["bank_state"])
        # Keep the trajectory consistent with the resumed state: the baseline and iterations up to
        # the saved one.
        # A preempted VM can leave a partial or zero-filled last line: unreadable lines are dropped,
        # never fatal.
        lines = trajectory.read_text(errors="replace").splitlines() if trajectory.exists() else []
        parsed = []
        for line in lines:
            try:
                parsed.append((line, json.loads(line)))
            except ValueError:
                continue
        kept = [line for line, d in parsed if "baseline" in d or d.get("iteration", -1) < start]
        trajectory.write_text("".join(line + "\n" for line in kept))
        print(json.dumps({"resumed_at": start}), flush=True)
    elif args.eval_every > 0:
        baseline = evaluate(args.init)
        with trajectory.open("w") as f:
            f.write(json.dumps({"baseline": baseline, "eval_worlds": eval_worlds}) + "\n")
        print(json.dumps({"baseline": baseline}), flush=True)
    if args.critic_warmup is not None:
        hyper.critic_warmup = args.critic_warmup
    if args.round_start_warmup is not None:
        hyper.round_start_warmup = args.round_start_warmup
    if args.branch_share is not None:
        hyper.branch_share = args.branch_share
    resumed = args.resume and resume_path.exists()
    hyper.kl_refit = (
        args.kl_refit
        if args.kl_refit is not None
        else bool(r.get("kl_refit", False))
        if resumed
        else True
    )
    hyper.cost_bootstrap = resume_setting("cost_bootstrap", resume_state, args.cost_bootstrap, True)
    if args.film_bound:
        for net in (actor, teacher, critic, cost_critic):
            if hasattr(net.encoder, "film_extent"):
                net.encoder.film_bound = args.film_bound
    # Extra iterations, so the actor still gets --iterations updates.
    warmup = effective_warmup
    economic_trace = None
    if args.economic_trace is not None:
        from .economic_trace import EconomicTraceWriter
        from .provenance import checkpoint_hash, code_commit

        economic_trace = EconomicTraceWriter(
            args.economic_trace,
            {
                "runId": str(args.out.resolve()),
                "arm": args.arm or "diagnostic",
                "optimizerSeed": args.seed,
                "nonqualifying": bool(args.nonqualifying),
                "executionAuthority": qualified,
                "localExecutionDesignSha256": checkpoint_hash(args.local_execution_design),
                "executionManifestSha256": checkpoint_hash(args.execution_manifest),
                "learnerCommit": code_commit(),
                "initialCheckpointSha256": checkpoint_hash(args.init),
                "runManifest": frozen_run,
            },
        )
    paused_for_calibration = False
    for it in range(start, args.iterations + warmup):
        clock = time.time()
        worlds = draw(args.worlds)
        lagrange, stats = iteration(
            actor,
            critic,
            cost_critic,
            teacher,
            normalizer,
            worlds,
            args.turns,
            stage,
            hyper,
            optimizers,
            lagrange,
            generator,
            rng,
            grassland_set,
            search,
            train_actor=it >= warmup,
            branch=branch,
            bank=bank,
            economic_trace=economic_trace,
            observed_train=observed_train,
            observed_gate=observed_gate,
            enabled_value_heads=enabled_value_heads,
        )
        stats["critic_warmup"] = it < warmup
        stats.update({"iteration": it, "seconds": time.time() - clock, "worlds": worlds})
        if args.curriculum == "r3-balanced-v1":
            stats["curriculum_sources"] = draw.last_sources
        save(args.out / "latest.pt")
        calibration_pause = False
        if hyper.learner_contract == "observed-anchor-v1":
            pause = calibration_pause_after_update(
                actor,
                behavior_policy_configuration(stage, hyper, actor),
                stats["value_calibration"],
                observed_gate.metadata["source"]["sourcePolicyDigest"],
                args.out / "latest.pt",
                args.out / "calibration-refresh-request.json",
                observed_binding,
                enabled_value_heads,
            )
            if pause is not None:
                stats["calibration_pause"] = pause
                calibration_pause = True
        if not calibration_pause and args.eval_every > 0 and (it + 1) % args.eval_every == 0:
            ev = evaluate(args.out / "latest.pt")
            ev["welfare_ratio_vs_init"] = ev["welfare"] / max(1e-9, baseline["welfare"])
            ratios = {
                name: r["welfare"] / max(1e-9, baseline["regimes"][name]["welfare"])
                for name, r in ev["regimes"].items()
            }
            ev["regime_ratios"] = ratios
            ev["selection_score"] = float(
                np.exp(np.mean(np.log(np.maximum(1e-9, list(ratios.values())))))
            )
            eligible = ev["extinctions"] <= baseline["extinctions"]
            if hyper.personality != "default":
                # Personality runs select on how far personalities separate, not on default welfare
                # (the default
                # point is ev itself, the Elder at [1, 0, 0, 0]).
                points = {
                    "welfare": ev,
                    **{
                        name: evaluate(args.out / "latest.pt", p)
                        for name, p in SELECTION_POINTS.items()
                        if name != "welfare"
                    },
                }
                spread = spread_selection(points, baseline)
                spread["points"] = {
                    name: {
                        k: p[k]
                        for k in (
                            "welfare",
                            "population",
                            "qol",
                            "below_30",
                            "extinctions",
                            "worker_contract",
                            "producer",
                        )
                    }
                    for name, p in points.items()
                }
                ev["spread"] = spread
                ev["selection_score"] = spread["score"]
                eligible = spread["eligible"]
            eligible = bool(eligible) and not args.nonqualifying
            ev["eligible"] = bool(eligible)
            stats["eval"] = ev
            if ev["selection_score"] > best and eligible:
                best = ev["selection_score"]
                save(args.out / "best.pt")
        log.append(stats)
        with trajectory.open("a") as f:
            f.write(json.dumps(stats) + "\n")
        resume_checkpoint = {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "cost_critic": cost_critic.state_dict(),
            "optimizers": [o.state_dict() for o in optimizers],
            "lagrange": lagrange,
            "best": best,
            "baseline": baseline,
            "log": log,
            "iteration": it,
            "numpy_rng": rng.bit_generator.state,
            "generator": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "kl_refit": hyper.kl_refit,
            "cost_bootstrap": hyper.cost_bootstrap,
            "region_head": actor.region_head,
            "legacy_actor_optimizer": len(optimizers[0].param_groups[0]["params"])
            != len(list(actor.parameters())),
            "contract": contracts.NATIVE_CONTRACT,
            "calibration_manifest": held_manifest,
            "family_lineage": lineage,
            "offline_compatibility": offline_compatibility,
            "curriculum": args.curriculum,
            "curriculum_state": draw.state_dict() if args.curriculum == "r3-balanced-v1" else None,
            "bank_state": bank_state(bank)
            if bank is not None and args.curriculum == "r3-balanced-v1"
            else None,
            "run_manifest": frozen_run,
        }
        if hyper.learner_contract != "legacy":
            resume_checkpoint.update(
                versioned_learner_checkpoint_metadata(
                    hyper.learner_contract,
                    state.get("feature_contract"),
                    normalizer=state["normalizer"],
                )
            )
        atomic(resume_checkpoint, resume_path)
        print(json.dumps({k: v for k, v in stats.items() if k != "worlds"}), flush=True)
        if calibration_pause:
            paused_for_calibration = True
            break
    if economic_trace is not None:
        economic_trace.close()
    if paused_for_calibration:
        (args.out / "calibration.paused").write_text(
            json.dumps(
                {
                    "learnerContract": hyper.learner_contract,
                    "status": "actor-gate-closed-no-completion-claim",
                    "nonqualifying": True,
                }
            )
        )
    elif args.arm and not args.nonqualifying:
        from .completion import build

        completed = build(args.out, frozen_run)
        target = args.out / "completion.json"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(completed, sort_keys=True))
        os.replace(tmp, target)
        (args.out / "done").write_text(json.dumps({"completion_sha256": checkpoint_hash(target)}))
    else:
        (args.out / ("diagnostic.done" if args.nonqualifying else "done")).write_text(
            json.dumps({"iterations": args.iterations, "nonqualifying": args.nonqualifying})
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
