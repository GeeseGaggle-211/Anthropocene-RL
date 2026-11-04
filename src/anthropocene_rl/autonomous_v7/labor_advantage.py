"""Intervention B: the action-conditioned (dueling) advantage critic for labor search.

G0 manifest amendment r5-b makes the action-conditioned critic core:
``Q_j(C, a) = V_j(C) + A_j(C, a)`` for the two task objectives (QoL-years,
population-years) and the two costs (feasibility, settlement guard). ``V_j`` is
the existing critic head and is never trained here: the advantage head reads the
frozen critic's context (no gradient reaches the critic encoders or V heads).

What lives here:

* the fixed-layout semantic **action vector** (:data:`ACTION_FEATURE_WIDTH`
  floats) built from an executed schema-12 action, rows bound to the catalog by
  stable ID, and the same vector built from the actor's *output distribution*
  as a differentiable reparameterized sample (the Q-pi path). That path is
  behind :class:`QPiGate`, closed by default: with it closed the actor outputs
  are detached, so no gradient reaches an actor parameter;
* :class:`LaborActionEncoder` (MLP to a 128-d embedding) and
  :class:`LaborAdvantageHead` (one 2-layer MLP per objective over
  ``[village context + privileged readout + action embedding]``), centred over
  the candidate set at each root;
* the **candidate-payload schema** (:data:`CANDIDATE_PAYLOAD_SCHEMA`), the
  consumer contract the search exporter writes to, with a strict loader;
* same-root pair construction, the pairwise loss and the fit, which refuses
  reserved (90154-90159), calibration, development and sealed worlds;
* :class:`AdvantageRanker`, the search hook (score with A, natively evaluate the
  top four plus H12, keep full evaluation on a declared fraction of roots);
* bounded-QoL-head helpers: TD(lambda) targets through the head and the
  saturation monitor.

Encoding note: with a root population (every search export declares one) the reader
encodes specialties after native apportionment and unstaffed roles' activity as
neutral, so legacy pair-fit and ``labor_advantage_eval`` numbers from before that
change are not directly comparable (``labor_a_stage1`` module docstring).

Nothing here writes an actor loss, a PPO advantage, a critic outcome label or a
G2C V criterion. See ``docs/LABOR_ADVANTAGE.md``.

Candidate-payload schema (``labor-advantage-candidates-v1``)
-------------------------------------------------------------

One shard is a ``torch.save``-d dict loadable with ``weights_only=True``::

    {
      "schema": "labor-advantage-candidates-v1",
      "producer": {...},                  # free-form provenance (search config, commits)
      "component_order": ["qol_years", "population_years",
                          "feasibility", "settlement_guard"],
      "returns_units": "raw-discounted-component-return",
      "discount_years": 400.0,
      "specialty_ids": [14 str],          # order of laborSpecialtyAllocation entries
      "activity_ids": [22 str],           # order of each laborActivityAllocation row
      "trade_groups": [7 str],            # order of tradeValue / tradeGate
      "tensors": {name: Tensor[R, ...]},  # root observation rows exactly as
                                          # labor_policy_tensors builds them (incl.
                                          # labor_personality); NO privileged key
      "privileged": {name: Tensor[R, ...]} | None,  # raw critic-only arrays
                                          # (labor_privileged.shard_keys) of the same rows
      "roots": [
        {
          "root_id": str,                 # unique within the shard
          "world": int, "turn": int, "camp": int, "society": int,
          "source": "h12" | "actor",      # whose state the root is
          "tensor_row": int,              # row in tensors/privileged
          "personality": [q, p],          # root personality (also its FiLM input)
          "full_evaluation": bool,        # every candidate natively evaluated on the
                                          # declared horizons/streams (regret subset)
          "pair_sd": {"<H>": [4 float]} | None,  # optional SD of a single-rollout
                                          # paired difference at horizon H, raw units
          "candidates": [
            {
              "index": int,               # 0..n-1, position in this list
              "family": str,              # labor_search FAMILIES name
              "is_h12": bool,             # exactly one per root
              "digest": str,              # labor_search.payload_digest(action)
              "action": {schema-12 payload: movement, fission, tradeValue[7],
                         tradeGate[7], tradeShare, laborSpecialtyAllocation[14],
                         laborActivityAllocation[13][22]},
              "control": None | "all-idle" | "delayed-benefit",
              "bookkeeping_twin": None | int,  # candidate identical in physics,
                                          # differing only in bookkeeping fields
              "evaluations": [
                {
                  "horizon": int,         # decisions the branch was scheduled for
                  "reached_horizon": int, # exact decisions resolved (<= horizon)
                  "terminated": bool,     # the focal lineage ended before horizon
                  "stream": int,          # 0: deterministic selection stream;
                                          # >= 1: independent re-evaluation streams
                  "returns": [4 float],   # discounted raw returns, component_order
                  "step_years": [float],  # physical years of each resolved step
                  "se": [4 float] | None, # optional per-evaluation SE (raw units)
                }
              ]
            }
          ]
        }
      ]
    }

Optional T2 plan rows (``labor_search --plan-candidates``) follow every
first-action candidate and carry ``plan_id``, ``plan_role`` (``plan`` or
``first-step-control``), ``plan_steps`` (the executed focal action per step) and
``step_returns`` ({stream: [[4] per step]}); their ``action`` is the shared first
step. They are parsed into ``Root.plans`` and never paired as first-action
candidates (a plan's return is not a forced-first-action-then-H12 return).
Shards without plan rows parse unchanged.

An evaluation with ``reached_horizon < horizon`` and ``terminated`` false is a
truncated branch and is ignored. Pair SE precedence: paired re-evaluation
streams (both candidates, >= 2 shared streams >= 1), else per-evaluation ``se``
of the selection stream, else the root ``pair_sd``; a pair with none is not
comparable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import contracts
from .labor_bridge import ACTIVITIES, ROLE_ROWS, SPECIALTIES, LaborPolicySample
from .labor_model import (
    LABOR_COST_REFERENCE_SCALES,
    LABOR_COSTS,
    LABOR_OBJECTIVE_REFERENCE_SCALES,
    LABOR_OBJECTIVES,
    LaborCostCritics,
    LaborTaskCritic,
    _masked_softmax,
    _masks,
    critic_privileged_batch,
)
from .labor_privileged import (
    G0_CALIBRATION_WORLDS,
    RESERVED_HELDOUT_WORLDS,
    assert_actor_batch,
    is_privileged_key,
)

ADVANTAGE_ARTIFACT_KIND = "model9-labor-advantage-head"
ADVANTAGE_ARCHITECTURE = "labor-dueling-advantage-mlp-v1"
CANDIDATE_PAYLOAD_SCHEMA = "labor-advantage-candidates-v1"
# v2: specialty block after native apportionment, unstaffed roles idle (A spec, 4 Nov 2026).
# A head bound to v1 (raw requested shares) is refused on load.
ACTION_LAYOUT_VERSION = "labor-action-semantic-vector-v2"
SPECIALTY_ENCODING = "post-native-largest-remainder-apportionment"
UNSTAFFED_ACTIVITY = "idle"
RETURNS_UNITS = "raw-discounted-component-return"
COMPONENTS = (*LABOR_OBJECTIVES, *LABOR_COSTS)
REFERENCE_SCALES = {**LABOR_OBJECTIVE_REFERENCE_SCALES, **LABOR_COST_REFERENCE_SCALES}
CONTROLS = ("all-idle", "delayed-benefit")
# T2 plan rows (labor_search): parsed into Root.plans, never first-action candidates.
PLAN_ROLES = ("plan", "first-step-control")

# Keyed-stream tags.
SPLIT_TAG = 0x4153  # "AS": world split
SHUFFLE_TAG = 0x4146  # "AF": fit shuffles
RANKER_TAG = 0x4152  # "AR": full-evaluation draw

# Movement union (labor actor ``movement_codes`` order): stay, seven site rows,
# keep the destination intent, then the eight region rows.
MOVEMENT_UNION_CODES: tuple[int, ...] = (
    -1,
    *range(contracts.MOVEMENT_SLOTS - 1),
    contracts.KEEP_INTENT,
    *(contracts.REGION_BASE + row for row in range(contracts.REGION_ROWS)),
)
MOVEMENT_UNION = len(MOVEMENT_UNION_CODES)
TRADE_GROUP_COUNT = len(contracts.TRADE_GROUPS)
_LOG_TRADE_LOW = math.log(contracts.TRADE_VALUE_RANGE[0])
_LOG_TRADE_HIGH = math.log(contracts.TRADE_VALUE_RANGE[1])
_LOG_TRADE_SCALE = max(abs(_LOG_TRADE_LOW), abs(_LOG_TRADE_HIGH))

# Canonical neutral activity of an unstaffed role: all worker time idle.
NEUTRAL_ACTIVITY = ACTIVITIES.index("idle")

# Fixed action layout: (block, width) in vector order.
ACTION_BLOCKS: tuple[tuple[str, int], ...] = (
    ("specialty_share", len(SPECIALTIES)),
    ("activity_share", ROLE_ROWS * len(ACTIVITIES)),
    ("movement_onehot", MOVEMENT_UNION),
    ("fission_onehot", contracts.FISSION_POLICIES),
    ("trade_gate_onehot", TRADE_GROUP_COUNT * contracts.TRADE_GATES),
    ("trade_value_log", TRADE_GROUP_COUNT),
    ("trade_share", 1),
)
ACTION_FEATURE_WIDTH = sum(width for _, width in ACTION_BLOCKS)
ACTION_FIELDS = (
    "movement",
    "fission",
    "tradeValue",
    "tradeGate",
    "tradeShare",
    "laborSpecialtyAllocation",
    "laborActivityAllocation",
)

# Preregistered numbers (manifest section 8 / amendment r5-b).
RANKING_GATE = 0.70
FALSE_IMPROVEMENT_GATE = 0.15
FALSE_IMPROVEMENT_MARGIN = 0.01
PAIR_SE_MULTIPLIER = 2.0
REGRESSION_WEIGHT = 0.1
TOP_K = 4
FULL_EVALUATION_FRACTION = 0.20
EMBEDDING_WIDTH = 128
# Target sources of the fit: "pairs" (historical same-root pairs, pairwise_loss) and
# the stage-1 search-pretraining lambda-return targets (labor_a_stage1, anchored Huber).
PAIR_TARGET_SOURCE = "pairs"
LAMBDA_TARGET_SOURCE = "lambda-successor-v1"
TARGET_SOURCES = (PAIR_TARGET_SOURCE, LAMBDA_TARGET_SOURCE)
# Stage-1 loss/mask defaults (A(s,a) spec, approved 4 Nov 2026; see labor_a_stage1).
HUBER_SE_MULTIPLIER = 1.345  # Huber delta = 1.345 x the row's target noise SE
STAGE1_MASKED_BLOCKS = ("trade_value_log", "trade_share")  # never perturbed by search (D5)
BLOCK_SUPPORT_FLOOR = 30  # fit rows whose block differs from H12's; fewer -> block held at 0
VARIANCE_FLOOR_FRACTION = 0.25  # per-candidate variance >= 0.25 x pooled single-stream variance
SE_ABS_FLOOR = 1e-3  # reference-scale units
DEFAULT_OBJECTIVE_HORIZONS: dict[str, tuple[int, ...] | None] = {
    # QoL is predictable over 1-3 decisions and noise over 25 (qol-diag report).
    "qol_years": (1, 2, 3),
    "population_years": None,
    "feasibility": None,
    "settlement_guard": None,
}


def action_layout_state() -> dict[str, Any]:
    return {
        "version": ACTION_LAYOUT_VERSION,
        "width": ACTION_FEATURE_WIDTH,
        "blocks": [[name, width] for name, width in ACTION_BLOCKS],
        "specialtyIds": list(SPECIALTIES),
        "activityIds": list(ACTIVITIES),
        "tradeGroups": list(contracts.TRADE_GROUPS),
        "movementUnionCodes": list(MOVEMENT_UNION_CODES),
        "tradeValueTransform": "log(clamp(v, 0.25, 4)) / log(4)",
        "specialtyEncoding": SPECIALTY_ENCODING,
        "unstaffedActivity": UNSTAFFED_ACTIVITY,
    }


# ---------------------------------------------------------------------------
# Action vector
# ---------------------------------------------------------------------------


def _binding(declared: Sequence[str] | None, canonical: Sequence[str], label: str) -> list[int]:
    """Positions in ``declared`` order of each canonical ID (stable-ID binding)."""
    if declared is None:
        return list(range(len(canonical)))
    declared = [str(item) for item in declared]
    if sorted(declared) != sorted(canonical) or len(set(declared)) != len(declared):
        raise ValueError(f"{label} ids must be a permutation of the catalog registry")
    return [declared.index(item) for item in canonical]


def movement_union_index(code: int) -> int:
    """Union slot of a schema-12 movement code (native -2 is not an actor action)."""
    if isinstance(code, bool) or int(code) != code:
        raise ValueError("movement code must be an integer")
    try:
        return MOVEMENT_UNION_CODES.index(int(code))
    except ValueError:
        raise ValueError(f"movement code {code} is not in the actor movement union") from None


def apportion_specialty_counts(shares: Sequence[float], population: int) -> np.ndarray:
    """Native residents per specialty for canonical-order ``shares`` and ``population``.

    The shares go through the native ingestion canonicalization
    (``labor_bridge._simplex`` with every coordinate legal: a masked coordinate
    carries zero mass, so the result equals the catalog-masked parse) and then
    the native stable largest-remainder apportionment
    (``labor_bridge.apportion_largest_remainder``) -- exactly
    ``labor_search.labor_state``. Shares that miss the simplex by more than the
    native tolerance (float32 softmax output of the reparameterized path) are
    renormalized first; an exact simplex is never renormalized, so a serialized
    action and its parsed copy apportion identically. A population of zero (or
    an all-zero share vector) apportions to zero residents.
    """
    from .labor_bridge import SIMPLEX_TOLERANCE, _simplex, apportion_largest_remainder

    values = [max(0.0, float(x)) for x in shares]
    if len(values) != len(SPECIALTIES):
        raise ValueError("specialty shares differ from the catalog registry")
    population = int(population)
    total = math.fsum(values)
    if population <= 0 or not total > 0.0:
        return np.zeros(len(values), dtype=np.int64)
    if abs(total - 1.0) > SIMPLEX_TOLERANCE:
        values = [min(1.0, v / total) for v in values]
    canonical = _simplex(values, (True,) * len(values), "specialty shares")
    return np.asarray(apportion_largest_remainder(canonical, population), dtype=np.int64)


def neutral_unstaffed_activity(activity: np.ndarray, counts: Sequence[int]) -> np.ndarray:
    """Canonical-order activity rows with every role the apportionment leaves unstaffed
    (zero residents) set to the canonical neutral value, all worker time ``idle``
    (``labor_search.transfer`` / ``compose_payload`` convention). An unstaffed role's
    activity simplex has no physical effect, so two actions with the same residents and
    the same staffed-role activities (H12 and its control twin) encode identically."""
    out = np.array(activity, dtype=np.float64, copy=True)
    counts = np.asarray(counts).reshape(-1)
    if counts.shape != (len(SPECIALTIES),) or out.shape != (ROLE_ROWS, len(ACTIVITIES)):
        raise ValueError("unstaffed-role canonicalization needs catalog-shaped inputs")
    for role in range(ROLE_ROWS):
        if int(counts[role + 1]) == 0:
            out[role] = 0.0
            out[role, NEUTRAL_ACTIVITY] = 1.0
    return out


def apportioned_specialty(shares: Sequence[float], population: int) -> np.ndarray:
    """Post-apportionment specialty shares: native residents / population (float64)."""
    population = int(population)
    counts = apportion_specialty_counts(shares, population)
    if population <= 0:
        return np.zeros(len(SPECIALTIES), dtype=np.float64)
    return counts.astype(np.float64) / float(population)


def action_features(
    action: Mapping[str, Any],
    *,
    specialty_ids: Sequence[str] | None = None,
    activity_ids: Sequence[str] | None = None,
    trade_groups: Sequence[str] | None = None,
    population: int | None = None,
) -> np.ndarray:
    """The fixed-layout semantic vector of one executed schema-12 action.

    Rows are bound to the catalog by stable ID: ``specialty_ids`` and
    ``activity_ids`` name the order the payload's entries use (default the
    catalog order), and each value lands in its canonical slot. Only the seven
    semantic fields are read; every other key (schema version, descriptions,
    digests) is bookkeeping and cannot change the vector.

    With ``population`` (the root camp's residents) the specialty block is
    encoded **after** native largest-remainder apportionment
    (:func:`apportioned_specialty`, in canonical order): the share of residents
    the native actually staffs. Two payloads that apportion to the same
    residents (an actor sample and the search's composed action, or H12 and its
    bookkeeping twin) then encode identically. The activity rows of roles that
    apportion to zero residents are set to the neutral value (all idle,
    :func:`neutral_unstaffed_activity`). Without it the raw requested
    shares are encoded (the historical layout; used only for payloads that do
    not declare a population).
    """
    if not isinstance(action, Mapping) or any(name not in action for name in ACTION_FIELDS):
        raise ValueError(f"action must carry the schema-12 fields {list(ACTION_FIELDS)}")
    spec_pos = _binding(specialty_ids, SPECIALTIES, "specialty")
    act_pos = _binding(activity_ids, ACTIVITIES, "activity")
    group_pos = _binding(trade_groups, contracts.TRADE_GROUPS, "trade group")
    specialty = np.asarray(action["laborSpecialtyAllocation"], dtype=np.float64)
    activity = np.asarray(action["laborActivityAllocation"], dtype=np.float64)
    if specialty.shape != (len(SPECIALTIES),) or activity.shape != (ROLE_ROWS, len(ACTIVITIES)):
        raise ValueError("labor allocation shapes differ from the catalog registry")
    if not (np.isfinite(specialty).all() and np.isfinite(activity).all()):
        raise ValueError("labor allocations must be finite")
    if (specialty < -1e-9).any() or (activity < -1e-9).any():
        raise ValueError("labor allocations must be non-negative")
    # Role rows follow the declared specialty order minus idle.
    declared_roles = [s for s in (specialty_ids or SPECIALTIES) if s != "idle"]
    role_pos = [declared_roles.index(name) for name in SPECIALTIES[1:]]
    if (specialty_ids or SPECIALTIES)[0] != "idle":
        raise ValueError("the declared specialty order must start with idle")
    canonical_specialty = specialty[spec_pos]
    canonical_activity = activity[role_pos][:, act_pos]
    if population is not None:
        counts = apportion_specialty_counts(canonical_specialty, population)
        canonical_specialty = apportioned_specialty(canonical_specialty, population)
        canonical_activity = neutral_unstaffed_activity(canonical_activity, counts)
    movement = np.zeros(MOVEMENT_UNION)
    movement[movement_union_index(action["movement"])] = 1.0
    fission_code = action["fission"]
    if isinstance(fission_code, bool) or int(fission_code) != fission_code:
        raise ValueError("fission must be an integer policy")
    if not 0 <= int(fission_code) < contracts.FISSION_POLICIES:
        raise ValueError("fission policy out of range")
    fission = np.zeros(contracts.FISSION_POLICIES)
    fission[int(fission_code)] = 1.0
    gates = [int(g) for g in action["tradeGate"]]
    values = np.asarray(action["tradeValue"], dtype=np.float64)
    if len(gates) != TRADE_GROUP_COUNT or values.shape != (TRADE_GROUP_COUNT,):
        raise ValueError("trade posture must have one entry per native trade group")
    if any(not 0 <= g < contracts.TRADE_GATES for g in gates):
        raise ValueError("trade gate out of range")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("trade values must be finite and positive")
    gate = np.zeros((TRADE_GROUP_COUNT, contracts.TRADE_GATES))
    for row, position in enumerate(group_pos):
        gate[row, gates[position]] = 1.0
    log_value = np.clip(np.log(values[group_pos]), _LOG_TRADE_LOW, _LOG_TRADE_HIGH)
    share = float(action["tradeShare"])
    if not math.isfinite(share) or not -1e-9 <= share <= 1.0 + 1e-9:
        raise ValueError("trade share must be in [0, 1]")
    vector = np.concatenate(
        [
            canonical_specialty,
            canonical_activity.reshape(-1),
            movement,
            fission,
            gate.reshape(-1),
            log_value / _LOG_TRADE_SCALE,
            [min(1.0, max(0.0, share))],
        ]
    )
    assert vector.shape == (ACTION_FEATURE_WIDTH,)
    return vector.astype(np.float32)


def encode_actions(
    actions: Sequence[Mapping[str, Any]],
    population: int | None,
    *,
    specialty_ids: Sequence[str] | None = None,
    activity_ids: Sequence[str] | None = None,
    trade_groups: Sequence[str] | None = None,
) -> np.ndarray:
    """Actor-side encoding of concrete schema-12 actions at one root, ``[n, width]``.

    The same function the search payload reader uses (:func:`action_features`
    with the root ``population``), so an action encodes identically whether it
    came from the search export or from the actor.
    """
    return np.stack(
        [
            action_features(
                a,
                specialty_ids=specialty_ids,
                activity_ids=activity_ids,
                trade_groups=trade_groups,
                population=population,
            )
            for a in actions
        ]
    )


def block_slices() -> dict[str, slice]:
    """Column range of every action block in the fixed layout."""
    out, cursor = {}, 0
    for name, width in ACTION_BLOCKS:
        out[name] = slice(cursor, cursor + width)
        cursor += width
    return out


@dataclass(frozen=True)
class QPiGate:
    """Admission record for actor updates through Q (amendment r5-b item 2).

    The Q-pi plumbing exists but stays closed: :func:`reparameterized_action_features`
    detaches the actor's outputs unless it receives an *open* gate, which
    requires an evaluator report on exactly the reserved worlds 90154-90159
    passing every preregistered gate. Nothing in this repository constructs one
    from a live run; an open gate also requires the user's explicit decision.
    """

    report_digest: str
    worlds: tuple[int, ...]
    ranking: Mapping[str, float]
    false_improvement: float
    idle_control: bool
    delayed_benefit: bool
    authorized_by: str = ""

    @property
    def open(self) -> bool:
        return (
            bool(self.authorized_by)
            and set(self.worlds) == set(RESERVED_HELDOUT_WORLDS)
            and bool(self.ranking)
            and all(value >= RANKING_GATE for value in self.ranking.values())
            and self.false_improvement <= FALSE_IMPROVEMENT_GATE
            and self.idle_control
            and self.delayed_benefit
        )

    @classmethod
    def from_report(cls, report: Mapping[str, Any], authorized_by: str = "") -> QPiGate:
        gates = report["gates"]
        return cls(
            report_digest=str(report["report_digest"]),
            worlds=tuple(int(w) for w in report["worlds"]),
            ranking={
                key: float(gates[key]["value"])
                for key in ("ranking:qol_years", "ranking:utility:root")
                if gates.get(key, {}).get("value") is not None
            },
            false_improvement=float(gates["false_improvement"]["value"])
            if gates["false_improvement"]["value"] is not None
            else math.inf,
            idle_control=bool(gates["idle_control"]["pass"]),
            delayed_benefit=bool(gates["delayed_benefit"]["pass"]),
            authorized_by=authorized_by,
        )


def q_pi_enabled(gate: QPiGate | None) -> bool:
    return isinstance(gate, QPiGate) and gate.open


def _gumbel_softmax(
    logits: torch.Tensor, mask: torch.Tensor | None, temperature: float, generator: torch.Generator
) -> torch.Tensor:
    uniform = torch.rand(
        logits.shape, dtype=logits.dtype, device=logits.device, generator=generator
    ).clamp(1e-10, 1.0 - 1e-10)
    noisy = (logits - torch.log(-torch.log(uniform))) / temperature
    if mask is not None:
        noisy = noisy.masked_fill(~mask, -torch.inf)
    return torch.softmax(noisy, dim=-1)


def reparameterized_action_features(
    output: Mapping[str, torch.Tensor],
    tensors: Mapping[str, torch.Tensor],
    *,
    generator: torch.Generator,
    gate: QPiGate | None = None,
    temperature: float = 1.0,
    population: torch.Tensor | Sequence[int] | None = None,
) -> torch.Tensor:
    """The action vector of a reparameterized sample from the actor's output heads.

    Labor shares are the additive-log-ratio Gaussian draws the actor samples
    (``mean + eps * std``, masked softmax); categorical heads (movement union,
    fission, trade gates) use Gumbel-softmax relaxations; trade values and the
    trade share use their Gaussian latents through the executed transforms.

    With the gate closed (the default) every actor output is detached first:
    the vector is a sample of the actor's distribution but no gradient can reach
    an actor parameter. Only an open :class:`QPiGate` makes it differentiable.

    ``population`` (one root camp population per row): the specialty block is
    the **post-apportionment** share (:func:`apportioned_specialty`), exactly the
    encoding of the executed action on the search side. Largest-remainder
    apportionment is piecewise constant, so it has no useful gradient: the block
    is a straight-through estimator ``hard + (soft - soft.detach())`` whose
    forward value is exactly the apportioned share and whose gradient is the
    gradient of the continuous share (gradient-only; never used as a value). The
    activity rows of roles apportioned zero residents forward the neutral row (all
    idle) exactly, with the same straight-through gradient.
    Without ``population`` the block is the continuous requested share: use
    that only on the gradient path, never to evaluate or encode an action.
    """
    assert_actor_batch(tensors)
    if not q_pi_enabled(gate):
        output = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in output.items()
        }
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Gumbel-softmax temperature must be positive")
    specialty_logits = output["specialty_logits"]
    batch = specialty_logits.shape[0]
    specialty_mask, activity_mask = _masks(tensors, batch)
    if specialty_mask.shape[1] != len(SPECIALTIES) or activity_mask.shape[1:] != (
        ROLE_ROWS,
        len(ACTIVITIES),
    ):
        raise ValueError("actor output rows differ from the catalog registry")

    def draw(logits: torch.Tensor, log_std: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mean = logits - logits[..., :1]
        other = mask.clone()
        other[..., 0] = False
        noise = torch.randn(
            logits.shape, dtype=logits.dtype, device=logits.device, generator=generator
        )
        latent = torch.where(other, mean + noise * log_std.exp(), torch.zeros_like(mean))
        return _masked_softmax(latent, mask)

    specialty = draw(specialty_logits, output["specialty_log_std"], specialty_mask)
    if population is not None:
        counts = [int(p) for p in torch.as_tensor(population).reshape(-1).tolist()]
        if len(counts) != batch:
            raise ValueError("one root population per reparameterized row")
        soft = specialty.detach().double().cpu().numpy()
        hard = torch.as_tensor(
            np.stack([apportioned_specialty(row, p) for row, p in zip(soft, counts, strict=True)]),
            dtype=torch.float64,
        ).to(dtype=specialty.dtype, device=specialty.device)
        # Forward value exactly ``hard`` (x - x == 0); gradient of the continuous share.
        specialty = hard + (specialty - specialty.detach())
    activity = draw(output["activity_logits"], output["activity_log_std"], activity_mask)
    if population is not None:
        # Unstaffed roles (zero apportioned residents) forward the neutral activity row
        # exactly; straight-through gradient of the continuous row (gradient-only).
        staffed = torch.as_tensor(
            np.stack(
                [
                    apportion_specialty_counts(row, p)[1:] > 0
                    for row, p in zip(soft, counts, strict=True)
                ]
            ),
            device=activity.device,
        )
        neutral = torch.zeros_like(activity)
        neutral[..., NEUTRAL_ACTIVITY] = 1.0
        hard_activity = torch.where(staffed.unsqueeze(-1), activity.detach(), neutral)
        activity = hard_activity + (activity - activity.detach())
    movement_logits = output["movement_union_logits"]
    movement_mask = output["movement_union_mask"]
    if not torch.equal(
        output["movement_codes"].cpu(), torch.tensor(MOVEMENT_UNION_CODES, dtype=torch.long)
    ):
        raise ValueError("actor movement codes differ from the action-vector movement union")
    movement = _gumbel_softmax(movement_logits, movement_mask, temperature, generator)
    fission = _gumbel_softmax(output["fission"], None, temperature, generator)
    gate_rows = _gumbel_softmax(output["trade_gate"], None, temperature, generator)
    value_mean, value_std = output["trade_value_mean"], output["trade_value_log_std"].exp()
    value_noise = torch.randn(
        value_mean.shape, dtype=value_mean.dtype, device=value_mean.device, generator=generator
    )
    log_value = (value_mean + value_noise * value_std).clamp(_LOG_TRADE_LOW, _LOG_TRADE_HIGH)
    share_mean, share_std = output["trade_share_mean"], output["trade_share_log_std"].exp()
    share_noise = torch.randn(
        share_mean.shape, dtype=share_mean.dtype, device=share_mean.device, generator=generator
    )
    share = torch.sigmoid(share_mean + share_noise * share_std)
    vector = torch.cat(
        [
            specialty,
            activity.reshape(batch, -1),
            movement,
            fission,
            gate_rows.reshape(batch, -1),
            log_value / _LOG_TRADE_SCALE,
            share.unsqueeze(-1),
        ],
        dim=-1,
    )
    if vector.shape[-1] != ACTION_FEATURE_WIDTH:
        raise ValueError("reparameterized action vector width differs from the fixed layout")
    return vector


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class LaborActionEncoder(nn.Module):
    """Fixed-layout action vector -> 128-d action embedding (2-layer MLP)."""

    def __init__(self, embedding: int = EMBEDDING_WIDTH, hidden: int = 256) -> None:
        super().__init__()
        self.embedding = embedding
        self.net = nn.Sequential(
            nn.Linear(ACTION_FEATURE_WIDTH, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, embedding),
            nn.SiLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != ACTION_FEATURE_WIDTH:
            raise ValueError(f"action features must be [N, {ACTION_FEATURE_WIDTH}]")
        return self.net(features)


@dataclass(frozen=True)
class AdvantageWidths:
    """Context widths per objective (critic hidden + privileged readout) and head sizes."""

    context: Mapping[str, int]
    embedding: int = EMBEDDING_WIDTH
    hidden: int = 256
    encoder_hidden: int = 256

    def state_dict(self) -> dict[str, Any]:
        return {
            "context": {name: int(self.context[name]) for name in COMPONENTS},
            "embedding": int(self.embedding),
            "hidden": int(self.hidden),
            "encoderHidden": int(self.encoder_hidden),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> AdvantageWidths:
        return cls(
            {name: int(state["context"][name]) for name in COMPONENTS},
            int(state["embedding"]),
            int(state["hidden"]),
            int(state["encoderHidden"]),
        )


class LaborAdvantageHead(nn.Module):
    """Scalar dueling advantage A_j(C, a) per objective, centred per root before use.

    ``contexts[j]`` is the frozen critic's per-root context for objective j:
    the critic encoder's trunk output (the value head's input, a function of
    the contextual village token) concatenated with its privileged readout.
    Each objective has its own 2-layer MLP; the action encoder is shared. The
    last layer is zero-initialised, so an untrained head predicts A = 0 and
    Q = V exactly. With a separate QoL encoder each task objective reads the
    encoder that feeds its own value head (``LaborTaskCritic.objective_encoder``):
    the QoL component the QoL encoder's context, population-years the population
    encoder's. The widths are unchanged (both encoders share one architecture).
    """

    def __init__(self, widths: AdvantageWidths) -> None:
        super().__init__()
        self.widths = widths
        self.action_encoder = LaborActionEncoder(widths.embedding, widths.encoder_hidden)
        self.heads = nn.ModuleDict()
        for name in COMPONENTS:
            head = nn.Sequential(
                nn.Linear(widths.context[name] + widths.embedding, widths.hidden),
                nn.SiLU(),
                nn.Linear(widths.hidden, 1),
            )
            nn.init.zeros_(head[2].weight)
            nn.init.zeros_(head[2].bias)
            self.heads[name] = head
        # Masked action blocks (stage 1: trade value/share and unsupported blocks)
        # are zeroed before the encoder, so A's response to them is exactly zero.
        # Non-persistent: the state_dict (and every historical checkpoint) is
        # unchanged; the mask travels as ``masked_blocks`` in the artifact.
        self.register_buffer("feature_mask", torch.ones(ACTION_FEATURE_WIDTH), persistent=False)
        self.masked_blocks: tuple[str, ...] = ()

    def set_masked_blocks(self, blocks: Iterable[str]) -> None:
        names = tuple(sorted(set(blocks)))
        slices = block_slices()
        unknown = [b for b in names if b not in slices]
        if unknown:
            raise ValueError(f"unknown action blocks {unknown}")
        mask = torch.ones(ACTION_FEATURE_WIDTH, device=self.feature_mask.device)
        for name in names:
            mask[slices[name]] = 0.0
        self.feature_mask = mask
        self.masked_blocks = names

    @classmethod
    def for_critics(
        cls,
        task_critic: LaborTaskCritic,
        cost_critics: LaborCostCritics,
        *,
        embedding: int = EMBEDDING_WIDTH,
        hidden: int = 256,
    ) -> LaborAdvantageHead:
        context = {}
        for name in LABOR_OBJECTIVES:
            encoder = task_critic.objective_encoder(name)
            context[name] = int(encoder.width + encoder.token_width)
        for name in LABOR_COSTS:
            encoder = cost_critics.encoders[name]
            context[name] = int(encoder.width + encoder.token_width)
        return cls(AdvantageWidths(context, embedding, hidden, 256))

    def forward(
        self,
        contexts: Mapping[str, torch.Tensor],
        features: torch.Tensor,
        root_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Raw (uncentred) A per objective, in reference-scale units, one per candidate."""
        if root_index.dtype != torch.long or root_index.shape != features.shape[:1]:
            raise ValueError("root_index must be a long vector aligned with the features")
        if self.masked_blocks:
            features = features * self.feature_mask.to(features.dtype)
        embedding = self.action_encoder(features)
        out = {}
        for name in COMPONENTS:
            context = contexts[name]
            if context.ndim != 2 or context.shape[1] != self.widths.context[name]:
                raise ValueError(f"{name} advantage context width differs from the head")
            joined = torch.cat([context.detach()[root_index], embedding], dim=-1)
            out[name] = self.heads[name](joined).squeeze(-1)
        return out


def centre(
    values: torch.Tensor, root_index: torch.Tensor, roots: int | None = None
) -> torch.Tensor:
    """Subtract each root's mean over its scored candidate batch."""
    roots = int(root_index.max()) + 1 if roots is None else int(roots)
    total = torch.zeros(roots, dtype=values.dtype, device=values.device).index_add(
        0, root_index, values
    )
    count = torch.zeros(roots, dtype=values.dtype, device=values.device).index_add(
        0, root_index, torch.ones_like(values)
    )
    return values - (total / count.clamp_min(1))[root_index]


def utility_weights(personality: Sequence[float]) -> dict[str, float]:
    """Normalized-unit weights of the personality-weighted sum (labor_search utility)."""
    q, p = float(personality[0]), float(personality[1])
    return {"qol_years": q, "population_years": p, "feasibility": -1.0, "settlement_guard": -1.0}


def critic_contexts(
    task_critic: LaborTaskCritic,
    cost_critics: LaborCostCritics,
    tensors: Mapping[str, torch.Tensor],
    privileged_raw: Mapping[str, Any] | None = None,
    *,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Frozen per-root contexts for every objective (no gradient reaches the critic)."""
    assert_actor_batch(tensors)
    from .labor_device import module_device, to_device

    device = module_device(task_critic)
    rows = int(tensors["core"].shape[0])
    parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    # Train mode (the critics have no dropout) keeps PyTorch's eval-only fused
    # transformer fast path off, so contexts are exactly the critic's fitting graph.
    was = (task_critic.training, cost_critics.training)
    task_critic.train()
    cost_critics.train()
    try:
        with torch.no_grad():
            for start in range(0, rows, batch_size):
                index = torch.arange(start, min(rows, start + batch_size))
                batch = to_device({key: value[index] for key, value in tensors.items()}, device)
                task_priv = (
                    None
                    if privileged_raw is None
                    else critic_privileged_batch(task_critic, privileged_raw, index, device=device)
                )
                cost_priv = (
                    None
                    if privileged_raw is None
                    else critic_privileged_batch(cost_critics, privileged_raw, index, device=device)
                )
                if task_critic.privileged_layout is not None and task_priv is None:
                    raise ValueError("privileged critic contexts need the root privileged arrays")
                task = task_critic.encoder.context(batch, task_priv)
                task_context = torch.cat([task["hidden"], task["privileged_readout"]], dim=-1)
                for name in LABOR_OBJECTIVES:
                    if task_critic.objective_encoder(name) is task_critic.encoder:
                        parts[name].append(task_context)
                        continue
                    own = task_critic.objective_encoder(name).context(batch, task_priv)
                    parts[name].append(torch.cat([own["hidden"], own["privileged_readout"]], -1))
                for name in LABOR_COSTS:
                    cost = cost_critics.encoders[name].context(batch, cost_priv)
                    parts[name].append(torch.cat([cost["hidden"], cost["privileged_readout"]], -1))
    finally:
        task_critic.train(was[0])
        cost_critics.train(was[1])
    return {name: torch.cat(parts[name]) for name in COMPONENTS}


def state_values(
    task_critic: LaborTaskCritic,
    cost_critics: LaborCostCritics,
    tensors: Mapping[str, torch.Tensor],
    privileged_raw: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """V_j per root in reference-scale units (the existing heads, unchanged); on CPU."""
    from .labor_device import module_device, to_device

    device = module_device(task_critic)
    with torch.no_grad():
        task_priv = (
            None
            if privileged_raw is None
            else critic_privileged_batch(task_critic, privileged_raw, device=device)
        )
        cost_priv = (
            None
            if privileged_raw is None
            else critic_privileged_batch(cost_critics, privileged_raw, device=device)
        )
        tensors = to_device(tensors, device)
        logits = task_critic(tensors, task_priv)
        costs = cost_critics(tensors, cost_priv)
    values = {
        name: (task_critic.expectation(name, logits[name]) / REFERENCE_SCALES[name]).cpu()
        for name in LABOR_OBJECTIVES
    }
    values.update({name: (costs[name] / REFERENCE_SCALES[name]).cpu() for name in LABOR_COSTS})
    return values


def q_values(
    state_value: Mapping[str, torch.Tensor],
    advantage: Mapping[str, torch.Tensor],
    root_index: torch.Tensor,
    roots: int | None = None,
) -> dict[str, torch.Tensor]:
    """Q_j = V_j(root) + A_j centred over the root's candidate batch."""
    return {
        name: state_value[name][root_index] + centre(advantage[name], root_index, roots)
        for name in COMPONENTS
    }


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


TASK_CONTEXT_PER_OBJECTIVE = "per-objective-value-head-encoder"


def task_context_record(task_critic: LaborTaskCritic) -> str | None:
    """The payload's ``task_context`` for ``task_critic`` (None: historical shared context)."""
    return TASK_CONTEXT_PER_OBJECTIVE if task_critic.separate_qol_encoder else None


def advantage_checkpoint_payload(
    head: LaborAdvantageHead,
    *,
    critic_digest: str,
    config: AdvantageFitConfig,
    fit: Mapping[str, Any],
    task_context: str | None = None,
) -> dict[str, Any]:
    """Separate artifact bound to the frozen critic it reads (its labor_critic_digest).

    ``task_context`` (:func:`task_context_record`) is recorded only for a critic
    with a separate QoL encoder; every other payload keeps its historical keys.
    """
    return {
        "artifact_kind": ADVANTAGE_ARTIFACT_KIND,
        "architecture": ADVANTAGE_ARCHITECTURE,
        "action_layout": action_layout_state(),
        "components": list(COMPONENTS),
        "reference_scales": dict(REFERENCE_SCALES),
        "units": "reference-scale-normalized-return-difference",
        "centring": "mean-over-scored-candidate-batch-per-root",
        "widths": head.widths.state_dict(),
        "critic_digest": str(critic_digest),
        "critic_trained": False,
        "actor_gradient": "closed-q-pi-gate",
        "config": config.state_dict(),
        "fit": json.loads(json.dumps(fit)),
        "state_dict": head.state_dict(),
        # Only for a separate-QoL-encoder critic: old payloads keep their bytes.
        **({} if task_context is None else {"task_context": str(task_context)}),
        # Only for a head with masked action blocks (stage 1).
        **({} if not head.masked_blocks else {"masked_blocks": list(head.masked_blocks)}),
    }


def load_advantage_head(
    state: Mapping[str, Any],
    task_critic: LaborTaskCritic,
    cost_critics: LaborCostCritics,
) -> tuple[LaborAdvantageHead, AdvantageFitConfig]:
    """Strictly restore an advantage head against the exact critic it was fitted on."""
    from .labor_value_calibration import labor_critic_digest

    if not isinstance(state, Mapping) or state.get("artifact_kind") != ADVANTAGE_ARTIFACT_KIND:
        raise ValueError("not a labor advantage-head artifact")
    if state.get("architecture") != ADVANTAGE_ARCHITECTURE:
        raise ValueError("unsupported labor advantage-head architecture")
    if state.get("action_layout") != action_layout_state():
        raise ValueError("advantage head action layout differs from the executing registry")
    if state.get("components") != list(COMPONENTS):
        raise ValueError("advantage head components differ")
    digest = labor_critic_digest(task_critic, cost_critics)
    if state.get("critic_digest") != digest:
        raise ValueError("advantage head was fitted on a different critic (labor_critic_digest)")
    if state.get("task_context") != task_context_record(task_critic):
        raise ValueError("advantage head task context differs from the critic's QoL encoder")
    head = LaborAdvantageHead(AdvantageWidths.from_state(state["widths"]))
    expected = LaborAdvantageHead.for_critics(task_critic, cost_critics).widths.context
    if dict(head.widths.context) != dict(expected):
        raise ValueError("advantage head context widths differ from the critic")
    head.load_state_dict(state["state_dict"], strict=True)
    if state.get("masked_blocks"):
        head.set_masked_blocks(state["masked_blocks"])
    return head, AdvantageFitConfig.from_state(state["config"])


# ---------------------------------------------------------------------------
# Candidate payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Evaluation:
    horizon: int
    stream: int
    returns: np.ndarray  # raw, COMPONENTS order
    se: np.ndarray | None = None


@dataclass
class CandidateRecord:
    index: int
    family: str
    is_h12: bool
    digest: str
    action: dict[str, Any]
    features: np.ndarray
    control: str | None
    twin: int | None
    evaluations: dict[tuple[int, int], Evaluation]
    # T2 plan fields (labor_search --plan-candidates); None for first-action candidates.
    plan_id: str | None = None
    plan_role: str | None = None  # "plan" | "first-step-control"
    plan_steps: list[dict[str, Any]] | None = None
    step_returns: dict[int, np.ndarray] | None = None  # stream -> [steps, 4] raw rewards


@dataclass
class Root:
    root_id: str
    world: int
    turn: int
    camp: int
    source: str
    tensor_row: int
    personality: tuple[float, float]
    full_evaluation: bool
    pair_sd: dict[int, np.ndarray]
    candidates: list[CandidateRecord]
    shard: int = 0
    # T2 plan rows (plan and paired first-step control): diagnostic, never paired as
    # first-action candidates. Their ``index`` is their position in the raw row list.
    plans: list[CandidateRecord] = field(default_factory=list)
    # Curated scenario families of the root (labor_scenarios via labor_search
    # --source scenario); empty for keyed h12/actor roots.
    families: tuple[str, ...] = ()
    # Root camp residents (exporter ``population``, else the candidates' compiled
    # population); None for a payload that declares neither. With it every
    # candidate's specialty block is encoded after native apportionment.
    population: int | None = None

    @property
    def h12(self) -> int:
        return next(c.index for c in self.candidates if c.is_h12)


@dataclass
class CandidateSet:
    """Roots of one or more shards with their stacked root tensors."""

    roots: list[Root]
    tensors: dict[str, torch.Tensor]
    privileged: dict[str, torch.Tensor] | None
    rows: list[int]  # global tensor row of each root
    producers: list[Any] = field(default_factory=list)

    @property
    def worlds(self) -> list[int]:
        return sorted({root.world for root in self.roots})

    def subset(self, worlds: Iterable[int]) -> CandidateSet:
        keep = {int(w) for w in worlds}
        pairs = [(r, row) for r, row in zip(self.roots, self.rows, strict=True) if r.world in keep]
        return CandidateSet(
            [r for r, _ in pairs],
            self.tensors,
            self.privileged,
            [row for _, row in pairs],
            self.producers,
        )

    def root_tensors(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
        index = torch.as_tensor(self.rows, dtype=torch.long)
        tensors = {key: value[index] for key, value in self.tensors.items()}
        privileged = (
            None
            if self.privileged is None
            else {key: value[index] for key, value in self.privileged.items()}
        )
        return tensors, privileged


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _returns(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (len(COMPONENTS),) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be {len(COMPONENTS)} finite returns")
    return array


def _root_population(
    raw: Mapping[str, Any], raw_candidates: Sequence[Mapping[str, Any]], label: str
) -> int | None:
    """The root camp population: the exporter's root ``population``, else the one
    population every candidate's ``compiled`` block declares, else None."""
    if raw.get("population") is not None:
        population = _int(raw["population"], f"{label} population")
    else:
        declared = {
            _int(c["compiled"]["population"], f"{label} compiled population")
            for c in raw_candidates
            if isinstance(c.get("compiled"), Mapping)
            and c["compiled"].get("population") is not None
        }
        if len(declared) > 1:
            raise ValueError(f"{label} candidates declare different compiled populations")
        population = next(iter(declared)) if declared else None
    if population is not None and population < 0:
        raise ValueError(f"{label} population must be nonnegative")
    return population


def _check_compiled_counts(
    cand: Mapping[str, Any],
    action: Mapping[str, Any],
    spec_pos: Sequence[int],
    population: int | None,
    label: str,
) -> None:
    """The reader's apportionment must equal the exporter's native residents."""
    compiled = cand.get("compiled")
    if population is None or not isinstance(compiled, Mapping):
        return
    if compiled.get("specialty_counts") is None or compiled.get("population") is None:
        return
    if int(compiled["population"]) != int(population):
        return  # compiled at another population (plan steps); nothing to compare
    declared = np.asarray(compiled["specialty_counts"], dtype=np.int64)[list(spec_pos)]
    shares = np.asarray(action["laborSpecialtyAllocation"], dtype=np.float64)[list(spec_pos)]
    if not np.array_equal(apportion_specialty_counts(shares, population), declared):
        raise ValueError(f"{label}: apportioned residents differ from the compiled residents")


def parse_candidate_shard(shard: Mapping[str, Any], *, shard_index: int = 0) -> CandidateSet:
    """Strictly parse one ``labor-advantage-candidates-v1`` shard."""
    if not isinstance(shard, Mapping) or shard.get("schema") != CANDIDATE_PAYLOAD_SCHEMA:
        raise ValueError(f"candidate shard schema must be {CANDIDATE_PAYLOAD_SCHEMA!r}")
    if list(shard.get("component_order", [])) != list(COMPONENTS):
        raise ValueError(f"candidate shard component_order must be {list(COMPONENTS)}")
    if shard.get("returns_units") != RETURNS_UNITS:
        raise ValueError(f"candidate shard returns_units must be {RETURNS_UNITS!r}")
    if float(shard.get("discount_years", -1)) != contracts.HORIZON_YEARS:
        raise ValueError("candidate shard discount differs from the critic's 400-year constant")
    specialty_ids = list(shard.get("specialty_ids") or SPECIALTIES)
    activity_ids = list(shard.get("activity_ids") or ACTIVITIES)
    trade_groups = list(shard.get("trade_groups") or contracts.TRADE_GROUPS)
    tensors = shard.get("tensors")
    if not isinstance(tensors, Mapping) or "core" not in tensors:
        raise ValueError("candidate shard must carry root observation tensors")
    leaked = sorted(key for key in tensors if is_privileged_key(key))
    if leaked:
        raise ValueError(f"privileged keys in the candidate shard observation tensors: {leaked}")
    tensors = {key: torch.as_tensor(value) for key, value in tensors.items()}
    rows = int(tensors["core"].shape[0])
    if any(int(value.shape[0]) != rows for value in tensors.values()):
        raise ValueError("candidate shard tensors disagree on the root row count")
    privileged = shard.get("privileged")
    if privileged is not None:
        privileged = {key: torch.as_tensor(value) for key, value in privileged.items()}
        if any(int(value.shape[0]) != rows for value in privileged.values()):
            raise ValueError("candidate shard privileged arrays disagree with the root rows")
    roots: list[Root] = []
    seen: set[str] = set()
    for r, raw in enumerate(shard.get("roots") or []):
        label = f"root {r}"
        root_id = str(raw["root_id"])
        if root_id in seen:
            raise ValueError(f"duplicate root_id {root_id}")
        seen.add(root_id)
        source = str(raw["source"])
        if source not in ("h12", "actor"):
            raise ValueError(f"{label} source must be h12 or actor")
        row = _int(raw["tensor_row"], f"{label} tensor_row")
        if not 0 <= row < rows:
            raise ValueError(f"{label} tensor_row out of range")
        personality = tuple(float(x) for x in raw["personality"])
        if len(personality) != 2 or not math.isclose(sum(personality), 1.0, abs_tol=1e-6):
            raise ValueError(f"{label} personality must be a two-objective simplex")
        pair_sd = {
            _int(int(h), f"{label} pair_sd horizon"): _returns(v, f"{label} pair_sd")
            for h, v in (raw.get("pair_sd") or {}).items()
        }
        candidates = []
        plans: list[CandidateRecord] = []
        raw_candidates = raw.get("candidates") or []
        population = _root_population(raw, raw_candidates, label)
        spec_pos = _binding(specialty_ids, SPECIALTIES, "specialty")
        for i, cand in enumerate(raw_candidates):
            clabel = f"{label} candidate {i}"
            if _int(cand["index"], f"{clabel} index") != i:
                raise ValueError(f"{clabel} index must equal its position")
            control = cand.get("control")
            if control is not None and control not in CONTROLS:
                raise ValueError(f"{clabel} control must be one of {CONTROLS}")
            twin = cand.get("bookkeeping_twin")
            if twin is not None:
                twin = _int(twin, f"{clabel} bookkeeping_twin")
                if not 0 <= twin < len(raw_candidates) or twin == i:
                    raise ValueError(f"{clabel} bookkeeping_twin out of range")
            evaluations: dict[tuple[int, int], Evaluation] = {}
            for e, ev in enumerate(cand.get("evaluations") or []):
                horizon = _int(ev["horizon"], f"{clabel} evaluation {e} horizon")
                reached = _int(ev["reached_horizon"], f"{clabel} evaluation {e} reached")
                stream = _int(ev["stream"], f"{clabel} evaluation {e} stream")
                if horizon < 1 or stream < 0 or not 0 <= reached <= horizon:
                    raise ValueError(f"{clabel} evaluation {e} horizon/stream malformed")
                if reached < horizon and not bool(ev.get("terminated", False)):
                    continue  # truncated branch: not comparable at its declared horizon
                if (horizon, stream) in evaluations:
                    raise ValueError(f"{clabel} repeats horizon {horizon} stream {stream}")
                se = ev.get("se")
                evaluations[(horizon, stream)] = Evaluation(
                    horizon,
                    stream,
                    _returns(ev["returns"], f"{clabel} returns"),
                    None if se is None else _returns(se, f"{clabel} se"),
                )
            action = dict(cand["action"])
            plan_role = cand.get("plan_role")
            if plan_role is not None and plan_role not in PLAN_ROLES:
                raise ValueError(f"{clabel} plan_role must be one of {PLAN_ROLES}")
            if plan_role is None and plans:
                raise ValueError(f"{clabel}: plan rows must follow every first-action candidate")
            if plan_role is not None and (bool(cand["is_h12"]) or not cand.get("plan_id")):
                raise ValueError(f"{clabel}: a plan row needs a plan_id and cannot be H12")
            record = CandidateRecord(
                i,
                str(cand.get("family", "")),
                bool(cand["is_h12"]),
                str(cand.get("digest", "")),
                action,
                action_features(
                    action,
                    specialty_ids=specialty_ids,
                    activity_ids=activity_ids,
                    trade_groups=trade_groups,
                    population=population,
                ),
                control,
                twin,
                evaluations,
            )
            _check_compiled_counts(cand, action, spec_pos, population, clabel)
            if plan_role is None:
                candidates.append(record)
                continue
            record.plan_id = str(cand["plan_id"])
            record.plan_role = str(plan_role)
            record.plan_steps = [dict(s) for s in cand.get("plan_steps") or []]
            record.step_returns = {
                _int(int(s), f"{clabel} step_returns stream"): np.asarray(
                    v, dtype=np.float64
                ).reshape(-1, len(COMPONENTS))
                for s, v in (cand.get("step_returns") or {}).items()
            }
            plans.append(record)
        if sum(c.is_h12 for c in candidates) != 1:
            raise ValueError(f"{label} must have exactly one H12 candidate")
        roots.append(
            Root(
                root_id,
                _int(raw["world"], f"{label} world"),
                _int(raw["turn"], f"{label} turn"),
                _int(raw["camp"], f"{label} camp"),
                source,
                row,
                (personality[0], personality[1]),
                bool(raw.get("full_evaluation", False)),
                pair_sd,
                candidates,
                shard_index,
                plans,
                tuple(str(f) for f in raw.get("scenario_family") or ()),
                population,
            )
        )
    return CandidateSet(
        roots, tensors, privileged, [root.tensor_row for root in roots], [shard.get("producer")]
    )


def merge_candidate_sets(sets: Sequence[CandidateSet]) -> CandidateSet:
    """Concatenate shards; tensor rows are re-offset, keys must agree."""
    if not sets:
        raise ValueError("no candidate shards")
    keys = set(sets[0].tensors)
    has_priv = sets[0].privileged is not None
    roots: list[Root] = []
    rows: list[int] = []
    offset = 0
    ids: set[str] = set()
    for index, part in enumerate(sets):
        if set(part.tensors) != keys or (part.privileged is not None) != has_priv:
            raise ValueError("candidate shards disagree on tensor or privileged keys")
        for root, row in zip(part.roots, part.rows, strict=True):
            if root.root_id in ids:
                raise ValueError(f"root_id {root.root_id} repeats across shards")
            ids.add(root.root_id)
            root.shard = index
            roots.append(root)
            rows.append(row + offset)
        offset += int(part.tensors["core"].shape[0])
    tensors = {key: torch.cat([part.tensors[key] for part in sets]) for key in keys}
    privileged = (
        None
        if not has_priv
        else {key: torch.cat([part.privileged[key] for part in sets]) for key in sets[0].privileged}
    )
    return CandidateSet(roots, tensors, privileged, rows, [p for s in sets for p in s.producers])


def load_candidate_shards(paths: Sequence[Path | str]) -> CandidateSet:
    """Candidate shards from files and directories, in the listed order.

    Several payload directories (for example one per collection batch) are concatenated
    in the given order, each directory's ``*.pt`` files sorted. A file reached twice
    (listed twice, or through a directory and directly) is refused, as is a root id
    repeated across shards (:func:`merge_candidate_sets`); the fit refuses reserved,
    calibration, development and sealed worlds (:func:`require_advantage_fit_worlds`).
    """
    files: list[Path] = []
    for path in paths:
        path = Path(path)
        files.extend(sorted(path.glob("*.pt")) if path.is_dir() else [path])
    if not files:
        raise ValueError("no candidate shard files found")
    resolved = [f.resolve() for f in files]
    if len(set(resolved)) != len(resolved):
        repeated = sorted({str(f) for f in resolved if resolved.count(f) > 1})
        raise ValueError(f"candidate shard files listed more than once: {repeated[:3]}")
    return merge_candidate_sets(
        [
            parse_candidate_shard(
                torch.load(f, map_location="cpu", weights_only=True), shard_index=i
            )
            for i, f in enumerate(files)
        ]
    )


# ---------------------------------------------------------------------------
# Pairs and loss
# ---------------------------------------------------------------------------


def target_weights(target: str, personality: Sequence[float] | None = None) -> dict[str, float]:
    """Normalized-unit coefficients of a ranking target: one component or a utility."""
    if target in COMPONENTS:
        return {name: (1.0 if name == target else 0.0) for name in COMPONENTS}
    if target.startswith("utility") and personality is not None:
        return utility_weights(personality)
    raise ValueError(f"unknown ranking target {target!r}")


@dataclass(frozen=True)
class Pairs:
    """Same-root, same-horizon, same-stream candidate pairs for one target."""

    root: np.ndarray  # root position
    i: np.ndarray  # candidate index within the root
    k: np.ndarray
    horizon: np.ndarray
    reevaluated: np.ndarray  # bool
    delta: np.ndarray  # G_i - G_k, normalized units
    se: np.ndarray
    admitted: np.ndarray  # |delta| > multiplier * se

    def __len__(self) -> int:
        return int(self.root.size)


def _value(returns: np.ndarray, weights: Mapping[str, float]) -> float:
    return float(
        sum(
            weights[name] * returns[c] / REFERENCE_SCALES[name] for c, name in enumerate(COMPONENTS)
        )
    )


def _sd_value(sd: np.ndarray, weights: Mapping[str, float]) -> float:
    return math.sqrt(
        sum(
            (weights[name] * sd[c] / REFERENCE_SCALES[name]) ** 2
            for c, name in enumerate(COMPONENTS)
        )
    )


def root_value(
    root: Root,
    candidate: int,
    horizon: int,
    weights: Mapping[str, float],
    *,
    prefer_reeval: bool = True,
    streams: Sequence[int] | None = None,
) -> float | None:
    """A candidate's realized normalized value at ``horizon`` (re-evaluated mean if present)."""
    evals = root.candidates[candidate].evaluations
    reeval = sorted(s for (h, s) in evals if h == horizon and s >= 1)
    if streams is not None:
        reeval = [s for s in reeval if s in set(streams)]
    if prefer_reeval and len(reeval) >= 2:
        return float(np.mean([_value(evals[(horizon, s)].returns, weights) for s in reeval]))
    if (horizon, 0) in evals:
        return _value(evals[(horizon, 0)].returns, weights)
    return None


def build_pairs(
    roots: Sequence[Root],
    target: str,
    *,
    horizons: Sequence[int] | None = None,
    se_multiplier: float = PAIR_SE_MULTIPLIER,
    personality: Sequence[float] | str | None = None,
) -> Pairs:
    """All comparable same-root pairs i < k for ``target``.

    Pairs are formed per (root, horizon) and never pooled across horizons. For a
    pair both of whose candidates carry >= 2 shared re-evaluation streams, the
    re-evaluated paired mean and its paired standard error are used; otherwise
    the selection stream (0) with its declared SE; a pair is never formed from
    one re-evaluated and one single-rollout return. ``personality="root"`` uses
    each root's own personality for a utility target.
    """
    out: dict[str, list[Any]] = defaultdict(list)
    allowed = None if horizons is None else {int(h) for h in horizons}
    for r, root in enumerate(roots):
        pers = root.personality if personality == "root" else personality
        weights = target_weights(target, pers)
        available = sorted({h for c in root.candidates for (h, _) in c.evaluations})
        for horizon in available:
            if allowed is not None and horizon not in allowed:
                continue
            for i in range(len(root.candidates)):
                ei = root.candidates[i].evaluations
                for k in range(i + 1, len(root.candidates)):
                    ek = root.candidates[k].evaluations
                    shared = sorted(
                        s for (h, s) in ei if h == horizon and s >= 1 and (horizon, s) in ek
                    )
                    if len(shared) >= 2:
                        d = np.array(
                            [
                                _value(ei[(horizon, s)].returns, weights)
                                - _value(ek[(horizon, s)].returns, weights)
                                for s in shared
                            ]
                        )
                        delta = float(d.mean())
                        se = float(d.std(ddof=1) / math.sqrt(d.size))
                        reevaluated = True
                    elif (horizon, 0) in ei and (horizon, 0) in ek:
                        a, b = ei[(horizon, 0)], ek[(horizon, 0)]
                        delta = _value(a.returns, weights) - _value(b.returns, weights)
                        if a.se is not None and b.se is not None:
                            se = math.hypot(_sd_value(a.se, weights), _sd_value(b.se, weights))
                        elif horizon in root.pair_sd:
                            se = _sd_value(root.pair_sd[horizon], weights)
                        else:
                            continue  # no declared noise: not comparable
                        reevaluated = False
                    else:
                        continue
                    out["root"].append(r)
                    out["i"].append(i)
                    out["k"].append(k)
                    out["horizon"].append(horizon)
                    out["reevaluated"].append(reevaluated)
                    out["delta"].append(delta)
                    out["se"].append(se)
                    out["admitted"].append(abs(delta) > se_multiplier * se and delta != 0.0)

    def array(name: str, dtype: Any) -> np.ndarray:
        return np.asarray(out[name], dtype=dtype)

    return Pairs(
        array("root", np.int64),
        array("i", np.int64),
        array("k", np.int64),
        array("horizon", np.int64),
        array("reevaluated", bool),
        array("delta", np.float64),
        array("se", np.float64),
        array("admitted", bool),
    )


def pairwise_loss(
    a_i: torch.Tensor,
    a_k: torch.Tensor,
    delta: torch.Tensor,
    *,
    regression_weight: float = REGRESSION_WEIGHT,
) -> torch.Tensor:
    """sum softplus(-sign(dG)(A_i - A_k)) + 0.1 sum ((A_i - A_k) - dG)^2, averaged per pair.

    The caller passes only admitted pairs (w_ik = 1[|dG| > 2 SE]).
    """
    if a_i.numel() == 0:
        return a_i.sum() * 0.0
    diff = a_i - a_k
    ranking = torch.nn.functional.softplus(-torch.sign(delta) * diff)
    regression = (diff - delta).square()
    return (ranking + regression_weight * regression).mean()


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdvantageFitConfig:
    epochs: int = 50
    batch_roots: int = 32
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    regression_weight: float = REGRESSION_WEIGHT
    se_multiplier: float = PAIR_SE_MULTIPLIER
    horizons: Mapping[str, tuple[int, ...] | None] = field(
        default_factory=lambda: dict(DEFAULT_OBJECTIVE_HORIZONS)
    )
    holdout_fraction: float = 0.2
    seed: int = 0
    embedding: int = EMBEDDING_WIDTH
    hidden: int = 256
    # Stage-1 (lambda target source) fields; recorded only when that source is used,
    # so historical pair-fit configs keep their exact state dict. ``horizons``,
    # ``se_multiplier`` and ``regression_weight`` belong to the pair source only.
    target_source: str = PAIR_TARGET_SOURCE
    huber_se_multiplier: float = HUBER_SE_MULTIPLIER
    masked_blocks: tuple[str, ...] = STAGE1_MASKED_BLOCKS
    support_floor: int = BLOCK_SUPPORT_FLOOR

    def __post_init__(self) -> None:
        if set(self.horizons) != set(COMPONENTS):
            raise ValueError(f"advantage horizons must name exactly {list(COMPONENTS)}")
        if not 0.0 <= self.holdout_fraction < 1.0:
            raise ValueError("holdout fraction must be in [0, 1)")
        if self.target_source not in TARGET_SOURCES:
            raise ValueError(f"advantage target source must be one of {TARGET_SOURCES}")
        if not self.huber_se_multiplier > 0 or self.support_floor < 0:
            raise ValueError("Huber multiplier must be positive and the support floor >= 0")
        unknown = set(self.masked_blocks) - {name for name, _ in ACTION_BLOCKS}
        if unknown:
            raise ValueError(f"unknown masked action blocks {sorted(unknown)}")

    def state_dict(self) -> dict[str, Any]:
        state = self._base_state()
        if self.target_source != PAIR_TARGET_SOURCE:
            state.update(
                target_source=self.target_source,
                huber_se_multiplier=self.huber_se_multiplier,
                masked_blocks=list(self.masked_blocks),
                support_floor=self.support_floor,
            )
        return state

    def _base_state(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_roots": self.batch_roots,
            "lr": self.lr,
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": self.weight_decay,
            "regression_weight": self.regression_weight,
            "se_multiplier": self.se_multiplier,
            "horizons": {
                name: None if self.horizons[name] is None else list(self.horizons[name])
                for name in COMPONENTS
            },
            "holdout_fraction": self.holdout_fraction,
            "seed": self.seed,
            "embedding": self.embedding,
            "hidden": self.hidden,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> AdvantageFitConfig:
        data = dict(state)
        data["horizons"] = {
            name: None if value is None else tuple(int(h) for h in value)
            for name, value in data["horizons"].items()
        }
        if "masked_blocks" in data:
            data["masked_blocks"] = tuple(str(b) for b in data["masked_blocks"])
        return cls(**data)

    @staticmethod
    def parse_horizons(text: str | None) -> dict[str, tuple[int, ...] | None]:
        """``name=1,2,3;name=all`` overrides of :data:`DEFAULT_OBJECTIVE_HORIZONS`."""
        result = dict(DEFAULT_OBJECTIVE_HORIZONS)
        if not text:
            return result
        for part in text.split(";"):
            name, _, value = part.partition("=")
            name = name.strip()
            if name not in COMPONENTS:
                raise ValueError(f"unknown advantage objective {name!r}")
            value = value.strip()
            result[name] = None if value == "all" else tuple(int(h) for h in value.split(","))
        return result


def require_advantage_fit_worlds(worlds: Iterable[int]) -> None:
    """A never trains on reserved, calibration, development or sealed worlds."""
    from .labor_bc import DEVELOPMENT_WORLDS, SEALED_WORLDS

    worlds = sorted({int(w) for w in worlds})
    for name, bank in (
        ("reserved held-out (90154-90159)", RESERVED_HELDOUT_WORLDS),
        ("G0 calibration (90180-90199)", G0_CALIBRATION_WORLDS),
        ("development (2000-2059)", DEVELOPMENT_WORLDS),
        ("sealed confirmation (3000-3059)", SEALED_WORLDS),
    ):
        bad = [w for w in worlds if w in bank]
        if bad:
            raise ValueError(f"advantage fit refuses {name} worlds {bad}")


def split_worlds(
    worlds: Sequence[int], holdout_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Deterministic world-level split (before any pair is built)."""
    worlds = sorted({int(w) for w in worlds})
    if holdout_fraction <= 0 or len(worlds) < 2:
        return worlds, []
    order = np.random.default_rng([SPLIT_TAG, int(seed)]).permutation(len(worlds))
    count = max(1, math.ceil(holdout_fraction * len(worlds)))
    count = min(count, len(worlds) - 1)
    held = sorted(worlds[i] for i in order[:count])
    return [w for w in worlds if w not in set(held)], held


@dataclass
class _Flat:
    features: torch.Tensor  # [N, F]
    root_index: torch.Tensor  # [N] position of the root in the set
    offsets: list[int]  # first flat index of each root


def _flatten(roots: Sequence[Root]) -> _Flat:
    features, root_index, offsets = [], [], []
    for r, root in enumerate(roots):
        offsets.append(len(features))
        for cand in root.candidates:
            features.append(cand.features)
            root_index.append(r)
    return _Flat(
        torch.as_tensor(np.stack(features), dtype=torch.float32),
        torch.as_tensor(root_index, dtype=torch.long),
        offsets,
    )


def predict(
    head: LaborAdvantageHead,
    contexts: Mapping[str, torch.Tensor],
    roots: Sequence[Root],
) -> dict[str, list[np.ndarray]]:
    """Centred A per component, one array per root (candidate order)."""
    from .labor_device import module_device

    flat = _flatten(roots)
    device = module_device(head)
    root_index = flat.root_index.to(device)
    with torch.no_grad():
        raw = head(contexts, flat.features.to(device), root_index)
    out: dict[str, list[np.ndarray]] = {}
    for name in COMPONENTS:
        values = centre(raw[name], root_index, len(roots)).double().cpu().numpy()
        out[name] = [
            values[flat.offsets[r] : flat.offsets[r] + len(root.candidates)]
            for r, root in enumerate(roots)
        ]
    return out


def fit_advantage_head(
    task_critic: LaborTaskCritic,
    cost_critics: LaborCostCritics,
    candidates: CandidateSet,
    config: AdvantageFitConfig | None = None,
    *,
    stability: Any = "auto",
    frozen_actor: nn.Module | None = None,
    targets: Any = None,
) -> tuple[LaborAdvantageHead, dict[str, Any]]:
    """Fit A on same-root pairs of fitting worlds; the critic stays frozen.

    With ``config.target_source == LAMBDA_TARGET_SOURCE`` the fit is the stage-1
    search pretraining instead (``labor_a_stage1.fit_stage1``): ``targets`` (a
    ``labor_a_stage1.StageOneTargets`` built from export-v2 successor rows and a
    critic V) are fitted with the anchored, inverse-noise-variance Huber loss
    ``labor_a_stage1.anchored_huber_loss``; :func:`pairwise_loss` is then not used.

    Worlds are split (fit / held-out) before pairs are built; the held-out
    worlds' pairs are only scored. Reserved, calibration, development and
    sealed worlds are refused.

    ``stability`` (default ``"auto"``: on, the A head is an r5 component) wraps
    every step in the enforced stability monitor (``labor_stability``): finite
    checks with parameter+Adam rollback and LR halving, per-block gradient
    norms, frozen critic/actor gradients exactly zero, QoL/population-years
    dominance on the shared action encoder, and train versus held-out pair
    ranking every epoch for the overfit alarm. Its receipt is ``summary["stability"]``.
    """
    from .labor_device import module_device
    from .labor_value_calibration import labor_critic_digest

    config = config or AdvantageFitConfig()
    if config.target_source == LAMBDA_TARGET_SOURCE:
        from .labor_a_stage1 import fit_stage1

        if targets is None:
            raise ValueError("the lambda target source needs stage-1 targets")
        return fit_stage1(
            task_critic,
            cost_critics,
            candidates,
            targets,
            config,
            stability=stability,
            frozen_actor=frozen_actor,
        )
    if targets is not None:
        raise ValueError("stage-1 targets need target_source=LAMBDA_TARGET_SOURCE")
    require_advantage_fit_worlds(candidates.worlds)
    device = module_device(task_critic)
    digest_before = labor_critic_digest(task_critic, cost_critics)
    fit_worlds, held_worlds = split_worlds(candidates.worlds, config.holdout_fraction, config.seed)
    fit_set = candidates.subset(fit_worlds)
    held_set = candidates.subset(held_worlds)
    tensors, privileged = fit_set.root_tensors()
    contexts = critic_contexts(task_critic, cost_critics, tensors, privileged)
    pairs = {
        name: build_pairs(
            fit_set.roots, name, horizons=config.horizons[name], se_multiplier=config.se_multiplier
        )
        for name in COMPONENTS
    }
    flat = _flatten(fit_set.roots)
    flat_features = flat.features.to(device)
    flat_root_index = flat.root_index.to(device)
    # Admitted pairs as flat candidate indices, grouped by root for batching.
    by_root: dict[int, dict[str, list[tuple[int, int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for name, p in pairs.items():
        for idx in np.flatnonzero(p.admitted):
            r = int(p.root[idx])
            by_root[r][name].append(
                (
                    flat.offsets[r] + int(p.i[idx]),
                    flat.offsets[r] + int(p.k[idx]),
                    float(p.delta[idx]),
                )
            )
    trainable = sorted(by_root)
    torch.manual_seed(config.seed)
    head = LaborAdvantageHead.for_critics(
        task_critic, cost_critics, embedding=config.embedding, hidden=config.hidden
    )
    head.to(device)  # initialized on CPU (same initial parameters on every device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rng = np.random.default_rng([SHUFFLE_TAG, int(config.seed)])
    history = []
    from .labor_stability import StabilityMonitor, resolve_stability

    stability_config = resolve_stability(stability, r5=True)
    monitor = None
    h_contexts = None
    if stability_config is not None:
        blocks = {"action_encoder": list(head.action_encoder.parameters())}
        for name in COMPONENTS:
            blocks[f"head:{name}"] = list(head.heads[name].parameters())
        frozen = {"task_critic": task_critic, "cost_critics": cost_critics}
        if frozen_actor is not None:
            frozen["actor"] = frozen_actor
        monitor = StabilityMonitor(
            stability_config,
            fit="advantage-fit",
            optimizer=optimizer,
            parameters=list(head.parameters()),
            blocks=blocks,
            frozen=frozen,
            shared_heads={
                "action_encoder": (blocks["action_encoder"], ("qol_years", "population_years"))
            },
            applicable={"advantage_overfit": True},
        )
        if held_set.roots:
            h_tensors_e, h_priv_e = held_set.root_tensors()
            h_contexts = critic_contexts(task_critic, cost_critics, h_tensors_e, h_priv_e)

    def epoch_rankings() -> dict[str, dict[str, float | None]]:
        from .labor_advantage_eval import pairwise_ranking

        out: dict[str, dict[str, float | None]] = {}
        for label, roots, ctx in (
            ("train", fit_set.roots, contexts),
            ("heldout", held_set.roots, h_contexts),
        ):
            if not roots or ctx is None:
                out[label] = {name: None for name in ("qol_years", "population_years")}
                continue
            prediction = predict(head, ctx, roots)
            out[label] = {
                name: pairwise_ranking(
                    roots,
                    prediction,
                    name,
                    horizons=config.horizons[name],
                    se_multiplier=config.se_multiplier,
                )["accuracy"]
                for name in ("qol_years", "population_years")
            }
        return out

    for epoch_number in range(config.epochs if trainable else 0):
        order = [trainable[i] for i in rng.permutation(len(trainable))]
        total = 0.0
        for start in range(0, len(order), config.batch_roots):
            batch = order[start : start + config.batch_roots]
            optimizer.zero_grad()
            # Forward only the batch roots' candidates; pairs index them locally.
            members = torch.cat(
                [
                    torch.arange(
                        flat.offsets[r], flat.offsets[r] + len(fit_set.roots[r].candidates)
                    )
                    for r in batch
                ]
            )
            local = torch.full((len(flat.root_index),), -1, dtype=torch.long)
            local[members] = torch.arange(len(members))
            members_on = members.to(device)
            raw = head(contexts, flat_features[members_on], flat_root_index[members_on])
            loss = raw["qol_years"].sum() * 0.0
            component_losses: dict[str, torch.Tensor] = {}
            for name in COMPONENTS:
                triples = [t for r in batch for t in by_root[r][name]]
                if not triples:
                    continue
                index = local[
                    torch.as_tensor([[t[0], t[1]] for t in triples], dtype=torch.long)
                ].to(device)
                delta = torch.as_tensor([t[2] for t in triples], dtype=torch.float32).to(device)
                component = pairwise_loss(
                    raw[name][index[:, 0]],
                    raw[name][index[:, 1]],
                    delta,
                    regression_weight=config.regression_weight,
                )
                component_losses[name] = component
                loss = loss + component
            if monitor is None:
                if not torch.isfinite(loss):
                    raise ValueError("advantage fit produced a nonfinite loss")
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), config.max_grad_norm)
                optimizer.step()
            else:
                both = all(n in component_losses for n in ("qol_years", "population_years"))
                if not monitor.step(
                    loss,
                    max_grad_norm=config.max_grad_norm,
                    head_losses=component_losses if both else None,
                ):
                    continue
            total += float(loss.detach()) * len(batch)
        history.append(total / max(1, len(order)))
        if monitor is not None:
            ranks = epoch_rankings()
            monitor.epoch_ranking(
                epoch_number,
                ranks["train"]["qol_years"],
                ranks["heldout"]["qol_years"],
                population_years={k: ranks[k]["population_years"] for k in ranks},
                loss=history[-1],
            )
    if labor_critic_digest(task_critic, cost_critics) != digest_before:
        raise AssertionError("advantage fit changed the frozen critic")
    summary: dict[str, Any] = {
        "fit_worlds": fit_worlds,
        "heldout_worlds": held_worlds,
        "fit_roots": len(fit_set.roots),
        "heldout_roots": len(held_set.roots),
        "trainable_roots": len(trainable),
        "fit_pairs": {name: int(p.admitted.sum()) for name, p in pairs.items()},
        "comparable_pairs": {name: len(p) for name, p in pairs.items()},
        "reevaluated_pairs": {
            name: int((p.admitted & p.reevaluated).sum()) for name, p in pairs.items()
        },
        "loss_history": history,
        "critic_digest": digest_before,
        "parameters": parameter_counts(head),
    }
    if monitor is not None:
        summary["stability"] = monitor.finish()
    if held_set.roots:
        from .labor_advantage_eval import pairwise_ranking

        h_tensors, h_priv = held_set.root_tensors()
        h_contexts = critic_contexts(task_critic, cost_critics, h_tensors, h_priv)
        prediction = predict(head, h_contexts, held_set.roots)
        summary["heldout_ranking"] = {
            name: pairwise_ranking(
                held_set.roots,
                prediction,
                name,
                horizons=config.horizons[name],
                se_multiplier=config.se_multiplier,
            )["accuracy"]
            for name in COMPONENTS
        }
    return head, summary


def parameter_counts(head: LaborAdvantageHead) -> dict[str, int]:
    counts = {"action_encoder": sum(p.numel() for p in head.action_encoder.parameters())}
    for name in COMPONENTS:
        counts[f"head:{name}"] = sum(p.numel() for p in head.heads[name].parameters())
    counts["total"] = sum(p.numel() for p in head.parameters())
    return counts


# ---------------------------------------------------------------------------
# Search ranker hook
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankerPlan:
    """Which candidates a search root evaluates natively, and every predicted A."""

    advantage: dict[str, list[float]]  # centred A per component, candidate order
    utility: list[list[float]]  # [personality][candidate] predicted utility advantage
    order: list[int]  # best-rank-over-personalities order
    evaluate: list[int]  # sorted indices to evaluate natively
    full_evaluation: bool
    h12: int

    def record(self, index: int, realized: Sequence[float], horizon: int) -> dict[str, Any]:
        """Calibration record for an evaluated candidate (predicted A next to its return)."""
        return {
            "candidate": int(index),
            "horizon": int(horizon),
            "predicted_advantage": {name: self.advantage[name][index] for name in COMPONENTS},
            "predicted_utility": [row[index] for row in self.utility],
            "realized_returns": [float(x) for x in realized],
            "full_evaluation": self.full_evaluation,
        }


class AdvantageRanker:
    """labor_search hook: score with A, evaluate top-k plus H12, keep full evaluation on a fraction.

    The full-evaluation draw is keyed (``key`` plus the caller's root key), so
    it is reproducible and independent of the scores. Candidates must already
    be sampled (frozen actor, H12, structured perturbations); A never selects
    what is sampled, only which samples are natively evaluated.
    """

    def __init__(
        self,
        task_critic: LaborTaskCritic,
        cost_critics: LaborCostCritics,
        head: LaborAdvantageHead,
        *,
        top_k: int = TOP_K,
        full_fraction: float = FULL_EVALUATION_FRACTION,
        key: int = 0,
    ) -> None:
        if top_k < 1 or not 0.0 <= full_fraction <= 1.0:
            raise ValueError("ranker needs top_k >= 1 and a full-evaluation fraction in [0, 1]")
        self.task_critic, self.cost_critics, self.head = task_critic, cost_critics, head
        self.top_k, self.full_fraction, self.key = int(top_k), float(full_fraction), int(key)

    def plan(
        self,
        root_tensors: Mapping[str, torch.Tensor],
        actions: Sequence[Mapping[str, Any]],
        *,
        h12_index: int = 0,
        personalities: Sequence[Sequence[float]] = ((0.5, 0.5),),
        privileged_raw: Mapping[str, Any] | None = None,
        root_key: Sequence[int] = (),
        population: int | None = None,
    ) -> RankerPlan:
        """``population``: the root camp's residents; with it every candidate is encoded
        after native apportionment (:func:`encode_actions`), as the payload reader does."""
        if int(root_tensors["core"].shape[0]) != 1:
            raise ValueError("the ranker scores one root at a time")
        if not 0 <= h12_index < len(actions) or len(actions) < 1:
            raise ValueError("h12_index must name one of the candidates")
        contexts = critic_contexts(
            self.task_critic, self.cost_critics, root_tensors, privileged_raw
        )
        features = torch.as_tensor(encode_actions(actions, population), dtype=torch.float32)
        root_index = torch.zeros(len(actions), dtype=torch.long)
        with torch.no_grad():
            raw = self.head(contexts, features, root_index)
        advantage = {
            name: centre(raw[name], root_index, 1).double().tolist() for name in COMPONENTS
        }
        utility = []
        for pers in personalities:
            weights = utility_weights(pers)
            utility.append(
                [
                    sum(weights[name] * advantage[name][i] for name in COMPONENTS)
                    for i in range(len(actions))
                ]
            )
        u = np.asarray(utility).T  # [candidates, personalities]
        ranks = (u[None, :, :] > u[:, None, :]).sum(axis=1)
        order = sorted(
            range(len(actions)), key=lambda c: (int(ranks[c].min()), int(ranks[c].sum()), c)
        )
        draw = np.random.default_rng([RANKER_TAG, self.key, *[int(x) for x in root_key]]).random()
        full = bool(draw < self.full_fraction)
        evaluate = (
            list(range(len(actions))) if full else sorted({*order[: self.top_k], int(h12_index)})
        )
        return RankerPlan(advantage, utility, order, evaluate, full, int(h12_index))


# ---------------------------------------------------------------------------
# Bounded QoL head: TD(lambda) targets and saturation
# ---------------------------------------------------------------------------


def qol_td_lambda_targets(
    dataset: Mapping[str, Any], values: torch.Tensor, lam: float
) -> torch.Tensor:
    """TD(lambda) QoL-years targets bootstrapped through the bounded head.

    ``values`` are the head's decoded raw QoL-years predictions (detached) for
    every dataset row. Per (world, camp) trajectory, backward in turn order:
    ``G_i = r_i + gamma_i [(1 - lam) V(s_{i+1}) + lam G_{i+1}]``, with the
    state after the last row valued by its declared tail (the randomized tail
    estimate, or zero at a terminal/truncated-zero cutoff). ``lam = 1`` is the
    Monte Carlo return of ``labor_bc.discounted_returns`` exactly.
    """
    from .labor_bc import (
        CRITIC_DISCOUNT_YEARS,
        RETURN_TAIL_RANDOMIZED,
        TAIL_METHOD_FIELD,
        TAIL_RANDOMIZED,
        TAIL_TERMINAL_EXACT,
        TAIL_VALUE_PREFIX,
        TRAJECTORY_FIELD,
        elapsed_years,
        return_tail_method,
    )

    if not 0.0 <= float(lam) <= 1.0:
        raise ValueError("TD lambda must be in [0, 1]")
    n = int(dataset["row_world"].shape[0])
    values = values.reshape(-1).double().tolist()
    if len(values) != n:
        raise ValueError("TD(lambda) needs one value per dataset row")
    randomized = return_tail_method(dataset) == RETURN_TAIL_RANDOMIZED
    reward = dataset["reward_qol_years"].reshape(-1).double().tolist()
    elapsed = elapsed_years(dataset).reshape(-1).tolist()
    terminal = dataset["transition_terminal"].reshape(-1).tolist()
    turns = dataset["row_turn"].reshape(-1).tolist()
    if randomized:
        codes = dataset[TAIL_METHOD_FIELD].reshape(-1).tolist()
        tails = dataset[TAIL_VALUE_PREFIX + "qol_years"].reshape(-1).double().tolist()
    # Same trajectory keys as labor_bc.discounted_returns: a learner-state teacher
    # continuation (row_trajectory >= 0) is its own trajectory within its (world, camp),
    # so it is never chained onto the teacher trajectory or another continuation.
    trajectories = (
        dataset[TRAJECTORY_FIELD].reshape(-1).tolist()
        if TRAJECTORY_FIELD in dataset
        else [-1] * n
    )
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, (world, camp, trajectory) in enumerate(
        zip(
            dataset["row_world"].reshape(-1).tolist(),
            dataset["row_camp"].reshape(-1).tolist(),
            trajectories,
            strict=True,
        )
    ):
        groups[(int(world), int(camp), int(trajectory))].append(index)
    target = torch.zeros(n, dtype=torch.float64)
    for indices in groups.values():
        indices.sort(key=lambda i: turns[i])
        last = indices[-1]
        tail = 0.0
        if randomized and codes[last] in (TAIL_TERMINAL_EXACT, TAIL_RANDOMIZED):
            tail = tails[last]
        g_next, v_next = tail, tail
        for i in reversed(indices):
            gamma = math.exp(-float(elapsed[i]) / CRITIC_DISCOUNT_YEARS)
            if terminal[i]:
                g = reward[i]
            else:
                g = reward[i] + gamma * ((1.0 - lam) * v_next + lam * g_next)
            target[i] = g
            g_next, v_next = g, values[i]
    return target.to(torch.float32)


def _qol_trajectories(dataset: Mapping[str, Any]) -> list[list[int]]:
    """Row indices per (world, camp, row_trajectory), each sorted by ``row_turn``.

    The same keys and order as :func:`qol_td_lambda_targets` and
    ``labor_bc.discounted_returns``.
    """
    from .labor_bc import TRAJECTORY_FIELD

    n = int(dataset["row_world"].shape[0])
    turns = dataset["row_turn"].reshape(-1).tolist()
    trajectories = (
        dataset[TRAJECTORY_FIELD].reshape(-1).tolist()
        if TRAJECTORY_FIELD in dataset
        else [-1] * n
    )
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, (world, camp, trajectory) in enumerate(
        zip(
            dataset["row_world"].reshape(-1).tolist(),
            dataset["row_camp"].reshape(-1).tolist(),
            trajectories,
            strict=True,
        )
    ):
        groups[(int(world), int(camp), int(trajectory))].append(index)
    out = []
    for indices in groups.values():
        indices.sort(key=lambda i: turns[i])
        out.append(indices)
    return out


def qol_tail_terms(dataset: Mapping[str, Any], lam: float) -> dict[str, torch.Tensor]:
    """Per-row tail value and its coefficient in the TD(lambda) QoL target.

    ``tail`` is the trajectory's declared tail (zero without one), ``has_tail`` marks rows
    of trajectories closed by a terminal-exact or randomized tail and ``coefficient`` is
    c_i with G_i = (terms without the tail) + c_i * tail: c_last = gamma_last,
    c_i = lam * gamma_i * c_{i+1}, and zero at or before a terminal row. ``lam = 1``
    gives the Monte Carlo return's discount from row i to the cutoff state.
    """
    from .labor_bc import (
        CRITIC_DISCOUNT_YEARS,
        RETURN_TAIL_RANDOMIZED,
        TAIL_METHOD_FIELD,
        TAIL_RANDOMIZED,
        TAIL_TERMINAL_EXACT,
        TAIL_VALUE_PREFIX,
        elapsed_years,
        return_tail_method,
    )

    n = int(dataset["row_world"].shape[0])
    coefficient = torch.zeros(n, dtype=torch.float64)
    tail_value = torch.zeros(n, dtype=torch.float64)
    has_tail = torch.zeros(n, dtype=torch.bool)
    if return_tail_method(dataset) != RETURN_TAIL_RANDOMIZED:
        return {"coefficient": coefficient, "tail": tail_value, "has_tail": has_tail}
    codes = dataset[TAIL_METHOD_FIELD].reshape(-1).tolist()
    tails = dataset[TAIL_VALUE_PREFIX + "qol_years"].reshape(-1).double().tolist()
    elapsed = elapsed_years(dataset).reshape(-1).tolist()
    terminal = dataset["transition_terminal"].reshape(-1).tolist()
    for indices in _qol_trajectories(dataset):
        last = indices[-1]
        if codes[last] not in (TAIL_TERMINAL_EXACT, TAIL_RANDOMIZED):
            continue
        c_next = 1.0
        for position, i in enumerate(reversed(indices)):
            gamma = math.exp(-float(elapsed[i]) / CRITIC_DISCOUNT_YEARS)
            if terminal[i]:
                c = 0.0
            elif position == 0:
                c = gamma  # (1 - lam) tail + lam tail
            else:
                c = float(lam) * gamma * c_next
            coefficient[i] = c
            tail_value[i] = tails[last]
            has_tail[i] = True
            c_next = c
    return {"coefficient": coefficient, "tail": tail_value, "has_tail": has_tail}


def qol_tail_weight(
    dataset: Mapping[str, Any], target: torch.Tensor, lam: float
) -> torch.Tensor:
    """Share of each row's target value carried by the tail: c_i tail_i / G_i in [0, 1]
    (zero without a tail or at G_i = 0). ``target`` is one G_i per dataset row."""
    terms = qol_tail_terms(dataset, lam)
    target = target.reshape(-1).double()
    part = terms["coefficient"] * terms["tail"]
    weight = torch.where(target > 0, part / target.clamp_min(1e-300), torch.zeros_like(part))
    return torch.where(terms["has_tail"], weight.clamp(0.0, 1.0), torch.zeros_like(part))


def qol_td_admissibility(
    dataset: Mapping[str, Any],
    values: torch.Tensor,
    lam: float,
    *,
    head_max: float,
    weight: float,
    discount_years: float,
    rows: torch.Tensor | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Per-row admissible upper bound U_i of the TD(lambda) QoL targets, and the bug checks.

    U_i is the backward recursion of :func:`qol_td_lambda_targets` with the head maximum
    (scale * W) in place of every bootstrapped V and max(tail, head maximum) after the last
    row: ``U_i = r_i + gamma_i [(1 - lam) V_max + lam U_{i+1}]`` (``U_i = r_i`` at a
    terminal row). It never falls below the target while rewards, discounts and the head
    are what they claim to be, and it admits correct randomized-roulette overshoot (single
    tail estimates above W). Those claims are checked separately: rewards outside
    [0, elapsed], a recursion discount other than exp(-elapsed / ``discount_years``) (the
    head's declared tau) and decoded head values outside [0, head maximum], each with
    relative slack ``tolerance``. ``rows`` restricts the returned bound and every count
    to those rows (the fit rows); the result is what ``TDDriftMonitor.update`` takes as
    ``admissibility``. Diagnostics (never gating): rows whose tail exceeds W and the
    largest tail / W.
    """
    from .labor_bc import (
        CRITIC_DISCOUNT_YEARS,
        RETURN_TAIL_RANDOMIZED,
        TAIL_METHOD_FIELD,
        TAIL_RANDOMIZED,
        TAIL_TERMINAL_EXACT,
        TAIL_VALUE_PREFIX,
        elapsed_years,
        return_tail_method,
    )

    if not 0.0 <= float(lam) <= 1.0:
        raise ValueError("TD lambda must be in [0, 1]")
    n = int(dataset["row_world"].shape[0])
    values = values.reshape(-1).double()
    if int(values.numel()) != n:
        raise ValueError("TD(lambda) admissibility needs one value per dataset row")
    head_max, weight, tol = float(head_max), float(weight), float(tolerance)
    randomized = return_tail_method(dataset) == RETURN_TAIL_RANDOMIZED
    reward_t = dataset["reward_qol_years"].reshape(-1).double()
    elapsed_t = elapsed_years(dataset).reshape(-1).double()
    reward = reward_t.tolist()
    elapsed = elapsed_t.tolist()
    terminal = dataset["transition_terminal"].reshape(-1).tolist()
    tail_t = torch.zeros(n, dtype=torch.float64)
    has_tail = torch.zeros(n, dtype=torch.bool)
    if randomized:
        codes = dataset[TAIL_METHOD_FIELD].reshape(-1).tolist()
        tails = dataset[TAIL_VALUE_PREFIX + "qol_years"].reshape(-1).double().tolist()
    upper = torch.zeros(n, dtype=torch.float64)
    for indices in _qol_trajectories(dataset):
        last = indices[-1]
        tail = 0.0
        if randomized and codes[last] in (TAIL_TERMINAL_EXACT, TAIL_RANDOMIZED):
            tail = tails[last]
            for i in indices:
                tail_t[i] = tail
                has_tail[i] = True
        tail_term = max(tail, head_max)
        u_next, v_next = tail_term, tail_term
        for i in reversed(indices):
            gamma = math.exp(-float(elapsed[i]) / CRITIC_DISCOUNT_YEARS)
            if terminal[i]:
                u = reward[i]
            else:
                u = reward[i] + gamma * ((1.0 - float(lam)) * v_next + float(lam) * u_next)
            upper[i] = u
            u_next, v_next = u, head_max
    select = torch.arange(n) if rows is None else rows.reshape(-1).long()
    r, dt = reward_t[select], elapsed_t[select]
    reward_bad = ~torch.isfinite(r) | (r < -tol * dt.abs()) | (r > dt * (1.0 + tol))
    used = torch.exp(-dt / CRITIC_DISCOUNT_YEARS)
    declared = torch.exp(-dt / float(discount_years))
    discount_bad = (
        ~torch.isfinite(dt) | (dt < 0) | ((used - declared).abs() > tol)
    )
    v = values[select]
    head_bad = ~torch.isfinite(v) | (v < -tol * weight) | (v > head_max * (1.0 + tol))
    tails_sel = tail_t[select][has_tail[select]]
    count = int(select.numel())
    above = int((tails_sel > weight).sum())
    return {
        "upper": upper[select],
        "reward_violations": int(reward_bad.sum()),
        "discount_violations": int(discount_bad.sum()),
        "head_violations": int(head_bad.sum()),
        "diagnostics": {
            "gating": False,
            "head_max_over_w": head_max / weight,
            "tail_rows": int(tails_sel.numel()),
            "tail_above_w_rows": above,
            "tail_above_w_share": above / count if count else 0.0,
            "max_tail_over_w": float(tails_sel.max()) / weight if tails_sel.numel() else None,
        },
    }


def early_decision_mask(dataset: Mapping[str, Any], early_decisions: int) -> torch.Tensor:
    """Rows among the first ``early_decisions`` decisions of each trajectory.

    Trajectory keys match ``labor_bc.discounted_returns`` and ``qol_td_lambda_targets``:
    (world, camp, row_trajectory), with row_trajectory = -1 when absent (then rows are
    ranked by ``row_turn`` within (world, camp)).
    """
    from .labor_bc import TRAJECTORY_FIELD

    n = int(dataset["row_world"].shape[0])
    turns = dataset["row_turn"].reshape(-1).tolist()
    trajectories = (
        dataset[TRAJECTORY_FIELD].reshape(-1).tolist()
        if TRAJECTORY_FIELD in dataset
        else [-1] * n
    )
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(
        zip(
            dataset["row_world"].reshape(-1).tolist(),
            dataset["row_camp"].reshape(-1).tolist(),
            trajectories,
            strict=True,
        )
    ):
        groups[(int(key[0]), int(key[1]), int(key[2]))].append(index)
    mask = torch.zeros(n, dtype=torch.bool)
    for indices in groups.values():
        indices.sort(key=lambda i: turns[i])
        for i in indices[: max(0, int(early_decisions))]:
            mask[i] = True
    return mask


def qol_saturation_report(
    task_critic: LaborTaskCritic,
    logits: torch.Tensor,
    *,
    target_fraction: torch.Tensor | None = None,
    early: torch.Tensor | None = None,
    thresholds: Any = None,
) -> dict[str, Any]:
    """Saturation monitor for the bounded QoL head: share of |z| > 6 and z quantiles.

    With ``target_fraction`` (G/W) and ``early`` the report adds the stratified
    components of ``labor_stability.saturation_components`` under ``components``.
    """
    z = logits.detach().reshape(-1).double().cpu()
    quantiles = (
        torch.quantile(z, torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0], dtype=torch.float64)).tolist()
        if z.numel()
        else []
    )
    report = {
        "output": task_critic.qol_head.output,
        "share_abs_z_gt_6": task_critic.qol_saturation(logits),
        "z_quantiles_0_1_50_99_100": quantiles,
        "rows": int(z.numel()),
    }
    if target_fraction is not None and early is not None:
        from .labor_stability import saturation_components

        report["components"] = saturation_components(
            z,
            target_fraction,
            early,
            thresholds=thresholds,
            head_scale=float(getattr(task_critic.qol_head, "scale", 1.0)),
        )
    return report


# ---------------------------------------------------------------------------
# Synthetic payloads (local smoke and schema example; never real data)
# ---------------------------------------------------------------------------


def synthetic_root_tensors(
    rows: int, core_width: int = 10, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Observation-shaped root tensors for smoke tests (not a bridge observation)."""
    from .labor_bridge import (
        ACTIVITY_ROWS,
        LABOR_ACTIVITY_FEATURE_WIDTH,
        LABOR_RESOURCE_FEATURE_WIDTH,
        LABOR_SPECIALTY_FEATURE_WIDTH,
        RESOURCE_ROWS,
        SPECIALTY_ROWS,
    )

    g = torch.Generator().manual_seed(seed)
    activity = torch.ones(rows, SPECIALTY_ROWS - 1, ACTIVITY_ROWS, dtype=torch.bool)
    tensors = {
        "core": torch.randn(rows, core_width, generator=g),
        "labor_personality": torch.full((rows, 2), 0.5),
        "labor_specialty_features": torch.randn(
            rows, SPECIALTY_ROWS, LABOR_SPECIALTY_FEATURE_WIDTH, generator=g
        ),
        "labor_specialty_present": torch.ones(rows, SPECIALTY_ROWS, dtype=torch.bool),
        "labor_activity_features": torch.randn(
            rows, (SPECIALTY_ROWS - 1) * ACTIVITY_ROWS, LABOR_ACTIVITY_FEATURE_WIDTH, generator=g
        ),
        "labor_activity_present": torch.ones(
            rows, (SPECIALTY_ROWS - 1) * ACTIVITY_ROWS, dtype=torch.bool
        ),
        "labor_resource_features": torch.randn(
            rows, RESOURCE_ROWS, LABOR_RESOURCE_FEATURE_WIDTH, generator=g
        ),
        "labor_resource_present": torch.ones(rows, RESOURCE_ROWS, dtype=torch.bool),
        "labor_specialty_mask": torch.ones(rows, SPECIALTY_ROWS, dtype=torch.bool),
        "labor_activity_mask": activity,
        "movement_mask": torch.ones(rows, contracts.MOVEMENT_SLOTS, dtype=torch.bool),
        "region_movement_mask": torch.ones(rows, contracts.REGION_MOVEMENT_SLOTS, dtype=torch.bool),
    }
    for name, set_rows, width in contracts.SETS:
        tensors[name] = torch.zeros(rows, set_rows, width)
        tensors[name + "_present"] = torch.zeros(rows, set_rows, dtype=torch.bool)
    return tensors


def _synthetic_action(rng: np.random.Generator, *, idle: bool = False, farmer: float = 0.0) -> dict:
    specialty = rng.dirichlet(np.ones(len(SPECIALTIES)))
    if idle:
        specialty = np.zeros(len(SPECIALTIES))
        specialty[0] = 1.0
    if farmer:
        specialty = specialty * (1 - farmer)
        specialty[SPECIALTIES.index("Farmer")] += farmer
    activity = rng.dirichlet(np.ones(len(ACTIVITIES)), size=ROLE_ROWS)
    if idle:
        activity = np.zeros((ROLE_ROWS, len(ACTIVITIES)))
        activity[:, 0] = 1.0
    sample = LaborPolicySample(
        tuple(float(x) for x in specialty), tuple(tuple(float(x) for x in r) for r in activity)
    )
    return {
        "schemaVersion": 12,
        "movement": -1 if idle else int(rng.choice(MOVEMENT_UNION_CODES)),
        "fission": int(rng.integers(contracts.FISSION_POLICIES)),
        "tradeValue": [float(x) for x in np.exp(rng.uniform(-1.0, 1.0, TRADE_GROUP_COUNT))],
        "tradeGate": [0 if idle else int(x) for x in rng.integers(0, 3, TRADE_GROUP_COUNT)],
        "tradeShare": 0.0 if idle else float(rng.uniform()),
        **sample.payload(),
    }


def synthetic_candidate_shard(
    *,
    worlds: Sequence[int] = (90100, 90101, 90102, 90103, 90104),
    roots_per_world: int = 8,
    candidates: int = 8,
    horizons: Sequence[int] = (1, 2, 3),
    streams: int = 3,
    noise: float = 0.01,
    seed: int = 0,
    core_width: int = 10,
) -> dict[str, Any]:
    """A schema-valid shard with a planted, action-dependent return structure.

    Each component's return is a fixed random linear function of the action
    vector plus a root offset and per-stream noise, so a correct pipeline can
    learn the within-root ranking. Every root has an all-idle control (worst
    on every task objective) and, at even roots, a delayed-benefit Farmer
    candidate and a bookkeeping twin of H12.
    """
    rng = np.random.default_rng([0x5359, int(seed)])
    direction = rng.normal(size=(len(COMPONENTS), ACTION_FEATURE_WIDTH))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    farmer_slot = SPECIALTIES.index("Farmer")
    roots, row = [], 0
    for world in worlds:
        for r in range(roots_per_world):
            actions = [_synthetic_action(rng) for _ in range(candidates - 1)]
            controls: list[str | None] = [None] * (candidates - 1)
            twins: list[int | None] = [None] * (candidates - 1)
            actions.append(_synthetic_action(rng, idle=True))
            controls.append("all-idle")
            twins.append(None)
            if r % 2 == 0:
                actions.append(_synthetic_action(rng, farmer=0.5))
                controls.append("delayed-benefit")
                twins.append(None)
                twin = copy.deepcopy(actions[0])
                twin["description"] = "bookkeeping-only twin of H12"
                actions.append(twin)
                controls.append(None)
                twins.append(0)
                twins[0] = len(actions) - 1
            offset = rng.normal(size=len(COMPONENTS))
            records = []
            for i, action in enumerate(actions):
                features = action_features(action).astype(np.float64)
                base = 2.0 * direction @ features
                if controls[i] == "all-idle":
                    base = np.array([-3.0, -3.0, 3.0, 3.0])
                if controls[i] == "delayed-benefit":
                    # Planted to beat H12 on both task objectives, costs equal.
                    base = 2.0 * direction @ action_features(actions[0]).astype(np.float64)
                    base[:2] += 1.0 + features[farmer_slot]
                if twins[i] is not None and i > twins[i]:
                    base = 2.0 * direction @ action_features(actions[twins[i]]).astype(np.float64)
                evaluations = []
                for h in horizons:
                    for s in range(streams + 1):
                        draw = rng.normal(scale=noise, size=len(COMPONENTS))
                        value = (base + offset) * h * 0.1 + draw
                        raw = [
                            float(value[c] * REFERENCE_SCALES[name])
                            for c, name in enumerate(COMPONENTS)
                        ]
                        evaluations.append(
                            {
                                "horizon": h,
                                "reached_horizon": h,
                                "terminated": False,
                                "stream": s,
                                "returns": raw,
                                "step_years": [contracts.TURN_YEARS] * h,
                                "se": None,
                            }
                        )
                records.append(
                    {
                        "index": i,
                        "family": "h12" if i == 0 else "transfer",
                        "is_h12": i == 0,
                        "digest": hashlib.sha256(
                            json.dumps(action, sort_keys=True).encode()
                        ).hexdigest(),
                        "action": action,
                        "control": controls[i],
                        "bookkeeping_twin": twins[i],
                        "evaluations": evaluations,
                    }
                )
            roots.append(
                {
                    "root_id": f"{world}:{r}:synthetic",
                    "world": int(world),
                    "turn": r,
                    "camp": 1,
                    "society": 1,
                    "source": "h12",
                    "tensor_row": row,
                    "personality": [0.5, 0.5],
                    "full_evaluation": r % 5 == 0,
                    "pair_sd": {
                        str(h): [noise * REFERENCE_SCALES[n] for n in COMPONENTS] for h in horizons
                    },
                    "candidates": records,
                }
            )
            row += 1
    return {
        "schema": CANDIDATE_PAYLOAD_SCHEMA,
        "producer": {"kind": "synthetic-smoke", "seed": int(seed)},
        "component_order": list(COMPONENTS),
        "returns_units": RETURNS_UNITS,
        "discount_years": contracts.HORIZON_YEARS,
        "specialty_ids": list(SPECIALTIES),
        "activity_ids": list(ACTIVITIES),
        "trade_groups": list(contracts.TRADE_GROUPS),
        "tensors": synthetic_root_tensors(row, core_width, seed),
        "privileged": None,
        "roots": roots,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_critics(checkpoint: Path) -> tuple[LaborTaskCritic, LaborCostCritics]:
    from .labor_training import _models_from_checkpoint

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _, _, task, costs, _ = _models_from_checkpoint(state)
    for module in (task, costs):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return task, costs


def _load_actor(checkpoint: Path) -> nn.Module:
    """The checkpoint's actor, trainable, so the A fit can prove it receives no gradient."""
    from .labor_training import _models_from_checkpoint

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return _models_from_checkpoint(state)[1]


def argument_parser() -> argparse.ArgumentParser:
    from .labor_stability import add_stability_arguments

    parser = argparse.ArgumentParser(description="Fit the labor dueling advantage head (B)")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="fit A on candidate-payload shards (fitting worlds only)")
    fit.add_argument(
        "--checkpoint", type=Path, required=True, help="frozen labor critic checkpoint"
    )
    fit.add_argument("--payloads", type=Path, nargs="+", required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=50)
    fit.add_argument("--batch-roots", type=int, default=32)
    fit.add_argument("--lr", type=float, default=3e-4)
    fit.add_argument("--holdout-fraction", type=float, default=0.2)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument(
        "--horizons",
        default=None,
        help="per-objective horizons, e.g. 'qol_years=1,2,3;population_years=all' "
        "(default: QoL 1-3 decisions, every other objective all horizons)",
    )
    add_stability_arguments(fit)
    from .labor_device import add_device_argument

    add_device_argument(fit)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from .labor_device import cpu_copy, device_record, setup
    from .labor_value_calibration import labor_critic_digest

    args = argument_parser().parse_args(argv)
    device = setup(getattr(args, "device", "cpu"))
    task, costs = _load_critics(args.checkpoint)
    task.to(device)  # frozen critics; the frozen actor stays on CPU (not in the graph)
    costs.to(device)
    candidates = load_candidate_shards(args.payloads)
    config = AdvantageFitConfig(
        epochs=args.epochs,
        batch_roots=args.batch_roots,
        lr=args.lr,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        horizons=AdvantageFitConfig.parse_horizons(args.horizons),
    )
    from .labor_stability import StabilityAbort, stability_config_from_args

    try:
        head, summary = fit_advantage_head(
            task,
            costs,
            candidates,
            config,
            stability=stability_config_from_args(args),
            frozen_actor=_load_actor(args.checkpoint),
        )
    except StabilityAbort as error:
        abort = args.out.with_suffix(args.out.suffix + ".stability-abort.json")
        abort.parent.mkdir(parents=True, exist_ok=True)
        abort.write_text(json.dumps(error.summary, indent=2, sort_keys=True), encoding="utf-8")
        raise
    if device.type != "cpu":
        summary["device"] = device_record(device)
    head.to("cpu")  # device-agnostic artifact
    payload = advantage_checkpoint_payload(
        head,
        critic_digest=labor_critic_digest(task, costs),
        config=config,
        fit=summary,
        task_context=task_context_record(task),
    )
    payload["state_dict"] = cpu_copy(payload["state_dict"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    shown = {k: v for k, v in summary.items() if k not in ("loss_history", "stability")}
    print(json.dumps(shown, indent=2))
    if "stability" in summary:
        print(f"stability admitted: {summary['stability']['admitted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
