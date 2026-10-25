"""Opt-in Model 9 labor/retry model contract.

This module is deliberately separate from :mod:`model` and :mod:`contracts`.
Historical Model 9 actors and checkpoints do not acquire new inputs, heads, or
defaults by importing it.  The labor action encoder is added only after the
native producer publishes its exact ordered specialty/activity registries.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from . import contracts
from .labor_bridge import (
    LABOR_ACTIVITY_FEATURE_WIDTH,
    LABOR_REGISTRY_DIGEST,
    LABOR_RESOURCE_FEATURE_WIDTH,
    LABOR_SPECIALTY_FEATURE_WIDTH,
    LaborAttempt,
    LaborCatalog,
    LaborFeedback,
    LaborPolicySample,
    LaborProducerContract,
    labor_observation_batch,
)
from .labor_personality import (
    DEFAULT_PAIR,
    LaborPersonalityAssignment,
    LaborPersonalityConditioning,
    WorldPersonalities,
)
from .model import (
    BIN_HIGH,
    BIN_LOW,
    BINS,
    BOUNDED_FEATURE_CONTRACT,
    MASKED,
    TRADE_GATE_START,
    TRADE_LOG_STD_START,
    TRADE_SHARE_START,
    Normalizer,
    load_checkpoint_normalizer,
    normalizer_feature_contract,
    symexp,
    symlog,
    to_tensors,
)

LABOR_ARTIFACT_KIND = "model9-labor-retry-checkpoint"
LABOR_ARCHITECTURE = "model9-labor-retry-v1"
LABOR_MODEL_INPUT_CONTRACT = "model9-bounded-labor-feedback-v1"
LABOR_MODEL_CONTRACT_SCHEMA = 1

LABOR_OBJECTIVES = ("qol_years", "population_years")
LABOR_COSTS = ("feasibility", "settlement_guard")
LABOR_COST_REFERENCE_SCALES = {name: 1.0 for name in LABOR_COSTS}
LABOR_OBJECTIVE_REFERENCE_SCALES = {
    "qol_years": contracts.OBJECTIVE_REFERENCE_SCALES["qol_years"],
    "population_years": contracts.OBJECTIVE_REFERENCE_SCALES["population_years"],
}
DEFAULT_LABOR_PERSONALITY = {name: 0.5 for name in LABOR_OBJECTIVES}


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


# Versioned task-critic value representation (brief section 17.2). The
# historical clamp [-2, 9] with 255 two-hot bins (shared with Model 9) is the
# default and is never serialized, so existing checkpoints and contracts keep
# their exact bytes. Any other support is a declared, versioned representation
# change recorded in the checkpoint's labor model contract.
LABOR_VALUE_SUPPORT_VERSION = "labor-task-symlog-two-hot-support-v2"
LABOR_VALUE_SUPPORT_TRANSFORM = "symlog-of-return-over-reference-scale"


@dataclass(frozen=True)
class LaborValueSupport:
    """Clamp and bin count of the two-hot task-critic target ``symlog(G / s)``.

    Targets are clamped to ``[low, high]`` in symlog space and decoded as
    ``s * sum_j p_j symexp(b_j)`` with ``b_j = linspace(low, high, bins)``.
    Narrowing ``high`` removes decodable tail mass (``symexp(9)`` is about
    8102 reference scales); it is a representation change, not a tuning knob,
    so a checkpoint's support is fixed by its contract.
    """

    low: float = BIN_LOW
    high: float = BIN_HIGH
    bins: int = BINS

    def __post_init__(self) -> None:
        low = _finite_number(self.low, "value support low")
        high = _finite_number(self.high, "value support high")
        if isinstance(self.bins, bool) or not isinstance(self.bins, int) or self.bins < 3:
            raise ValueError("value support bins must be an integer >= 3")
        if not low < 0.0 < high:
            raise ValueError("value support must satisfy low < 0 < high in symlog space")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def is_default(self) -> bool:
        return (self.low, self.high, self.bins) == (BIN_LOW, BIN_HIGH, BINS)

    def bin_values(self) -> torch.Tensor:
        return torch.linspace(self.low, self.high, self.bins)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": LABOR_VALUE_SUPPORT_VERSION,
            "transform": LABOR_VALUE_SUPPORT_TRANSFORM,
            "low": self.low,
            "high": self.high,
            "bins": self.bins,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any] | None) -> LaborValueSupport:
        if state is None:
            return cls()
        if not isinstance(state, Mapping) or set(state) != {
            "version",
            "transform",
            "low",
            "high",
            "bins",
        }:
            raise ValueError("labor value support fields are malformed")
        if (
            state["version"] != LABOR_VALUE_SUPPORT_VERSION
            or state["transform"] != LABOR_VALUE_SUPPORT_TRANSFORM
        ):
            raise ValueError("labor value support version is unsupported")
        support = cls(state["low"], state["high"], state["bins"])
        if support.is_default:
            raise ValueError("the default value support is implicit and must not be serialized")
        return support

    @classmethod
    def parse(cls, text: str | None) -> LaborValueSupport:
        """Parse a ``LOW,HIGH,BINS`` command-line value (``None`` = default)."""
        if text is None:
            return cls()
        parts = [part.strip() for part in str(text).split(",")]
        if len(parts) != 3:
            raise ValueError("value support must be LOW,HIGH,BINS")
        return cls(float(parts[0]), float(parts[1]), int(parts[2]))


# Declared task-critic support for new campaign-r5 checkpoints (intervention
# brief 2026-10-26, item 1; audit candidate from ``labor_critic_support``):
# symlog clamp [-0.5, 2.5] with 255 two-hot bins. ``labor_training initialize
# --campaign-r5`` selects it. It is serialized in the checkpoint contract like
# any non-default support; old checkpoints, which omit the field, keep the
# implicit historical [-2, 9] default and their exact bytes.
LABOR_CAMPAIGN_R5_VALUE_SUPPORT = LaborValueSupport(-0.5, 2.5, 255)


@dataclass(frozen=True)
class LaborPersonality:
    """Declared fixed weighting over the two terminal task objectives.

    These are conditioning/combination weights, not learned reward weights.
    Values must already form a simplex; this boundary never silently
    normalizes a malformed declaration.
    """

    qol_years: float = DEFAULT_LABOR_PERSONALITY["qol_years"]
    population_years: float = DEFAULT_LABOR_PERSONALITY["population_years"]

    def __post_init__(self) -> None:
        values = tuple(
            _finite_number(getattr(self, name), f"personality {name}") for name in LABOR_OBJECTIVES
        )
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("labor personality weights must be in [0, 1]")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("labor personality weights must sum to one")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LaborPersonality:
        if not isinstance(value, Mapping) or set(value) != set(LABOR_OBJECTIVES):
            raise ValueError(f"labor personality fields must be exactly {list(LABOR_OBJECTIVES)}")
        return cls(
            **{
                name: _finite_number(value[name], f"personality {name}")
                for name in LABOR_OBJECTIVES
            }
        )

    def state_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in LABOR_OBJECTIVES}

    def tensor(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        return torch.tensor(
            [getattr(self, name) for name in LABOR_OBJECTIVES], dtype=torch.float32, device=device
        )

    def combine(self, objective_returns: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if set(objective_returns) != set(LABOR_OBJECTIVES):
            raise ValueError(f"labor objective values must be exactly {list(LABOR_OBJECTIVES)}")
        first = objective_returns[LABOR_OBJECTIVES[0]]
        if not isinstance(first, torch.Tensor) or not first.is_floating_point():
            raise ValueError("labor objective values must be floating tensors")
        result = torch.zeros_like(first)
        for name in LABOR_OBJECTIVES:
            value = objective_returns[name]
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_floating_point()
                or value.shape != first.shape
                or value.device != first.device
                or not torch.isfinite(value).all()
            ):
                raise ValueError("labor objective tensors must have matching finite shapes/devices")
            result = result + (getattr(self, name) / LABOR_OBJECTIVE_REFERENCE_SCALES[name]) * value
        return result


@dataclass(frozen=True)
class LaborModelContract:
    """Strict metadata embedded in every new labor/retry checkpoint.

    Native schema/registry identity is intentionally *not* guessed here.  The
    final checkpoint wrapper must add a separately validated producer contract
    once the native specialty/activity registry is emitted.  This record only
    prevents a legacy or differently-conditioned model from being loaded as a
    labor model.
    """

    personality: LaborPersonality = LaborPersonality()
    # Manifest-declared personality distribution and FiLM bound
    # (docs/LABOR_PERSONALITY.md). The default (balanced, unbounded) is not
    # serialized, so default checkpoints keep their pre-FiLM contract bytes.
    conditioning: LaborPersonalityConditioning = field(
        default_factory=LaborPersonalityConditioning
    )
    # Task-critic value support (brief section 17.2). The historical default
    # is not serialized, so default checkpoints keep their contract bytes.
    value_support: LaborValueSupport = field(default_factory=LaborValueSupport)

    def state_dict(self) -> dict[str, Any]:
        state = self._base_state()
        if not self.conditioning.is_default:
            state["personalityConditioning"] = self.conditioning.state_dict()
        if not self.value_support.is_default:
            state["criticValueSupport"] = self.value_support.state_dict()
        return state

    def _base_state(self) -> dict[str, Any]:
        return {
            "schemaVersion": LABOR_MODEL_CONTRACT_SCHEMA,
            "artifactKind": LABOR_ARTIFACT_KIND,
            "architecture": LABOR_ARCHITECTURE,
            "modelInputContract": LABOR_MODEL_INPUT_CONTRACT,
            "featureContract": BOUNDED_FEATURE_CONTRACT,
            "objectives": list(LABOR_OBJECTIVES),
            "objectiveReferenceScales": dict(LABOR_OBJECTIVE_REFERENCE_SCALES),
            "criticTargetTransform": "raw-objective-return-divided-by-frozen-reference-scale",
            "laborLatentGauge": "explicit-idle-coordinate-zero",
            "personality": self.personality.state_dict(),
            "costCritics": list(LABOR_COSTS),
            "costReferenceScales": dict(LABOR_COST_REFERENCE_SCALES),
            "costUnits": "dimensionless-normalized-cost-return",
            "registryDigest": LABOR_REGISTRY_DIGEST,
            "nativeProducerContractRequired": True,
            "retainedNonLaborHeads": [
                "movement",
                "fission",
                "tradeValue",
                "tradeGate",
                "tradeShare",
            ],
            "retiredLegacyHeads": [
                "strategic",
                "mix",
                "service",
                "clothing",
                "shelter",
                "commitment",
            ],
            "retryResampledHeads": ["laborSpecialtyAllocation", "laborActivityAllocation"],
            "retryFrozenHeads": ["movement", "fission", "tradeValue", "tradeGate", "tradeShare"],
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> LaborModelContract:
        template = cls().state_dict()
        if not isinstance(state, Mapping) or set(state) - {
            "personalityConditioning",
            "criticValueSupport",
        } != set(template):
            raise ValueError(f"labor model contract fields must be exactly {sorted(template)}")
        for key in (
            "schemaVersion",
            "artifactKind",
            "architecture",
            "modelInputContract",
            "featureContract",
            "objectives",
            "objectiveReferenceScales",
            "costCritics",
            "registryDigest",
            "nativeProducerContractRequired",
            "retainedNonLaborHeads",
            "retiredLegacyHeads",
            "retryResampledHeads",
            "retryFrozenHeads",
            "criticTargetTransform",
            "laborLatentGauge",
            "costReferenceScales",
            "costUnits",
        ):
            if state[key] != template[key]:
                raise ValueError(f"labor model contract {key} is unsupported")
        return cls(
            LaborPersonality.from_mapping(state["personality"]),
            LaborPersonalityConditioning.from_state(state.get("personalityConditioning")),
            LaborValueSupport.from_state(state.get("criticValueSupport")),
        )


def labor_checkpoint_contract(
    state: Mapping[str, Any], *, expected_layout: LaborEncoderLayout | None = None
) -> LaborModelContract:
    """Read the opt-in contract, rejecting legacy and partially tagged states."""
    if not isinstance(state, Mapping) or "labor_model_contract" not in state:
        raise ValueError("checkpoint is not an explicit labor/retry artifact")
    if state.get("artifact_kind") != LABOR_ARTIFACT_KIND:
        raise ValueError("checkpoint artifact kind is not labor/retry")
    contract = LaborModelContract.from_state(state["labor_model_contract"])
    if state.get("feature_contract") != BOUNDED_FEATURE_CONTRACT:
        raise ValueError("labor checkpoint requires the bounded feature contract")
    if state.get("model_input_contract") != LABOR_MODEL_INPUT_CONTRACT:
        raise ValueError("labor checkpoint input contract is missing or incompatible")
    LaborProducerContract.from_state(state.get("native_producer_contract"))
    layout = LaborEncoderLayout.from_state(state.get("labor_encoder_layout"))
    if expected_layout is not None and layout != expected_layout:
        raise ValueError("labor checkpoint encoder layout differs from the executing model")
    return contract


def labor_checkpoint_value_support(state: Mapping[str, Any]) -> LaborValueSupport:
    """The task-critic value support declared by a labor checkpoint's contract."""
    if not isinstance(state, Mapping) or "labor_model_contract" not in state:
        raise ValueError("checkpoint is not an explicit labor/retry artifact")
    return LaborModelContract.from_state(state["labor_model_contract"]).value_support


@dataclass(frozen=True)
class LaborEncoderLayout:
    """Feature widths from one reviewed producer/tensor contract.

    Row counts are intentionally absent: specialty, activity, and resource
    counts are runtime token dimensions bound by the registry digest.
    """

    core_input_width: int
    specialty_feature_width: int
    activity_feature_width: int
    resource_feature_width: int

    def __post_init__(self) -> None:
        for name in (
            "core_input_width",
            "specialty_feature_width",
            "activity_feature_width",
            "resource_feature_width",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def state_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in (
                "core_input_width",
                "specialty_feature_width",
                "activity_feature_width",
                "resource_feature_width",
            )
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> LaborEncoderLayout:
        names = {
            "core_input_width",
            "specialty_feature_width",
            "activity_feature_width",
            "resource_feature_width",
        }
        if not isinstance(state, Mapping) or set(state) != names:
            raise ValueError(f"labor encoder layout fields must be exactly {sorted(names)}")
        layout = cls(**{name: state[name] for name in names})
        expected = {
            "specialty_feature_width": LABOR_SPECIALTY_FEATURE_WIDTH,
            "activity_feature_width": LABOR_ACTIVITY_FEATURE_WIDTH,
            "resource_feature_width": LABOR_RESOURCE_FEATURE_WIDTH,
        }
        for name, width in expected.items():
            if getattr(layout, name) != width:
                raise ValueError(f"labor encoder {name} differs from the reviewed tensor contract")
        return layout


def labor_policy_tensors(
    villages: Sequence[Mapping[str, Any]],
    normalizer: Normalizer,
    *,
    device: torch.device | str = "cpu",
    personality: LaborPersonality | WorldPersonalities | Sequence[LaborPersonality] | None = None,
    retry_responses: Sequence[Mapping[str, Any] | None] | None = None,
    rejected_attempts: Sequence[LaborAttempt | None] | None = None,
) -> tuple[
    tuple[tuple[LaborCatalog, LaborFeedback | None, LaborAttempt | None], ...],
    dict[str, torch.Tensor],
]:
    """Authoritative contract-13 observation to actor/critic tensor path.

    Legacy core/entity inputs use the reviewed bounded normalizer. Labor token
    features come only from the strict producer-bound catalog/feedback parser.
    Immediate rejection context is accepted through ``retry_responses`` and is
    never substituted for historical post-execution feedback.
    """
    if normalizer_feature_contract(normalizer) != BOUNDED_FEATURE_CONTRACT:
        raise ValueError("labor policy inputs require the bounded normalization contract")
    from .bridge import observation_batch

    if (
        not isinstance(villages, Sequence)
        or isinstance(villages, (str, bytes))
        or not villages
        or any(
            not isinstance(row, Mapping) or not isinstance(row.get("observation"), Mapping)
            for row in villages
        )
    ):
        raise ValueError("labor policy tensor input must be nonempty bridge village rows")
    observations = [row["observation"] for row in villages]
    contexts, labor_batch = labor_observation_batch(
        observations, retry_responses, rejected_attempts=rejected_attempts
    )
    tensors = to_tensors(observation_batch(villages), normalizer, str(device))
    for name, values in labor_batch.items():
        dtype = torch.bool if values.dtype == bool else torch.float32
        tensors[name] = torch.as_tensor(values, dtype=dtype, device=device)
    tensors["labor_personality"] = labor_personality_rows(villages, personality, device=device)
    return contexts, tensors


def labor_personality_rows(
    villages: Sequence[Mapping[str, Any]],
    personality: LaborPersonality | WorldPersonalities | Sequence[LaborPersonality] | None,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Each village row's personality (qol_years, population_years) as float32.

    ``personality`` is one fixed personality for every row (the historical
    path, also used when ``None``), a :class:`WorldPersonalities` lineage
    resolver from a manifest-declared distribution, or one personality per row.
    """
    if personality is None or isinstance(personality, LaborPersonality):
        declared = LaborPersonality() if personality is None else personality
        return declared.tensor(device=device).expand(len(villages), -1).clone()
    if isinstance(personality, WorldPersonalities):
        pairs = personality.resolve(villages)
        rows = [LaborPersonality(*pair) for pair in pairs]
    elif isinstance(personality, Sequence) and not isinstance(personality, (str, bytes)):
        rows = list(personality)
    else:
        raise ValueError("labor personality must be fixed, per-row or a world resolver")
    if len(rows) != len(villages) or any(not isinstance(row, LaborPersonality) for row in rows):
        raise ValueError("labor per-row personalities must align with the village rows")
    return torch.stack([row.tensor(device=device) for row in rows])


def labor_checkpoint_personality(
    state: Mapping[str, Any],
) -> LaborPersonality | LaborPersonalityAssignment:
    """The checkpoint's declared personality: its fixed personality, or its
    distribution's assignment at the checkpoint iteration.

    Balanced runs and older checkpoints (no conditioning field) return the
    fixed contract personality, which reproduces their historical behavior
    exactly. Only a non-balanced distribution returns an assignment.
    """
    # Strict contract validation happens when the models load; this reads only
    # the two personality fields, as the historical fixed-personality read did.
    contract_state = state["labor_model_contract"]
    fixed = LaborPersonality.from_mapping(contract_state["personality"])
    conditioning = LaborPersonalityConditioning.from_state(
        contract_state.get("personalityConditioning")
    )
    if conditioning.alpha is None:
        return fixed
    if fixed != LaborPersonality():
        raise ValueError("a labor personality distribution requires the balanced reference")
    return conditioning.assignment(int(state.get("iteration", 0)))


def labor_personality_conditioning(
    personality: LaborPersonality | LaborPersonalityAssignment | None,
    film_bound: float | None = None,
) -> tuple[LaborPersonality, LaborPersonalityConditioning]:
    """Split a run personality into the contract's (reference, conditioning) pair."""
    if isinstance(personality, LaborPersonalityAssignment):
        conditioning = personality.conditioning
        if film_bound is not None and conditioning.film_bound != film_bound:
            raise ValueError("labor FiLM bound differs from the declared conditioning")
        return LaborPersonality(), conditioning
    fixed = LaborPersonality() if personality is None else personality
    return fixed, LaborPersonalityConditioning(film_bound=film_bound)


def set_labor_film_bound(bound: float | None, *modules: nn.Module) -> None:
    """Set the optional tanh FiLM bound on every labor encoder inside ``modules``."""
    if bound is not None and (not math.isfinite(float(bound)) or float(bound) <= 0):
        raise ValueError("labor FiLM bound must be finite and positive")
    for module in modules:
        for child in module.modules():
            if isinstance(child, LaborObservationEncoder):
                child.film_bound = None if bound is None else float(bound)


def labor_film_extent(module: nn.Module, tensors: Mapping[str, torch.Tensor]) -> float:
    """Largest |gamma| or |beta| any labor FiLM layer in ``module`` applies on this batch."""
    extents = [
        child.film_extent(tensors)
        for child in module.modules()
        if isinstance(child, LaborObservationEncoder)
    ]
    return max(extents) if extents else 0.0


class LaborObservationEncoder(nn.Module):
    """Permutation-equivariant token encoder for legacy and labor inputs.

    New specialty/activity/resource tokens use shared projectors and no
    ordinal embeddings. Stable identity and physical semantics must therefore
    be present in their reviewed numeric feature rows; a new ID is not an
    untrained output parameter. Legacy entity sets retain their established
    fixed-row positional embeddings.
    """

    def __init__(
        self,
        layout: LaborEncoderLayout,
        width: int = 256,
        token_width: int = 128,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if token_width % heads:
            raise ValueError("labor token width must be divisible by attention heads")
        self.layout = layout
        self.legacy_project = nn.ModuleDict(
            {name: nn.Linear(columns, token_width) for name, _, columns in contracts.SETS}
        )
        self.legacy_kind = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(token_width)) for name, _, _ in contracts.SETS}
        )
        self.legacy_position = nn.ModuleDict(
            {name: nn.Embedding(rows, token_width) for name, rows, _ in contracts.SETS}
        )
        self.specialty_project = nn.Linear(layout.specialty_feature_width, token_width)
        self.activity_project = nn.Linear(layout.activity_feature_width, token_width)
        self.resource_project = nn.Linear(layout.resource_feature_width, token_width)
        self.specialty_kind = nn.Parameter(torch.zeros(token_width))
        self.activity_kind = nn.Parameter(torch.zeros(token_width))
        self.resource_kind = nn.Parameter(torch.zeros(token_width))
        self.owner_to_activity = nn.Linear(token_width, token_width, bias=False)
        self.activities_to_specialty = nn.Linear(token_width, token_width, bias=False)
        self.global_owner_to_activity = nn.Linear(token_width, token_width, bias=False)
        self.global_village_to_activity = nn.Linear(token_width, token_width, bias=False)
        self.village = nn.Linear(layout.core_input_width + len(LABOR_OBJECTIVES), token_width)
        local_layer = nn.TransformerEncoderLayer(
            token_width,
            heads,
            dim_feedforward=2 * token_width,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        global_layer = nn.TransformerEncoderLayer(
            token_width,
            heads,
            dim_feedforward=2 * token_width,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.local_activity_attention = nn.TransformerEncoder(
            local_layer, layers, enable_nested_tensor=False
        )
        self.attention = nn.TransformerEncoder(global_layer, layers, enable_nested_tensor=False)
        self.local_activity_norm = nn.LayerNorm(token_width)
        self.activity_output_norm = nn.LayerNorm(token_width)
        self.norm = nn.LayerNorm(token_width)
        self.trunk = nn.Sequential(
            nn.Linear(token_width, width),
            nn.SiLU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )
        self.width = width
        for embedding in self.legacy_position.values():
            nn.init.normal_(embedding.weight, std=0.02)
        # FiLM personality conditioning (docs/LABOR_PERSONALITY.md), ported from
        # the contract-12 TokenEncoder. The condition is c = p - (0.5, 0.5). A
        # bias-free, zero-initialised linear map of c gives a per-channel scale
        # gamma and shift beta, h -> h * (1 + gamma) + beta, applied before every
        # local-activity attention layer, every global attention layer and each
        # trunk linear layer. At c = 0, and in any checkpoint saved before FiLM
        # (the load hook fills zero weights), it is exactly the identity. They
        # are built under a forked CPU RNG and zero-filled, so the global torch
        # RNG stream, and therefore every historical parameter's initial value
        # for a given seed (including heads built after this encoder), is
        # unchanged.
        objectives = len(LABOR_OBJECTIVES)
        with torch.random.fork_rng(devices=[]):
            self.film_local = nn.ModuleList(
                nn.Linear(objectives, 2 * token_width, bias=False) for _ in range(layers)
            )
            self.film_global = nn.ModuleList(
                nn.Linear(objectives, 2 * token_width, bias=False) for _ in range(layers)
            )
            self.film_trunk = nn.ModuleList(
                [
                    nn.Linear(objectives, 2 * token_width, bias=False),
                    nn.Linear(objectives, 2 * width, bias=False),
                ]
            )
        for film in self._films():
            nn.init.zeros_(film.weight)
        self.register_buffer(
            "default_personality",
            torch.tensor([DEFAULT_PAIR], dtype=torch.float32),
            persistent=False,
        )
        self._register_load_state_dict_pre_hook(self._identity_film)
        # Optional bound on gamma/beta (bound * tanh(x / bound)); off by default.
        self.film_bound: float | None = None

    def _films(self) -> list[nn.Linear]:
        return [*self.film_local, *self.film_global, *self.film_trunk]

    def _identity_film(self, state_dict, prefix, *args) -> None:
        """Checkpoints saved before FiLM load with identity (zero) FiLM weights."""
        for name, parameter in self.named_parameters():
            if name.startswith("film_") and prefix + name not in state_dict:
                state_dict[prefix + name] = torch.zeros_like(parameter)

    def film_condition(self, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return tensors["labor_personality"] - self.default_personality

    def film_extent(self, tensors: Mapping[str, torch.Tensor]) -> float:
        """The largest |gamma| or |beta| any FiLM layer applies on this batch."""
        condition = self.film_condition(tensors)
        with torch.no_grad():
            return float(
                max(self._bounded(film(condition)).abs().max() for film in self._films())
            )

    def _bounded(self, values: torch.Tensor) -> torch.Tensor:
        if self.film_bound:
            return self.film_bound * torch.tanh(values / self.film_bound)
        return values

    def _film(
        self, hidden: torch.Tensor, film: nn.Linear, condition: torch.Tensor
    ) -> torch.Tensor:
        scale, shift = self._bounded(film(condition)).chunk(2, dim=-1)
        if hidden.dim() == 3:
            scale, shift = scale.unsqueeze(1), shift.unsqueeze(1)
        return hidden * (1 + scale) + shift

    def _attend(
        self,
        encoder: nn.TransformerEncoder,
        films: nn.ModuleList | None,
        values: torch.Tensor,
        padding: torch.Tensor,
        condition: torch.Tensor | None,
    ) -> torch.Tensor:
        if films is None or condition is None:  # pre-FiLM reference path (parity tests)
            return encoder(values, src_key_padding_mask=padding)
        for layer, film in zip(encoder.layers, films, strict=True):
            values = layer(self._film(values, film, condition), src_key_padding_mask=padding)
        return values

    def _trunk(self, hidden: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if condition is None:  # pre-FiLM reference path (parity tests)
            return self.trunk(hidden)
        hidden = self.trunk[:3](self._film(hidden, self.film_trunk[0], condition))
        return self.trunk[3:](self._film(hidden, self.film_trunk[1], condition))

    @staticmethod
    def _tokens(
        tensors: Mapping[str, torch.Tensor],
        feature: str,
        present: str,
        width: int,
        rows: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values, mask = tensors.get(feature), tensors.get(present)
        if (
            not isinstance(values, torch.Tensor)
            or not values.is_floating_point()
            or values.ndim != 3
            or values.shape[-1] != width
            or not torch.isfinite(values).all()
            or not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.shape != values.shape[:2]
            or (rows is not None and values.shape[1] != rows)
        ):
            raise ValueError(f"{feature}/{present} labor token tensors are malformed")
        if (values.masked_select(~mask.unsqueeze(-1).expand_as(values)) != 0).any():
            raise ValueError(f"{feature} missing rows must be zero")
        return values, mask

    def forward(
        self, tensors: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        return self.encode(tensors, film=True)

    def encode(
        self, tensors: Mapping[str, torch.Tensor], *, film: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """``film=False`` is the exact pre-FiLM computation, kept as a parity reference."""
        core, personality = tensors.get("core"), tensors.get("labor_personality")
        if (
            not isinstance(core, torch.Tensor)
            or not core.is_floating_point()
            or core.ndim != 2
            or core.shape[1] != self.layout.core_input_width
            or not torch.isfinite(core).all()
        ):
            raise ValueError("labor core input is malformed")
        if (
            not isinstance(personality, torch.Tensor)
            or personality.shape != (len(core), 2)
            or not personality.is_floating_point()
            or not torch.isfinite(personality).all()
            or (personality < 0).any()
            or not torch.allclose(
                personality.sum(dim=1),
                torch.ones(len(core), device=core.device),
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise ValueError("labor personality must be a two-objective simplex")
        condition = personality - self.default_personality if film else None
        tokens = [self.village(torch.cat([core, personality], dim=1)).unsqueeze(1)]
        present = [torch.ones(len(core), 1, dtype=torch.bool, device=core.device)]
        for name, rows, columns in contracts.SETS:
            values, mask = self._tokens(tensors, name, name + "_present", columns, rows)
            positions = self.legacy_position[name].weight.unsqueeze(0)
            tokens.append(self.legacy_project[name](values) + self.legacy_kind[name] + positions)
            present.append(mask)
        specialty, specialty_present = self._tokens(
            tensors,
            "labor_specialty_features",
            "labor_specialty_present",
            self.layout.specialty_feature_width,
        )
        activity, activity_present = self._tokens(
            tensors,
            "labor_activity_features",
            "labor_activity_present",
            self.layout.activity_feature_width,
        )
        resources, resource_present = self._tokens(
            tensors,
            "labor_resource_features",
            "labor_resource_present",
            self.layout.resource_feature_width,
        )
        specialty_tokens = self.specialty_project(specialty) + self.specialty_kind
        role_count = specialty_tokens.shape[1] - 1
        if role_count <= 0 or activity.shape[1] % role_count:
            raise ValueError("activity tokens cannot be assigned to their owning specialties")
        activities_per_role = activity.shape[1] // role_count
        activity_tokens = (self.activity_project(activity) + self.activity_kind).view(
            len(core), role_count, activities_per_role, -1
        )
        activity_mask = activity_present.view(len(core), role_count, activities_per_role)
        owner = specialty_tokens[:, 1:].unsqueeze(2)
        local_input = activity_tokens + self.owner_to_activity(owner)
        local_shape = local_input.shape
        flat_local = local_input.reshape(-1, activities_per_role, local_shape[-1])
        flat_mask = activity_mask.reshape(-1, activities_per_role)
        safe_mask = flat_mask.clone()
        no_rows = ~safe_mask.any(dim=1)
        safe_mask[no_rows, 0] = True
        flat_local = flat_local.clone()
        flat_local[no_rows, 0] = 0.0
        local_output = self._attend(
            self.local_activity_attention,
            self.film_local,
            flat_local,
            ~safe_mask,
            None if condition is None else condition.repeat_interleave(role_count, dim=0),
        )
        local_output = self.local_activity_norm(local_output).reshape(local_shape)
        local_output = local_output * activity_mask.unsqueeze(-1)
        divisor = activity_mask.sum(dim=2, keepdim=True).clamp_min(1).to(local_output.dtype)
        owned_summary = local_output.sum(dim=2) / divisor
        specialty_tokens = torch.cat(
            [
                specialty_tokens[:, :1],
                specialty_tokens[:, 1:] + self.activities_to_specialty(owned_summary),
            ],
            dim=1,
        )
        resource_tokens = self.resource_project(resources) + self.resource_kind
        # Activities attend only within their owner. Global attention therefore
        # scales with village/specialty/resource tokens, not every pair of
        # activities, while role/activity permutations remain equivariant.
        tokens.extend([specialty_tokens, resource_tokens])
        present.extend([specialty_present, resource_present])
        lengths = [part.shape[1] for part in tokens]
        joined, mask = torch.cat(tokens, dim=1), torch.cat(present, dim=1)
        output = self.norm(
            self._attend(self.attention, self.film_global, joined, ~mask, condition)
        )
        output = output * mask.unsqueeze(-1)
        offsets = []
        cursor = 0
        for length in lengths:
            offsets.append((cursor, cursor + length))
            cursor += length
        specialty_tokens = output[:, offsets[-2][0] : offsets[-2][1]]
        resource_tokens = output[:, offsets[-1][0] : offsets[-1][1]]
        legacy_rows = {
            name: output[:, offsets[index + 1][0] : offsets[index + 1][1]]
            for index, (name, _, _) in enumerate(contracts.SETS)
        }
        contextual_activity = self.activity_output_norm(
            local_output
            + self.global_owner_to_activity(specialty_tokens[:, 1:].unsqueeze(2))
            + self.global_village_to_activity(output[:, :1].unsqueeze(2))
        )
        contextual_activity = contextual_activity * activity_mask.unsqueeze(-1)
        return (
            self._trunk(output[:, 0], condition),
            specialty_tokens,
            contextual_activity.reshape(len(core), role_count * activities_per_role, -1),
            resource_tokens,
            legacy_rows,
        )


class LaborTaskCritic(nn.Module):
    """One shared task-state encoder and one two-hot head per terminal objective.

    ``support`` is the declared, versioned value representation (clamp and bin
    count); the default reproduces the historical [-2, 9] x 255 critic exactly.
    """

    def __init__(
        self,
        layout: LaborEncoderLayout,
        width: int = 256,
        token_width: int = 128,
        *,
        support: LaborValueSupport | None = None,
    ) -> None:
        super().__init__()
        self.support = LaborValueSupport() if support is None else support
        if not isinstance(self.support, LaborValueSupport):
            raise ValueError("task critic support must be a LaborValueSupport")
        self.encoder = LaborObservationEncoder(layout, width, token_width)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(width, self.support.bins) for name in LABOR_OBJECTIVES}
        )
        self.register_buffer("bins", self.support.bin_values())

    def check_support(self) -> None:
        """Reject a bins buffer (e.g. a loaded one) that differs from the declared support."""
        if self.bins.shape != (self.support.bins,) or not torch.equal(
            self.bins.detach().cpu(), self.support.bin_values()
        ):
            raise ValueError("task critic bins differ from its declared value support")

    def support_diagnostics(
        self, objective: str, logits: torch.Tensor, observed_max: float
    ) -> dict[str, torch.Tensor]:
        """Per-row decoded tail terms in reference-scale (normalized) units.

        ``observed_max`` is the largest observed normalized target ``G / s``.
        ``upper_mass`` is the probability above it, ``upper_contribution`` the
        decoded value those bins add and ``upper_excess`` the part of it beyond
        placing the same mass at ``observed_max`` (bias no target can justify).
        """
        if objective not in LABOR_OBJECTIVES:
            raise ValueError(f"unknown labor objective {objective!r}")
        probabilities = torch.softmax(logits.double(), dim=-1)
        values = symexp(self.bins.double())
        upper = values > float(observed_max)
        upper_probability = probabilities * upper
        return {
            "upper_mass": upper_probability.sum(dim=-1),
            "upper_contribution": (upper_probability * values).sum(dim=-1),
            "upper_excess": (upper_probability * (values - float(observed_max))).sum(dim=-1),
            "top_bin_mass": probabilities[..., -1],
        }

    def forward(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden, _, _, _, _ = self.encoder(tensors)
        return {name: self.heads[name](hidden) for name in LABOR_OBJECTIVES}

    def expectation_normalized(self, logits: torch.Tensor) -> torch.Tensor:
        return (torch.softmax(logits, dim=-1) * symexp(self.bins)).sum(dim=-1)

    def expectation(self, objective: str, logits: torch.Tensor) -> torch.Tensor:
        if objective not in LABOR_OBJECTIVES:
            raise ValueError(f"unknown labor objective {objective!r}")
        return self.expectation_normalized(logits) * LABOR_OBJECTIVE_REFERENCE_SCALES[objective]

    def two_hot(self, objective: str, values: torch.Tensor) -> torch.Tensor:
        if objective not in LABOR_OBJECTIVES:
            raise ValueError(f"unknown labor objective {objective!r}")
        if (
            not isinstance(values, torch.Tensor)
            or values.ndim != 1
            or not torch.isfinite(values).all()
        ):
            raise ValueError("labor objective targets must be a finite vector")
        support = self.support
        normalized = values / LABOR_OBJECTIVE_REFERENCE_SCALES[objective]
        transformed = symlog(normalized).clamp(support.low, support.high)
        step = (support.high - support.low) / (support.bins - 1)
        position = (transformed - support.low) / step
        low = position.floor().long().clamp(0, support.bins - 2)
        upper = (position - low.float()).clamp(0, 1)
        target = torch.zeros(values.shape[0], support.bins, device=values.device)
        target.scatter_(1, low.unsqueeze(1), (1 - upper).unsqueeze(1))
        target.scatter_add_(1, (low + 1).unsqueeze(1), upper.unsqueeze(1))
        return target


class LaborCostCritics(nn.Module):
    """Separate feasibility and settlement-guard value functions.

    Each cost owns its encoder and scalar head.  There is no shared cost trunk
    through which one constraint can silently stand in for the other.
    """

    def __init__(
        self, layout: LaborEncoderLayout, width: int = 128, token_width: int = 128
    ) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict(
            {name: LaborObservationEncoder(layout, width, token_width) for name in LABOR_COSTS}
        )
        self.heads = nn.ModuleDict({name: nn.Linear(width, 1) for name in LABOR_COSTS})

    def forward(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: self.heads[name](self.encoders[name](tensors)[0]).squeeze(-1)
            for name in LABOR_COSTS
        }


def labor_checkpoint_payload(
    actor: LaborActor,
    task_critic: LaborTaskCritic,
    cost_critics: LaborCostCritics,
    normalizer: Normalizer,
    layout: LaborEncoderLayout,
    *,
    personality: LaborPersonality | LaborPersonalityAssignment | None = None,
) -> dict[str, Any]:
    """Build the strict model/normalizer portion of a new labor checkpoint.

    Optimizer, scheduler, rollout order, RNG, and run metadata belong to the
    trainer checkpoint layer. This function deliberately does not synthesize
    them or accept a historical artifact as a new labor checkpoint.
    """
    if normalizer_feature_contract(normalizer) != BOUNDED_FEATURE_CONTRACT:
        raise ValueError("labor checkpoints require a bounded normalizer")
    contract = LaborModelContract(
        *labor_personality_conditioning(personality, actor.encoder.film_bound),
        value_support=task_critic.support,
    )
    return {
        "artifact_kind": LABOR_ARTIFACT_KIND,
        "feature_contract": BOUNDED_FEATURE_CONTRACT,
        "model_input_contract": LABOR_MODEL_INPUT_CONTRACT,
        "labor_model_contract": contract.state_dict(),
        "native_producer_contract": LaborProducerContract().state_dict(),
        "labor_encoder_layout": layout.state_dict(),
        "normalizer": normalizer.state_dict(),
        "actor": actor.state_dict(),
        "task_critic": task_critic.state_dict(),
        "cost_critics": cost_critics.state_dict(),
    }


def load_labor_checkpoint_models(
    state: Mapping[str, Any],
    actor: LaborActor,
    task_critic: LaborTaskCritic,
    cost_critics: LaborCostCritics,
    *,
    expected_layout: LaborEncoderLayout,
) -> tuple[LaborModelContract, Normalizer]:
    """Strictly restore one new labor model bundle without legacy upgrading."""
    if not isinstance(state, dict):
        raise ValueError("labor checkpoint must be a dictionary")
    contract = labor_checkpoint_contract(state, expected_layout=expected_layout)
    normalizer = load_checkpoint_normalizer(
        state, expected_feature_contract=BOUNDED_FEATURE_CONTRACT
    )
    for name in ("actor", "task_critic", "cost_critics"):
        if not isinstance(state.get(name), Mapping):
            raise ValueError(f"labor checkpoint is missing strict {name} tensors")
    if contract.value_support != task_critic.support:
        raise ValueError(
            "labor checkpoint declares a different task-critic value support; construct "
            "the critic with labor_checkpoint_value_support(state)"
        )
    actor.load_state_dict(state["actor"], strict=True)
    task_critic.load_state_dict(state["task_critic"], strict=True)
    task_critic.check_support()
    cost_critics.load_state_dict(state["cost_critics"], strict=True)
    set_labor_film_bound(contract.conditioning.film_bound, actor, task_critic, cost_critics)
    return contract, normalizer


def _masks(tensors: Mapping[str, torch.Tensor], batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    specialty = tensors.get("labor_specialty_mask")
    activity = tensors.get("labor_activity_mask")
    if (
        not isinstance(specialty, torch.Tensor)
        or specialty.dtype != torch.bool
        or specialty.ndim != 2
        or specialty.shape[0] != batch
        or specialty.shape[1] < 1
    ):
        raise ValueError("labor_specialty_mask must be boolean [batch,specialties]")
    if (
        not isinstance(activity, torch.Tensor)
        or activity.dtype != torch.bool
        or activity.ndim != 3
        or activity.shape[0] != batch
        or activity.shape[1] != specialty.shape[1] - 1
        or activity.shape[2] < 1
    ):
        raise ValueError("labor_activity_mask must be boolean [batch,roles,activities]")
    specialty_present = tensors.get("labor_specialty_present")
    activity_present = tensors.get("labor_activity_present")
    if (
        not isinstance(specialty_present, torch.Tensor)
        or specialty_present.dtype != torch.bool
        or specialty_present.shape != specialty.shape
        or (specialty & ~specialty_present).any()
    ):
        raise ValueError("labor specialty actions require present specialty tokens")
    if (
        not isinstance(activity_present, torch.Tensor)
        or activity_present.dtype != torch.bool
        or activity_present.shape != (batch, activity.shape[1] * activity.shape[2])
        or (activity & ~activity_present.view_as(activity)).any()
    ):
        raise ValueError("labor activity actions require present activity tokens")
    if not specialty[:, 0].all():
        raise ValueError("the global idle specialty must be legal")
    role_present = specialty_present[:, 1:]
    activity_present_rows = activity_present.view_as(activity)
    # Current native roles are explicit even when unavailable: those rows keep
    # idle legal and advertise no non-idle action. A genuinely absent padded
    # role is different: it has no action or token coordinates at all.
    if (activity[:, :, 0] != role_present).any():
        raise ValueError("present role rows must preserve explicit idle; padded rows must be empty")
    if (activity[:, :, 1:] & ~specialty[:, 1:].unsqueeze(-1)).any():
        raise ValueError("masked specialties cannot advertise non-idle activities")
    if (activity_present_rows & ~role_present.unsqueeze(-1)).any():
        raise ValueError("padded specialties cannot advertise activity tokens")
    return specialty, activity


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != mask.shape or not torch.isfinite(logits).all():
        raise ValueError("labor logits/masks are malformed")
    has_legal = mask.any(dim=-1, keepdim=True)
    safe_mask = mask.clone()
    safe_mask[..., 0] |= ~has_legal.squeeze(-1)
    result = torch.softmax(logits.masked_fill(~safe_mask, -torch.inf), dim=-1)
    return torch.where(mask & has_legal, result, torch.zeros_like(result))


class LaborActor(nn.Module):
    """Specialty staffing and within-specialty activity distributions.

    The input is the future labor observation encoder's finite representation.
    This head does not infer output promises: activities are staffing/work-share
    requests and material scarcity may produce idle work or low receipts.
    """

    def __init__(
        self, layout: LaborEncoderLayout, width: int = 256, token_width: int = 128
    ) -> None:
        super().__init__()
        self.encoder = LaborObservationEncoder(layout, width, token_width)
        self.specialty = nn.Linear(token_width, 2)
        self.activity = nn.Linear(token_width, 2)
        self.stay = nn.Linear(width, 1)
        self.site_query = nn.Linear(width, token_width)
        self.region_keep = nn.Linear(width, 1)
        self.region_query = nn.Linear(width, token_width)
        self.fission = nn.Linear(width, contracts.FISSION_POLICIES)
        self.trade_token = nn.Linear(token_width, 2 + contracts.TRADE_GATES)
        self.trade_share = nn.Linear(width, 2)
        # Reviewed initial posture: native movement and conservative trade.
        with torch.no_grad():
            for layer in (self.region_keep, self.region_query):
                layer.weight.zero_()
                layer.bias.zero_()
            self.trade_token.weight.zero_()
            self.trade_token.bias.zero_()
            self.trade_token.bias[1] = TRADE_LOG_STD_START
            self.trade_token.bias[2 + contracts.TRADE_MAKE] = float(
                math.log(TRADE_GATE_START / (1 - TRADE_GATE_START) * (contracts.TRADE_GATES - 1))
            )
            self.trade_share.weight.zero_()
            self.trade_share.bias.zero_()
            self.trade_share.bias[0] = TRADE_SHARE_START
            self.trade_share.bias[1] = TRADE_LOG_STD_START

    def forward(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden, specialty_tokens, activity_tokens, _, legacy_rows = self.encoder(tensors)
        specialty_mask, activity_mask = _masks(tensors, len(specialty_tokens))
        if specialty_tokens.shape[:2] != specialty_mask.shape:
            raise ValueError("specialty token/mask registry differs")
        if activity_tokens.shape[1] != activity_mask.shape[1] * activity_mask.shape[2]:
            raise ValueError("activity tokens are not canonical specialty-major rows")
        specialty = self.specialty(specialty_tokens)
        activity = self.activity(activity_tokens).view(*activity_mask.shape, 2)
        sites = legacy_rows["sites"]
        movement_mask = tensors.get("movement_mask")
        if (
            not isinstance(movement_mask, torch.Tensor)
            or movement_mask.dtype != torch.bool
            or movement_mask.shape != (len(hidden), 1 + sites.shape[1])
        ):
            raise ValueError("labor movement mask differs from the site registry")
        site_scores = torch.einsum("bh,brh->br", self.site_query(hidden), sites) / math.sqrt(
            sites.shape[-1]
        )
        movement = torch.cat([self.stay(hidden), site_scores], dim=1).masked_fill(
            ~movement_mask, MASKED
        )
        regions = legacy_rows["regions"]
        region_mask = tensors.get("region_movement_mask")
        if (
            not isinstance(region_mask, torch.Tensor)
            or region_mask.dtype != torch.bool
            or region_mask.shape != (len(hidden), 2 + regions.shape[1])
        ):
            raise ValueError("labor region mask differs from the region registry")
        region_scores = torch.einsum("bh,brh->br", self.region_query(hidden), regions) / math.sqrt(
            regions.shape[-1]
        )
        regional = torch.cat([self.region_keep(hidden), region_scores], dim=1)
        union_mask = torch.cat([movement_mask, region_mask[:, 1:]], dim=1)
        if not union_mask.any(dim=1).all():
            raise ValueError("schema-12 movement union has no legal action")
        movement_union = torch.cat([movement, regional], dim=1).masked_fill(~union_mask, MASKED)
        movement_codes = torch.tensor(
            [
                -1,
                *range(contracts.MOVEMENT_SLOTS - 1),
                contracts.KEEP_INTENT,
                *(contracts.REGION_BASE + row for row in range(contracts.REGION_ROWS)),
            ],
            dtype=torch.long,
            device=hidden.device,
        )
        trade_rows = legacy_rows["trade"]
        if trade_rows.shape[1] != len(contracts.TRADE_GROUPS):
            raise ValueError("labor trade token registry differs from the native groups")
        trade = self.trade_token(trade_rows)
        trade_share = self.trade_share(hidden)
        return {
            "specialty_logits": specialty[..., 0].masked_fill(~specialty_mask, -30.0),
            "specialty_log_std": specialty[..., 1].clamp(-5.0, 1.0),
            "activity_logits": activity[..., 0].masked_fill(~activity_mask, -30.0),
            "activity_log_std": activity[..., 1].clamp(-5.0, 1.0),
            "movement_union_logits": movement_union,
            "movement_union_mask": union_mask,
            "movement_codes": movement_codes,
            "fission": self.fission(hidden),
            "trade_value_mean": trade[..., 0],
            "trade_value_log_std": trade[..., 1].clamp(-5.0, 1.0),
            "trade_gate": trade[..., 2:],
            "trade_share_mean": trade_share[:, 0],
            "trade_share_log_std": trade_share[:, 1].clamp(-5.0, 1.0),
        }


def deterministic_labor_policy(
    output: Mapping[str, torch.Tensor], tensors: Mapping[str, torch.Tensor]
) -> LaborPolicySample:
    """Decode exactly one row. Batch decoding remains tensor-native in PPO."""
    specialty_logits = output.get("specialty_logits")
    activity_logits = output.get("activity_logits")
    if not isinstance(specialty_logits, torch.Tensor) or specialty_logits.shape[0] != 1:
        raise ValueError("deterministic labor payload requires a one-row actor output")
    specialty_mask, activity_mask = _masks(tensors, 1)
    specialty = _masked_softmax(specialty_logits, specialty_mask)[0]
    activity = _masked_softmax(activity_logits, activity_mask)[0]
    return LaborPolicySample(
        tuple(float(value) for value in specialty.detach().cpu()),
        tuple(tuple(float(value) for value in row.detach().cpu()) for row in activity),
    )


def _nonlabor_output(
    output: Mapping[str, torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    movement = output.get("movement_union_logits")
    movement_mask = output.get("movement_union_mask")
    movement_codes = output.get("movement_codes")
    fission = output.get("fission")
    value_mean = output.get("trade_value_mean")
    value_log_std = output.get("trade_value_log_std")
    gate = output.get("trade_gate")
    share_mean = output.get("trade_share_mean")
    share_log_std = output.get("trade_share_log_std")
    if (
        not isinstance(movement, torch.Tensor)
        or movement.ndim != 2
        or not isinstance(movement_mask, torch.Tensor)
        or movement_mask.dtype != torch.bool
        or movement_mask.shape != movement.shape
        or not movement_mask.any(dim=1).all()
        or not isinstance(movement_codes, torch.Tensor)
        or movement_codes.dtype != torch.long
        or movement_codes.shape != (movement.shape[1],)
        or not isinstance(fission, torch.Tensor)
        or fission.shape != (movement.shape[0], contracts.FISSION_POLICIES)
        or not isinstance(value_mean, torch.Tensor)
        or value_mean.shape != (movement.shape[0], len(contracts.TRADE_GROUPS))
        or not isinstance(value_log_std, torch.Tensor)
        or value_log_std.shape != value_mean.shape
        or not isinstance(gate, torch.Tensor)
        or gate.shape != (movement.shape[0], len(contracts.TRADE_GROUPS), contracts.TRADE_GATES)
        or not isinstance(share_mean, torch.Tensor)
        or share_mean.shape != (movement.shape[0],)
        or not isinstance(share_log_std, torch.Tensor)
        or share_log_std.shape != share_mean.shape
        or any(
            not torch.isfinite(value).all()
            for value in (
                movement,
                fission,
                value_mean,
                value_log_std,
                gate,
                share_mean,
                share_log_std,
            )
        )
    ):
        raise ValueError("schema-12 retained non-labor actor output is malformed")
    return (
        movement,
        movement_mask,
        movement_codes,
        fission,
        value_mean,
        value_log_std,
        gate,
        torch.stack([share_mean, share_log_std], dim=-1),
    )


def _categorical_sample(logits: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    return torch.multinomial(flat, 1, generator=generator).reshape(logits.shape[:-1])


def deterministic_nonlabor_policy(output: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Retain the exact mode of every schema-12 non-labor head."""
    movement, movement_mask, codes, fission, value_mean, _, gate, share = _nonlabor_output(output)
    movement_index = movement.masked_fill(~movement_mask, -torch.inf).argmax(dim=1)
    return {
        "movement_index": movement_index,
        "movement": codes[movement_index],
        "fission": fission.argmax(dim=1),
        "trade_value_latent": value_mean,
        "trade_gate": gate.argmax(dim=2),
        "trade_share_latent": share[:, 0],
        "movement_mask": movement_mask.clone(),
        "movement_codes": codes.clone(),
    }


def sample_nonlabor_policy(
    output: Mapping[str, torch.Tensor], *, generator: torch.Generator
) -> dict[str, torch.Tensor]:
    """Sample retained non-labor heads using only the supplied generator."""
    movement, movement_mask, codes, fission, value_mean, value_log_std, gate, share = (
        _nonlabor_output(output)
    )
    movement_index = _categorical_sample(
        movement.masked_fill(~movement_mask, -torch.inf), generator=generator
    )
    fission_sample = _categorical_sample(fission, generator=generator)
    gate_sample = _categorical_sample(gate, generator=generator)
    value_noise = torch.randn(
        value_mean.shape, dtype=value_mean.dtype, device=value_mean.device, generator=generator
    )
    share_noise = torch.randn(
        share.shape[0], dtype=share.dtype, device=share.device, generator=generator
    )
    return {
        "movement_index": movement_index,
        "movement": codes[movement_index],
        "fission": fission_sample,
        "trade_value_latent": value_mean + value_noise * value_log_std.exp(),
        "trade_gate": gate_sample,
        "trade_share_latent": share[:, 0] + share_noise * share[:, 1].exp(),
        "movement_mask": movement_mask.clone(),
        "movement_codes": codes.clone(),
    }


_NONLABOR_SAMPLE_FIELDS = {
    "movement_index",
    "movement",
    "fission",
    "trade_value_latent",
    "trade_gate",
    "trade_share_latent",
    "movement_mask",
    "movement_codes",
}


def nonlabor_log_prob(
    output: Mapping[str, torch.Tensor], sample: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Score the one retained non-labor sample on the initial policy edge."""
    movement, movement_mask, codes, fission, value_mean, value_log_std, gate, share = (
        _nonlabor_output(output)
    )
    if not isinstance(sample, Mapping) or set(sample) != _NONLABOR_SAMPLE_FIELDS:
        raise ValueError("schema-12 non-labor sample fields are incomplete or unexpected")
    batch = movement.shape[0]
    movement_index, movement_code = sample["movement_index"], sample["movement"]
    fission_sample, value, gate_sample, share_value = (
        sample["fission"],
        sample["trade_value_latent"],
        sample["trade_gate"],
        sample["trade_share_latent"],
    )
    if (
        not isinstance(movement_index, torch.Tensor)
        or movement_index.dtype != torch.long
        or movement_index.shape != (batch,)
        or (movement_index < 0).any()
        or (movement_index >= movement.shape[1]).any()
        or not isinstance(movement_code, torch.Tensor)
        or movement_code.dtype != torch.long
        or movement_code.shape != (batch,)
        or not torch.equal(movement_code, codes[movement_index])
        or not isinstance(sample["movement_mask"], torch.Tensor)
        or not torch.equal(sample["movement_mask"], movement_mask)
        or not isinstance(sample["movement_codes"], torch.Tensor)
        or not torch.equal(sample["movement_codes"], codes)
        or not movement_mask.gather(1, movement_index[:, None]).all()
    ):
        raise ValueError("retained movement sample differs from the current legal union")
    if (
        not isinstance(fission_sample, torch.Tensor)
        or fission_sample.dtype != torch.long
        or fission_sample.shape != (batch,)
        or (fission_sample < 0).any()
        or (fission_sample >= contracts.FISSION_POLICIES).any()
        or not isinstance(value, torch.Tensor)
        or value.shape != value_mean.shape
        or not isinstance(gate_sample, torch.Tensor)
        or gate_sample.dtype != torch.long
        or gate_sample.shape != value_mean.shape
        or (gate_sample < 0).any()
        or (gate_sample >= contracts.TRADE_GATES).any()
        or not isinstance(share_value, torch.Tensor)
        or share_value.shape != (batch,)
        or not torch.isfinite(value).all()
        or not torch.isfinite(share_value).all()
    ):
        raise ValueError("retained schema-12 non-labor sample is malformed")
    movement_score = (
        torch.log_softmax(movement.masked_fill(~movement_mask, -torch.inf), dim=1)
        .gather(1, movement_index[:, None])
        .squeeze(1)
    )
    fission_score = torch.log_softmax(fission, dim=1).gather(1, fission_sample[:, None]).squeeze(1)
    gate_score = (
        torch.log_softmax(gate, dim=2).gather(2, gate_sample.unsqueeze(-1)).squeeze(-1).sum(dim=1)
    )
    normal_constant = 0.5 * math.log(2.0 * math.pi)
    value_score = (
        -0.5 * ((value - value_mean) / value_log_std.exp()) ** 2 - value_log_std - normal_constant
    ).sum(dim=1)
    share_score = (
        -0.5 * ((share_value - share[:, 0]) / share[:, 1].exp()) ** 2
        - share[:, 1]
        - normal_constant
    )
    result = {
        "movement": movement_score,
        "fission": fission_score,
        "trade_value": value_score,
        "trade_gate": gate_score,
        "trade_share": share_score,
    }
    result["joint"] = sum(result.values())
    return result


def nonlabor_entropy(output: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    movement, movement_mask, _, fission, _, value_log_std, gate, share = _nonlabor_output(output)

    def categorical(logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=-1)
        log_probabilities = torch.log_softmax(logits, dim=-1)
        return -torch.where(
            probabilities > 0, probabilities * log_probabilities, torch.zeros_like(probabilities)
        ).sum(dim=-1)

    constant = 0.5 * (1.0 + math.log(2.0 * math.pi))
    result = {
        "movement": categorical(movement.masked_fill(~movement_mask, -torch.inf)),
        "fission": categorical(fission),
        "trade_value": (value_log_std + constant).sum(dim=1),
        "trade_gate": categorical(gate).sum(dim=1),
        "trade_share": share[:, 1] + constant,
    }
    result["joint"] = sum(result.values())
    return result


def _labor_tensor_payload(sample: Mapping[str, torch.Tensor], index: int) -> dict[str, Any]:
    specialty, activity = sample.get("specialty_allocation"), sample.get("activity_allocation")
    if (
        not isinstance(specialty, torch.Tensor)
        or specialty.ndim != 2
        or not isinstance(activity, torch.Tensor)
        or activity.ndim != 3
        or activity.shape[0] != specialty.shape[0]
        or isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= specialty.shape[0]
    ):
        raise ValueError("labor sample cannot be serialized at the requested row")
    decoded = LaborPolicySample(
        tuple(float(value) for value in specialty[index].detach().cpu()),
        tuple(tuple(float(value) for value in row.detach().cpu()) for row in activity[index]),
    )
    return decoded.payload()


def schema12_retry_action(sample: Mapping[str, torch.Tensor], index: int = 0) -> dict[str, Any]:
    """Serialize only the two labor heads for a zero-time retry."""
    return _labor_tensor_payload(sample, index)


def schema12_initial_action(
    labor_sample: Mapping[str, torch.Tensor],
    nonlabor_sample: Mapping[str, torch.Tensor],
    index: int = 0,
) -> dict[str, Any]:
    """Serialize the exact complete schema-12 action; retired heads cannot leak in."""
    labor = _labor_tensor_payload(labor_sample, index)
    if not isinstance(nonlabor_sample, Mapping) or set(nonlabor_sample) != _NONLABOR_SAMPLE_FIELDS:
        raise ValueError("schema-12 non-labor sample fields are incomplete or unexpected")
    batch = nonlabor_sample["movement"].shape[0]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= batch:
        raise ValueError("non-labor sample cannot be serialized at the requested row")
    value = nonlabor_sample["trade_value_latent"][index]
    gate = nonlabor_sample["trade_gate"][index]
    trade_share = torch.sigmoid(nonlabor_sample["trade_share_latent"][index])
    low, high = contracts.TRADE_VALUE_RANGE
    return {
        "schemaVersion": 12,
        "movement": int(nonlabor_sample["movement"][index]),
        "fission": int(nonlabor_sample["fission"][index]),
        "tradeValue": torch.exp(value).clamp(low, high).detach().cpu().tolist(),
        "tradeGate": [int(item) for item in gate.detach().cpu()],
        "tradeShare": float(trade_share.detach().cpu()),
        **labor,
    }


def sample_labor_policy(
    output: Mapping[str, torch.Tensor],
    tensors: Mapping[str, torch.Tensor],
    *,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Sample gauge-fixed additive-log-ratio latents with a local generator.

    Explicit idle coordinate zero is the zero reference. Other legal
    coordinates are independent Gaussians. PPO retains these exact latents, so likelihood
    never has to invert a rounded serialized simplex.
    """
    specialty_logits, specialty_std = (
        output.get("specialty_logits"),
        output.get("specialty_log_std"),
    )
    activity_logits, activity_std = output.get("activity_logits"), output.get("activity_log_std")
    if any(
        not isinstance(value, torch.Tensor)
        for value in (specialty_logits, specialty_std, activity_logits, activity_std)
    ):
        raise ValueError("labor actor output is incomplete")
    batch = specialty_logits.shape[0]
    specialty_mask, activity_mask = _masks(tensors, batch)
    if specialty_std.shape != specialty_logits.shape or activity_std.shape != activity_logits.shape:
        raise ValueError("labor actor mean/std shapes differ")

    def draw(
        logits: torch.Tensor, log_std: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (mask[..., 1:].any(dim=-1) & ~mask[..., 0]).any():
            raise ValueError("labor ALR requires explicit idle whenever a row is active")
        reference = torch.zeros((*mask.shape[:-1], 1), dtype=torch.long, device=mask.device)
        mean = logits - logits[..., :1]
        other = mask.clone()
        other.scatter_(-1, reference, False)
        noise = torch.randn(
            logits.shape, dtype=logits.dtype, device=logits.device, generator=generator
        )
        latent = torch.where(other, mean + noise * log_std.exp(), torch.zeros_like(mean))
        return latent, _masked_softmax(latent, mask)

    specialty_latent, specialty_allocation = draw(specialty_logits, specialty_std, specialty_mask)
    activity_latent, activity_allocation = draw(activity_logits, activity_std, activity_mask)
    return {
        "specialty_allocation": specialty_allocation,
        "activity_allocation": activity_allocation,
        "specialty_latent": specialty_latent,
        "activity_latent": activity_latent,
        "specialty_mask": specialty_mask.clone(),
        "activity_mask": activity_mask.clone(),
    }


def _simplex_log_prob(
    logits: torch.Tensor,
    log_std: torch.Tensor,
    sample: torch.Tensor,
    latent: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if (
        logits.shape != log_std.shape
        or logits.shape != sample.shape
        or logits.shape != latent.shape
        or logits.shape != mask.shape
    ):
        raise ValueError("labor likelihood tensor shapes differ")
    if (
        not torch.isfinite(logits).all()
        or not torch.isfinite(log_std).all()
        or not torch.isfinite(sample).all()
        or not torch.isfinite(latent).all()
        or (sample < 0).any()
        or (sample.masked_select(~mask) != 0).any()
    ):
        raise ValueError("labor likelihood tensors are invalid")
    mass = sample.sum(dim=-1)
    expected_mass = mask.any(dim=-1).to(sample.dtype)
    if not torch.allclose(mass, expected_mass, rtol=0.0, atol=1e-6):
        raise ValueError("labor samples must be masked simplexes")
    # Additive log-ratios against the last legal coordinate.  A one-coordinate
    # (idle-only) row is deterministic and contributes zero likelihood.
    if (mask[..., 1:].any(dim=-1) & ~mask[..., 0]).any():
        raise ValueError("labor ALR requires explicit idle whenever a row is active")
    reference = torch.zeros((*mask.shape[:-1], 1), dtype=torch.long, device=mask.device)
    if not torch.allclose(sample, _masked_softmax(latent, mask), rtol=1e-6, atol=1e-7):
        raise ValueError("labor sample does not match its retained latent")
    mean = logits - logits[..., :1]
    other = mask.clone()
    other.scatter_(-1, reference, False)
    if (latent.masked_select(~other) != 0).any():
        raise ValueError("labor latent is not in the declared zero-reference gauge")
    std = log_std.exp()
    gaussian = -0.5 * ((latent - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
    # PPO scores the retained ALR latent density.  The simplex Jacobian is
    # action-only and cancels in the old/new likelihood ratio; adding it here
    # would also be wrong if it were counted more than once.
    return (gaussian * other).sum(dim=-1)


def labor_log_prob(
    output: Mapping[str, torch.Tensor],
    sample: Mapping[str, torch.Tensor],
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    specialty = sample.get("specialty_allocation")
    activity = sample.get("activity_allocation")
    specialty_latent = sample.get("specialty_latent")
    activity_latent = sample.get("activity_latent")
    logits = output.get("specialty_logits")
    if not isinstance(logits, torch.Tensor):
        raise ValueError("labor actor output is incomplete")
    specialty_mask, activity_mask = _masks(tensors, len(logits))
    if set(sample) != {
        "specialty_allocation",
        "activity_allocation",
        "specialty_latent",
        "activity_latent",
        "specialty_mask",
        "activity_mask",
    }:
        raise ValueError("labor sample fields are incomplete or unexpected")
    if not isinstance(sample["specialty_mask"], torch.Tensor) or not torch.equal(
        sample["specialty_mask"], specialty_mask
    ):
        raise ValueError("labor sample is bound to another specialty mask")
    if not isinstance(sample["activity_mask"], torch.Tensor) or not torch.equal(
        sample["activity_mask"], activity_mask
    ):
        raise ValueError("labor sample is bound to another activity mask")
    if (
        not isinstance(specialty, torch.Tensor)
        or not isinstance(activity, torch.Tensor)
        or not isinstance(specialty_latent, torch.Tensor)
        or not isinstance(activity_latent, torch.Tensor)
        or specialty.shape != specialty_mask.shape
        or activity.shape != activity_mask.shape
    ):
        raise ValueError("labor sample shapes differ from their masks")
    specialty_logp = _simplex_log_prob(
        logits, output["specialty_log_std"], specialty, specialty_latent, specialty_mask
    )
    activity_logp = _simplex_log_prob(
        output["activity_logits"],
        output["activity_log_std"],
        activity,
        activity_latent,
        activity_mask,
    ).sum(dim=1)
    return {
        "specialty": specialty_logp,
        "activity": activity_logp,
        "joint": specialty_logp + activity_logp,
    }


def labor_entropy(
    output: Mapping[str, torch.Tensor], tensors: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Entropy of the additive-log-ratio Gaussian latents used for sampling."""
    logits = output.get("specialty_logits")
    if not isinstance(logits, torch.Tensor):
        raise ValueError("labor actor output is incomplete")
    specialty_mask, activity_mask = _masks(tensors, len(logits))
    constant = 0.5 * (1.0 + math.log(2.0 * math.pi))

    def entropy(log_std: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if (mask[..., 1:].any(dim=-1) & ~mask[..., 0]).any():
            raise ValueError("labor ALR requires explicit idle whenever a row is active")
        reference = torch.zeros((*mask.shape[:-1], 1), dtype=torch.long, device=mask.device)
        other = mask.clone()
        other.scatter_(-1, reference, False)
        return ((log_std + constant) * other).sum(dim=-1)

    specialty = entropy(output["specialty_log_std"], specialty_mask)
    activity = entropy(output["activity_log_std"], activity_mask).sum(dim=1)
    return {"specialty": specialty, "activity": activity, "joint": specialty + activity}
