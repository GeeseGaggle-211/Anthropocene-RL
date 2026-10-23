"""Checkpoint-bound Gate 3/4 continuation collection.

This is a deliberately narrow runner for the reviewed recovered T3 endpoints.  It is
not a training, readiness, or gate-passage entry point.  The runner:

* resolves every historical ``latest.pt`` endpoint (and the T3 initialization critic)
  from the hash-bound Gate-1 DAG;
* loads actors through the reviewed recovered-contract-12 adapter and loads critics
  with an exact key/dtype/shape check;
* executes the sampled PPO policy with ``ppo.explore`` and a persistent CPU generator;
* uses exact in-process bridge forks, keeps PPO personality and mix-shaping history,
  and derives rewards from every completed transition before selecting opening camps;
* records baseline policy-value calibration separately from the paired, one-time
  development-budget sensitivity intervention; and
* stores compact values, returns, action receipts, and optional first/end economic rows,
  never full observation histories.

No checkpoint is selected from outcomes.  The world/regime/state/path grid and every
Russian-roulette stopping schedule are derived from the frozen protocol below before a
simulator result is observed.  Worlds are the independent units; regimes, states, camps,
paths, and stopping draws are nested measurements.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import exp, isfinite
from pathlib import Path
from time import monotonic
from typing import Any, Literal, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from . import contracts, ppo
from .bridge import DEFAULT_BRIDGE, Model9Bridge
from .evaluate import load_strict_state
from .gate1_production_adapter import load_recovered_contract12
from .gate34_continuation_collector import (
    CampOneStepValue,
    CampReward,
    ForkedAlternatives,
    FrozenGeometricSchedule,
    RandomizedTailEstimate,
    TailAssumptions,
    aggregate_camp_prefix,
    aggregate_ranking_prediction,
    bounded_geometric_tail_assumptions,
    combine_independent_path_returns,
    estimate_randomized_tail,
    freeze_geometric_schedule,
    raw_camp_reward_absolute_bound,
    shaped_nonnegative_component_absolute_bound,
    shaped_welfare_reward_absolute_bound,
)
from .model import Actor, Critic, Normalizer, VectorCritic, symexp, to_tensors
from .provenance import ProvenanceError, checkpoint_hash, digest

Json = dict[str, Any]
ObjectiveName = Literal["welfare", "population_years", "trade_utility", "qol_years"]

FROZEN_WORLD_SEEDS = tuple(range(50000, 50016))
FROZEN_REGIMES: tuple[tuple[str, int], ...] = (
    ("default", 0),
    ("farming", contracts.CAPABILITY_MASKS["farming"]),
    ("fishing", contracts.CAPABILITY_MASKS["fishing"]),
    ("herding", contracts.CAPABILITY_MASKS["herding"]),
)
FROZEN_DECISION_STEPS = (5, 15, 25)
FROZEN_PATH_REPLICATES = 4
FROZEN_PREFIX_TERMS = 25
FROZEN_TAIL_DRAWS = 16
FROZEN_TAIL_SURVIVAL = 0.95
FROZEN_TAIL_HARD_CAP = 200
FROZEN_ALTERNATIVES = ("sampled-policy", "development-budget-native-v1")
FROZEN_REWARD_SOURCE_SHA256 = "d195a6f152efc6ad152548998fb84d8d6bcc02bc32b97fef2e60adbfaaa017ca"
FROZEN_PPO_SOURCE_SHA256 = "75f842e88de735695c79af1417de355b51f4d270fda68fb67563f6f87d6f9f1e"
FROZEN_T2_SCRIPT_SHA256 = "42f0e1820d2d0080197490ce74f1f76ea3b14962838aa3d1873c14b635aa1993"
FROZEN_STAGE = "mixed"
FROZEN_INIT_LOG_STD_FLOOR = -2.0

TRAINED_ENDPOINT_IDS = tuple(
    [f"t3per-s{seed}" for seed in range(70, 75)]
    + [f"t3def-s{seed}" for seed in range(75, 80)]
    + [f"t3lo-s{seed}" for seed in range(80, 85)]
)
FROZEN_ENDPOINT_IDS = ("t3-initialization-clone", *TRAINED_ENDPOINT_IDS)


def _read_json(path: Path) -> Json:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"cannot read sealed JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProvenanceError(f"sealed JSON {path} must contain an object")
    return cast(Json, value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProvenanceError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _sealed_reference(base: Path, reference: object, *, kind: str) -> tuple[Path, Json]:
    if not isinstance(reference, dict):
        raise ProvenanceError(f"missing sealed {kind} reference")
    relative = reference.get("path")
    expected = _require_sha256(reference.get("sha256"), f"{kind} reference hash")
    if not isinstance(relative, str) or not relative:
        raise ProvenanceError(f"sealed {kind} reference lacks a path")
    path = (base / relative).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as error:
        raise ProvenanceError(f"sealed {kind} reference escapes its package") from error
    if checkpoint_hash(path) != expected:
        raise ProvenanceError(f"sealed {kind} bytes differ from their reference")
    value = _read_json(path)
    _validate_sealed(value, kind=kind)
    return path, value


def _validate_sealed(value: Json, *, kind: str) -> None:
    if (
        value.get("schema") != 1
        or value.get("kind") != kind
        or value.get("digest")
        != digest({key: item for key, item in value.items() if key != "digest"})
    ):
        raise ProvenanceError(f"{kind} is not a canonical schema-1 sealed artifact")


@dataclass(frozen=True)
class EndpointReceipt:
    """One exact Gate-1 DAG node and its reviewed loading dictionary."""

    node_id: str
    checkpoint: Path
    checkpoint_sha256: str
    artifact_metadata: Json
    resolved_semantics: Json
    loading: Json
    critic_typed_content_sha256: str
    arm_id: str | None
    optimizer_seed: int | None


@dataclass(frozen=True)
class EndpointRegistry:
    """The immutable all-endpoint receipt used even by a calibration-first run."""

    freeze_root: Path
    freeze_candidate_sha256: str
    artifact_dag_sha256: str
    adapter_protocol_sha256: str
    retrieval_root: Path
    bridge_path: Path
    bridge_sha256: str
    source_hashes: Mapping[str, str]
    endpoints: tuple[EndpointReceipt, ...]

    @property
    def registry_sha256(self) -> str:
        return _canonical_sha256(
            {
                "freezeCandidate": self.freeze_candidate_sha256,
                "artifactDag": self.artifact_dag_sha256,
                "adapterProtocol": self.adapter_protocol_sha256,
                "bridge": self.bridge_sha256,
                "sources": dict(self.source_hashes),
                "endpoints": [
                    {
                        "id": item.node_id,
                        "checkpoint": item.checkpoint_sha256,
                        "critic": item.critic_typed_content_sha256,
                        "semantics": item.resolved_semantics,
                        "loading": item.loading,
                    }
                    for item in self.endpoints
                ],
            }
        )


def _artifact_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ProvenanceError("Gate-1 artifact path is missing")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ProvenanceError("Gate-1 artifact path escapes retrieval root") from error
    return path


def load_endpoint_registry(
    freeze_root: Path,
    *,
    retrieval_root: Path | None = None,
    verify_checkpoint_bytes: bool = True,
) -> EndpointRegistry:
    """Resolve the all-15-plus-initialization dictionary from the sealed Gate-1 DAG.

    ``verify_checkpoint_bytes=False`` exists only for synthetic tests.  A collection
    call always requires full byte verification.
    """

    freeze_root = freeze_root.resolve()
    candidate_path = freeze_root / "freeze-candidate.json"
    candidate = _read_json(candidate_path)
    _validate_sealed(candidate, kind="gate1_production_freeze_candidate")
    proposals = candidate.get("proposals")
    if not isinstance(proposals, dict):
        raise ProvenanceError("freeze package lacks proposal references")
    dag_path, dag = _sealed_reference(
        freeze_root,
        proposals.get("artifact_dag"),
        kind="gate1_artifact_dag_proposal",
    )
    adapter_path, adapter = _sealed_reference(
        freeze_root,
        proposals.get("recovered_contract12_adapter"),
        kind="gate1_recovered_contract12_adapter_protocol",
    )
    declared_root = candidate.get("inputs", {}).get("retrieval_root")
    if retrieval_root is None:
        if not isinstance(declared_root, str) or not declared_root:
            raise ProvenanceError("freeze package lacks its retrieval root")
        retrieval_root = Path(declared_root)
    retrieval_root = retrieval_root.resolve()

    current_bridge = candidate.get("inputs", {}).get("current_bridge")
    if not isinstance(current_bridge, dict):
        raise ProvenanceError("freeze package lacks its current bridge identity")
    bridge_value = current_bridge.get("path")
    if not isinstance(bridge_value, str) or not bridge_value:
        raise ProvenanceError("freeze package current bridge path is invalid")
    bridge_path = Path(bridge_value).resolve()
    bridge_sha256 = _require_sha256(
        current_bridge.get("sha256"), "freeze package current bridge hash"
    )
    if checkpoint_hash(bridge_path) != bridge_sha256:
        raise ProvenanceError("current bridge bytes differ from the freeze package")

    allowed = adapter.get("allowed_checkpoints")
    if not isinstance(allowed, dict) or set(allowed) != set(FROZEN_ENDPOINT_IDS):
        raise ProvenanceError("adapter does not bind exactly the reviewed endpoint cohort")
    nodes = dag.get("nodes")
    if not isinstance(nodes, list):
        raise ProvenanceError("Gate-1 DAG lacks nodes")
    by_id = {
        node.get("id"): node
        for node in nodes
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }
    if any(node_id not in by_id for node_id in FROZEN_ENDPOINT_IDS):
        raise ProvenanceError("Gate-1 DAG omits a reviewed historical endpoint")

    source_root = retrieval_root / "producer"
    sources = {
        "reward": source_root / "src/anthropocene_rl/autonomous_v7/data.py",
        "ppo": source_root / "src/anthropocene_rl/autonomous_v7/ppo.py",
        "t2_script": source_root / "t2.sh",
    }
    expected_sources = {
        "reward": FROZEN_REWARD_SOURCE_SHA256,
        "ppo": FROZEN_PPO_SOURCE_SHA256,
        "t2_script": FROZEN_T2_SCRIPT_SHA256,
    }
    actual_sources = {name: checkpoint_hash(path) for name, path in sources.items()}
    if actual_sources != expected_sources:
        raise ProvenanceError("recovered T3 reward/PPO/schedule source bytes differ")

    receipts: list[EndpointReceipt] = []
    protocol_reference = proposals["recovered_contract12_adapter"]
    for node_id in FROZEN_ENDPOINT_IDS:
        node = cast(Json, by_id[node_id])
        artifact = node.get("artifact")
        loading = node.get("loading")
        semantics = node.get("historical_semantics")
        if not all(isinstance(value, dict) for value in (artifact, loading, semantics)):
            raise ProvenanceError(f"{node_id} lacks artifact/loading/semantics")
        artifact = cast(Json, artifact)
        loading = cast(Json, loading)
        semantics = cast(Json, semantics)
        if (
            loading.get("mode") != "reviewed-adapter"
            or loading.get("adapter") != "recovered-contract12-inactive-region"
            or loading.get("node_id") != node_id
            or loading.get("protocol") != protocol_reference
        ):
            raise ProvenanceError(f"{node_id} has another checkpoint loading receipt")
        expected_checkpoint = _require_sha256(artifact.get("sha256"), f"{node_id} checkpoint hash")
        if allowed.get(node_id) != expected_checkpoint:
            raise ProvenanceError(f"{node_id} differs from the adapter allow-list")
        checkpoint = _artifact_path(retrieval_root, artifact.get("path"))
        if verify_checkpoint_bytes and checkpoint_hash(checkpoint) != expected_checkpoint:
            raise ProvenanceError(f"{node_id} checkpoint bytes differ")
        metadata = artifact.get("metadata")
        components = artifact.get("components")
        if not isinstance(metadata, dict) or not isinstance(components, dict):
            raise ProvenanceError(f"{node_id} lacks artifact metadata/components")
        critic = components.get("critic")
        if not isinstance(critic, dict):
            raise ProvenanceError(f"{node_id} does not contain a critic")
        critic_hash = _require_sha256(
            critic.get("typed_content_sha256"), f"{node_id} critic content hash"
        )
        identity = node.get("run_identity")
        if not isinstance(identity, dict):
            raise ProvenanceError(f"{node_id} lacks run identity")
        seed = identity.get("optimizer_seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ProvenanceError(f"{node_id} optimizer seed is invalid")
        arm = identity.get("arm_id")
        if arm is not None and not isinstance(arm, str):
            raise ProvenanceError(f"{node_id} arm identity is invalid")
        receipts.append(
            EndpointReceipt(
                node_id=node_id,
                checkpoint=checkpoint,
                checkpoint_sha256=expected_checkpoint,
                artifact_metadata=cast(Json, metadata),
                resolved_semantics=semantics,
                loading=loading,
                critic_typed_content_sha256=critic_hash,
                arm_id=arm,
                optimizer_seed=seed,
            )
        )

    return EndpointRegistry(
        freeze_root=freeze_root,
        freeze_candidate_sha256=checkpoint_hash(candidate_path),
        artifact_dag_sha256=checkpoint_hash(dag_path),
        adapter_protocol_sha256=checkpoint_hash(adapter_path),
        retrieval_root=retrieval_root,
        bridge_path=bridge_path,
        bridge_sha256=bridge_sha256,
        source_hashes=actual_sources,
        endpoints=tuple(receipts),
    )


@dataclass(frozen=True)
class EndpointRuntime:
    receipt: EndpointReceipt
    actor: Actor
    critic: Critic | VectorCritic
    normalizer: Normalizer
    stage: ppo.Stage
    vector_critic: bool
    objectives: tuple[ObjectiveName, ...]
    personality_spec: str
    personality_drift: float
    log_std_floor: float
    mix_change_cost: float
    break_cost: float
    reward_semantics: Literal["raw-stationary", "history-conditioned-shaped"]
    return_contract_id: str


def _strict_module_state(module: nn.Module, state: object, *, component: str) -> None:
    if not isinstance(state, dict):
        raise ProvenanceError(f"checkpoint lacks {component} state")
    expected = module.state_dict()
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise ProvenanceError(
            f"{component} tensor keys differ (missing={missing}, unexpected={unexpected})"
        )
    for name, reference in expected.items():
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != reference.dtype
            or value.shape != reference.shape
            or value.layout != torch.strided
        ):
            raise ProvenanceError(f"{component} tensor {name} differs in dtype/device/shape/layout")
    module.load_state_dict(cast(dict[str, torch.Tensor], state), strict=True)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ProvenanceError(f"{name} must be a finite number")
    return float(value)


def load_endpoint_runtime(registry: EndpointRegistry, receipt: EndpointReceipt) -> EndpointRuntime:
    """Safely load one receipt; no current ``upgrade()`` or implicit actor defaults."""

    if receipt not in registry.endpoints:
        raise ProvenanceError("endpoint is outside the frozen registry")
    if checkpoint_hash(receipt.checkpoint) != receipt.checkpoint_sha256:
        raise ProvenanceError("checkpoint changed after registry construction")
    protocol_reference = receipt.loading.get("protocol")
    if not isinstance(protocol_reference, dict):
        raise ProvenanceError("endpoint loading receipt lacks adapter protocol")
    # The DAG artifact reference also carries byte size.  The executing Gate-1
    # adapter deliberately accepts only {path, sha256}; derive that exact runtime
    # dictionary from the sealed DAG instead of inventing or copying defaults.
    runtime_loading = dict(receipt.loading)
    runtime_loading["protocol"] = {
        "path": protocol_reference.get("path"),
        "sha256": protocol_reference.get("sha256"),
    }
    raw = load_recovered_contract12(
        receipt.checkpoint,
        runtime_loading,
        receipt.resolved_semantics,
        base=registry.freeze_root,
    )
    actor, normalizer = load_strict_state(
        raw,
        resolved_semantics=receipt.resolved_semantics,
        diagnostic_loading=None,
    )
    core_width = raw.get("core_width")
    privileged_width = raw.get("privileged_width", 0)
    token = receipt.resolved_semantics.get("token")
    if (
        isinstance(core_width, bool)
        or not isinstance(core_width, int)
        or core_width <= 0
        or isinstance(privileged_width, bool)
        or not isinstance(privileged_width, int)
        or privileged_width < 0
        or type(token) is not bool
    ):
        raise ProvenanceError("checkpoint critic architecture metadata is incomplete")
    vector = raw.get("vector_critic", False)
    if type(vector) is not bool:
        raise ProvenanceError("checkpoint vector_critic metadata is invalid")
    critic: Critic | VectorCritic
    if vector:
        critic = VectorCritic(core_width, privileged_width=privileged_width, token=token)
    else:
        critic = Critic(core_width, privileged_width=privileged_width, token=token)
    _strict_module_state(critic, raw.get("critic"), component="critic")
    actor.eval()
    critic.eval()

    if receipt.node_id == "t3-initialization-clone":
        # The recovered t2.sh binds this checkpoint as the mixed-stage entry.  Its
        # scalar critic was pretrained on unshaped lineage welfare; it has no PPO
        # hyper mapping and must not inherit later checkpoint hyperparameters.
        personality: str = "default"
        drift = 0.0
        floor = FROZEN_INIT_LOG_STD_FLOOR
        mix_cost = 0.0
        break_cost = 0.0
        reward_semantics: Literal["raw-stationary", "history-conditioned-shaped"] = "raw-stationary"
        return_contract = "recovered-t3-init-scalar-lineage-welfare-v1"
    else:
        if raw.get("stage") != FROZEN_STAGE:
            raise ProvenanceError("historical endpoint is not the recovered mixed stage")
        hyper = raw.get("hyper")
        if not isinstance(hyper, dict):
            raise ProvenanceError("historical endpoint lacks exact PPO hyperparameters")
        personality_value = hyper.get("personality")
        if personality_value not in {"default", "dirichlet:1.0"}:
            raise ProvenanceError("historical endpoint personality contract is unsupported")
        assert isinstance(personality_value, str)
        personality = personality_value
        drift = _finite_number(hyper.get("personality_drift"), "personality_drift")
        floor = _finite_number(hyper.get("log_std_floor"), "log_std_floor")
        mix_cost = _finite_number(hyper.get("mix_change_cost"), "mix_change_cost")
        break_cost = _finite_number(hyper.get("break_cost"), "break_cost")
        reward_semantics = "history-conditioned-shaped"
        return_contract = (
            "recovered-t3-vector-shaped-components-v1"
            if vector
            else "recovered-t3-scalar-shaped-lineage-welfare-v1"
        )

    objectives: tuple[ObjectiveName, ...] = (
        cast(tuple[ObjectiveName, ...], tuple(contracts.OBJECTIVES)) if vector else ("welfare",)
    )
    return EndpointRuntime(
        receipt=receipt,
        actor=actor,
        critic=critic,
        normalizer=normalizer,
        stage=ppo.STAGES[FROZEN_STAGE],
        vector_critic=vector,
        objectives=objectives,
        personality_spec=personality,
        personality_drift=drift,
        log_std_floor=floor,
        mix_change_cost=mix_cost,
        break_cost=break_cost,
        reward_semantics=reward_semantics,
        return_contract_id=return_contract,
    )


def _derived_seed(*parts: object) -> int:
    payload = json.dumps(parts, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


class PersonalityState:
    """Per-world PPO personality state with explicit daughter inheritance."""

    def __init__(
        self,
        spec: str,
        drift: float,
        seed: int,
        *,
        traits: Mapping[int, NDArray[np.float32]] | None = None,
    ) -> None:
        self.spec = spec
        self.drift = float(drift)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.traits = {
            int(society): np.asarray(value, dtype=np.float32).copy()
            for society, value in (traits or {}).items()
        }
        self._sample = ppo.personality_sampler(spec, self.rng)

    @property
    def active(self) -> bool:
        return self.spec != "default"

    def clone_for_path(self, seed: int) -> PersonalityState:
        """Preserve existing lineages; independently draw only future lineages/daughters."""
        return PersonalityState(self.spec, self.drift, seed, traits=self.traits)

    def add_to_batch(self, batch: Json, villages: Sequence[Json]) -> Json:
        if not self.active:
            return batch
        rows: list[NDArray[np.float32]] = []
        for village in villages:
            society = int(village["society"])
            parent = int(village.get("parentSociety", 0))
            if society not in self.traits:
                if parent in self.traits:
                    self.traits[society] = ppo.personality_drift(
                        self.traits[parent], self.rng, self.drift
                    )
                else:
                    self.traits[society] = self._sample()
            rows.append(self.traits[society])
        batch["personality"] = np.asarray(rows, dtype=np.float32)
        return batch


@dataclass(frozen=True)
class SampledActions:
    actions: Mapping[int, Json]
    mixes: Mapping[int, tuple[float, ...]]
    spans: Mapping[int, int]
    personality: Mapping[int, tuple[float, ...]]


def _live_villages(step: Json) -> list[Json]:
    villages = step.get("villages")
    if not isinstance(villages, list):
        raise ValueError("bridge step lacks villages")
    return [
        cast(Json, village)
        for village in villages
        if isinstance(village, dict)
        and village.get("observation") is not None
        and not bool(village.get("terminated"))
    ]


def sample_training_actions(
    runtime: EndpointRuntime,
    step: Json,
    generator: torch.Generator,
    personalities: PersonalityState,
) -> SampledActions:
    """Sample the exact recovered PPO action distribution with a persistent RNG."""

    live = _live_villages(step)
    if not live:
        return SampledActions({}, {}, {}, {})
    batch = cast(Json, ppo.village_batch(live))
    personalities.add_to_batch(batch, live)
    tensors = to_tensors(batch, runtime.normalizer)
    with torch.no_grad():
        output = ppo.explore(runtime.actor(tensors), runtime.log_std_floor)
        draw = ppo.sample(output, tensors, runtime.stage, generator=generator)
        decoded = ppo.to_bridge_actions(draw, tensors)
        mix = ppo.mix_from_ratios(draw["mix_z"], tensors["economy_mix_mask"]).numpy()
    actions: dict[int, Json] = {}
    mixes: dict[int, tuple[float, ...]] = {}
    spans: dict[int, int] = {}
    traits: dict[int, tuple[float, ...]] = {}
    for row, (village, action) in enumerate(zip(live, decoded, strict=True)):
        camp = int(village["camp"])
        actions[camp] = action
        mixes[camp] = tuple(float(value) for value in mix[row])
        spans[camp] = int(draw["span"][row])
        if personalities.active:
            traits[camp] = tuple(float(value) for value in batch["personality"][row])
    return SampledActions(actions, mixes, spans, traits)


class FirstActionCallback(Protocol):
    def __call__(self, action: Json, native_label: Json, village: Json) -> Json: ...


@dataclass(frozen=True)
class FrozenInterventionSpec:
    """A reviewed one-time decoded-action mutation and its immutable allowed leaves."""

    intervention_id: str
    version: int
    allowed_changed_paths: tuple[str, ...]
    description: str
    callback: FirstActionCallback
    spec_sha256: str

    def identity_payload(self) -> Json:
        return {
            "id": self.intervention_id,
            "version": self.version,
            "allowedChangedPaths": list(self.allowed_changed_paths),
            "description": self.description,
        }

    def validate_identity(self) -> None:
        if _canonical_sha256(self.identity_payload()) != self.spec_sha256:
            raise ProvenanceError("intervention callback differs from its frozen specification")


def _development_budget_native(action: Json, native_label: Json, village: Json) -> Json:
    del village
    strategic = action.get("strategic")
    native = native_label.get("strategic")
    if (
        not isinstance(strategic, list)
        or len(strategic) != 5
        or not isinstance(native, list)
        or len(native) != 5
    ):
        raise ValueError("development intervention requires five strategic controls")
    value = native[2]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError("native development-budget label is nonfinite")
    changed = copy.deepcopy(action)
    cast(list[Any], changed["strategic"])[2] = float(value)
    return changed


_DEVELOPMENT_IDENTITY: Json = {
    "id": "development-budget-native-v1",
    "version": 1,
    "allowedChangedPaths": ["strategic[2]"],
    "description": (
        "replace only the first decoded development-budget control with the native "
        "teacher label from the identical forked state"
    ),
}
DEVELOPMENT_NATIVE_INTERVENTION = FrozenInterventionSpec(
    intervention_id=cast(str, _DEVELOPMENT_IDENTITY["id"]),
    version=cast(int, _DEVELOPMENT_IDENTITY["version"]),
    allowed_changed_paths=("strategic[2]",),
    description=cast(str, _DEVELOPMENT_IDENTITY["description"]),
    callback=_development_budget_native,
    spec_sha256=_canonical_sha256(_DEVELOPMENT_IDENTITY),
)


def _leaf_values(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_leaf_values(value[key], child))
        return out
    if isinstance(value, list):
        out = {}
        for index, item in enumerate(value):
            out.update(_leaf_values(item, f"{prefix}[{index}]"))
        return out
    return {prefix: value}


@dataclass(frozen=True)
class ActionInterventionReceipt:
    camp_id: int
    before_sha256: str
    after_sha256: str
    changed_paths: tuple[str, ...]
    exact_zero_change: bool


def apply_first_action_intervention(
    spec: FrozenInterventionSpec,
    *,
    camp_id: int,
    action: Json,
    native_label: Json,
    village: Json,
) -> tuple[Json, ActionInterventionReceipt]:
    """Run a callback, then prove complete equality outside its frozen leaf allow-list."""

    spec.validate_identity()
    before = copy.deepcopy(action)
    after = spec.callback(copy.deepcopy(action), copy.deepcopy(native_label), village)
    if not isinstance(after, dict):
        raise ValueError("first-action intervention must return a decoded action object")
    left = _leaf_values(before)
    right = _leaf_values(after)
    if set(left) != set(right):
        raise ValueError("first-action intervention changed the decoded action schema")
    changed = tuple(sorted(path for path in left if left[path] != right[path]))
    forbidden = sorted(set(changed) - set(spec.allowed_changed_paths))
    if forbidden:
        raise ValueError(f"intervention changed leaves outside its frozen scope: {forbidden}")
    return after, ActionInterventionReceipt(
        camp_id=camp_id,
        before_sha256=_canonical_sha256(before),
        after_sha256=_canonical_sha256(after),
        changed_paths=changed,
        exact_zero_change=not changed,
    )


def probe_native_labels(bridge: Model9Bridge, decision_step: Json) -> Mapping[int, Json]:
    """Advance one forked native transition to expose the action labels, then restore outside."""

    opening = {int(village["camp"]) for village in _live_villages(decision_step)}
    successor = bridge.step({})
    labels: dict[int, Json] = {}
    for village in cast(list[Json], successor.get("villages", [])):
        camp = int(village["camp"])
        teacher = village.get("teacher")
        if camp in opening and isinstance(teacher, dict):
            labels[camp] = cast(Json, teacher)
    if set(labels) != opening:
        raise ValueError("native fork probe did not return every opening camp label")
    return labels


@dataclass(frozen=True)
class FrozenCollectionProtocol:
    world_seeds: tuple[int, ...] = FROZEN_WORLD_SEEDS
    regimes: tuple[tuple[str, int], ...] = FROZEN_REGIMES
    decision_steps: tuple[int, ...] = FROZEN_DECISION_STEPS
    path_replicates: int = FROZEN_PATH_REPLICATES
    prefix_terms: int = FROZEN_PREFIX_TERMS
    tail_draws: int = FROZEN_TAIL_DRAWS
    tail_survival: float = FROZEN_TAIL_SURVIVAL
    tail_hard_cap: int = FROZEN_TAIL_HARD_CAP
    alternatives: tuple[str, ...] = FROZEN_ALTERNATIVES

    def __post_init__(self) -> None:
        if (
            self.world_seeds != FROZEN_WORLD_SEEDS
            or self.regimes != FROZEN_REGIMES
            or self.decision_steps != FROZEN_DECISION_STEPS
            or self.path_replicates != FROZEN_PATH_REPLICATES
            or self.prefix_terms != FROZEN_PREFIX_TERMS
            or self.tail_draws != FROZEN_TAIL_DRAWS
            or self.tail_survival != FROZEN_TAIL_SURVIVAL
            or self.tail_hard_cap != FROZEN_TAIL_HARD_CAP
            or self.alternatives != FROZEN_ALTERNATIVES
        ):
            raise ProvenanceError("Gate 3/4 collection protocol axes are immutable")

    @property
    def frozen_state_count(self) -> int:
        return len(self.world_seeds) * len(self.regimes) * len(self.decision_steps)

    @property
    def native_label_probe_steps_per_checkpoint(self) -> int:
        return self.frozen_state_count

    @property
    def bridge_step_hard_cap_per_checkpoint(self) -> int:
        # Per world/regime: one main path to step 25, one native-label probe at each
        # frozen state, and paired alternatives for four paths to prefix+tail cap.
        return (
            len(self.world_seeds)
            * len(self.regimes)
            * (
                max(self.decision_steps)
                + len(self.decision_steps)
                + len(self.decision_steps)
                * len(self.alternatives)
                * self.path_replicates
                * (self.prefix_terms + self.tail_hard_cap)
            )
        )

    @property
    def protocol_sha256(self) -> str:
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> Json:
        return {
            "kind": "gate34-checkpoint-continuation-protocol",
            "version": 1,
            "independentUnit": "world",
            "worldSeeds": list(self.world_seeds),
            "regimes": [{"id": regime_id, "withheld": mask} for regime_id, mask in self.regimes],
            "decisionSteps": list(self.decision_steps),
            "pathReplicates": self.path_replicates,
            "fixedPrefixTerms": self.prefix_terms,
            "tail": {
                "method": "independent-geometric-russian-roulette",
                "drawsPerPath": self.tail_draws,
                "survivalProbability": self.tail_survival,
                "hardCapTerms": self.tail_hard_cap,
                "hardCapRule": "unresolved; never redraw, discard, or truncate",
            },
            "alternatives": list(self.alternatives),
            "interventionSpecSha256": DEVELOPMENT_NATIVE_INTERVENTION.spec_sha256,
            "nativeLabelProbeStepsPerCheckpoint": self.native_label_probe_steps_per_checkpoint,
            "bridgeStepHardCapPerCheckpoint": self.bridge_step_hard_cap_per_checkpoint,
            "baselineCalibrationArm": "sampled-policy",
            "sensitivityScope": "single-first-action decoded development-budget substitution",
            "selectionAllowed": False,
            "readinessClaimed": False,
        }

    def stopping_schedule(
        self,
        endpoint_id: str,
        world_seed: int,
        regime_id: str,
        decision_step: int,
        path_replicate: int,
    ) -> FrozenGeometricSchedule:
        identity = (
            "gate34-rr-v1",
            endpoint_id,
            world_seed,
            regime_id,
            decision_step,
            path_replicate,
        )
        return freeze_geometric_schedule(
            schedule_id="/".join(str(value) for value in identity),
            rng_seed=_derived_seed(*identity),
            draw_count=self.tail_draws,
            survival_probability=self.tail_survival,
            hard_cap_terms=self.tail_hard_cap,
        )


FROZEN_PROTOCOL = FrozenCollectionProtocol()


def _recovered_lineage_rewards(rows: Sequence[Json], *, qol: bool = True) -> NDArray[np.float64]:
    """Exact pure mirror of recovered data.py at its bound SHA-256."""

    reward = np.zeros(len(rows), dtype=np.float64)
    first_turn: dict[tuple[int, int], int] = {}
    order = sorted(range(len(rows)), key=lambda index: int(rows[index]["turn"]))
    for index in order:
        row = rows[index]
        key = (int(row["world"]), int(row["camp"]))
        first_turn.setdefault(key, int(row["turn"]))
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, row in enumerate(rows):
        world = int(row["world"])
        camp = int(row["camp"])
        turn = int(row["turn"])
        parent = int(row["parent_society"])
        recent = parent > 0 and turn - first_turn[(world, camp)] < contracts.LINEAGE_TURNS
        group = parent if recent else int(row["society"])
        groups.setdefault((world, turn, group), []).append(index)
    for indices in groups.values():
        population = np.asarray(
            [max(0.0, float(rows[index]["prior_population"])) for index in indices],
            dtype=np.float64,
        )
        alive = np.asarray([not bool(rows[index]["terminated"]) for index in indices])
        quality = (
            np.asarray(
                [float(rows[index]["qol"]) if qol else 1.0 for index in indices],
                dtype=np.float64,
            )
            * alive
        )
        denominator = float(population.sum())
        lineage_quality = float((population * quality).sum() / denominator) if denominator else 0.0
        for index, prior, is_alive in zip(indices, population, alive, strict=True):
            reward[index] = (
                prior * lineage_quality * float(rows[index]["years"]) if is_alive else 0.0
            )
    return reward


def _derived_components(rows: Sequence[Json]) -> Mapping[ObjectiveName, NDArray[np.float64]]:
    welfare = _recovered_lineage_rewards(rows)
    population_years = _recovered_lineage_rewards(rows, qol=False)
    trade = np.asarray([float(row.get("trade_utility", 0.0)) for row in rows])
    prior = np.asarray([float(row["prior_population"]) for row in rows])
    alive = np.asarray([not bool(row["terminated"]) for row in rows])
    qol_years = np.where(alive & (prior > 0), welfare / np.maximum(1.0, prior), 0.0)
    return {
        "welfare": welfare,
        "population_years": population_years,
        "trade_utility": trade,
        "qol_years": qol_years,
    }


def _critic_predictions(
    runtime: EndpointRuntime,
    villages: Sequence[Json],
    personalities: PersonalityState,
) -> tuple[
    Mapping[ObjectiveName, Mapping[int, float]],
    Mapping[ObjectiveName, Mapping[int, Json]],
]:
    live = [
        village
        for village in villages
        if village.get("observation") is not None and not bool(village.get("terminated"))
    ]
    if not live:
        empty: dict[ObjectiveName, dict[int, float]] = {
            objective: {} for objective in runtime.objectives
        }
        empty_diagnostics: dict[ObjectiveName, dict[int, Json]] = {
            objective: {} for objective in runtime.objectives
        }
        return empty, empty_diagnostics
    batch = cast(Json, ppo.village_batch(live))
    personalities.add_to_batch(batch, live)
    # At a decision state the opening residents are the current village population.
    # Training's outcome row calls the same quantity ``prior_population`` after the
    # transition.  Supplying current population preserves the critic's raw-unit scale.
    population = np.asarray([int(village["population"]) for village in live], dtype=np.float64)
    batch["prior_population"] = population
    tensors = to_tensors(batch, runtime.normalizer)
    with torch.no_grad():
        raw_logits = runtime.critic(tensors)
    logits_by_objective: Mapping[str, torch.Tensor]
    if runtime.vector_critic:
        if not isinstance(raw_logits, dict):
            raise ValueError("vector critic returned scalar logits")
        logits_by_objective = raw_logits
    else:
        if not isinstance(raw_logits, torch.Tensor):
            raise ValueError("scalar critic returned vector logits")
        logits_by_objective = {"welfare": raw_logits}
    bins = runtime.critic.state_dict()["bins"].detach().cpu()
    if (
        bins.ndim != 1
        or bins.numel() < 2
        or not torch.isfinite(bins).all()
        or not bool(torch.all(bins[1:] > bins[:-1]))
    ):
        raise ProvenanceError("loaded critic has invalid support bins")
    support = symexp(bins).numpy().astype(np.float64)
    camps = [int(village["camp"]) for village in live]
    values: dict[ObjectiveName, dict[int, float]] = {}
    diagnostics: dict[ObjectiveName, dict[int, Json]] = {}
    for objective_name, logits in logits_by_objective.items():
        objective = cast(ObjectiveName, objective_name)
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        per_resident = (
            (torch.softmax(logits, dim=-1) * symexp(bins.to(logits.device)))
            .sum(dim=-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        scale = (
            np.maximum(1.0, population)
            if objective in ppo.PER_RESIDENT
            else np.ones_like(population)
        )
        raw_values = per_resident * scale
        values[objective] = {
            camp: float(value) for camp, value in zip(camps, raw_values, strict=True)
        }
        diagnostics[objective] = {
            camp: {
                "predictedPerResidentMean": float(mean),
                "predictedRawMean": float(raw),
                "lowerBinProbabilityMass": float(probabilities[row, 0]),
                "upperBinProbabilityMass": float(probabilities[row, -1]),
                "supportPerResident": [float(support[0]), float(support[-1])],
                "supportRaw": [float(support[0] * scale[row]), float(support[-1] * scale[row])],
                "scale": float(scale[row]),
                "scaleRule": "per-opening-resident" if objective in ppo.PER_RESIDENT else "raw",
            }
            for row, (camp, mean, raw) in enumerate(
                zip(camps, per_resident, raw_values, strict=True)
            )
        }
    return values, diagnostics


def _critic_values(
    runtime: EndpointRuntime,
    villages: Sequence[Json],
    personalities: PersonalityState,
) -> Mapping[ObjectiveName, Mapping[int, float]]:
    return _critic_predictions(runtime, villages, personalities)[0]


def _outcome_rows(
    *,
    world_seed: int,
    before: Json,
    after: Json,
    sampled: SampledActions,
    previous_mixes: Mapping[int, tuple[float, ...]],
    runtime: EndpointRuntime,
) -> list[Json]:
    """Create rows for every completed decision before any opening-camp filtering."""

    before_live = {int(village["camp"]): village for village in _live_villages(before)}
    after_by_camp = {
        int(village["camp"]): village
        for village in cast(list[Json], after.get("villages", []))
        if isinstance(village, dict)
    }
    if set(before_live) != set(sampled.actions):
        raise ValueError("sampled actions do not cover every live pre-transition camp")
    rows: list[Json] = []
    for camp, prior_village in before_live.items():
        if camp not in after_by_camp:
            raise ValueError("completed transition omitted a pre-transition camp")
        village = after_by_camp[camp]
        current_mix = sampled.mixes[camp]
        previous = previous_mixes.get(camp)
        mix_change = (
            0.0
            if previous is None
            else 0.5
            * float(np.abs(np.asarray(current_mix) - np.asarray(previous, dtype=np.float64)).sum())
        )
        observation = prior_village.get("observation")
        hold_legal = bool(
            isinstance(observation, dict) and observation.get("commitHoldMask", False)
        )
        voluntary_break = bool(
            runtime.stage.commitments and hold_legal and sampled.spans[camp] != contracts.SPAN_HOLD
        )
        row: Json = {
            "world": world_seed,
            "camp": camp,
            "turn": int(before["turn"]),
            "society": int(village["society"]),
            "parent_society": int(village.get("parentSociety", 0)),
            "prior_population": int(village["priorPopulation"]),
            "population": int(village["population"]),
            "qol": float(village["qol"]),
            "nutrition": float(village["nutrition"]),
            "years": float(village["years"]),
            "welfare": float(village["welfare"]),
            "terminated": bool(village["terminated"]),
            "trade_utility": float(village.get("tradeUtility", 0.0)),
            "mix_change": mix_change,
            "voluntary_break": voluntary_break,
            "personality": sampled.personality.get(camp),
        }
        if float(row["years"]) != float(contracts.TURN_YEARS):
            raise ValueError("collector reward bound requires the bound 20-year transition")
        rows.append(row)
    return rows


def _shape_branch_components(
    runtime: EndpointRuntime,
    history: Sequence[Json],
    branch: Sequence[Json],
) -> Mapping[ObjectiveName, NDArray[np.float64]]:
    all_rows = [*history, *branch]
    raw = _derived_components(all_rows)
    offset = len(history)
    components = {objective: values[offset:].copy() for objective, values in raw.items()}
    if runtime.reward_semantics == "raw-stationary":
        return {objective: components[objective] for objective in runtime.objectives}
    mix_change = np.asarray([float(row["mix_change"]) for row in branch])
    breaks = np.asarray([bool(row["voluntary_break"]) for row in branch])
    share = runtime.mix_change_cost * mix_change + runtime.break_cost * breaks
    welfare_now = np.maximum(
        0.0, np.asarray([float(row["welfare"]) for row in branch], dtype=np.float64)
    )
    components["welfare"] -= share * welfare_now
    if runtime.vector_critic and runtime.personality_spec != "default":
        for objective in runtime.objectives:
            if objective != "welfare":
                values = components[objective]
                components[objective] = values - share * np.maximum(0.0, values)
    return {objective: components[objective] for objective in runtime.objectives}


def _compact_economic_rows(bridge: Model9Bridge) -> list[Json]:
    rows = getattr(bridge, "economic_diagnostics", {})
    if not isinstance(rows, dict):
        raise ValueError("bridge economic diagnostics have another schema")
    copied = [copy.deepcopy(rows[camp]) for camp in sorted(rows)]
    encoded = json.dumps(copied, sort_keys=True, allow_nan=False).encode("utf-8")
    if len(encoded) > 8 * 1024 * 1024:
        raise ValueError("single economic boundary exceeds the 8 MiB capture cap")
    return cast(list[Json], copied)


def _compact_end_state(step: Json) -> Json:
    villages = cast(list[Json], step.get("villages", []))
    rows = [
        {
            "camp": int(village["camp"]),
            "society": int(village["society"]),
            "parentSociety": int(village.get("parentSociety", 0)),
            "population": int(village["population"]),
            "qol": float(village["qol"]),
            "terminated": bool(village["terminated"]),
        }
        for village in villages
    ]
    return {
        "turn": int(step["turn"]),
        "fingerprint": str(step["fingerprint"]),
        "villages": rows,
    }


def _tail_assumptions(
    runtime: EndpointRuntime,
    objective: ObjectiveName,
    opening_camps: int,
    schedule: FrozenGeometricSchedule,
) -> TailAssumptions:
    provenance = f"signed-int32-population+qol-unit-bound/{runtime.receipt.node_id}/{objective}"
    if objective == "trade_utility":
        return TailAssumptions(False, False, "no-source-backed-signed-trade-bound")
    raw_bound = raw_camp_reward_absolute_bound(objective, opening_camps)
    if runtime.reward_semantics == "history-conditioned-shaped":
        if objective == "welfare":
            raw_welfare = raw_camp_reward_absolute_bound("welfare", opening_camps)
            raw_bound = shaped_welfare_reward_absolute_bound(
                raw_bound,
                raw_welfare,
                mix_change_cost=runtime.mix_change_cost,
                break_cost=runtime.break_cost,
                commitments_active=runtime.stage.commitments,
            )
        elif runtime.personality_spec != "default":
            raw_bound = shaped_nonnegative_component_absolute_bound(
                raw_bound,
                mix_change_cost=runtime.mix_change_cost,
                break_cost=runtime.break_cost,
                commitments_active=runtime.stage.commitments,
            )
    gamma = exp(-float(contracts.TURN_YEARS) / float(contracts.HORIZON_YEARS))
    return bounded_geometric_tail_assumptions(
        per_turn_absolute_bound=raw_bound,
        discount=gamma,
        survival_probability=schedule.survival_probability,
        provenance_id=provenance,
    )


def _discounted_terms(
    rows: Sequence[CampReward],
    opening_camps: Sequence[int],
    *,
    start: int,
    required: int,
) -> tuple[float, ...]:
    by_key = {(row.camp_id, row.step_index): row for row in rows}
    alive = {camp: True for camp in opening_camps}
    discount = {camp: 1.0 for camp in opening_camps}
    terms: list[float] = []
    for step_index in range(start + required):
        total = 0.0
        for camp in opening_camps:
            if not alive[camp]:
                continue
            row = by_key.get((camp, step_index))
            if row is None:
                raise ValueError("opening camp branch sequence is incomplete")
            total += discount[camp] * row.reward
            discount[camp] *= exp(-row.years / float(contracts.HORIZON_YEARS))
            if row.terminated:
                alive[camp] = False
        if step_index >= start:
            terms.append(float(total))
    return tuple(terms)


def _zero_tail(schedule: FrozenGeometricSchedule) -> RandomizedTailEstimate:
    zeros = tuple(0.0 for _ in schedule.stop_indices)
    return RandomizedTailEstimate(
        status="identified",
        schedule_id=schedule.schedule_id,
        estimate=0.0,
        standard_error=0.0,
        draw_estimates=zeros,
        standard_error_scope="exact terminal zero tail",
        stopping_randomization_contribution_to_standard_error=0.0,
    )


def _exact_terminal_tail(
    rows: Sequence[CampReward],
    opening_camps: Sequence[int],
    *,
    start: int,
    schedule: FrozenGeometricSchedule,
) -> RandomizedTailEstimate:
    """Return the exact globally discounted suffix when every camp terminates.

    This is an observed finite tail, not a randomized-truncation estimate.  Rewards
    before ``start`` still advance the discount from the original frozen state.
    """

    if start < 0:
        raise ValueError("terminal tail start must be nonnegative")
    camps = tuple(int(camp) for camp in opening_camps)
    if not camps or len(camps) != len(set(camps)):
        raise ValueError("opening_camps must be nonempty and unique")
    by_camp: dict[int, list[CampReward]] = {camp: [] for camp in camps}
    for row in rows:
        if row.camp_id in by_camp:
            by_camp[row.camp_id].append(row)

    total = 0.0
    for camp in camps:
        sequence = sorted(by_camp[camp], key=lambda row: row.step_index)
        indices = [row.step_index for row in sequence]
        if not sequence or indices != list(range(len(sequence))):
            raise ValueError(f"opening camp {camp} has an incomplete terminal sequence")
        ended = [index for index, row in enumerate(sequence) if row.terminated]
        if ended != [len(sequence) - 1]:
            raise ValueError(f"opening camp {camp} lacks one final terminal outcome")
        discount = 1.0
        for row in sequence:
            if row.step_index >= start:
                total += discount * row.reward
            discount *= exp(-row.years / float(contracts.HORIZON_YEARS))

    draws = tuple(float(total) for _ in schedule.stop_indices)
    return RandomizedTailEstimate(
        status="identified",
        schedule_id=schedule.schedule_id,
        estimate=float(total),
        standard_error=0.0,
        draw_estimates=draws,
        standard_error_scope=(
            "exact observed finite terminal suffix; no stopping randomization, "
            "action/simulator-path, or world uncertainty included"
        ),
        stopping_randomization_contribution_to_standard_error=0.0,
    )


@dataclass(frozen=True)
class BranchResult:
    alternative_id: str
    path_replicate: int
    schedule: FrozenGeometricSchedule
    prefix_returns: Mapping[ObjectiveName, float]
    tail_estimates: Mapping[ObjectiveName, RandomizedTailEstimate]
    camp_prefix_returns: Mapping[ObjectiveName, Mapping[int, float]]
    camp_tail_estimates: Mapping[ObjectiveName, Mapping[int, RandomizedTailEstimate]]
    ranking_predictions: Mapping[ObjectiveName, float]
    action_receipts: tuple[ActionInterventionReceipt, ...]
    collected_steps: int
    opening_camp_count: int
    terminal: bool
    economic: Json | None
    # In-memory producer payload.  The historical Gate3/4 JSON deliberately omits
    # these arrays; the Stage-1 observed-return producer consumes them directly.
    anchor_successor_inputs: Mapping[int, Mapping[str, NDArray[Any]]] | None = None
    anchor_branch_rows: tuple[Json, ...] = ()
    anchor_one_step_rewards: Mapping[ObjectiveName, Mapping[int, CampReward]] | None = None
    anchor_successor_world_digest: str | None = None


def _branch_rewards(
    runtime: EndpointRuntime,
    history: Sequence[Json],
    branch_rows: Sequence[Json],
    opening_camps: Sequence[int],
) -> Mapping[ObjectiveName, list[CampReward]]:
    components = _shape_branch_components(runtime, history, branch_rows)
    camp_step: dict[int, int] = {camp: 0 for camp in opening_camps}
    result: dict[ObjectiveName, list[CampReward]] = {
        objective: [] for objective in runtime.objectives
    }
    for row_index, row in enumerate(branch_rows):
        camp = int(row["camp"])
        if camp not in camp_step:
            continue
        step_index = camp_step[camp]
        camp_step[camp] += 1
        for objective in runtime.objectives:
            result[objective].append(
                CampReward(
                    camp_id=camp,
                    step_index=step_index,
                    reward=float(components[objective][row_index]),
                    years=float(row["years"]),
                    terminated=bool(row["terminated"]),
                )
            )
    return result


def _execute_branch(
    *,
    bridge: Model9Bridge,
    decision_step: Json,
    runtime: EndpointRuntime,
    protocol: FrozenCollectionProtocol,
    world_seed: int,
    history: Sequence[Json],
    main_personalities: PersonalityState,
    previous_mixes: Mapping[int, tuple[float, ...]],
    native_labels: Mapping[int, Json],
    alternative_id: str,
    path_replicate: int,
    schedule: FrozenGeometricSchedule,
    action_seed: int,
    personality_seed: int,
    capture_economic: bool,
    capture_anchor: bool = False,
) -> BranchResult:
    opening = tuple(int(village["camp"]) for village in _live_villages(decision_step))
    if not opening:
        raise ValueError("frozen state has no live opening camps")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(action_seed)
    personalities = main_personalities.clone_for_path(personality_seed)
    mixes = dict(previous_mixes)
    current = decision_step
    branch_rows: list[Json] = []
    first_receipts: list[ActionInterventionReceipt] = []
    first_economic: list[Json] | None = None
    successor_values: Mapping[ObjectiveName, Mapping[int, float]] | None = None
    successor_inputs: dict[int, dict[str, NDArray[Any]]] = {}
    successor_world_digest: str | None = None
    maximum_steps = protocol.prefix_terms + (
        0 if schedule.exceeds_hard_cap else schedule.required_terms
    )
    terminated_opening: set[int] = set()

    for step_index in range(maximum_steps):
        sampled = sample_training_actions(runtime, current, generator, personalities)
        actions = {camp: copy.deepcopy(action) for camp, action in sampled.actions.items()}
        if step_index == 0 and alternative_id == DEVELOPMENT_NATIVE_INTERVENTION.intervention_id:
            by_camp = {int(village["camp"]): village for village in _live_villages(current)}
            for camp in sorted(actions):
                changed, receipt = apply_first_action_intervention(
                    DEVELOPMENT_NATIVE_INTERVENTION,
                    camp_id=camp,
                    action=actions[camp],
                    native_label=native_labels[camp],
                    village=by_camp[camp],
                )
                actions[camp] = changed
                first_receipts.append(receipt)
        elif step_index == 0:
            first_receipts.extend(
                ActionInterventionReceipt(
                    camp_id=camp,
                    before_sha256=_canonical_sha256(action),
                    after_sha256=_canonical_sha256(action),
                    changed_paths=(),
                    exact_zero_change=True,
                )
                for camp, action in sorted(actions.items())
            )
        after = bridge.step(actions)
        rows = _outcome_rows(
            world_seed=world_seed,
            before=current,
            after=after,
            sampled=sampled,
            previous_mixes=mixes,
            runtime=runtime,
        )
        branch_rows.extend(rows)
        mixes.update(sampled.mixes)
        for row in rows:
            if int(row["camp"]) in opening and bool(row["terminated"]):
                terminated_opening.add(int(row["camp"]))
        if step_index == 0:
            successor_world_digest = str(after.get("fingerprint", "")) or None
            opening_successors = [
                village
                for village in cast(list[Json], after.get("villages", []))
                if int(village["camp"]) in opening
            ]
            successor_values = _critic_values(runtime, opening_successors, personalities)
            live_successors = (
                [
                    village
                    for village in opening_successors
                    if village.get("observation") is not None
                    and not bool(village.get("terminated"))
                ]
                if capture_anchor
                else []
            )
            if live_successors:
                successor_batch = cast(Json, ppo.village_batch(live_successors))
                personalities.add_to_batch(successor_batch, live_successors)
                successor_batch["prior_population"] = np.asarray(
                    [int(village["population"]) for village in live_successors],
                    dtype=np.float64,
                )
                for row, village in enumerate(live_successors):
                    successor_inputs[int(village["camp"])] = {
                        key: np.asarray(values[row]).copy()
                        for key, values in successor_batch.items()
                    }
            if capture_economic:
                first_economic = _compact_economic_rows(bridge)
        current = after
        if len(terminated_opening) == len(opening):
            break

    rewards = _branch_rewards(runtime, history, branch_rows, opening)
    prefix_returns: dict[ObjectiveName, float] = {}
    tails: dict[ObjectiveName, RandomizedTailEstimate] = {}
    camp_prefix_returns: dict[ObjectiveName, dict[int, float]] = {}
    camp_tail_estimates: dict[ObjectiveName, dict[int, RandomizedTailEstimate]] = {}
    rankings: dict[ObjectiveName, float] = {}
    assert successor_values is not None
    one_step_by_objective: dict[ObjectiveName, dict[int, CampReward]] = {}
    for objective in runtime.objectives:
        objective_rows = rewards[objective]
        prefix_rows = [row for row in objective_rows if row.step_index < protocol.prefix_terms]
        prefix = aggregate_camp_prefix(prefix_rows, opening, protocol.prefix_terms)
        prefix_returns[objective] = prefix.prefix_return
        if prefix.terminal:
            tails[objective] = _zero_tail(schedule)
        elif len(terminated_opening) == len(opening):
            tails[objective] = _exact_terminal_tail(
                objective_rows,
                opening,
                start=protocol.prefix_terms,
                schedule=schedule,
            )
        elif schedule.exceeds_hard_cap:
            tails[objective] = RandomizedTailEstimate(
                status="unresolved-hard-cap", schedule_id=schedule.schedule_id
            )
        else:
            terms = _discounted_terms(
                objective_rows,
                opening,
                start=protocol.prefix_terms,
                required=schedule.required_terms,
            )
            tails[objective] = estimate_randomized_tail(
                terms,
                schedule,
                _tail_assumptions(runtime, objective, len(opening), schedule),
            )
        one_step_rows = {row.camp_id: row for row in objective_rows if row.step_index == 0}
        one_step_by_objective[objective] = one_step_rows
        one_step = [
            CampOneStepValue(
                camp_id=camp,
                immediate_reward=one_step_rows[camp].reward,
                years=one_step_rows[camp].years,
                successor_value=(
                    0.0
                    if one_step_rows[camp].terminated
                    else float(successor_values[objective].get(camp, 0.0))
                ),
                terminated=one_step_rows[camp].terminated,
            )
            for camp in opening
        ]
        rankings[objective] = aggregate_ranking_prediction(one_step, opening).value
        camp_prefix_returns[objective] = {}
        camp_tail_estimates[objective] = {}
        for camp in opening:
            camp_rows = [row for row in objective_rows if row.camp_id == camp]
            camp_prefix_rows = [row for row in camp_rows if row.step_index < protocol.prefix_terms]
            camp_prefix = aggregate_camp_prefix(camp_prefix_rows, [camp], protocol.prefix_terms)
            camp_prefix_returns[objective][camp] = camp_prefix.prefix_return
            if camp_prefix.terminal:
                camp_tail = _zero_tail(schedule)
            elif camp_rows[-1].terminated:
                camp_tail = _exact_terminal_tail(
                    camp_rows,
                    [camp],
                    start=protocol.prefix_terms,
                    schedule=schedule,
                )
            elif schedule.exceeds_hard_cap:
                camp_tail = RandomizedTailEstimate(
                    status="unresolved-hard-cap", schedule_id=schedule.schedule_id
                )
            else:
                camp_terms = _discounted_terms(
                    camp_rows,
                    [camp],
                    start=protocol.prefix_terms,
                    required=schedule.required_terms,
                )
                # The aggregate opening-camp magnitude proof also bounds each
                # individual camp and is deliberately conservative here.
                camp_tail = estimate_randomized_tail(
                    camp_terms,
                    schedule,
                    _tail_assumptions(runtime, objective, len(opening), schedule),
                )
            camp_tail_estimates[objective][camp] = camp_tail

    economic: Json | None = None
    if capture_economic:
        economic = {
            "firstTransitionRows": first_economic,
            "endTransitionRows": _compact_economic_rows(bridge),
            "endState": _compact_end_state(current),
        }
    return BranchResult(
        alternative_id=alternative_id,
        path_replicate=path_replicate,
        schedule=schedule,
        prefix_returns=prefix_returns,
        tail_estimates=tails,
        camp_prefix_returns=camp_prefix_returns,
        camp_tail_estimates=camp_tail_estimates,
        ranking_predictions=rankings,
        action_receipts=tuple(first_receipts),
        collected_steps=max((int(row["turn"]) for row in branch_rows), default=-1)
        - int(decision_step["turn"])
        + 1,
        opening_camp_count=len(opening),
        terminal=len(terminated_opening) == len(opening),
        economic=economic,
        anchor_successor_inputs=successor_inputs,
        anchor_branch_rows=tuple(copy.deepcopy(branch_rows)),
        anchor_one_step_rewards=one_step_by_objective,
        anchor_successor_world_digest=successor_world_digest,
    )


def direction_counts(values: Sequence[float]) -> Json:
    """Count every paired probe; exact zeros stay in the denominator."""

    checked = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if checked.ndim != 1 or checked.size == 0 or not np.all(np.isfinite(checked)):
        raise ValueError("direction values must be a nonempty finite sequence")
    positive = int(np.sum(checked > 0))
    negative = int(np.sum(checked < 0))
    zero = int(np.sum(checked == 0))
    return {
        "denominator": int(checked.size),
        "positive": positive,
        "negative": negative,
        "exactZeroUninformativeButRetained": zero,
        "positiveFraction": positive / int(checked.size),
        "negativeFraction": negative / int(checked.size),
        "exactZeroFraction": zero / int(checked.size),
    }


def _paired_tail_from_results(
    first: RandomizedTailEstimate,
    second: RandomizedTailEstimate,
    schedule: FrozenGeometricSchedule,
) -> RandomizedTailEstimate:
    """Subtract corresponding frozen stopping draws from paired physical paths."""

    if not first.identified:
        return RandomizedTailEstimate(first.status, schedule.schedule_id)
    if not second.identified:
        return RandomizedTailEstimate(second.status, schedule.schedule_id)
    if (
        first.schedule_id != schedule.schedule_id
        or second.schedule_id != schedule.schedule_id
        or len(first.draw_estimates) != len(second.draw_estimates)
    ):
        raise ValueError("paired alternatives do not share one frozen stopping schedule")
    differences = np.asarray(first.draw_estimates) - np.asarray(second.draw_estimates)
    standard_error = float(differences.std(ddof=1) / np.sqrt(len(differences)))
    return RandomizedTailEstimate(
        status="identified",
        schedule_id=schedule.schedule_id,
        estimate=float(differences.mean()),
        standard_error=standard_error,
        draw_estimates=tuple(float(value) for value in differences),
        standard_error_scope=(
            "paired stopping-randomization conditional on paired observed paths; "
            "excludes action/simulator-path and world uncertainty"
        ),
        stopping_randomization_contribution_to_standard_error=standard_error,
    )


def _randomized_estimate_json(value: RandomizedTailEstimate) -> Json:
    return {
        "status": value.status,
        "scheduleId": value.schedule_id,
        "estimate": value.estimate,
        "standardError": value.standard_error,
        "standardErrorScope": value.standard_error_scope,
        "stoppingRandomizationContributionToStandardError": (
            value.stopping_randomization_contribution_to_standard_error
        ),
        "estimateKind": value.estimate_kind,
        "drawCount": len(value.draw_estimates),
    }


def paired_path_diagnostic(
    intervention: BranchResult,
    baseline: BranchResult,
    objective: ObjectiveName,
    *,
    uncertainty_z: float = 1.96,
    tolerance: float = 0.0,
) -> Json:
    """Compare one paired nested path without promoting it to an independent world.

    The prediction is the tested critic's ``r + gamma * V(successor)`` difference.
    The target is the critic-independent fixed-prefix plus paired randomized-tail
    difference.  A nonzero stopping-randomization interval that crosses the tie
    tolerance remains uncertain; it is never coerced to a sign or tie.
    """

    uncertainty_z = _finite_number(uncertainty_z, "uncertainty_z")
    tolerance = _finite_number(tolerance, "tolerance")
    if uncertainty_z < 0 or tolerance < 0:
        raise ValueError("uncertainty_z and tolerance must be nonnegative")
    if intervention.path_replicate != baseline.path_replicate:
        raise ValueError("paired paths have different replicate identities")
    if intervention.schedule != baseline.schedule:
        raise ValueError("paired alternatives use different stopping schedules")
    tail = _paired_tail_from_results(
        intervention.tail_estimates[objective],
        baseline.tail_estimates[objective],
        intervention.schedule,
    )
    predicted = (
        intervention.ranking_predictions[objective] - baseline.ranking_predictions[objective]
    )
    predicted_relation = (
        "positive"
        if predicted > tolerance
        else "negative"
        if predicted < -tolerance
        else "predicted-tie"
    )
    prefix = intervention.prefix_returns[objective] - baseline.prefix_returns[objective]
    realised: float | None = None
    interval: list[float] | None = None
    realised_relation = "unresolved"
    agreement: bool | None = None
    if tail.identified:
        assert tail.estimate is not None and tail.standard_error is not None
        realised = prefix + tail.estimate
        radius = uncertainty_z * tail.standard_error
        lower, upper = realised - radius, realised + radius
        interval = [float(lower), float(upper)]
        if lower > tolerance:
            realised_relation = "positive"
        elif upper < -tolerance:
            realised_relation = "negative"
        elif intervention.terminal and baseline.terminal and realised == 0.0:
            realised_relation = "exact-terminal-tie"
        elif lower >= -tolerance and upper <= tolerance:
            realised_relation = "estimated-within-tolerance"
        else:
            realised_relation = "uncertain"
        if realised_relation in {"positive", "negative"}:
            agreement = predicted_relation == realised_relation
        elif realised_relation == "exact-terminal-tie":
            agreement = predicted_relation == "predicted-tie"
    return {
        "pathReplicate": intervention.path_replicate,
        "nestedUnit": "stochastic path within one world; not an independent world",
        "predictionDefinition": "intervention minus baseline r + gamma * V(successor)",
        "predictedDifference": float(predicted),
        "predictedRelation": predicted_relation,
        "targetDefinition": (
            "intervention minus baseline critic-independent fixed prefix plus paired "
            "randomized continuation tail"
        ),
        "fixedPrefixDifference": float(prefix),
        "pairedTailDifference": _randomized_estimate_json(tail),
        "realisedFullReturnDifference": realised,
        "realisedDifferenceInterval": interval,
        "intervalScope": (
            "approximate stopping-randomization Monte Carlo interval conditional on "
            "this paired observed path; not a full-policy or finite-world confidence interval"
        ),
        "uncertaintyZ": uncertainty_z,
        "tieTolerance": tolerance,
        "realisedRelation": realised_relation,
        "predictionTargetRelationAgreement": agreement,
    }


def _action_receipt_json(value: ActionInterventionReceipt) -> Json:
    return {
        "camp": value.camp_id,
        "beforeSha256": value.before_sha256,
        "afterSha256": value.after_sha256,
        "changedPaths": list(value.changed_paths),
        "exactZeroChange": value.exact_zero_change,
    }


def _branch_json(value: BranchResult) -> Json:
    return {
        "alternative": value.alternative_id,
        "pathReplicate": value.path_replicate,
        "schedule": {
            "id": value.schedule.schedule_id,
            "rngSeed": value.schedule.rng_seed,
            "stopIndices": list(value.schedule.stop_indices),
            "requiredTailTerms": value.schedule.required_terms,
            "exceedsHardCap": value.schedule.exceeds_hard_cap,
        },
        "prefixReturns": dict(value.prefix_returns),
        "tailEstimates": {
            objective: _randomized_estimate_json(estimate)
            for objective, estimate in value.tail_estimates.items()
        },
        "rankingPredictions": dict(value.ranking_predictions),
        "actionReceipts": [_action_receipt_json(item) for item in value.action_receipts],
        "collectedSteps": value.collected_steps,
        "openingCampCount": value.opening_camp_count,
        "terminal": value.terminal,
        "economic": value.economic,
    }


def _aggregate_state(
    runtime: EndpointRuntime,
    state_values: Mapping[ObjectiveName, Mapping[int, float]],
    critic_diagnostics: Mapping[ObjectiveName, Mapping[int, Json]],
    paths: Sequence[BranchResult],
) -> Json:
    by_alternative = {
        alternative: sorted(
            (path for path in paths if path.alternative_id == alternative),
            key=lambda path: path.path_replicate,
        )
        for alternative in FROZEN_ALTERNATIVES
    }
    if any(len(items) != FROZEN_PATH_REPLICATES for items in by_alternative.values()):
        raise ValueError("frozen state lacks its complete paired path set")
    combined: dict[str, Json] = {}
    paired: dict[str, Json] = {}
    for objective in runtime.objectives:
        for alternative, items in by_alternative.items():
            estimate = combine_independent_path_returns(
                [item.prefix_returns[objective] for item in items],
                [item.tail_estimates[objective] for item in items],
                f"{items[0].schedule.schedule_id.rsplit('/', 1)[0]}/{alternative}/paths",
            )
            combined.setdefault(alternative, {})[objective] = _randomized_estimate_json(estimate)

        baseline = by_alternative["sampled-policy"]
        intervention = by_alternative["development-budget-native-v1"]
        paired_tails: list[RandomizedTailEstimate] = []
        prefix_differences: list[float] = []
        ranking_differences: list[float] = []
        complete_differences: list[float] = []
        per_path_diagnostics: list[Json] = []
        for first, second in zip(intervention, baseline, strict=True):
            if first.schedule != second.schedule:
                raise ValueError("paired alternatives use different stopping schedules")
            tail = _paired_tail_from_results(
                first.tail_estimates[objective], second.tail_estimates[objective], first.schedule
            )
            paired_tails.append(tail)
            prefix_difference = first.prefix_returns[objective] - second.prefix_returns[objective]
            prefix_differences.append(prefix_difference)
            ranking_differences.append(
                first.ranking_predictions[objective] - second.ranking_predictions[objective]
            )
            if tail.identified:
                assert tail.estimate is not None
                complete_differences.append(prefix_difference + tail.estimate)
            per_path_diagnostics.append(paired_path_diagnostic(first, second, objective))
        paired_estimate = combine_independent_path_returns(
            prefix_differences,
            paired_tails,
            f"{baseline[0].schedule.schedule_id.rsplit('/', 1)[0]}/paired-difference/paths",
        )
        paired[objective] = {
            "interventionMinusBaseline": _randomized_estimate_json(paired_estimate),
            "oneStepRankingDirectionAllPaths": direction_counts(ranking_differences),
            "fullReturnDirectionIdentifiedPaths": (
                direction_counts(complete_differences) if complete_differences else None
            ),
            "fullReturnDirectionDenominatorNote": (
                "only identified paths; unresolved tails are reported, never treated as ties"
            ),
            "perPathPredictionVsIndependentReturn": per_path_diagnostics,
            "nestedPathAgreement": {
                "identifiedNumerator": sum(
                    item["predictionTargetRelationAgreement"] is not None
                    for item in per_path_diagnostics
                ),
                "agreementNumerator": sum(
                    item["predictionTargetRelationAgreement"] is True
                    for item in per_path_diagnostics
                ),
                "pathDenominator": len(per_path_diagnostics),
                "identifiedFraction": (
                    sum(
                        item["predictionTargetRelationAgreement"] is not None
                        for item in per_path_diagnostics
                    )
                    / len(per_path_diagnostics)
                ),
                "agreementAmongIdentified": (
                    sum(
                        item["predictionTargetRelationAgreement"] is True
                        for item in per_path_diagnostics
                    )
                    / sum(
                        item["predictionTargetRelationAgreement"] is not None
                        for item in per_path_diagnostics
                    )
                    if any(
                        item["predictionTargetRelationAgreement"] is not None
                        for item in per_path_diagnostics
                    )
                    else None
                ),
                "independentUnit": "world",
                "pathRole": "nested stochastic continuation replicate",
            },
        }

    calibration: dict[str, Json] = {}
    baseline_combined = combined["sampled-policy"]
    for objective in runtime.objectives:
        raw_value = float(sum(state_values[objective].values()))
        estimate = baseline_combined[objective]
        calibration[objective] = {
            "criticRawWorldValueOverOpeningCamps": raw_value,
            "target": estimate,
            "signedTargetMinusCritic": (
                float(estimate["estimate"]) - raw_value
                if estimate["status"] == "identified"
                else None
            ),
            "scope": "baseline sampled-policy path only",
        }
    support_diagnostics: dict[str, Json] = {}
    baseline_paths = by_alternative["sampled-policy"]
    for objective in runtime.objectives:
        camps: list[Json] = []
        below = 0
        above = 0
        identified = 0
        for camp, prediction in sorted(critic_diagnostics[objective].items()):
            camp_target = combine_independent_path_returns(
                [path.camp_prefix_returns[objective][camp] for path in baseline_paths],
                [path.camp_tail_estimates[objective][camp] for path in baseline_paths],
                (
                    f"{baseline_paths[0].schedule.schedule_id.rsplit('/', 1)[0]}/"
                    f"baseline/camp-{camp}/paths"
                ),
            )
            target_raw = camp_target.estimate if camp_target.identified else None
            scale = float(prediction["scale"])
            target_scaled = None if target_raw is None else float(target_raw) / scale
            lower, upper = cast(list[float], prediction["supportPerResident"])
            is_below = target_scaled is not None and target_scaled < lower
            is_above = target_scaled is not None and target_scaled > upper
            identified += int(target_scaled is not None)
            below += int(is_below)
            above += int(is_above)
            camps.append(
                {
                    "camp": camp,
                    **prediction,
                    "baselineTarget": _randomized_estimate_json(camp_target),
                    "targetPerResidentOrRaw": target_scaled,
                    "irreducibleTargetSupportExceedance": {
                        "identified": target_scaled is not None,
                        "belowLower": is_below if target_scaled is not None else None,
                        "aboveUpper": is_above if target_scaled is not None else None,
                        "amountBelow": (
                            max(0.0, lower - target_scaled) if target_scaled is not None else None
                        ),
                        "amountAbove": (
                            max(0.0, target_scaled - upper) if target_scaled is not None else None
                        ),
                    },
                }
            )
        support_diagnostics[objective] = {
            "camps": camps,
            "identifiedTargetCount": identified,
            "belowSupportCount": below,
            "aboveSupportCount": above,
            "interpretation": (
                "target outside loaded checkpoint bins is irreducible support clipping; "
                "inside-support error is ordinary calibration error. This diagnostic "
                "does not attribute any historical performance change."
            ),
        }
    return {
        "baselineValueCalibration": calibration,
        "criticDistributionAndSupport": support_diagnostics,
        "alternativePolicyExpectations": combined,
        "pairedSensitivity": paired,
    }


def _collect_frozen_state(
    *,
    bridge: Model9Bridge,
    current: Json,
    runtime: EndpointRuntime,
    protocol: FrozenCollectionProtocol,
    world_seed: int,
    regime_id: str,
    decision_step: int,
    history: Sequence[Json],
    main_personalities: PersonalityState,
    previous_mixes: Mapping[int, tuple[float, ...]],
    capture_economic: bool,
    path_progress: Callable[[Json], None] | None = None,
    anchor_capture: Callable[
        [Sequence[int], Mapping[str, NDArray[Any]], Sequence[BranchResult]], None
    ]
    | None = None,
) -> Json:
    opening = _live_villages(current)
    state_values, critic_diagnostics = _critic_predictions(runtime, opening, main_personalities)
    state_record: Json = {
        "worldSeed": world_seed,
        "regime": regime_id,
        "decisionStep": decision_step,
        "turn": int(current["turn"]),
        "fingerprint": str(current["fingerprint"]),
        "openingCamps": [
            {
                "camp": int(village["camp"]),
                "priorPopulation": int(village.get("population", 0)),
            }
            for village in opening
        ],
        "criticValuesByCamp": {
            objective: {str(camp): value for camp, value in values.items()}
            for objective, values in state_values.items()
        },
        "openingPersonalityContext": [
            {
                "camp": int(village["camp"]),
                "society": int(village["society"]),
                "parentSociety": int(village.get("parentSociety", 0)),
                "weights": (
                    [float(value) for value in main_personalities.traits[int(village["society"])]]
                    if main_personalities.active
                    else None
                ),
            }
            for village in opening
        ],
        "personalityContextScope": (
            "traits used for the frozen-state critic predictions and inherited by "
            "each branch clone; future daughter draws use each path's bound seed"
        ),
    }
    parent_batch = cast(Json, ppo.village_batch(opening))
    main_personalities.add_to_batch(parent_batch, opening)
    parent_batch["prior_population"] = np.asarray(
        [int(village["population"]) for village in opening], dtype=np.float64
    )
    paths: list[BranchResult] = []
    with ForkedAlternatives(bridge) as forks:
        probe_step = forks.begin("native-label-probe")
        native_labels = probe_native_labels(bridge, probe_step)
        for path_replicate in range(protocol.path_replicates):
            schedule = protocol.stopping_schedule(
                runtime.receipt.node_id,
                world_seed,
                regime_id,
                decision_step,
                path_replicate,
            )
            action_seed = _derived_seed(
                "gate34-action-v1",
                runtime.receipt.node_id,
                world_seed,
                regime_id,
                decision_step,
                path_replicate,
            )
            personality_seed = _derived_seed(
                "gate34-personality-v1",
                runtime.receipt.node_id,
                world_seed,
                regime_id,
                decision_step,
                path_replicate,
            )
            for alternative_id in protocol.alternatives:
                branch_step = forks.begin(f"path-{path_replicate}/{alternative_id}")
                started = monotonic()
                result = _execute_branch(
                    bridge=bridge,
                    decision_step=branch_step,
                    runtime=runtime,
                    protocol=protocol,
                    world_seed=world_seed,
                    history=history,
                    main_personalities=main_personalities,
                    previous_mixes=previous_mixes,
                    native_labels=native_labels,
                    alternative_id=alternative_id,
                    path_replicate=path_replicate,
                    schedule=schedule,
                    action_seed=action_seed,
                    personality_seed=personality_seed,
                    capture_economic=capture_economic,
                    capture_anchor=anchor_capture is not None,
                )
                paths.append(result)
                if path_progress is not None:
                    progress = _branch_json(result)
                    progress["maximumSteps"] = protocol.prefix_terms + (
                        0 if schedule.exceeds_hard_cap else schedule.required_terms
                    )
                    progress["durationSeconds"] = monotonic() - started
                    path_progress(progress)
    if anchor_capture is not None:
        anchor_capture(
            tuple(int(village["camp"]) for village in opening),
            {key: np.asarray(value).copy() for key, value in parent_batch.items()},
            tuple(paths),
        )
    state_record["paths"] = [_branch_json(path) for path in paths]
    state_record["summary"] = _aggregate_state(runtime, state_values, critic_diagnostics, paths)
    return state_record


BridgeFactory = Callable[[bool], Model9Bridge]


def _default_bridge_factory(economic: bool) -> Model9Bridge:
    return Model9Bridge(economic_diagnostics=economic)


def collect_endpoint(
    registry: EndpointRegistry,
    receipt: EndpointReceipt,
    *,
    protocol: FrozenCollectionProtocol = FROZEN_PROTOCOL,
    capture_economic: bool = False,
    bridge_factory: BridgeFactory = _default_bridge_factory,
    only_world_seed: int | None = None,
) -> Json:
    """Collect one endpoint without changing the frozen endpoint/world registry.

    ``only_world_seed`` is the explicitly labelled runtime-calibration mode.  It may
    only choose the first prescribed world and its output remains incomplete; the
    protocol and endpoint registry in the receipt are unchanged.
    """

    if only_world_seed not in {None, protocol.world_seeds[0]}:
        raise ProvenanceError("runtime calibration may use only the first prescribed world")
    if checkpoint_hash(registry.bridge_path) != registry.bridge_sha256:
        raise ProvenanceError("bound bridge bytes changed before collection")
    if DEFAULT_BRIDGE.resolve() != registry.bridge_path:
        configured = os.environ.get("ANTHROPOCENE_MODEL9_BRIDGE")
        if not configured or Path(configured).resolve() != registry.bridge_path:
            raise ProvenanceError("runtime bridge path differs from the Gate-1 receipt")
    runtime = load_endpoint_runtime(registry, receipt)
    selected_worlds = (
        (protocol.world_seeds[0],) if only_world_seed is not None else protocol.world_seeds
    )
    states: list[Json] = []
    producer_receipts: list[Json] = []
    for world_seed in selected_worlds:
        for regime_id, withheld in protocol.regimes:
            bridge = bridge_factory(capture_economic)
            try:
                schema_hash = getattr(bridge.schema, "bridge_sha256", None)
                if schema_hash is not None and schema_hash != registry.bridge_sha256:
                    raise ProvenanceError("live bridge schema differs from the Gate-1 receipt")
                current = bridge.reset(
                    world_seed,
                    movement=runtime.stage.bridge_movement,
                    withheld=withheld,
                )
                if int(current["turn"]) != 9:
                    raise ProvenanceError(
                        "fresh generated world post-reset boundary differs from turn 9"
                    )
                producer_receipts.append(
                    {
                        "worldSeed": world_seed,
                        "regime": regime_id,
                        "freshWorldDescriptor": {
                            "turn": None,
                            "snapshot": None,
                            "expectedPostResetTurn": 9,
                        },
                        "resetIdentity": copy.deepcopy(bridge.last_reset_identity),
                        "simulatorCommit": getattr(bridge.schema, "simulator_commit", None),
                        "simulatorRng": (
                            "conditional on exact reset and in-process fork state; "
                            "no invented resampling"
                        ),
                    }
                )
                main_action_seed = _derived_seed(
                    "gate34-main-action-v1", receipt.node_id, world_seed, regime_id
                )
                main_personality_seed = _derived_seed(
                    "gate34-main-personality-v1", receipt.node_id, world_seed, regime_id
                )
                generator = torch.Generator(device="cpu")
                generator.manual_seed(main_action_seed)
                personalities = PersonalityState(
                    runtime.personality_spec,
                    runtime.personality_drift,
                    main_personality_seed,
                )
                previous_mixes: dict[int, tuple[float, ...]] = {}
                history: list[Json] = []
                decision_count = 0
                while decision_count <= max(protocol.decision_steps):
                    if decision_count in protocol.decision_steps:
                        states.append(
                            _collect_frozen_state(
                                bridge=bridge,
                                current=current,
                                runtime=runtime,
                                protocol=protocol,
                                world_seed=world_seed,
                                regime_id=regime_id,
                                decision_step=decision_count,
                                history=history,
                                main_personalities=personalities,
                                previous_mixes=previous_mixes,
                                capture_economic=capture_economic,
                            )
                        )
                    if decision_count == max(protocol.decision_steps):
                        break
                    sampled = sample_training_actions(runtime, current, generator, personalities)
                    if not sampled.actions:
                        raise ValueError("main path terminated before a prescribed frozen state")
                    after = bridge.step(sampled.actions)
                    history.extend(
                        _outcome_rows(
                            world_seed=world_seed,
                            before=current,
                            after=after,
                            sampled=sampled,
                            previous_mixes=previous_mixes,
                            runtime=runtime,
                        )
                    )
                    previous_mixes.update(sampled.mixes)
                    current = after
                    decision_count += 1
            finally:
                bridge.close()
    return {
        "kind": "gate34-checkpoint-continuation-evidence",
        "schema": 1,
        "status": (
            "runtime-calibration-incomplete"
            if only_world_seed is not None
            else "collection-complete-no-gate-claim"
        ),
        "registrySha256": registry.registry_sha256,
        "protocol": protocol.as_dict(),
        "protocolSha256": protocol.protocol_sha256,
        "allFrozenEndpointIds": list(FROZEN_ENDPOINT_IDS),
        "executedEndpointId": receipt.node_id,
        "executedWorldSeeds": list(selected_worlds),
        "checkpoint": receipt.checkpoint_sha256,
        "critic": receipt.critic_typed_content_sha256,
        "returnContract": runtime.return_contract_id,
        "rewardSemantics": runtime.reward_semantics,
        "personality": runtime.personality_spec,
        "logStdFloor": runtime.log_std_floor,
        "producerReceipts": producer_receipts,
        "states": states,
        "inferenceScope": (
            "preparation/current-simulator conditional diagnostic; worlds are independent "
            "units; no readiness, treatment selection, or Gate 3/4 passage claim"
        ),
    }


def _write_sealed_json(path: Path, value: Json) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite collection evidence: {path}")
    body = dict(value)
    if "digest" in body:
        raise ValueError("collection output is already sealed")
    body["digest"] = digest(body)
    encoded = json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale collection temporary exists: {temporary}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def run_frozen_collection(
    freeze_root: Path,
    output_directory: Path,
    *,
    retrieval_root: Path | None = None,
    capture_economic: bool = False,
    runtime_calibration_first_only: bool = False,
) -> Json:
    """Execute the frozen registry, or only its explicit first endpoint/world calibration.

    The calibration switch is not an endpoint selector: its manifest retains the full
    endpoint/world protocol and is permanently labelled incomplete.  Full execution has
    no arbitrary endpoint subset option, preventing post-outcome pruning.
    """

    registry = load_endpoint_registry(freeze_root, retrieval_root=retrieval_root)
    output_directory.mkdir(parents=True, exist_ok=False)
    receipts = registry.endpoints[:1] if runtime_calibration_first_only else registry.endpoints
    outputs: list[Json] = []
    for receipt in receipts:
        result = collect_endpoint(
            registry,
            receipt,
            capture_economic=capture_economic,
            only_world_seed=(FROZEN_WORLD_SEEDS[0] if runtime_calibration_first_only else None),
        )
        filename = f"{receipt.node_id}.json"
        _write_sealed_json(output_directory / filename, result)
        outputs.append(
            {
                "endpoint": receipt.node_id,
                "path": filename,
                "sha256": checkpoint_hash(output_directory / filename),
            }
        )
    manifest = {
        "kind": "gate34-checkpoint-continuation-manifest",
        "schema": 1,
        "status": (
            "runtime-calibration-incomplete"
            if runtime_calibration_first_only
            else "collection-complete-no-gate-claim"
        ),
        "registrySha256": registry.registry_sha256,
        "protocol": FROZEN_PROTOCOL.as_dict(),
        "protocolSha256": FROZEN_PROTOCOL.protocol_sha256,
        "allFrozenEndpointIds": list(FROZEN_ENDPOINT_IDS),
        "executed": outputs,
        "captureEconomic": capture_economic,
        "selectionAllowed": False,
        "readinessClaimed": False,
        "gatePassedClaimed": False,
    }
    _write_sealed_json(output_directory / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen, non-readiness Gate 3/4 checkpoint diagnostic collector"
    )
    parser.add_argument("--freeze-package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--retrieval-root", type=Path)
    parser.add_argument("--capture-economic", action="store_true")
    parser.add_argument(
        "--runtime-calibration-first-only",
        action="store_true",
        help=(
            "execute only the first prescribed endpoint/world as an incomplete runtime "
            "calibration; it cannot modify the frozen design"
        ),
    )
    parser.add_argument(
        "--intervention",
        choices=(DEVELOPMENT_NATIVE_INTERVENTION.intervention_id,),
        default=DEVELOPMENT_NATIVE_INTERVENTION.intervention_id,
        help="the only CLI-authorized, hash-bound first-action intervention",
    )
    args = parser.parse_args(argv)
    DEVELOPMENT_NATIVE_INTERVENTION.validate_identity()
    run_frozen_collection(
        args.freeze_package,
        args.out,
        retrieval_root=args.retrieval_root,
        capture_economic=args.capture_economic,
        runtime_calibration_first_only=args.runtime_calibration_first_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
