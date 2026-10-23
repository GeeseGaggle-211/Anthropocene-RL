"""Bounded observed-state retention for the seed-4242 optimizer-direction replay.

This is deliberately a smoke diagnostic helper, not a training or gate runner.
It observes one existing six-turn PPO smoke without consuming RNG, stores six
world-only native snapshots and observed policy contexts, and captures the fresh-
Adam actor's first real minibatch mutation.  Proposed decision reconstruction uses
the original generated reset arguments plus the ordered baseline action prefix;
captured snapshots are never represented as complete bridge state.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import torch

from .economic_trace import state_digest

Json = dict[str, Any]

RETAINED_TURNS = tuple(range(9, 15))
STRATEGIC_DEVELOPMENT_INDEX = 2
STRATEGIC_NEIGHBORHOOD_RADIUS = 0.01
_ROOT_STEP_METADATA_KEYS = frozenset(
    {"fingerprint", "nativeMs", "restoredObservation"}
)
_BRANCH_LINEAGE_KEYS = frozenset(
    {
        "trajectoryId",
        "transitionId",
        "sourceTransitionId",
        "parentTransitionId",
        "branchId",
        "forkId",
        "controlId",
        "eventId",
        "requestId",
        "transactionId",
        "legId",
        "receiptId",
    }
)
_PURE_BRANCH_KEYS = frozenset({"parentTransitionId", "branchId", "forkId"})


def _encoded(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _encoded(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_encoded(child) for child in value]
    return value


def value_digest(value: Any) -> str:
    blob = json.dumps(
        _encoded(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(blob).hexdigest()


def _keyed_values(
    value: Any, names: frozenset[str], path: str = "$"
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key) in names:
                result[child_path] = copy.deepcopy(child)
            result.update(_keyed_values(child, names, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            result.update(_keyed_values(child, names, f"{path}[{index}]"))
    return result


def _canonical_branch_semantics(
    value: Any,
    trajectory: str | None,
    *,
    path: tuple[str, ...] = (),
    lineage_key: str | None = None,
) -> Any:
    """Normalize only documented branch metadata paths and lineage-key values."""

    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            name = str(key)
            if not path and name in _ROOT_STEP_METADATA_KEYS:
                continue
            # Native economic rows document these three branch-only fields at
            # exactly economicDiagnostics.<camp>.<field>.  Same-named fields at
            # any other path remain content-bound.
            if (
                len(path) == 2
                and path[0] == "economicDiagnostics"
                and name in _PURE_BRANCH_KEYS
            ):
                continue
            result[name] = _canonical_branch_semantics(
                child,
                trajectory,
                path=(*path, name),
                lineage_key=name,
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _canonical_branch_semantics(
                child,
                trajectory,
                path=(*path, str(index)),
                lineage_key=lineage_key,
            )
            for index, child in enumerate(value)
        ]
    if (
        isinstance(value, str)
        and trajectory
        and lineage_key in _BRANCH_LINEAGE_KEYS
    ):
        return value.replace(trajectory, "$TRAJECTORY")
    return value


def _step_identity_bundle(step: Json) -> Json:
    """Separate observed step content from branch-local identity metadata.

    The native save/load boundary may change the full-world ``fingerprint`` and
    marks an observation as restored.  A fork also creates new trajectory-derived
    IDs.  Those values are retained below and are not called physical-state
    equality.  Public observations and realised reward inputs are bound separately.
    """

    if not isinstance(step, Mapping) or "turn" not in step or "villages" not in step:
        raise ValueError("native step identity requires turn and villages")
    trajectory_value = step.get("trajectoryId")
    trajectory = str(trajectory_value) if trajectory_value is not None else None
    lineage = _keyed_values(step, _BRANCH_LINEAGE_KEYS)
    branch_only = _keyed_values(step, _PURE_BRANCH_KEYS)
    observed = _canonical_branch_semantics(step, trajectory)
    return {
        "fullStepDigest": value_digest(step),
        "observedStepDigest": value_digest(observed),
        "policyObservationDigest": value_digest(
            [
                {
                    "camp": village.get("camp"),
                    "society": village.get("society"),
                    "parentSociety": village.get("parentSociety"),
                    "player": village.get("player"),
                    "terminated": village.get("terminated"),
                    "observation": village.get("observation"),
                }
                for village in step["villages"]
            ]
        ),
        "rewardInputDigest": value_digest(
            {
                "turn": step["turn"],
                "villages": [
                    {
                        key: village.get(key)
                        for key in (
                            "camp",
                            "society",
                            "parentSociety",
                            "priorPopulation",
                            "population",
                            "qol",
                            "nutrition",
                            "years",
                            "welfare",
                            "terminated",
                            "tradeUtility",
                        )
                    }
                    for village in step["villages"]
                ],
            }
        ),
        "branchMetadata": {
            "fingerprint": step.get("fingerprint"),
            "nativeMs": step.get("nativeMs"),
            "restoredObservation": step.get("restoredObservation"),
            "trajectoryId": trajectory_value,
            "lineageFieldPaths": sorted(lineage),
            "lineageFieldDigest": value_digest(lineage),
            "pureBranchFieldPaths": sorted(branch_only),
            "pureBranchFieldDigest": value_digest(branch_only),
        },
    }


def _compare_step_identity(expected: Json, actual_step: Json, label: str) -> Json:
    actual = _step_identity_bundle(actual_step)
    observed_fields = (
        "observedStepDigest",
        "policyObservationDigest",
        "rewardInputDigest",
    )
    differing = [
        field for field in observed_fields if expected.get(field) != actual[field]
    ]
    if differing:
        raise ValueError(f"{label} observed step/policy/reward content differs: {differing}")
    expected_metadata = expected["branchMetadata"]
    actual_metadata = actual["branchMetadata"]
    for field in ("lineageFieldPaths", "pureBranchFieldPaths"):
        if expected_metadata.get(field) != actual_metadata[field]:
            raise ValueError(f"{label} branch metadata shape differs: {field}")
    return {
        "observedStepDigest": actual["observedStepDigest"],
        "policyObservationDigest": actual["policyObservationDigest"],
        "rewardInputDigest": actual["rewardInputDigest"],
        "recordedFullStepDigest": expected["fullStepDigest"],
        "replayedFullStepDigest": actual["fullStepDigest"],
        "branchMetadata": {
            "recorded": expected_metadata,
            "replayed": actual_metadata,
            "lineageMapping": {
                "recordedTrajectoryId": expected_metadata.get("trajectoryId"),
                "replayedTrajectoryId": actual_metadata.get("trajectoryId"),
                "canonicalToken": "$TRAJECTORY",
            },
        },
        "fullWorldEquivalenceProven": False,
        "fullWorldEquivalenceReason": (
            "native fingerprint is a full-world hash and differs after fork/save-load "
            "normalization; public JSON equality cannot establish latent-world equality"
        ),
    }


def _module_state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _clone_arrays(value: Mapping[str, Any]) -> Json:
    return {
        str(key): (
            child.detach().cpu().clone()
            if isinstance(child, torch.Tensor)
            else child.copy()
            if isinstance(child, np.ndarray)
            else copy.deepcopy(child)
        )
        for key, child in value.items()
    }


def _rng_state(
    generator: torch.Generator, numpy_generator: np.random.Generator
) -> Json:
    return {
        "torchGlobal": torch.get_rng_state().clone(),
        "policyGenerator": generator.get_state().clone(),
        "numpyMinibatch": copy.deepcopy(numpy_generator.bit_generator.state),
        "numpyGlobal": copy.deepcopy(np.random.get_state()),
        "python": copy.deepcopy(random.getstate()),
    }


class _RetainedBridge:
    def __init__(self, bridge: Any, owner: SmokeAuditRetention) -> None:
        self._bridge = bridge
        self._owner = owner
        self._current: Json | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bridge, name)

    def reset(self, *args: Any, **kwargs: Any) -> Json:
        if self._owner.reset_record is not None:
            raise RuntimeError("optimizer audit retention observed more than one reset")
        seed = kwargs.get("seed", args[0] if args else 0)
        players = kwargs.get("players", 8)
        movement = kwargs.get("movement", False)
        withheld = kwargs.get("withheld", 0)
        size = kwargs.get("size")
        snapshot = kwargs.get("snapshot")
        if snapshot is not None or size is not None:
            raise ValueError(
                "bounded optimizer retention requires a generated default-size world"
            )
        current = self._bridge.reset(*args, **kwargs)
        self._current = copy.deepcopy(current)
        self._owner.reset_record = {
            "arguments": {
                "seed": int(seed),
                "players": int(players),
                "movement": bool(movement),
                "withheld": int(withheld),
                "size": None,
                "snapshotProvided": False,
            },
            "lastResetIdentity": copy.deepcopy(
                getattr(self._bridge, "last_reset_identity", None)
            ),
            "observedStepIdentity": _step_identity_bundle(current),
        }
        return current

    def step(self, actions: Mapping[int, Json]) -> Json:
        if self._current is None:
            raise RuntimeError("retained bridge was stepped before reset")
        snapshot = base64.b64decode(self._bridge.capture(), validate=True)
        before = self._current
        successor = self._bridge.step(actions)
        self._owner.native_records.append(
            {
                "preTurn": int(before["turn"]),
                "preFingerprint": str(before["fingerprint"]),
                "preStepIdentity": _step_identity_bundle(before),
                "snapshot": snapshot,
                "snapshotSha256": hashlib.sha256(snapshot).hexdigest(),
                "completeActionMap": copy.deepcopy(dict(actions)),
                "postTurn": int(successor["turn"]),
                "postFingerprint": str(successor["fingerprint"]),
                "successorStepIdentity": _step_identity_bundle(successor),
            }
        )
        self._current = copy.deepcopy(successor)
        return successor


class SmokeAuditRetention:
    """Observe one six-turn smoke while restoring every monkeypatch on exit."""

    def __init__(
        self,
        ppo_module: Any,
        actor: torch.nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        generator: torch.Generator,
        numpy_generator: np.random.Generator,
    ) -> None:
        self.ppo = ppo_module
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.generator = generator
        self.numpy_generator = numpy_generator
        self.policy_records: list[Json] = []
        self.native_records: list[Json] = []
        self.reset_record: Json | None = None
        self.first_actor_step: Json | None = None
        self._last_batch: Json | None = None
        self._last_output: Json | None = None
        self._originals: Json = {}

    def __enter__(self) -> SmokeAuditRetention:
        self._originals = {
            "Model9Bridge": self.ppo.Model9Bridge,
            "to_tensors": self.ppo.to_tensors,
            "explore": self.ppo.explore,
            "to_bridge_actions": self.ppo.to_bridge_actions,
            "optimizer_step": self.actor_optimizer.step,
        }
        owner = self

        def bridge_factory(*args: Any, **kwargs: Any) -> _RetainedBridge:
            return _RetainedBridge(
                owner._originals["Model9Bridge"](*args, **kwargs), owner
            )

        def to_tensors(batch: Json, *args: Any, **kwargs: Any) -> Json:
            tensors = owner._originals["to_tensors"](batch, *args, **kwargs)
            owner._last_batch = _clone_arrays(batch)
            return tensors

        def explore(output: Json, *args: Any, **kwargs: Any) -> Json:
            explored = owner._originals["explore"](output, *args, **kwargs)
            owner._last_output = _clone_arrays(explored)
            return explored

        def to_bridge_actions(draw: Json, tensors: Json) -> list[Json]:
            actions = owner._originals["to_bridge_actions"](draw, tensors)
            if owner._last_batch is None or owner._last_output is None:
                raise RuntimeError("policy context capture is incomplete")
            owner.policy_records.append(
                {
                    "rawPolicyBatch": _clone_arrays(owner._last_batch),
                    "packedPolicyInputs": _clone_arrays(tensors),
                    "actorDistribution": _clone_arrays(owner._last_output),
                    "sampledLatents": _clone_arrays(draw),
                    "decodedActionList": copy.deepcopy(actions),
                }
            )
            return actions

        def optimizer_step(*args: Any, **kwargs: Any) -> Any:
            if owner.first_actor_step is not None:
                return owner._originals["optimizer_step"](*args, **kwargs)
            before_weights = copy.deepcopy(owner.actor.state_dict())
            before_adam = copy.deepcopy(owner.actor_optimizer.state_dict())
            before_rng = _rng_state(owner.generator, owner.numpy_generator)
            result = owner._originals["optimizer_step"](*args, **kwargs)
            owner.first_actor_step = {
                "weightsBefore": before_weights,
                "weightsAfter": copy.deepcopy(owner.actor.state_dict()),
                "traceActorHashBefore": _module_state_digest(before_weights),
                "traceActorHashAfter": state_digest(owner.actor),
                "adamBefore": before_adam,
                "adamAfter": copy.deepcopy(owner.actor_optimizer.state_dict()),
                "rngBefore": before_rng,
                "rngAfter": _rng_state(owner.generator, owner.numpy_generator),
                "semantics": (
                    "prospective replay with freshly constructed Adam; not a recovered "
                    "historical optimizer state"
                ),
            }
            return result

        self.ppo.Model9Bridge = bridge_factory
        self.ppo.to_tensors = to_tensors
        self.ppo.explore = explore
        self.ppo.to_bridge_actions = to_bridge_actions
        self.actor_optimizer.step = optimizer_step
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.ppo.Model9Bridge = self._originals["Model9Bridge"]
        self.ppo.to_tensors = self._originals["to_tensors"]
        self.ppo.explore = self._originals["explore"]
        self.ppo.to_bridge_actions = self._originals["to_bridge_actions"]
        self.actor_optimizer.step = self._originals["optimizer_step"]

    def payload(
        self,
        *,
        normalizer_state: Json,
        rollout: Any,
        stage: Any,
        log_std_floor: float,
    ) -> Json:
        if self.first_actor_step is None:
            raise ValueError("retained smoke did not perform an actor optimizer step")
        if self.reset_record is None:
            raise ValueError("retained smoke did not observe its generated reset")
        if len(self.native_records) != len(RETAINED_TURNS):
            raise ValueError("retained smoke did not produce exactly six native decisions")
        if len(self.policy_records) != len(self.native_records):
            raise ValueError("native and policy retention counts differ")
        if tuple(record["preTurn"] for record in self.native_records) != RETAINED_TURNS:
            raise ValueError("retained native decisions are not turns 9 through 14")
        records = []
        for native, policy in zip(
            self.native_records, self.policy_records, strict=True
        ):
            action_map = native["completeActionMap"]
            if value_digest(list(action_map.values())) != value_digest(
                policy["decodedActionList"]
            ):
                raise ValueError("policy action order differs from native complete map")
            rows = len(policy["rawPolicyBatch"]["core"])
            if rows != len(action_map):
                raise ValueError("retained policy rows do not cover every native action")
            records.append(
                {
                    **native,
                    **policy,
                    "campIdsInPolicyRowOrder": [int(camp) for camp in action_map],
                    "rawPolicyBatchDigest": value_digest(policy["rawPolicyBatch"]),
                    "packedPolicyInputDigest": value_digest(
                        policy["packedPolicyInputs"]
                    ),
                    "sampledLatentDigest": value_digest(policy["sampledLatents"]),
                    "completeActionMapDigest": value_digest(action_map),
                }
            )
        return {
            "schemaVersion": 2,
            "kind": "seed-4242-bounded-optimizer-audit-retention",
            "records": records,
            "resetReconstruction": copy.deepcopy(self.reset_record),
            "tracePostResetWorldSnapshotSha256": records[0]["snapshotSha256"],
            "snapshotSemantics": (
                "all captures are post-Observe world-only evidence because capture omits "
                "bridge-local previous-decision context; observed replay generates "
                "the original seed/config reset and replays the complete action prefix"
            ),
            "firstActorStep": self.first_actor_step,
            "normalizer": copy.deepcopy(normalizer_state),
            "stage": copy.deepcopy(vars(stage)),
            "logStdFloor": float(log_std_floor),
            "rolloutObservationDigest": value_digest(rollout.obs),
            "rolloutDrawDigest": value_digest(rollout.draws),
            "rolloutLogProbabilityDigest": value_digest(rollout.logp),
            "rolloutOutcomeDigest": value_digest(rollout.outcome),
            "neighborhoodPlan": {
                "head": "strategic",
                "coordinate": STRATEGIC_DEVELOPMENT_INDEX,
                "decodedUnitRadius": STRATEGIC_NEIGHBORHOOD_RADIUS,
                "interval": "[max(0,a-0.01), min(1,a+0.01)]",
                "transform": "latent-normal-through-sigmoid",
                "endpointRule": (
                    "the analytic continuous-logistic-normal reference maps a bound at "
                    "0 or 1 to the corresponding infinite latent tail"
                ),
                "finitePrecisionExecutionExact": False,
                "finitePrecisionCaveat": (
                    "torch sigmoid and sampled latents use finite precision and can round "
                    "to an endpoint; this analytic mass is not a bit-exact probability "
                    "over that discrete floating-point execution"
                ),
                "densityKeptSeparate": True,
            },
        }


def decoded_unit_neighborhood_probability(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    decoded_center: torch.Tensor,
    *,
    radius: float = STRATEGIC_NEIGHBORHOOD_RADIUS,
    log_std_floor: float | None = None,
) -> torch.Tensor:
    """Analytic logistic-normal mass in a clipped decoded-unit neighborhood.

    Boundaries at zero and one map to infinite latent bounds, thereby retaining
    the complete tails of the continuous reference distribution.  This is not
    a bit-exact probability over finite-precision ``randn`` and ``sigmoid``;
    those operations can round decoded values to an endpoint.  The diagnostic
    mass remains deliberately separate from sampled-action log density.
    """

    if not math.isfinite(radius) or radius <= 0.0 or radius > 1.0:
        raise ValueError("decoded neighborhood radius must be in (0, 1]")
    if mean.shape != log_std.shape or mean.shape != decoded_center.shape:
        raise ValueError("mean, log_std and decoded_center shapes differ")
    if not (
        torch.isfinite(mean).all()
        and torch.isfinite(log_std).all()
        and torch.isfinite(decoded_center).all()
    ):
        raise ValueError("decoded neighborhood inputs must be finite")
    if bool(((decoded_center < 0.0) | (decoded_center > 1.0)).any()):
        raise ValueError("decoded center must lie in [0, 1]")
    effective_log_std = (
        log_std.clamp(min=float(log_std_floor))
        if log_std_floor is not None
        else log_std
    )
    std = effective_log_std.exp()
    lower = (decoded_center - radius).clamp(0.0, 1.0)
    upper = (decoded_center + radius).clamp(0.0, 1.0)
    lower_z = torch.where(
        lower <= 0.0,
        torch.full_like(lower, -torch.inf),
        torch.logit(lower),
    )
    upper_z = torch.where(
        upper >= 1.0,
        torch.full_like(upper, torch.inf),
        torch.logit(upper),
    )
    lower_cdf = torch.special.ndtr((lower_z - mean) / std)
    upper_cdf = torch.special.ndtr((upper_z - mean) / std)
    return (upper_cdf - lower_cdf).clamp(0.0, 1.0)


def _verify_retained_policy_context(current: Json, record: Json, state: Json) -> Json:
    from . import ppo
    from .model import load_checkpoint_normalizer, to_tensors

    live = [
        village
        for village in current.get("villages", [])
        if village.get("observation") is not None and not bool(village.get("terminated"))
    ]
    camps = [int(village["camp"]) for village in live]
    if camps != [int(camp) for camp in record["campIdsInPolicyRowOrder"]]:
        raise ValueError("prefix replay live-camp policy row order differs")
    reconstructed = ppo.village_batch(live)
    retained = record["rawPolicyBatch"]
    if "personality" in retained:
        reconstructed["personality"] = np.asarray(retained["personality"]).copy()
    if set(reconstructed) != set(retained):
        raise ValueError("prefix replay policy input keys differ")
    differing = [
        key
        for key in reconstructed
        if value_digest(reconstructed[key]) != value_digest(retained[key])
    ]
    if differing:
        raise ValueError(f"prefix replay raw policy inputs differ: {differing}")
    if value_digest(reconstructed) != record["rawPolicyBatchDigest"]:
        raise ValueError("prefix replay raw policy batch digest differs")
    normalizer = load_checkpoint_normalizer(state)
    packed = to_tensors(reconstructed, normalizer)
    if value_digest(packed) != record["packedPolicyInputDigest"]:
        raise ValueError("prefix replay packed policy input digest differs")
    return {
        "campIdsInPolicyRowOrder": camps,
        "rawPolicyBatchDigest": record["rawPolicyBatchDigest"],
        "packedPolicyInputDigest": record["packedPolicyInputDigest"],
    }


def replay_retained_snapshots(
    package: Path,
    bridge_factory: Callable[[], Any],
    *,
    world_seed: int,
    movement: bool,
    withheld: int,
) -> Json:
    """Verify observed prefix policy/reward content from the generated reset.

    Every captured native snapshot omits bridge-local previous-decision context
    and is never reset here.  At every reconstructed target this verifies the
    observed step/policy/reward projections and full observation-derived raw and
    packed policy inputs.  Branch-local full-world fingerprints, restoration
    flags, and trajectory-derived IDs are retained and mapped separately.  This
    is not proof that latent simulator worlds are equal.  Later independent
    alternatives would have to execute inside the same in-process fork scope.
    """

    state = torch.load(package / "retained-state.pt", map_location="cpu", weights_only=False)
    rows = []
    bridge = bridge_factory()
    try:
        for record in state["records"]:
            snapshot_path = package / record["snapshotRelativePath"]
            snapshot = snapshot_path.read_bytes()
            if hashlib.sha256(snapshot).hexdigest() != record["snapshotSha256"]:
                raise ValueError("retained world-only snapshot content hash differs")
        reset = state["resetReconstruction"]
        arguments = reset["arguments"]
        requested = {
            "seed": int(world_seed),
            "players": 8,
            "movement": bool(movement),
            "withheld": int(withheld),
            "size": None,
            "snapshotProvided": False,
        }
        if requested != arguments:
            raise ValueError("replay reset arguments differ from retained smoke")
        current = bridge.reset(
            world_seed,
            movement=movement,
            withheld=withheld,
        )
        if getattr(bridge, "last_reset_identity", None) != reset["lastResetIdentity"]:
            raise ValueError("generated replay reset native identity differs")
        reset_identity = _compare_step_identity(
            reset["observedStepIdentity"], current, "generated replay reset"
        )
        expected_reset_metadata = reset["observedStepIdentity"]["branchMetadata"]
        actual_reset_metadata = reset_identity["branchMetadata"]["replayed"]
        for field in ("fingerprint", "trajectoryId", "restoredObservation"):
            if expected_reset_metadata.get(field) != actual_reset_metadata.get(field):
                raise ValueError(f"generated replay reset metadata differs: {field}")
        from .gate34_continuation_collector import ForkedAlternatives

        for record in state["records"]:
            if int(current["turn"]) != int(record["preTurn"]):
                raise ValueError("ordered prefix replay turn differs")
            decision_identity = _compare_step_identity(
                record["preStepIdentity"], current, "ordered prefix decision"
            )
            context = _verify_retained_policy_context(current, record, state)
            with ForkedAlternatives(bridge) as forks:
                restored = forks.begin(
                    f"verify-turn-{int(record['preTurn'])}-context"
                )
                fork_identity = _compare_step_identity(
                    record["preStepIdentity"], restored, "in-process fork restoration"
                )
                restored_context = _verify_retained_policy_context(
                    restored, record, state
                )
                if restored_context != context:
                    raise ValueError("in-process fork restoration changed policy context")
            successor = bridge.step(record["completeActionMap"])
            successor_identity = _compare_step_identity(
                record["successorStepIdentity"],
                successor,
                "ordered baseline prefix successor",
            )
            rows.append(
                {
                    "turn": int(record["preTurn"]),
                    **context,
                    "decisionIdentity": decision_identity,
                    "forkIdentity": fork_identity,
                    "successorIdentity": successor_identity,
                    "observedDecisionPolicyRewardMatch": True,
                    "forkPolicyContextMatch": True,
                    "fullWorldEquivalenceProven": False,
                }
            )
            current = successor
    finally:
        bridge.close()
    return {
        "status": "observed-step-policy-reward-prefix-verified",
        "generatedResetIdentity": reset_identity,
        "records": rows,
        "scope": (
            "observed baseline step/policy/reward prefix and fork-context check only; "
            "branch-local full-world fingerprints and lineage IDs are retained but not "
            "equated; latent full-world equivalence, independent outcome labels, and "
            "standalone restoration from native snapshots remain unproven"
        ),
    }


def write_retention_package(root: Path, payload: Json, binding: Json) -> Json:
    """Write one immutable bounded retention directory after replay binding passes."""

    if root.exists():
        raise FileExistsError(f"refusing to overwrite optimizer audit retention: {root}")
    root.mkdir(parents=True)
    snapshots = root / "snapshots"
    snapshots.mkdir()
    state_records = []
    snapshot_manifest = []
    for record in payload["records"]:
        record = dict(record)
        snapshot = record.pop("snapshot")
        relative = Path("snapshots") / f"turn-{int(record['preTurn']):02d}.anthro"
        path = root / relative
        path.write_bytes(snapshot)
        record["snapshotRelativePath"] = relative.as_posix()
        state_records.append(record)
        snapshot_manifest.append(
            {
                "turn": int(record["preTurn"]),
                "path": relative.as_posix(),
                "sha256": record["snapshotSha256"],
                "bytes": len(snapshot),
            }
        )
    state = {**payload, "records": state_records}
    state_path = root / "retained-state.pt"
    torch.save(state, state_path)
    manifest = {
        "schemaVersion": 2,
        "kind": payload["kind"],
        "status": "retained-observed-smoke-context-replay-deferred",
        "binding": binding,
        "snapshots": snapshot_manifest,
        "snapshotSemantics": {
            "tracePostResetWorldSnapshotSha256": payload[
                "tracePostResetWorldSnapshotSha256"
            ],
            "allCapturesAreWorldOnlyEvidence": True,
            "standaloneCompleteReplay": False,
            "observedTargetReconstructionPlan": (
                "fresh generated reset with retained seed/config plus ordered complete "
                "baseline action prefix; alternative pairs use in-process fork/restore"
            ),
        },
        "state": {
            "path": state_path.name,
            "sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        },
        "neighborhoodPlan": payload["neighborhoodPlan"],
        "claimsExcluded": [
            "full-world-equivalence",
            "complete-optimizer-replay",
            "historical-optimizer-recovery",
            "beneficial-action-label",
            "readiness",
            "gate-passage",
        ],
    }
    manifest["digest"] = value_digest(manifest)
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest
