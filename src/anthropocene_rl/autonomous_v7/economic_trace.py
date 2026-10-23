"""Prospective diagnostics only: no field is ever passed to an actor or reward function."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from itertools import pairwise
from pathlib import Path

from .diagnostics import canonical

SCHEMA_VERSION = 1
POLICY_INPUT_HEALTH_SCHEMA_VERSION = 1
POLICY_INPUT_OUTLIER_LIMIT = 8
PREVIOUS_DECISION_CORE_INDICES = tuple(range(40, 56))
MOVEMENT_CORE_INDEX = 144
_KNOWN_CORE_LABELS = {
    47: "previous-development-request",
    53: "compiler-intervention-count",
    144: "movement-flag",
}


def canonical_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def merkle_root(leaves: list[bytes]) -> str:
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0].hex()


def _finite(value, path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _finite(child, f"{path}.{i}")
    elif (
        isinstance(value, (float, int)) and not isinstance(value, bool) and not math.isfinite(value)
    ):
        raise ValueError(f"nonfinite diagnostic field {path}")


def validate_economic_step(
    step: dict, *, required=False, initial=False, gate2_required=False
) -> dict[int, dict]:
    raw = step.get("economicDiagnostics", {})
    if not isinstance(raw, dict):
        raise ValueError("economic diagnostics must be keyed by camp")
    if initial or step.get("diagnosticSchemaVersion", 0) == 0:
        if raw:
            raise ValueError("reset cannot carry economic transitions")
        return {}
    if required and step.get("diagnosticSchemaVersion") != 3:
        raise ValueError("economic diagnostics requires envelope 3")
    if raw and step.get("diagnosticSchemaVersion") != 3:
        raise ValueError("economic sidecar outside envelope 3")
    result, seen = {}, set()
    for key, value in raw.items():
        row = canonical(value)
        if (
            row.get("schemaVersion") not in {1, 2}
            or row.get("campId") != int(key)
            or row.get("decisionTurn") != int(step["turn"]) - 1
            or row.get("trajectoryId") != step.get("trajectoryId")
        ):
            raise ValueError("economic schema/camp/turn/trajectory mismatch")
        if gate2_required and row.get("schemaVersion") != 2:
            raise ValueError("Gate 2 requires economic accounting schema 2")
        identity = row.get("transitionId")
        if not identity or identity in seen:
            raise ValueError("missing/duplicate economic transition identity")
        seen.add(identity)
        if not isinstance(row.get("domains"), dict) or not isinstance(row.get("unavailable"), dict):
            raise ValueError("economic domains/missingness absent")
        _finite(row)
        for domain, fields in row["domains"].items():
            for name, item in fields.items():
                if (
                    isinstance(item, (int, float))
                    and not isinstance(item, bool)
                    and item < 0
                    and not any(
                        tag in name.lower()
                        for tag in ("residual", "return", "utility", "gain", "correction", "net")
                    )
                ):
                    raise ValueError(f"negative physical economic field {domain}.{name}")
                if item is None and f"domains.{domain}.{name}" not in row["unavailable"]:
                    raise ValueError(f"missing availability reason {domain}.{name}")
                if name.endswith("Denominator") and item == 0:
                    ratio = name[: -len("Denominator")]
                    if fields.get(ratio) is not None:
                        raise ValueError("economic zero-denominator ratio must be null")
        for group, field in (
            ("goods", "good"),
            ("materials", "material"),
            ("herds", "herdId"),
            ("requests", "requestId"),
        ):
            identities = [r[field] for r in row.get(group, [])]
            if len(set(identities)) != len(identities):
                raise ValueError(f"duplicate economic {group}")
        if row.get("schemaVersion") == 2:
            from .economic_accounting import validate_economic_accounting

            validate_economic_accounting(row)
        result[int(key)] = row
    if required and not result:
        raise ValueError("missing economic diagnostics")
    return result


class EconomicTraceWriter:
    """Synchronous streaming writer: applies backpressure, retains no rollout row queue.

    No done marker is written until both streams have been flushed and closed. Existing
    artifacts are never replaced, so failures remain visibly incomplete.
    """

    def __init__(self, directory: Path, manifest: dict):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest = {
            "schemaVersion": 1,
            "evidenceClass": "prospective_diagnostic_training_trace",
            **manifest,
        }
        (self.directory / "manifest.json").write_bytes(canonical_bytes(self.manifest) + b"\n")
        self.streams = {
            name: (self.directory / f"{name}.jsonl").open("xb")
            for name in ("transitions", "updates")
        }
        self.counts = dict.fromkeys(self.streams, 0)
        self.hashes = {name: hashlib.sha256() for name in self.streams}
        self.leaves = {name: [] for name in self.streams}
        self.transition_ids = []
        self.policy_version = 0
        self.rollout = 0
        self.snapshot_hashes = set()
        self.normalizer_moment_hashes = set()
        self._snapshot_lock = threading.Lock()

    def store_snapshot(self, raw: bytes) -> str:
        identity = hashlib.sha256(raw).hexdigest()
        with self._snapshot_lock:
            folder = self.directory / "snapshots"
            folder.mkdir(exist_ok=True)
            target = folder / f"{identity}.anthro"
            if not target.exists():
                with target.open("xb") as stream:
                    stream.write(raw)
            elif hashlib.sha256(target.read_bytes()).hexdigest() != identity:
                raise ValueError("content-addressed snapshot changed")
            self.snapshot_hashes.add(identity)
        return identity

    def store_normalizer_moments(self, state: dict) -> str:
        """Store immutable moments once per distinct state and return their digest."""
        blob = canonical_bytes(state)
        identity = hashlib.sha256(blob).hexdigest()
        with self._snapshot_lock:
            folder = self.directory / "normalizers"
            folder.mkdir(exist_ok=True)
            target = folder / f"{identity}.json"
            if not target.exists():
                with target.open("xb") as stream:
                    stream.write(blob)
            elif hashlib.sha256(target.read_bytes()).hexdigest() != identity:
                raise ValueError("content-addressed normalizer moments changed")
            self.normalizer_moment_hashes.add(identity)
        return identity

    def write(self, stream: str, row: dict):
        current_ids = set(self.transition_ids)
        if (
            stream == "updates"
            and "batchTransitionIds" in row
            and any(
                identity not in current_ids for identity in row["batchTransitionIds"]
            )
        ):
            raise ValueError("update references a transition outside the current rollout")
        _finite(row)
        blob = canonical_bytes(
            {
                "schemaVersion": 1,
                **self.manifest,
                "rollout": max(0, self.rollout - 1) if stream == "updates" else self.rollout,
                **row,
            }
        )
        if len(blob) > 256 * 1024 * 1024:
            raise ValueError("economic trace row exceeds 256 MiB bound")
        line = blob + b"\n"
        self.streams[stream].write(line)
        self.streams[stream].flush()
        self.counts[stream] += 1
        self.hashes[stream].update(line)
        self.leaves[stream].append(hashlib.sha256(blob).digest())

    def close(self):
        for stream in self.streams.values():
            stream.close()
        done = {
            "schemaVersion": 1,
            **{
                name: {"rows": self.counts[name], "sha256": self.hashes[name].hexdigest()}
                for name in self.streams
            },
            "merkleRoot": merkle_root(self.leaves["transitions"] + self.leaves["updates"]),
            "snapshotHashes": sorted(self.snapshot_hashes),
            "normalizerMomentHashes": sorted(self.normalizer_moment_hashes),
        }
        (self.directory / "done.json").write_bytes(canonical_bytes(done) + b"\n")

    def abort(self):
        for stream in self.streams.values():
            stream.close()


def state_digest(module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def capture_policy_input_health(batch, tensors, normalizer) -> list[dict]:
    """Describe the exact float tensors supplied to the actor, without mutating them.

    Raw/transformed/affine values are retained only for the small, reviewed core
    coordinate set.  Every reported final normalized value is read from ``tensors``
    itself, after float32 conversion, presence masking, clipping and dead-feature
    zeroing by :func:`model.to_tensors`.  The full input is represented only by
    bounded group summaries and the largest normalized coordinates.
    """
    import numpy as np
    import torch

    from . import contracts
    from .model import BOUNDED_FEATURE_CONTRACT, PHYSICAL_CORE_FEATURES

    raw_core = np.asarray(batch["core"])
    if raw_core.ndim != 2:
        raise ValueError("policy input health requires a two-dimensional core batch")
    rows, core_width = raw_core.shape
    core_tensor = tensors.get("core")
    if (
        not isinstance(core_tensor, torch.Tensor)
        or core_tensor.ndim != 2
        or tuple(core_tensor.shape) != (rows, 2 * core_width)
    ):
        raise ValueError("policy input health core tensor layout differs from to_tensors")
    core_actual = core_tensor.detach().cpu().numpy()
    normalized_core = core_actual[:, :core_width]
    core_presence = core_actual[:, core_width:]
    transformed_core = np.sign(raw_core) * np.log1p(np.abs(raw_core))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        affine_core = (transformed_core - normalizer.core_mean) / normalizer.core_std
    bounded_contract = normalizer.normalization_contract
    physical_features = {feature.index: feature for feature in PHYSICAL_CORE_FEATURES}
    supported_history = (set(bounded_contract.supported_history_indices) if bounded_contract is not None else set())

    def optional_number(value):
        result = float(value)
        return result if math.isfinite(result) else None

    def group_rows(name, values, *, normalized):
        array = values.detach().cpu().numpy()
        if len(array) != rows:
            raise ValueError(f"policy input health row mismatch for {name}")
        flat = array.reshape(rows, -1)
        output = []
        for row in flat:
            finite = np.isfinite(row)
            magnitudes = np.abs(row[finite])
            output.append(
                {
                    "group": name,
                    "semantics": (
                        "symlog-affine-normalized-actual-actor-input"
                        if normalized
                        else "identity-actual-actor-input"
                    ),
                    "scalarCount": int(row.size),
                    "finiteCount": int(finite.sum()),
                    "nonfiniteCount": int(row.size - finite.sum()),
                    "maximumAbsolute": float(magnitudes.max()) if magnitudes.size else None,
                }
            )
        return output

    normalized_groups: list[tuple[str, torch.Tensor]] = [("core", core_tensor[:, :core_width])]
    identity_groups: list[tuple[str, torch.Tensor]] = [
        ("corePresence", core_tensor[:, core_width:])
    ]
    for name, _set_rows, _width in contracts.SETS:
        value = tensors.get(name)
        if not isinstance(value, torch.Tensor) or value.dtype == torch.bool:
            raise ValueError(f"policy input health missing float actor set {name}")
        normalized_groups.append((name, value))
    personality = tensors.get("personality")
    if not isinstance(personality, torch.Tensor) or personality.dtype == torch.bool:
        raise ValueError("policy input health missing float actor personality")
    identity_groups.append(("personality", personality))

    group_health = {
        name: group_rows(name, value, normalized=True)
        for name, value in normalized_groups
    }
    if bounded_contract is not None:
        for row in group_health["core"]:
            row["semantics"] = "registered-physical-or-symlog-affine-actual-actor-input"
    group_health.update(
        {
            name: group_rows(name, value, normalized=False)
            for name, value in identity_groups
        }
    )

    outliers: list[list[dict]] = [[] for _ in range(rows)]
    for group, value in normalized_groups:
        array = value.detach().cpu().numpy()
        for row_index in range(rows):
            for coordinate in np.ndindex(array.shape[1:]):
                number = float(array[(row_index, *coordinate)])
                if math.isfinite(number):
                    outliers[row_index].append(
                        {
                            "group": group,
                            "coordinate": [int(index) for index in coordinate],
                            "normalized": number,
                            "absoluteNormalized": abs(number),
                        }
                    )
    for row in outliers:
        row.sort(
            key=lambda item: (
                -item["absoluteNormalized"],
                item["group"],
                item["coordinate"],
            )
        )
        del row[POLICY_INPUT_OUTLIER_LIMIT:]

    dead = normalizer.dead.get("core") if normalizer.guard is not None else None

    def core_coordinate(row_index: int, index: int) -> dict:
        raw = raw_core[row_index, index]
        actual = normalized_core[row_index, index]
        present = core_presence[row_index, index]
        is_dead = bool(dead[index]) if dead is not None and index < len(dead) else None
        feature = physical_features.get(index) if bounded_contract is not None else None
        if feature is None:
            transformed = transformed_core[row_index, index]
            before_guard = affine_core[row_index, index]
            transform = "symlog-affine"
            reference = None
            support_status = "not-gated"
            mean = optional_number(normalizer.core_mean[index])
            std = optional_number(normalizer.core_std[index])
            affine = optional_number(before_guard)
        else:
            if feature.transform in ("bool", "fraction"):
                transformed = float(raw)
            else:
                transformed = float(np.log1p(raw) / np.log1p(feature.reference))
            before_guard = transformed
            transform = feature.transform
            reference = feature.reference
            support_status = ("supported" if index in supported_history else "masked-unsupported"
                              if 40 <= index <= 55 else "not-gated")
            mean = std = affine = None  # fitted affine is not executed for registered physical coordinates
        result = {
            "index": index,
            "label": _KNOWN_CORE_LABELS.get(index, f"core[{index}]"),
            "raw": optional_number(raw),
            "rawFinite": bool(np.isfinite(raw)),
            "transformed": optional_number(transformed),
            "transformedFinite": bool(np.isfinite(transformed)),
            "normalizerMean": mean,
            "normalizerStd": std,
            "affineBeforeGuard": affine,
            "affineBeforeGuardFinite": bool(np.isfinite(before_guard)) if feature is None else None,
            "normalized": optional_number(actual),
            "normalizedFinite": bool(np.isfinite(actual)),
            "corePresent": bool(present != 0),
            "guardEnabled": normalizer.guard is not None,
            "guardLimit": float(normalizer.guard) if normalizer.guard is not None else None,
            "dead": is_dead,
            "clippedByGuard": bool(
                normalizer.guard is not None
                and np.isfinite(before_guard)
                and abs(float(before_guard)) > float(normalizer.guard)
            ),
            # Registered physical coordinates bypass the fitted dead mask. History may instead be zeroed by its
            # explicit closed-loop support mask; movement 144 remains live even when constant in the fit corpus.
            "zeroedByDeadMask": (False if feature is not None else bool(is_dead)
                                  if is_dead is not None else None),
            "zeroedBySupportMask": bool(feature is not None and support_status == "masked-unsupported"),
            "zeroedByPresenceMask": bool(present == 0),
        }
        if feature is not None:
            result.update({
                "normalizationContract": BOUNDED_FEATURE_CONTRACT,
                "transform": transform,
                "transformReference": reference,
                "beforeGuard": optional_number(before_guard),
                "supportStatus": support_status,
            })
        return result

    output = []
    for row_index in range(rows):
        previous_indices = [
            index for index in PREVIOUS_DECISION_CORE_INDICES if index < core_width
        ]
        previous_absent = [
            index for index in PREVIOUS_DECISION_CORE_INDICES if index >= core_width
        ]
        previous_status = (
            "complete"
            if not previous_absent
            else "absent-legacy-core-width"
            if not previous_indices
            else "partial-legacy-core-width"
        )
        movement_available = MOVEMENT_CORE_INDEX < core_width
        groups = [
            group_health[name][row_index]
            for name, _value in [*normalized_groups, *identity_groups]
        ]
        actor_nonfinite = sum(group["nonfiniteCount"] for group in groups)
        normalized_nonfinite = sum(
            group_health[name][row_index]["nonfiniteCount"]
            for name, _value in normalized_groups
        )
        row_output = {
                "schemaVersion": POLICY_INPUT_HEALTH_SCHEMA_VERSION,
                "source": "exact-actor-input-tensors",
                "floatTensorScope": (
                    "actor-consumed core, entity sets, and personality; boolean masks excluded; "
                    "critic-only privileged tensor excluded"
                ),
                "previousDecision": {
                    "status": previous_status,
                    "expectedIndices": list(PREVIOUS_DECISION_CORE_INDICES),
                    "absentIndices": previous_absent,
                    "coordinates": [
                        core_coordinate(row_index, index) for index in previous_indices
                    ],
                },
                "movement": {
                    "status": "complete" if movement_available else "absent-legacy-core-width",
                    "expectedIndex": MOVEMENT_CORE_INDEX,
                    "coordinate": (
                        core_coordinate(row_index, MOVEMENT_CORE_INDEX)
                        if movement_available
                        else None
                    ),
                },
                "groups": groups,
                "actorFloatInputNonfiniteCount": actor_nonfinite,
                "normalizedInputNonfiniteCount": normalized_nonfinite,
                "maximumAbsoluteNormalized": (
                    outliers[row_index][0]["absoluteNormalized"]
                    if outliers[row_index]
                    else None
                ),
                "topNormalizedOutliers": outliers[row_index],
            }
        if bounded_contract is not None:
            row_output["normalizationContract"] = BOUNDED_FEATURE_CONTRACT
        output.append(row_output)
    return output


def _objective_combination_details(objectives, batch, hyper):
    """Mirror PPO objective combination for diagnostics without changing the update.

    Returned component contributions sum to the *pre-standardization* actor
    objective advantage.  They are deliberately kept separate from the cost
    constraint and the later regime/global balancing performed by ``ppo.update``.
    """
    import numpy as np

    from . import contracts

    advantages = {
        name: np.asarray(values[4], dtype=np.float64) for name, values in objectives.items()
    }
    if not advantages:
        raise ValueError("economic trace requires at least one objective")
    row_count = len(next(iter(advantages.values())))
    if any(len(value) != row_count for value in advantages.values()):
        raise ValueError("objective advantages have inconsistent row counts")

    personal = "personality" in batch
    if not personal:
        mode = "configured-objective-weights"
        scales = {name: 1.0 for name in advantages}
        outer = 1.0
        personality = None
    else:
        personality = np.asarray(batch["personality"], dtype=np.float64)
        if personality.shape != (row_count, len(contracts.OBJECTIVES)):
            raise ValueError("personality rows do not match objective contract")
        if hyper.personality_scales == "reference":
            mode = "personality-reference-normalized"
            scales = {
                name: float(contracts.OBJECTIVE_REFERENCE_SCALES[name]) for name in advantages
            }
            outer = 1.0
        elif hyper.personality_scales == "batch":
            mode = "personality-batch-normalized-welfare-rescaled"
            scales = {name: float(np.std(value)) + 1e-9 for name, value in advantages.items()}
            outer = scales.get("welfare", 1.0)
        else:
            raise ValueError(f"unknown personality scales {hyper.personality_scales!r}")

    total = np.zeros(row_count, dtype=np.float64)
    components = {}
    names = (
        [name for name in contracts.OBJECTIVES if name in advantages]
        if personal
        else list(advantages)
    )
    for name in names:
        advantage = advantages[name]
        configured = float(hyper.objective_weights.get(name, 0.0))
        scale = scales[name]
        scaled = advantage / scale
        clipped = scaled
        personality_weight = None
        if personal:
            objective_index = contracts.OBJECTIVES.index(name)
            personality_weight = personality[:, objective_index]
            if hyper.personality_scales == "reference" and hyper.personality_clip > 0:
                clipped = np.clip(scaled, -hyper.personality_clip, hyper.personality_clip)
            contribution = personality_weight * clipped * outer
            pre_clip_weight = personality_weight * outer / scale
        else:
            contribution = configured * advantage
            pre_clip_weight = np.full(row_count, configured, dtype=np.float64)
        effective_weight = np.divide(
            contribution,
            advantage,
            out=pre_clip_weight.copy(),
            where=advantage != 0,
        )
        total = total + contribution
        components[name] = {
            "scale": scale,
            "scaled": scaled,
            "clipped": clipped,
            "clipApplied": clipped != scaled,
            "personalityWeight": personality_weight,
            "preClipEffectiveWeight": pre_clip_weight,
            "effectiveWeight": effective_weight,
            "contribution": contribution,
            "configuredWeight": configured,
        }
    return {
        "mode": mode,
        "configuredWeightsApplied": not personal,
        "batchWelfareSpreadMultiplier": outer
        if personal and hyper.personality_scales == "batch"
        else None,
        "total": total,
        "components": components,
    }


def emit_rollout(
    writer,
    roll,
    batch,
    draws,
    old_logp,
    objectives,
    cost,
    combined,
    weights,
    regimes,
    hyper,
    lagrange,
    actor,
    critic,
    cost_critic,
    normalizer,
):
    """Write actual pre-update values/targets; no historical reconstruction."""
    import numpy as np

    from . import contracts, data

    provenance = {
        "actorCheckpointHash": state_digest(actor),
        "criticCheckpointHash": state_digest(critic),
        "costCheckpointHash": state_digest(cost_critic),
    }
    normalizer_state = (
        normalizer.state_dict()
        if hasattr(normalizer, "state_dict")
        else normalizer.state()
        if hasattr(normalizer, "state")
        else None
    )
    provenance["normalizerHash"] = (
        writer.store_normalizer_moments(normalizer_state)
        if normalizer_state is not None
        else None
    )
    combination = _objective_combination_details(objectives, batch, hyper)
    objective_advantage = combination["total"]
    objective_scale = max(1e-8, float(np.std(objective_advantage)))
    cv, cb, ca, ct = cost
    constraint_scale = max(1e-8, float(np.std(ca)) + 1e-8)
    objective_standardized = objective_advantage / objective_scale
    constraint_contribution = -float(lagrange) * np.asarray(ca) / constraint_scale
    pre_balance_advantage = objective_standardized + constraint_contribution
    next_rows = {}
    for sequence in data.sequences(batch):
        for i, successor in pairwise(sequence):
            if not batch["terminated"][i]:
                next_rows[int(i)] = int(successor)
    writer.transition_ids = []
    for i in range(len(combined)):
        captured = roll.economic_rows[i]
        economic = captured["economicDiagnostics"]
        transition = (
            f"{writer.manifest['runId']}/{writer.rollout}/"
            f"{batch['world'][i]}/{economic['transitionId']}"
        )
        economic = dict(economic)
        economic["sourceTransitionId"] = economic["transitionId"]
        economic["transitionId"] = transition
        writer.transition_ids.append(transition)
        successor = next_rows.get(i)
        output_objectives = {}
        for name, values in objectives.items():
            reward, raw_reward, prediction, bootstrap, advantage, target = values
            detail = combination["components"][name]
            vnext = (
                0.0
                if batch["terminated"][i]
                else prediction[successor]
                if successor is not None
                else bootstrap[i]
            )
            reference_scale = contracts.OBJECTIVE_REFERENCE_SCALES.get(name)
            scale = detail["scale"]
            scaled = detail["scaled"][i]
            output_objectives[name] = {
                "reward": float(raw_reward[i]),
                "shapedReward": float(reward[i]),
                "shapingDebit": float(raw_reward[i] - reward[i]),
                "prediction": float(prediction[i]),
                "bootstrap": float(vnext),
                "tdResidual": float(
                    reward[i] + data.discount(batch["years"][i]) * vnext - prediction[i]
                ),
                "target": float(target[i]),
                "advantage": float(advantage[i]),
                "referenceScale": float(reference_scale) if reference_scale else None,
                "combinationScale": float(scale) if scale else None,
                "referenceScaledAdvantage": float(advantage[i] / reference_scale)
                if reference_scale
                else None,
                # Compatibility field: this is the configured static weight, not
                # the applied per-row personality coefficient.
                "objectiveWeight": float(hyper.objective_weights.get(name, 0.0)),
                "configuredObjectiveWeight": float(detail["configuredWeight"]),
                "configuredObjectiveWeightApplied": bool(
                    combination["configuredWeightsApplied"]
                ),
                "personalityWeight": float(detail["personalityWeight"][i])
                if detail["personalityWeight"] is not None
                else None,
                "preClipEffectiveCombinationWeight": float(
                    detail["preClipEffectiveWeight"][i]
                ),
                "effectiveCombinationWeight": float(detail["effectiveWeight"][i]),
                "effectiveCombinationWeightSemantics": (
                    "weighted-contribution/advantage; zero-advantage rows use the "
                    "pre-clip linear coefficient"
                ),
                "weightedAdvantageContribution": float(detail["contribution"][i]),
                "combinationClipApplied": bool(detail["clipApplied"][i]),
                "valuePopulationScale": 1.0
                if name == "qol_years"
                else float(max(1, batch["prior_population"][i])),
                "scaledAdvantage": float(scaled) if scaled is not None else None,
                "clippedAdvantage": float(detail["clipped"][i]),
                "zeroShapedRewardNonzeroBootstrap": bool(
                    float(reward[i]) == 0.0 and float(vnext) != 0.0
                ),
            }
        cost_next = (
            0.0 if batch["terminated"][i] else cv[successor] if successor is not None else cb[i]
        )
        world = int(batch["world"][i])
        unavailable = {
            "independentReturn": "no-independent-continuation-bound",
            "personalityBand": "no-frozen-band-manifest",
            "populationQuintile": "no-frozen-baseline-quintile-manifest",
            "qolQuintile": "no-frozen-baseline-quintile-manifest",
            "actorHeadLogProbabilities": "distribution-api-computes-joint-log-probability-only",
            "actorHeadEntropy": "current-distribution-api-computes-joint-entropy-only",
        }
        if provenance["normalizerHash"] is None:
            unavailable["provenance.normalizerHash"] = "normalizer-state-api-unavailable"
        policy_input_health = captured.get("policyInputHealth")
        if policy_input_health is None:
            unavailable["policyInputHealth"] = "not-emitted-by-legacy-economic-trace"
        row = {
            "schemaVersion": 1,
            "evidenceClass": "prospective_diagnostic_training_trace",
            **writer.manifest,
            "transitionId": transition,
            "trajectoryId": economic["trajectoryId"],
            "decisionTurn": int(batch["turn"][i]),
            "campId": int(batch["camp"][i]),
            "societyId": int(batch["society"][i]),
            "worldSeed": int(batch["seed"][i]),
            "rollout": writer.rollout,
            "workerSlot": world,
            "policyVersion": writer.policy_version,
            "years": float(batch["years"][i]),
            "gamma": float(data.discount(batch["years"][i])),
            "horizonYears": contracts.HORIZON_YEARS,
            "gaeLambda": 0.95,
            "terminated": bool(batch["terminated"][i]),
            "truncated": successor is None and not bool(batch["terminated"][i]),
            "bootstrapEligible": not bool(batch["terminated"][i]),
            "bootstrapDestinationRow": successor,
            "terminationReason": "terminal-village"
            if batch["terminated"][i]
            else "rollout-horizon"
            if successor is None
            else "continuing",
            "bootstrapCheckpointHash": provenance["criticCheckpointHash"],
            "bootstrapPolicyVersion": writer.policy_version,
            "successorTransitionId": None
            if successor is None
            else (
                f"{writer.manifest['runId']}/{writer.rollout}/{batch['world'][successor]}/"
                f"{roll.economic_rows[successor]['economicDiagnostics']['transitionId']}"
            ),
            "economicDiagnostics": economic,
            "actionDiagnostics": captured.get("actionDiagnostics"),
            "actorPolicyActiveMask": captured.get("actorPolicyActiveMask"),
            "policyInputHealth": policy_input_health,
            "objectives": output_objectives,
            "objectiveCombinationMode": combination["mode"],
            "objectiveCombination": {
                "mode": combination["mode"],
                "configuredWeightsApplied": bool(combination["configuredWeightsApplied"]),
                "batchWelfareSpreadMultiplier": combination[
                    "batchWelfareSpreadMultiplier"
                ],
                "preStandardizationObjectiveAdvantage": float(objective_advantage[i]),
                "objectiveStandardizationScale": float(objective_scale),
                "objectiveAfterStandardization": float(objective_standardized[i]),
                "costAdvantageStandardizationScale": float(constraint_scale),
                "costConstraintContributionBeforeBalancing": float(
                    constraint_contribution[i]
                ),
                "preBalanceActorAdvantage": float(pre_balance_advantage[i]),
                "finalPolicyAdvantage": float(combined[i]),
                "balanceWeightIsSeparateLossWeight": True,
            },
            "finalPolicyAdvantage": float(combined[i]),
            "balanceWeight": float(weights[i]),
            "lagrangeMultiplier": float(lagrange),
            "objectiveWeights": hyper.objective_weights,
            "personality": batch["personality"][i].tolist()
            if "personality" in batch
            else [1.0, 0.0, 0.0, 0.0],
            "capabilityRegime": str(regimes[i]),
            "actorDistribution": captured["actorDistribution"],
            "sampledAction": captured["sampledAction"],
            "actorJointLogProbability": float(old_logp[i]),
            "cost": {
                "prediction": float(cv[i]),
                "bootstrap": float(cost_next),
                "target": float(ct[i]),
                "advantage": float(ca[i]),
                "tdResidual": float(
                    batch["cost"][i] + data.discount(batch["years"][i]) * cost_next - cv[i]
                ),
            },
            "provenance": {
                **provenance,
                **captured["provenance"],
                "workerContract": roll.worker_contracts[world],
                "worldContract": roll.world_contracts[world],
            },
            "normalizerMomentsRef": provenance["normalizerHash"],
            "costBootstrapEnabled": hyper.cost_bootstrap,
            "unavailable": unavailable,
        }
        writer.write("transitions", row)
    writer.rollout += 1
