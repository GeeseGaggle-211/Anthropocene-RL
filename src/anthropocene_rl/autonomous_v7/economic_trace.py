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


def validate_economic_step(step: dict, *, required=False, initial=False) -> dict[int, dict]:
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
            row.get("schemaVersion") != 1
            or row.get("campId") != int(key)
            or row.get("decisionTurn") != int(step["turn"]) - 1
            or row.get("trajectoryId") != step.get("trajectoryId")
        ):
            raise ValueError("economic schema/camp/turn/trajectory mismatch")
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
    if hasattr(normalizer, "state_dict"):
        provenance["normalizerHash"] = hashlib.sha256(
            canonical_bytes(normalizer.state_dict())
        ).hexdigest()
    else:
        provenance["normalizerHash"] = (
            hashlib.sha256(canonical_bytes(normalizer.state())).hexdigest()
            if hasattr(normalizer, "state")
            else None
        )
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
            vnext = (
                0.0
                if batch["terminated"][i]
                else prediction[successor]
                if successor is not None
                else bootstrap[i]
            )
            reference_scale = contracts.OBJECTIVE_REFERENCE_SCALES.get(name)
            scale = (
                (
                    reference_scale
                    if hyper.personality_scales == "reference"
                    else float(np.std(advantage)) + 1e-9
                )
                if "personality" in batch
                else 1.0
            )
            scaled = advantage[i] / scale if scale else None
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
                "objectiveWeight": float(hyper.objective_weights.get(name, 0.0)),
                "valuePopulationScale": 1.0
                if name == "qol_years"
                else float(max(1, batch["prior_population"][i])),
                "scaledAdvantage": float(scaled) if scaled is not None else None,
                "clippedAdvantage": float(
                    np.clip(scaled, -hyper.personality_clip, hyper.personality_clip)
                )
                if scaled is not None
                and "personality" in batch
                and hyper.personality_clip > 0
                and hyper.personality_scales == "reference"
                else float(scaled)
                if scaled is not None
                else None,
            }
        cv, cb, ca, ct = cost
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
            "objectives": output_objectives,
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
            "normalizerMoments": normalizer.state_dict(),
            "costBootstrapEnabled": hyper.cost_bootstrap,
            "unavailable": unavailable,
        }
        writer.write("transitions", row)
    writer.rollout += 1
