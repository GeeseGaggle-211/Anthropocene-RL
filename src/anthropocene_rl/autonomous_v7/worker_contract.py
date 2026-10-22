"""Immutable pre-execution producer/world contracts. Metadata reads never Reset, generate, observe or act."""
import base64
import copy
import hashlib
import os
import re
from . import contracts
from .provenance import ProvenanceError, digest, simulator_identity


def seal(body):
    body = copy.deepcopy(body)
    return dict(body, digest=digest(body))


def checked(contract):
    if not isinstance(contract, dict) or contract.get("digest") != digest({k: v for k, v in contract.items() if k != "digest"}):
        raise ProvenanceError("worker producer/world contract is missing or mutated")
    return contract


def observe(bridge):
    from .teacher_data import binary_identity
    schema = bridge.schema
    flags = {"sim_rules": os.environ.get("ANTHROPOCENE_SIM_RULES", ""), "sim_statics": os.environ.get("ANTHROPOCENE_SIM_STATICS", "")}
    binary = binary_identity()["sha256"]
    if getattr(schema, "producer_flags", None) != flags:
        raise ProvenanceError("actual native process rule/static flags differ before reset")
    if getattr(schema, "bridge_sha256", None) != binary:
        raise ProvenanceError("actual loaded bridge SHA differs from configured binary before reset")
    versions = {"contract": contracts.NATIVE_CONTRACT, "action": contracts.ACTION_SCHEMA,
                "observation": contracts.OBSERVATION_SCHEMA, "environment": contracts.ENVIRONMENT_SCHEMA,
                "head": schema.decision_diagnostics_schema, "envelope": schema.diagnostic_envelope_schema,
                "raw": schema.raw_material_diagnostics_schema}
    if versions["head"] != 1 or versions["envelope"] != 2 or versions["raw"] != 2:
        raise ProvenanceError("worker diagnostics/head/envelope versions differ before reset")
    features = getattr(schema, "feature_flags", None)
    if not isinstance(features, dict) or features.get("SimulationConfig.ElderRawMaterialsEnabled") is not False:
        raise ProvenanceError("worker feature flags missing or raw materials enabled before reset")
    configuration = {"sha256": getattr(schema, "configuration_hash", ""), "players": getattr(schema, "players", 0)}
    if (not re.fullmatch(r"[0-9a-fA-F]{64}", configuration["sha256"])
            or type(configuration["players"]) is not int or configuration["players"] <= 0):
        raise ProvenanceError("worker full configuration identity is missing before reset")
    return seal({"version": 1, "simulator": simulator_identity(schema), "bridge_sha256": binary,
                 "bridge_build_hash": schema.bridge_build_hash, "schemas": versions, "flags": flags, "feature_flags": features,
                 "configuration": configuration})


def from_state(state, *, required=False):
    value = state.get("run_manifest", {}).get("worker_contract")
    if value is not None: return checked(value)
    if required or state.get("run_manifest", {}).get("options", {}).get("arm"):
        raise ProvenanceError("registered checkpoint lacks immutable pre-reset worker contract")
    return None  # legacy diagnostic-only weights do not establish readiness


def world(seed, withheld=0, *, snapshot_sha256="", turn=None, players=8, movement=False, head=None, cohort_digest=""):
    from .protocol import seeds
    seeds([seed], "pre-execution world")
    return seal({"seed": int(seed), "withheld": int(withheld), "snapshot_sha256": snapshot_sha256,
                 "turn": turn, "players": int(players), "movement": bool(movement), "region_head": head, "cohort_digest": cohort_digest})


def validate(bridge, expected, descriptor, *, snapshot=None, head=None):
    descriptor = checked(descriptor)
    actual = observe(bridge)
    if expected is not None and actual != checked(expected):
        raise ProvenanceError("worker producer differs from immutable expected contract before reset")
    if descriptor["region_head"] is not None and head != descriptor["region_head"]:
        raise ProvenanceError("worker actor head differs from frozen contract before reset")
    configuration = actual["configuration"]
    if descriptor["players"] != configuration["players"]:
        raise ProvenanceError("worker reset players differ from frozen full configuration before reset")
    payload = {"op": "reset_metadata", "seed": descriptor["seed"], "withheld": descriptor["withheld"], "players": descriptor["players"]}
    if snapshot is not None:
        raw = base64.b64decode(snapshot) if isinstance(snapshot, str) else bytes(snapshot)
        if hashlib.sha256(raw).hexdigest() != descriptor["snapshot_sha256"]:
            raise ProvenanceError("worker snapshot differs from frozen cohort before reset")
        payload["snapshot"] = base64.b64encode(raw).decode("ascii")
    elif descriptor["snapshot_sha256"]: raise ProvenanceError("worker snapshot is missing before reset")
    metadata = bridge.request(payload)
    from .protocol import seeds
    seeds([metadata["sourceSeed"]], "snapshot metadata before reset")
    if (metadata["sourceSeed"] != descriptor["seed"] or metadata.get("rawMaterialsEnabled") is not False
            or metadata.get("featureFlags") != actual["feature_flags"]
            or metadata.get("configurationHash") != configuration["sha256"]
            or metadata.get("players") != descriptor["players"]
            or metadata.get("turn") != descriptor["turn"]):
        raise ProvenanceError("worker snapshot/root/turn/feature flags/full configuration/players differ before reset")
    # Re-read the disk SHA after metadata validation; a swapped process image cannot be excused by a later report.
    if observe(bridge) != actual: raise ProvenanceError("worker binary/flags changed before reset")
    bridge._expected_worker_contract = copy.deepcopy(expected or actual)
    bridge._expected_world_contract = copy.deepcopy(descriptor)
    return actual
