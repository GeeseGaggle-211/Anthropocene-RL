"""Native teacher collection through the exact production bridge; immutable publish-last datasets, never migration."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

from . import contracts
from .bridge import Model9Bridge, DEFAULT_BRIDGE, observation_batch
from .protocol import seeds, scan, seed_list
from .provenance import ProvenanceError, checkpoint_hash, code_commit, digest, simulator_identity

SCHEMA = 1
SCHEMAS = {"contract": contracts.NATIVE_CONTRACT, "action": contracts.ACTION_SCHEMA,
           "observation": contracts.OBSERVATION_SCHEMA, "environment": contracts.ENVIRONMENT_SCHEMA,
           "labels": 1, "outcomes": 1, "utility": 2}


def binary_identity() -> dict:
    path = Path(os.environ.get("ANTHROPOCENE_MODEL9_BRIDGE", str(DEFAULT_BRIDGE)))
    sha = checkpoint_hash(path)
    if not re.fullmatch(r"[0-9a-f]{64}", sha): raise ProvenanceError("production bridge binary missing")
    return {"path": str(path.resolve()), "sha256": sha}


def source_record(options: dict, schema, binary: dict, learner: str | None = None) -> dict:
    scan(options); learner = code_commit() if learner is None else learner
    if not re.fullmatch(r"[0-9a-f]{40}", learner): raise ProvenanceError("teacher generation requires clean learner commit")
    body = {"schema": SCHEMA, "source_kind": "native_teacher", "learner_commit": learner,
            "simulator": simulator_identity(schema, production=True), "bridge_sha256": binary["sha256"],
            "schemas": SCHEMAS, "core_width": schema.core_width, "core_features_digest": digest(schema.core_features), "options": options,
            "flags": {"sim_rules": os.environ.get("ANTHROPOCENE_SIM_RULES", ""),
                      "sim_statics": os.environ.get("ANTHROPOCENE_SIM_STATICS", "")}}
    return dict(body, source_run=digest(body))


def validate_source(record: dict, *, live: dict | None = None, binary_sha256: str | None = None) -> None:
    scan(record)
    if (record.get("schema") != SCHEMA or record.get("source_kind") != "native_teacher"
            or record.get("schemas") != SCHEMAS
            or not isinstance(record.get("core_width"), int) or record["core_width"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("core_features_digest", "")))
            or not re.fullmatch(r"[0-9a-f]{40}", str(record.get("learner_commit", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("bridge_sha256", "")))
            or record.get("source_run") != digest({k: v for k, v in record.items() if k != "source_run"})
            or not re.fullmatch(r"[0-9a-f]{40}", str(record.get("simulator", {}).get("commit", "")))
            or record.get("simulator", {}).get("clean") is not True
            or not record.get("simulator", {}).get("buildHash")):
        raise ProvenanceError("unattributed/dirty/legacy teacher source")
    options = record.get("options", {})
    if (options.get("movement") is not True or options.get("raw_materials") is not False
            or options.get("opening_limit") != 40 or not options.get("worlds")
            or not isinstance(options.get("players"), int) or options["players"] <= 0
            or not isinstance(options.get("turns"), int) or options["turns"] <= 0):
        raise ProvenanceError("teacher generation options missing or incompatible")
    if live is not None and record["simulator"] != live:
        raise ProvenanceError("teacher source differs from exact live production bridge")
    if binary_sha256 is not None and record["bridge_sha256"] != binary_sha256:
        raise ProvenanceError("teacher source bridge binary SHA differs")


def trajectory(meta: dict) -> str:
    return digest({k: meta[k] for k in ("source_run", "seed", "withheld", "warmup", "opening_turns",
        "first_turn", "players", "turns", "initial_fingerprint", "configuration_hash", "initial_snapshot_sha256")})


def payload(directory: Path) -> dict:
    out = {}
    for path in sorted(directory.glob("*.npy")):
        array = np.load(path, allow_pickle=False)
        out[path.name] = {"sha256": checkpoint_hash(path), "dtype": array.dtype.str,
                          "shape": list(array.shape), "rows": int(len(array))}
    return out


def write_shard(directory: Path, arrays: dict, meta: dict, source: dict) -> dict:
    """A collision never overwrites an existing shard, including an interrupted unpublished one."""
    validate_source(source); scan(meta)
    if directory.exists(): raise ProvenanceError("teacher shard output already exists; no overwrite/migration")
    n = len(arrays["camp"])
    if n <= 0 or any(len(a) != n or a.dtype.hasobject for a in arrays.values()):
        raise ProvenanceError("invalid teacher payload row layout")
    directory.mkdir(parents=True)
    for key, array in sorted(arrays.items()):
        if not re.fullmatch(r"[a-z][a-z_]*", key): raise ProvenanceError("invalid teacher array name")
        np.save(directory / (key + ".npy"), array, allow_pickle=False)
    record = dict(meta, schema=SCHEMA, source_record=source, source_run=source["source_run"], rows=n,
                  contract=contracts.NATIVE_CONTRACT, raw_materials=False)
    record["trajectory_id"] = trajectory(record); record["payload"] = payload(directory)
    record["data_digest"] = digest(record["payload"])
    (directory / "meta.json.tmp").write_text(json.dumps(record, sort_keys=True, allow_nan=False))
    os.replace(directory / "meta.json.tmp", directory / "meta.json")
    return record


def verify_shard(directory: Path, *, live=None, binary_sha256=None) -> dict:
    if not (directory / "meta.json").is_file(): raise ProvenanceError("teacher shard not published")
    meta = json.loads((directory / "meta.json").read_text()); scan(meta)
    validate_source(meta.get("source_record", {}), live=live, binary_sha256=binary_sha256)
    if (meta.get("schema") != SCHEMA or meta.get("source_run") != meta["source_record"]["source_run"]
            or meta.get("contract") != contracts.NATIVE_CONTRACT or meta.get("raw_materials") is not False
            or meta.get("trajectory_id") != trajectory(meta)):
        raise ProvenanceError("teacher shard context/trajectory mismatch")
    actual = payload(directory)
    if not actual or actual != meta.get("payload") or digest(actual) != meta.get("data_digest"):
        raise ProvenanceError("teacher payload hash/dtype/shape mismatch")
    n = meta.get("rows")
    if not isinstance(n, int) or n <= 0: raise ProvenanceError("teacher shard must have positive rows")
    for key in ("initial_fingerprint", "configuration_hash", "initial_snapshot_sha256"):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(meta.get(key, ""))):
            raise ProvenanceError("teacher shard lacks full initial configuration/snapshot identity")
    if any(p["rows"] != n for p in actual.values()): raise ProvenanceError("teacher row counts differ")
    from .data import OBSERVATION_KEYS, LABEL_KEYS, OUTCOME_KEYS
    required = {k + ".npy" for k in OBSERVATION_KEYS + LABEL_KEYS + OUTCOME_KEYS}
    if not required <= set(actual): raise ProvenanceError("teacher schema arrays missing")
    widths = {"core": (meta["source_record"]["core_width"],), "core_present": (meta["source_record"]["core_width"],),
              "strategic": (5,), "strategic_mask": (5,), "service_levels": (6,), "service_mask": (6,)}
    for name, rows, width in contracts.SETS:
        widths[name] = (rows, width); widths[name + "_present"] = (rows,)
    for key, shape in widths.items():
        if actual[key + ".npy"]["shape"] != [n, *shape]: raise ProvenanceError("teacher contracted array shape mismatch")
    identities = list(zip(np.load(directory / "turn.npy").tolist(), np.load(directory / "camp.npy").tolist()))
    if len(set(identities)) != len(identities): raise ProvenanceError("duplicate teacher decisions")
    world = {"seed": meta["seed"], "withheld": meta["withheld"], "warmup": meta["warmup"]}
    if world not in meta["source_record"]["options"]["worlds"]:
        raise ProvenanceError("teacher shard world outside source generation list")
    return meta


def dataset(directory: Path, shards: list[dict], source: dict) -> dict:
    body = {"schema": SCHEMA, "source_record": source, "source_run": source["source_run"],
            "shards": {m["trajectory_id"]: {"path": "world-" + m["trajectory_id"],
                       "meta_sha256": checkpoint_hash(directory / ("world-" + m["trajectory_id"]) / "meta.json"),
                       "data_digest": m["data_digest"]} for m in shards}}
    if len(body["shards"]) != len(shards): raise ProvenanceError("duplicate dataset trajectories")
    body["dataset_digest"] = digest(body)
    (directory / "dataset.json.tmp").write_text(json.dumps(body, sort_keys=True))
    os.replace(directory / "dataset.json.tmp", directory / "dataset.json")
    return body


def verify_dataset(directory: Path) -> dict:
    obj = json.loads((directory / "dataset.json").read_text())
    validate_source(obj.get("source_record", {}))
    if obj.get("schema") != SCHEMA or not obj.get("shards"):
        raise ProvenanceError("teacher dataset schema/positive shard coverage missing")
    if obj.get("dataset_digest") != digest({k: v for k, v in obj.items() if k != "dataset_digest"}):
        raise ProvenanceError("teacher dataset digest mismatch")
    listed = {entry["path"] for entry in obj["shards"].values()}
    if listed != {p.name for p in directory.iterdir() if p.is_dir()}:
        raise ProvenanceError("teacher dataset contains missing/undeclared/unpublished shard")
    for identity, entry in obj["shards"].items():
        if not re.fullmatch(r"[0-9a-f]{64}", identity) or entry["path"] != "world-" + identity or checkpoint_hash(directory / entry["path"] / "meta.json") != entry["meta_sha256"]:
            raise ProvenanceError("teacher dataset shard identity/hash mismatch")
        meta = verify_shard(directory / entry["path"])
        if meta["source_run"] != obj["source_run"] or meta["trajectory_id"] != identity or meta["data_digest"] != entry["data_digest"]:
            raise ProvenanceError("teacher dataset source/payload binding mismatch")
    return obj


def collect_world(bridge, world: dict, options: dict) -> tuple[dict, dict]:
    from .worker_contract import observe, validate, world as describe_world
    expected = observe(bridge)
    validate(bridge, expected, describe_world(world["seed"], world["withheld"], players=options["players"], movement=True))
    start = bridge.reset(world["seed"], players=options["players"], movement=True, withheld=world["withheld"])
    opening = int(start["turn"])
    identity = bridge.last_reset_identity
    if identity.get("rawMaterialsEnabled") is not False:
        raise ProvenanceError("teacher collector requires actual raw-material flag OFF")
    for _ in range(world["warmup"]): start = bridge.step({})
    raw = base64.b64decode(bridge.capture()); first = int(start["turn"])
    meta = dict(world, opening_turns=opening, first_turn=first, players=options["players"], turns=options["turns"],
                initial_fingerprint=start["fingerprint"], configuration_hash=identity["configurationHash"],
                initial_snapshot_sha256=hashlib.sha256(raw).hexdigest())
    observations, labels, outcomes, decision_turns = [], [], [], []
    current = start
    for _ in range(options["turns"]):
        by_camp = {v["camp"]: v for v in current["villages"] if v.get("observation") is not None and not v["terminated"]}
        if not by_camp: break
        step = bridge.step({})
        for v in step["villages"]:
            if v["camp"] in by_camp and v.get("teacher") is not None:
                observations.append(by_camp[v["camp"]]); labels.append(v["teacher"]); outcomes.append(v)
                decision_turns.append(current["turn"])
        current = step
    arrays = observation_batch(observations)
    arrays["privileged"] = np.asarray([v["privileged"] for v in observations], np.float32)
    mapping = {"strategic": "strategic", "strategic_mask": "strategicMask", "service_levels": "service",
        "service_mask": "serviceMask", "clothing_route": "clothingRoute", "clothing_route_label_mask": "clothingRouteLabel",
        "shelter_route": "shelterRoute", "shelter_route_label_mask": "shelterRouteLabel", "movement_label_mask": "movementLabel",
        "fission_policy": "fission", "repair": "repair"}
    for key, field in mapping.items(): arrays[key] = np.asarray([v[field] for v in labels])
    arrays["movement"] = np.asarray([v["movement"] + 1 for v in labels], np.int32)
    fields = {"camp": "camp", "society": "society", "parent_society": "parentSociety", "prior_population": "priorPopulation",
        "population": "population", "qol": "qol", "nutrition": "nutrition", "years": "years", "welfare": "welfare",
        "terminated": "terminated", "trade_utility": "tradeUtility"}
    for key, field in fields.items(): arrays[key] = np.asarray([v[field] for v in outcomes])
    arrays["turn"] = np.asarray(decision_turns, np.int32)
    bridge.verify_replay()
    return arrays, meta


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--seeds", required=True)
    parser.add_argument("--withheld", type=int, default=0); parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--players", type=int, default=8); parser.add_argument("--turns", type=int, default=25)
    parser.add_argument("--prior-use-inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True); args = parser.parse_args()
    from .protocol import verify_inventory
    inventory = verify_inventory(args.prior_use_inventory)
    worlds = [{"seed": s, "withheld": args.withheld, "warmup": args.warmup} for s in seed_list(args.seeds)]
    options = {"worlds": worlds, "players": args.players, "turns": args.turns, "opening_limit": 40,
               "movement": True, "raw_materials": False, "prior_use_inventory": inventory["digest"]}
    if args.out.exists(): raise ProvenanceError("teacher output exists; regeneration uses a new immutable directory")
    binary = binary_identity(); records = []
    with Model9Bridge() as bridge:
        source = source_record(options, bridge.schema, binary); validate_source(source)
        args.out.mkdir(parents=True)
        for world in worlds:
            arrays, meta = collect_world(bridge, world, options); meta["source_run"] = source["source_run"]
            records.append(write_shard(args.out / ("world-" + trajectory(meta)), arrays, meta, source))
        if (binary_identity() != binary or code_commit() != source["learner_commit"]
                or simulator_identity(bridge.schema, production=True) != source["simulator"]):
            raise ProvenanceError("teacher producer changed during collection; dataset not published")
    dataset(args.out, records, source); return 0


if __name__ == "__main__": raise SystemExit(main())
