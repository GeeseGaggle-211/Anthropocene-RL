"""Versioned source curriculum and private, restart-exact daughter-bank images."""
from __future__ import annotations

import hashlib
import json
import shutil
import zlib
from pathlib import Path

import numpy as np

from . import contracts
from .protocol import seeds, artifact
from .provenance import ProvenanceError, digest

SCHEDULE = ("default", "daughter", "grassland", "valley", "fixture", "default", "daughter",
            "grassland", "valley", "fixture", "default", "daughter")
MASKS = tuple(contracts.CAPABILITY_MASKS.values())


def specification(stage) -> dict:
    return {"version": "r3-balanced-v1", "slots": list(SCHEDULE), "fallback": "default",
            "fixture_masks": list(MASKS), "slot_cursor": 0, "fixture_cursor": 0, "stage": stage.name}


class BalancedSampler:
    def __init__(self, stage, train, grassland, rng, valley=None, bank=None):
        self.stage, self.train, self.grassland, self.rng = stage, train, grassland, rng
        self.valley, self.bank = valley, bank
        self.slot_cursor = self.fixture_cursor = 0
        self.last_sources = []
        seeds(train, "curriculum TRAIN"); seeds(grassland, "curriculum grassland")
        seeds([s for s, _ in valley or []], "curriculum valley")
        if not len(train): raise ProvenanceError("curriculum has no fresh TRAIN openings")

    def state_dict(self):
        return {"version": "r3-balanced-v1", "slot_cursor": self.slot_cursor, "fixture_cursor": self.fixture_cursor}

    def load_state_dict(self, state):
        if (state.get("version") != "r3-balanced-v1" or any(not isinstance(state.get(k), int) or state[k] < 0
                for k in ("slot_cursor", "fixture_cursor"))):
            raise ProvenanceError("invalid curriculum resume state")
        self.slot_cursor, self.fixture_cursor = state["slot_cursor"], state["fixture_cursor"]

    def __call__(self, n):
        worlds, self.last_sources = [], []
        for _ in range(n):
            requested = SCHEDULE[self.slot_cursor % len(SCHEDULE)]; self.slot_cursor += 1
            actual = requested; world = None
            if requested == "daughter" and self.bank is not None:
                entry = self.bank.sample(self.rng, set(MASKS) if self.stage.name == "capability" else None)
                if entry is not None:
                    seeds([entry["seed"]], "curriculum daughter")
                    world = (int(entry["seed"]), int(entry["mask"]), 0, int(entry["id"]))
            elif requested == "grassland" and len(self.grassland):
                world = (int(self.rng.choice(self.grassland)), 0)
            elif requested == "valley" and self.valley and self.stage.region_movement:
                seed, warm = self.valley[int(self.rng.integers(len(self.valley)))]; world = (int(seed), 0, int(warm))
            elif requested == "fixture":
                world = (int(self.rng.choice(self.train)), int(MASKS[self.fixture_cursor % len(MASKS)]))
                self.fixture_cursor += 1
            if world is None:
                actual = "default"; world = (int(self.rng.choice(self.train)), 0)
            worlds.append(world)
            self.last_sources.append({"requested": requested, "realised": actual, "world": list(world),
                                      "snapshot_sha256": entry["snapshot_sha256"] if actual == "daughter" else ""})
        return worlds


def bank_state(bank) -> dict:
    entries = bank.entries()
    hashes = {}
    for entry in entries:
        seeds([entry["seed"]], "bank")
        raw = bank.snapshot(entry); hashes[entry["file"]] = hashlib.sha256(raw).hexdigest()
    body = {"entries": entries, "snapshots": hashes}
    return dict(body, digest=digest(body))


def freeze_bank(bank, output: Path) -> dict:
    """Publish a content-bound initial bank image without changing the source bank."""
    if output.exists(): raise ProvenanceError("frozen bank image already exists")
    entries = bank.entries()
    for entry in entries:
        seeds([entry["seed"]], "frozen bank")
        if entry.get("snapshot_sha256") != hashlib.sha256(bank.snapshot(entry)).hexdigest():
            raise ProvenanceError("bank entry lacks exact snapshot identity")
    from .provenance import checkpoint_hash
    obj = {"schema": 1, "entries": entries, "files": {entry["file"]: {
        "path": str((bank.directory / entry["file"]).resolve()),
        "sha256": checkpoint_hash(bank.directory / entry["file"])} for entry in entries}}
    obj["digest"] = digest(obj)
    with output.open("x") as stream: json.dump(obj, stream, sort_keys=True)
    return obj


def restore_bank(bank, state):
    if state.get("digest") != digest({k: v for k, v in state.items() if k != "digest"}):
        raise ProvenanceError("bank resume identity mismatch")
    for entry in state["entries"]:
        seeds([entry["seed"]], "restored bank")
        if hashlib.sha256(bank.snapshot(entry)).hexdigest() != state["snapshots"].get(entry["file"]):
            raise ProvenanceError("bank resume snapshot missing/mutated")
    bank._write(state["entries"])


def verify_image(image: Path) -> dict:
    """Read-only verification of every frozen compressed blob and decompressed snapshot."""
    obj = json.loads(image.read_text())
    if obj.get("schema") != 1 or obj.get("digest") != digest({k: v for k, v in obj.items() if k != "digest"}):
        raise ProvenanceError("invalid frozen bank image")
    entries = obj.get("entries", [])
    if len({e["id"] for e in entries}) != len(entries) or set(obj.get("files", {})) != {e["file"] for e in entries}:
        raise ProvenanceError("frozen bank has duplicate entries or incomplete/extra blob inventory")
    from .daughter_bank import check_partition
    check_partition(entries)
    for entry in entries:
        seeds([entry["seed"]], "frozen bank")
        if Path(entry["file"]).name != entry["file"]: raise ProvenanceError("unsafe bank blob name")
        path, _ = artifact(obj["files"][entry["file"]], image.parent)
        try: raw = zlib.decompress(path.read_bytes())
        except zlib.error as error: raise ProvenanceError("invalid compressed bank snapshot") from error
        if hashlib.sha256(raw).hexdigest() != entry.get("snapshot_sha256"):
            raise ProvenanceError("bank snapshot content mismatch")
    return obj


def private_bank(bank, image: Path, owner: Path, *, resume=False) -> dict:
    """Mutable bank is owned by one run; immutable snapshot blobs are retained for exact crash recovery."""
    if not bank.directory.resolve().is_relative_to(owner.resolve()):
        raise ProvenanceError("daughter bank must be private inside this run's output directory")
    obj = verify_image(image); root = image.parent
    marker = bank.directory / "owner.json"
    expected = {"owner": str(owner.resolve()), "image": obj["digest"]}
    if marker.exists() and json.loads(marker.read_text()) != expected:
        raise ProvenanceError("bank belongs to another run/image")
    if not marker.exists():
        if bank.entries(): raise ProvenanceError("cannot adopt an unowned mutable daughter bank")
        for entry in obj["entries"]:
            seeds([entry["seed"]], "frozen bank")
            ref = obj["files"].get(entry["file"])
            path, _ = artifact(ref, root)
            if Path(entry["file"]).name != entry["file"]: raise ProvenanceError("unsafe bank blob name")
            shutil.copyfile(path, bank.directory / entry["file"])
            if hashlib.sha256(bank.snapshot(entry)).hexdigest() != entry.get("snapshot_sha256"):
                raise ProvenanceError("bank snapshot content mismatch")
        bank._write(obj["entries"]); marker.write_text(json.dumps(expected, sort_keys=True))
    elif not resume:
        raise ProvenanceError("existing private bank requires explicit resume")
    bank.retain_snapshots = True
    return expected
