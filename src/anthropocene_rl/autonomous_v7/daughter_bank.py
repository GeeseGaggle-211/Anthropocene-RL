"""Daughter-start bank (docs/MODEL9_PERSONALITY.md, daughter bank): world snapshots captured when a daughter village
is founded, so training, search and evaluation can start from young daughters instead of waiting for fission.

A bank is a directory of zlib-compressed snapshot files (``<hash>.snap``, the bridge's ``capture`` bytes, named by
content so daughters founded in the same world and turn share one file) and an
``index.jsonl`` of entries: id, seed, mask (withheld technologies), movement, turn, the daughter's society and camp and
its parent society. It keeps at most ``cap`` entries, oldest first out, and persists across iterations and runs.

Held-out partition: whole root worlds, never single entries. Entries whose world seed has seed % HOLDOUT_EVERY == 0
(``held_out``) are reserved for ``daughter_eval`` and never drawn for training. Daughters of one world and turn share a
content-addressed snapshot, so an id-based split would put identical states on both sides; a seed split cannot, and
``check_partition`` asserts that no snapshot file appears in both partitions.
"""  # noqa: E501 - docstring/prose line kept verbatim

from __future__ import annotations

import base64
import hashlib
import json
import os
import zlib
from pathlib import Path
from typing import Any

import numpy as np

CAP = 500
CAPTURES_PER_WORLD = 3  # a reservoir sample over each world rollout's founded daughters
HOLDOUT_EVERY = (
    10  # entries from worlds with seed % HOLDOUT_EVERY == 0 are the held-out evaluation family
)


def held_out(entry: dict[str, Any]) -> bool:
    """The held-out family: the entry's root world seed (bank starts keep their root seed, so a capture made inside a
    bank world stays in its root's partition)."""  # noqa: E501 - docstring/prose line kept verbatim
    return int(entry["seed"]) % HOLDOUT_EVERY == 0


def check_partition(entries: list[dict[str, Any]]) -> None:
    """No snapshot (by content hash) may serve both training and held-out evaluation."""
    train = {e["file"] for e in entries if not held_out(e)}
    shared = train & {e["file"] for e in entries if held_out(e)}
    if shared:
        raise AssertionError(
            f"daughter bank leak: snapshots in both partitions: {sorted(shared)[:3]}"
        )


class Reservoir:
    """Reservoir sample (algorithm R) of ``size`` items over a stream of founded daughters, so a world's captures
    spread over the whole episode instead of its first foundings. ``offer`` returns the slot the new item takes (or
    None: not kept) and the item it evicts (or None). The keyed generator makes it deterministic."""  # noqa: E501 - docstring/prose line kept verbatim

    def __init__(self, size: int, rng: np.random.Generator) -> None:
        self.size, self.rng, self.seen, self.items = size, rng, 0, []

    def offer(self, item: Any) -> tuple[int | None, Any]:
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(item)
            return len(self.items) - 1, None
        j = int(self.rng.integers(self.seen))
        if j >= self.size:
            return None, None
        evicted, self.items[j] = self.items[j], item
        return j, evicted


class DaughterBank:
    def __init__(self, directory: Path | str, cap: int = CAP) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cap = cap
        self.index = self.directory / "index.jsonl"

    def entries(self) -> list[dict[str, Any]]:
        if not self.index.exists():
            return []
        entries = [json.loads(line) for line in self.index.read_text().splitlines() if line.strip()]
        from .protocol import seeds

        seeds([entry["seed"] for entry in entries], "daughter bank index")
        return entries

    def training_entries(self) -> list[dict[str, Any]]:
        entries = self.entries()
        check_partition(entries)
        return [e for e in entries if not held_out(e)]

    def evaluation_entries(self) -> list[dict[str, Any]]:
        entries = self.entries()
        check_partition(entries)
        return [e for e in entries if held_out(e)]

    def add(
        self,
        snapshot: str | bytes,
        seed: int,
        mask: int,
        turn: int,
        society: int,
        camp: int,
        parent_society: int,
        movement: bool = False,
    ) -> dict[str, Any]:
        """Store one capture (base64 from the bridge, or raw bytes) and rotate the bank to its cap."""  # noqa: E501 - docstring/prose line kept verbatim
        from .protocol import seeds

        seeds([seed], "daughter capture")
        raw = base64.b64decode(snapshot) if isinstance(snapshot, str) else bytes(snapshot)
        entries = self.entries()
        next_id = max((int(e["id"]) for e in entries), default=-1) + 1
        name = hashlib.sha1(raw).hexdigest()[:20] + ".snap"
        if not (self.directory / name).exists():
            tmp = self.directory / (name + ".tmp")
            tmp.write_bytes(zlib.compress(raw, 6))
            os.replace(tmp, self.directory / name)
        entry = {
            "id": next_id,
            "seed": int(seed),
            "mask": int(mask),
            "movement": bool(movement),
            "turn": int(turn),
            "society": int(society),
            "camp": int(camp),
            "parent_society": int(parent_society),
            "file": name,
            "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        }
        entries.append(entry)
        dropped, entries = (
            entries[: -self.cap] if len(entries) > self.cap else [],
            entries[-self.cap :],
        )
        self._write(entries)
        kept = {e["file"] for e in entries}
        for e in [] if getattr(self, "retain_snapshots", False) else dropped:
            if e["file"] in kept:
                continue
            try:
                (self.directory / e["file"]).unlink()
            except FileNotFoundError:
                pass
        return entry

    def remove(self, entry_id: int) -> None:
        """Drop one entry (a reservoir eviction) and its file unless another entry shares it. Absent ids are ignored."""  # noqa: E501 - docstring/prose line kept verbatim
        entries = self.entries()
        gone = [e for e in entries if int(e["id"]) == int(entry_id)]
        if not gone:
            return
        entries = [e for e in entries if int(e["id"]) != int(entry_id)]
        self._write(entries)
        if not getattr(self, "retain_snapshots", False) and gone[0]["file"] not in {
            e["file"] for e in entries
        }:
            try:
                (self.directory / gone[0]["file"]).unlink()
            except FileNotFoundError:
                pass

    def snapshot(self, entry: dict[str, Any]) -> bytes:
        return zlib.decompress((self.directory / entry["file"]).read_bytes())

    def get(self, entry_id: int) -> dict[str, Any]:
        for e in self.entries():
            if int(e["id"]) == int(entry_id):
                return e
        raise KeyError(f"daughter bank entry {entry_id} is not in {self.directory}")

    def sample(
        self, rng: np.random.Generator, masks: set[int] | None = None
    ) -> dict[str, Any] | None:
        """A uniformly drawn training entry (``masks``: only those withheld masks), or None when there is none."""  # noqa: E501 - docstring/prose line kept verbatim
        pool = [e for e in self.training_entries() if masks is None or int(e["mask"]) in masks]
        return pool[int(rng.integers(len(pool)))] if pool else None

    def _write(self, entries: list[dict[str, Any]]) -> None:
        tmp = self.index.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(e) + "\n" for e in entries))
        os.replace(tmp, self.index)
