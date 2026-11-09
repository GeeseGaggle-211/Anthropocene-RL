"""Capture the frozen Gate-1 cohort's reset boundaries without policy execution.

This is a deliberately narrow pre-evaluation operation: it loads no actor,
submits no action, advances no transition, and makes no outcome/selection
decision.  The resulting sealed manifest lets the next reviewed candidate bind
each seed/mask fixture to its authoritative native reset turn and fingerprint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import gate1_freeze_candidate as freeze
from .bridge import Model9Bridge, dotnet_command
from .provenance import checkpoint_hash, digest
from .worker_contract import observe, validate, world

SCHEMA = 1
EXPECTED_WORLD_COUNT = 64
EXPECTED_SEEDS = tuple(range(50_000, 50_016))
EXPECTED_REGIMES = ("default", "farming", "fishing", "herding")


class ResetFixtureError(ValueError):
    """The reset fixture or its reviewed inputs are incompatible."""


def _fail(message: str) -> None:
    raise ResetFixtureError("Gate 1 reset fixture capture: " + message)


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["digest"] = digest(result)
    return result


def _worlds(candidate: dict[str, Any], base: Path) -> list[dict[str, Any]]:
    evaluation = freeze._read_ref(
        candidate["proposals"]["evaluation"],
        base,
        kind="gate1_evaluation_claim_protocol_proposal",
    )
    rows = evaluation.get("cohort", {}).get("worlds")
    expected = {(seed, regime) for seed in EXPECTED_SEEDS for regime in EXPECTED_REGIMES}
    if (
        not isinstance(rows, list)
        or len(rows) != EXPECTED_WORLD_COUNT
        or {(row.get("seed"), row.get("regime")) for row in rows if isinstance(row, dict)}
        != expected
        or len({row.get("id") for row in rows if isinstance(row, dict)}) != EXPECTED_WORLD_COUNT
    ):
        _fail("reviewed cohort is not the exact 50000-50015 by four-regime inventory")
    for row in rows:
        if (
            row.get("family") != "gate1-evaluation-only-50000-50015"
            or row.get("players") != 8
            or row.get("snapshot") != {"kind": "none", "sha256": None}
            or type(row.get("withheld")) is not int
            or type(row.get("cohort_digest")) is not str
            or not row["cohort_digest"]
        ):
            _fail("reviewed world is not a fresh generated eight-player fixture")
    return rows


def capture(candidate_path: Path, review_path: Path, out: Path) -> dict[str, Any]:
    """Perform exactly one actor-free reset for every reviewed cohort world."""
    if out.exists():
        _fail("output already exists")
    candidate, base = freeze._load_reviewed_candidate(
        candidate_path.resolve(),
        review_path.resolve(),
    )
    frozen_bridge = candidate["inputs"]["current_bridge"]
    bridge_path = Path(frozen_bridge["path"])
    if (
        not bridge_path.is_file()
        or bridge_path.is_symlink()
        or bridge_path.stat().st_size != frozen_bridge["size"]
        or checkpoint_hash(bridge_path) != frozen_bridge["sha256"]
    ):
        _fail("current diagnostic bridge differs from the reviewed binary")
    command = dotnet_command()
    if Path(command[-1]).resolve() != bridge_path.resolve():
        _fail("configured bridge command differs from the reviewed current binary")
    reviewed_worlds = _worlds(candidate, base)
    fixtures = []
    with Model9Bridge(command=command) as bridge:
        worker = observe(bridge)
        if (
            worker.get("bridge_sha256") != frozen_bridge["sha256"]
            or worker.get("configuration", {}).get("players") != 8
        ):
            _fail("observed worker differs from the reviewed bridge or player count")
        for frozen in reviewed_worlds:
            descriptor = world(
                frozen["seed"],
                frozen["withheld"],
                turn=None,
                players=frozen["players"],
                movement=False,
                head=None,
                cohort_digest=frozen["cohort_digest"],
            )
            observed = validate(bridge, worker, descriptor, snapshot=None, head=None)
            if observed != worker:
                _fail("worker identity changed between reset fixtures")
            step = bridge.reset(
                frozen["seed"],
                players=frozen["players"],
                withheld=frozen["withheld"],
                movement=False,
            )
            identity = bridge.last_reset_identity
            turn, fingerprint = step.get("turn"), step.get("fingerprint")
            if (
                type(turn) is not int
                or turn < 0
                or not isinstance(fingerprint, str)
                or not fingerprint
                or identity.get("sourceSeed") != frozen["seed"]
                or identity.get("configurationHash") != worker["configuration"]["sha256"]
                or identity.get("rawMaterialsEnabled") is not False
            ):
                _fail("native reset returned an incomplete or incompatible boundary identity")
            fixtures.append(
                {
                    "world_id": frozen["id"],
                    "seed": frozen["seed"],
                    "regime": frozen["regime"],
                    "withheld": frozen["withheld"],
                    "players": frozen["players"],
                    "cohort_digest": frozen["cohort_digest"],
                    "pre_reset_metadata_turn": None,
                    "reset_turn": turn,
                    "reset_fingerprint": fingerprint,
                    "source_seed": identity["sourceSeed"],
                    "configuration_sha256": identity["configurationHash"],
                    "raw_materials_enabled": identity["rawMaterialsEnabled"],
                    "world_contract": descriptor,
                }
            )
            del step
    source = Path(__file__).resolve()
    body = _seal(
        {
            "schema": SCHEMA,
            "kind": "gate1_reset_fixture_capture",
            "candidate_digest": candidate["digest"],
            "candidate_sha256": checkpoint_hash(candidate_path),
            "review_sha256": checkpoint_hash(review_path),
            "capture_implementation": {
                "path": "src/anthropocene_rl/autonomous_v7/gate1_reset_fixture_capture.py",
                "sha256": checkpoint_hash(source),
                "size": source.stat().st_size,
            },
            "bridge": frozen_bridge,
            "worker_contract": worker,
            "world_count": len(fixtures),
            "fixtures": fixtures,
            "operation_counts": {
                "actor_loads": 0,
                "policy_actions": 0,
                "resets": len(fixtures),
                "additional_steps": 0,
            },
            "scope": "metadata-only frozen-cohort reset-boundary capture",
            "selection_allowed": False,
            "readiness_claimed": False,
            "outcome_evidence_claimed": False,
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    body = capture(args.candidate, args.review, args.out)
    print(json.dumps({"path": str(args.out), "digest": body["digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
