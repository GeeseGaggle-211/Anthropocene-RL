from __future__ import annotations

import json
from pathlib import Path

import pytest

from anthropocene_rl.autonomous_v7 import gate1_reset_fixture_capture as capture
from anthropocene_rl.autonomous_v7.provenance import checkpoint_hash, digest


def _worlds() -> list[dict]:
    masks = {"default": 0, "farming": 1, "fishing": 2, "herding": 4}
    return [
        {"id": f"s{seed}-{regime}", "seed": seed, "regime": regime,
         "withheld": masks[regime], "players": 8,
         "family": "gate1-evaluation-only-50000-50015",
         "cohort_digest": digest({"seed": seed, "regime": regime}),
         "snapshot": {"kind": "none", "sha256": None}}
        for seed in range(50_000, 50_016) for regime in capture.EXPECTED_REGIMES
    ]


def _fixture(tmp_path: Path, monkeypatch, *, missing_fingerprint: bool = False):
    bridge_path = tmp_path / "bridge.dll"
    bridge_path.write_bytes(b"reviewed-bridge")
    worker = {"bridge_sha256": checkpoint_hash(bridge_path),
              "configuration": {"players": 8, "sha256": "c" * 64}}
    candidate = {
        "digest": "candidate-digest",
        "inputs": {"current_bridge": {"path": str(bridge_path),
                                       "sha256": checkpoint_hash(bridge_path),
                                       "size": bridge_path.stat().st_size}},
        "proposals": {"evaluation": {"path": "evaluation.json",
                                      "sha256": "e" * 64, "size": 1}},
    }
    evaluation = {"cohort": {"worlds": _worlds()}}
    candidate_path = tmp_path / "candidate.json"
    review_path = tmp_path / "review.json"
    candidate_path.write_text("{}", encoding="utf-8")
    review_path.write_text("{}", encoding="utf-8")
    calls = []

    class FakeBridge:
        def __init__(self, *, command):
            assert command[-1] == str(bridge_path)
            self.last_reset_identity = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def reset(self, seed, *, players, withheld, movement):
            assert players == 8 and movement is False
            calls.append((seed, withheld))
            self.last_reset_identity = {
                "sourceSeed": seed, "configurationHash": "c" * 64,
                "rawMaterialsEnabled": False,
            }
            fingerprint = None if missing_fingerprint and len(calls) == 1 else digest(
                {"seed": seed, "withheld": withheld}
            )
            return {"turn": 8 + seed % 2, "fingerprint": fingerprint}

    monkeypatch.setattr(capture.freeze, "_load_reviewed_candidate",
                        lambda *_args: (candidate, tmp_path))
    monkeypatch.setattr(capture.freeze, "_read_ref", lambda *_args, **_kwargs: evaluation)
    monkeypatch.setattr(capture, "dotnet_command", lambda: ["dotnet", str(bridge_path)])
    monkeypatch.setattr(capture, "Model9Bridge", FakeBridge)
    monkeypatch.setattr(capture, "observe", lambda _bridge: worker)
    monkeypatch.setattr(capture, "validate", lambda _bridge, expected, *_args, **_kwargs: expected)
    monkeypatch.setattr(capture, "world", lambda seed, withheld, **kwargs: {
        "seed": seed, "withheld": withheld, **kwargs,
    })
    return candidate_path, review_path, calls


def test_capture_runs_exactly_the_frozen_64_resets_and_no_steps(tmp_path, monkeypatch):
    candidate, review, calls = _fixture(tmp_path, monkeypatch)
    out = tmp_path / "reset-fixtures.json"
    body = capture.capture(candidate, review, out)
    assert body["operation_counts"] == {
        "actor_loads": 0, "policy_actions": 0, "resets": 64, "additional_steps": 0,
    }
    assert len(calls) == len(set(calls)) == 64
    assert {row["reset_turn"] for row in body["fixtures"]} == {8, 9}
    assert all(row["pre_reset_metadata_turn"] is None for row in body["fixtures"])
    observed = json.loads(out.read_text(encoding="utf-8"))
    assert observed["digest"] == digest({key: value for key, value in observed.items()
                                         if key != "digest"})


def test_capture_rejects_missing_native_boundary_without_writing_output(tmp_path, monkeypatch):
    candidate, review, calls = _fixture(tmp_path, monkeypatch, missing_fingerprint=True)
    out = tmp_path / "reset-fixtures.json"
    with pytest.raises(capture.ResetFixtureError, match="incomplete"):
        capture.capture(candidate, review, out)
    assert calls == [(50_000, 0)]
    assert not out.exists()
