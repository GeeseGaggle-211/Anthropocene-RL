from __future__ import annotations

import base64
import gzip
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from anthropocene_rl.autonomous_v7 import bridge as bridge_module
from anthropocene_rl.autonomous_v7 import provenance
from anthropocene_rl.autonomous_v7.semantic_trace import (
    ACTION_CONTRACT,
    FINGERPRINT_CONTRACT,
    NORMALIZATION,
    TRACE_SCHEMA,
    complete_payload,
    verify,
)

FLAGS = {"sim_rules": "", "sim_statics": ""}


def _payload(turn: int, value: int) -> bytes:
    magic = b"ANTHROPOCENE-11"
    header = b"complete test header"
    return (
        bytes([len(magic)])
        + magic
        + struct.pack("<i", len(header))
        + header
        + struct.pack("<iii", 1, turn, value)
    )


def _frame(tmp_path: Path, name: str, turn: int, value: int, seed: int):
    raw = gzip.compress(_payload(turn, value), mtime=0)
    path = tmp_path / name
    path.write_bytes(raw)
    artifact = {"path": str(path), "sha256": provenance.checkpoint_hash(path)}
    payload, hashes = complete_payload(raw)
    return (
        {
            "turn": turn,
            "source_seed": seed,
            "fingerprint": hashlib.sha256(payload).hexdigest(),
            "snapshot": artifact,
            **hashes,
        },
        raw,
        artifact,
    )


def _fake_environment(monkeypatch, identities, factors=None):
    calls = []

    class ReplayBridge:
        def __init__(self, command):
            identity = identities[Path(command[1]).name]
            self.factor = (factors or {}).get(Path(command[1]).name, 1)
            self.schema = SimpleNamespace(
                build_hash=identity["buildHash"],
                simulator_commit=identity["commit"],
                simulator_source_clean=identity["clean"],
            )
            self.seed = self.turn = self.value = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def reset(self, seed, *, players, movement, withheld, snapshot):
            calls.append("reset")
            payload = gzip.decompress(snapshot)
            self.seed = seed
            self.turn, self.value = struct.unpack_from("<ii", payload, len(payload) - 8)
            return {"turn": self.turn, "villages": [{"camp": 7}, {"camp": 8}]}

        def step(self, actions):
            calls.append("step")
            self.value += int(actions[7]["delta"]) * self.factor
            self.turn += 1
            return {"turn": self.turn, "villages": [{"camp": 7}, {"camp": 8}]}

        def request(self, request):
            payload = _payload(self.turn, self.value)
            fingerprint = hashlib.sha256(payload).hexdigest()
            if request["op"] == "fingerprint_snapshot":
                return {
                    "snapshot": base64.b64encode(gzip.compress(payload, mtime=0)).decode("ascii"),
                    "fingerprint": fingerprint,
                    "turn": self.turn,
                    "sourceSeed": self.seed,
                }
            if request["op"] == "validate_fingerprint_snapshot":
                checked = gzip.decompress(base64.b64decode(request["snapshot"], validate=True))
                return {
                    "sourceSeed": self.seed,
                    "turn": self.turn,
                    "payloadFingerprint": hashlib.sha256(checked).hexdigest(),
                }
            raise AssertionError(request)

    monkeypatch.setattr(bridge_module, "Model9Bridge", ReplayBridge)
    monkeypatch.setattr(bridge_module, "dotnet_command", lambda: ["dotnet", "unused.dll"])
    return calls


def _trace(tmp_path: Path, side: str, identity: dict, outputs: set[str]):
    binary = tmp_path / (side + ".dll")
    binary.write_bytes((side + " pinned test bridge").encode())
    bridge_ref = {"path": str(binary), "sha256": provenance.checkpoint_hash(binary)}
    outputs.add(bridge_ref["sha256"])
    initial, initial_raw, initial_ref = _frame(tmp_path, side + "-initial.save", 3, 10, 2000)
    outputs.add(initial_ref["sha256"])
    reset_path = tmp_path / (side + "-reset.save")
    reset_path.write_bytes(initial_raw)
    reset_ref = {"path": str(reset_path), "sha256": provenance.checkpoint_hash(reset_path)}
    outputs.add(reset_ref["sha256"])
    successor, _, successor_ref = _frame(tmp_path, side + "-next.save", 4, 13, 2000)
    outputs.add(successor_ref["sha256"])
    trace = {
        "schema": TRACE_SCHEMA,
        "kind": "simulator_semantic_trace",
        "normalization": NORMALIZATION,
        "fingerprint_contract": FINGERPRINT_CONTRACT,
        "action_contract": ACTION_CONTRACT,
        "simulator": identity,
        "flags": FLAGS,
        "seed": 2000,
        "turns": 1,
        "bridge": bridge_ref,
        "reset": {"seed": 2000, "players": 2, "withheld": 0, "movement": False},
        "reset_snapshot": reset_ref,
        "initial_state": initial,
        "transitions": [
            {
                "from_turn": 3,
                "actions": {"7": {"delta": 3}},
                "background": {"mode": "native_teacher_for_unsubmitted_camps", "camps": [8]},
                "successor": successor,
            }
        ],
    }
    return trace, initial_raw


def test_two_pinned_bridge_replays_match_and_wrong_action_or_copied_frame_fail(
    tmp_path, monkeypatch
):
    source = {"commit": "a" * 40, "buildHash": "1" * 32, "clean": True}
    live = {"commit": "b" * 40, "buildHash": "2" * 32, "clean": True}
    _fake_environment(monkeypatch, {"source.dll": source, "live.dll": live})
    outputs: set[str] = set()
    traces = [_trace(tmp_path, "source", source, outputs), _trace(tmp_path, "live", live, outputs)]
    for side, (trace, _) in zip(("source", "live"), traces, strict=False):
        row = {
            "seed": 2000,
            "turns": 1,
            side + "_fingerprint": trace["transitions"][-1]["successor"]["fingerprint"],
        }
        assert len(verify(trace, tmp_path, trace["simulator"], FLAGS, row, side, outputs)) == 2

    trace, _ = traces[1]
    row = {
        "seed": 2000,
        "turns": 1,
        "live_fingerprint": trace["transitions"][-1]["successor"]["fingerprint"],
    }
    wrong_action = json.loads(json.dumps(trace))
    wrong_action["transitions"][0]["actions"]["7"]["delta"] = 2
    with pytest.raises(provenance.ProvenanceError, match="replay successor differs"):
        verify(wrong_action, tmp_path, live, FLAGS, row, "live", outputs)

    copied = json.loads(json.dumps(trace))
    copied["transitions"][0]["successor"] = copied["initial_state"]
    with pytest.raises(provenance.ProvenanceError, match="successor turn/root alignment mismatch"):
        verify(copied, tmp_path, live, FLAGS, row, "live", outputs)

    wrong_background = json.loads(json.dumps(trace))
    wrong_background["transitions"][0]["background"]["camps"] = []
    with pytest.raises(provenance.ProvenanceError, match="background input"):
        verify(wrong_background, tmp_path, live, FLAGS, row, "live", outputs)

    wrong_bridge = json.loads(json.dumps(trace))
    wrong_bridge["bridge"] = traces[0][0]["bridge"]
    with pytest.raises(provenance.ProvenanceError, match="actual binary differs"):
        verify(wrong_bridge, tmp_path, live, FLAGS, row, "live", outputs)

    wrong_root = json.loads(json.dumps(trace))
    wrong_root["initial_state"]["source_seed"] = 2001
    with pytest.raises(provenance.ProvenanceError, match="initial state root differs"):
        verify(wrong_root, tmp_path, live, FLAGS, row, "live", outputs)

    wrong_successor_root = json.loads(json.dumps(trace))
    wrong_successor_root["transitions"][0]["successor"]["source_seed"] = 2001
    with pytest.raises(provenance.ProvenanceError, match="successor root differs"):
        verify(wrong_successor_root, tmp_path, live, FLAGS, row, "live", outputs)

    wrong_turn = json.loads(json.dumps(trace))
    wrong_turn["transitions"][0]["from_turn"] = 2
    with pytest.raises(provenance.ProvenanceError, match="turn is not contiguous"):
        verify(wrong_turn, tmp_path, live, FLAGS, row, "live", outputs)

    incomplete = json.loads(json.dumps(trace))
    incomplete["transitions"].clear()
    with pytest.raises(provenance.ProvenanceError, match="versioned replayable"):
        verify(incomplete, tmp_path, live, FLAGS, row, "live", outputs)


def test_real_bridge_two_process_replay_is_reproducible(tmp_path, monkeypatch):
    from anthropocene_rl.autonomous_v7.bridge import DEFAULT_BRIDGE, Model9Bridge

    if not DEFAULT_BRIDGE.exists():
        pytest.skip("model9 bridge is not built")
    monkeypatch.setenv("ANTHROPOCENE_SIM_RULES", "")
    monkeypatch.setenv("ANTHROPOCENE_SIM_STATICS", "")
    flags = {"sim_rules": "", "sim_statics": ""}
    outputs: set[str] = set()
    binary_ref = {"path": str(DEFAULT_BRIDGE), "sha256": provenance.checkpoint_hash(DEFAULT_BRIDGE)}
    outputs.add(binary_ref["sha256"])
    with Model9Bridge() as producer:
        start = producer.reset(1005, players=8, movement=False)
        identity = provenance.simulator_identity(producer.schema, production=True)
        initial_meta = producer.request({"op": "fingerprint_snapshot"})
        initial_raw = base64.b64decode(initial_meta["snapshot"], validate=True)
        reset_raw = base64.b64decode(producer.capture(), validate=True)
        initial_path = tmp_path / "real-initial.save"
        initial_path.write_bytes(initial_raw)
        initial_ref = {
            "path": str(initial_path),
            "sha256": provenance.checkpoint_hash(initial_path),
        }
        outputs.add(initial_ref["sha256"])
        reset_path = tmp_path / "real-reset.save"
        reset_path.write_bytes(reset_raw)
        reset_ref = {"path": str(reset_path), "sha256": provenance.checkpoint_hash(reset_path)}
        outputs.add(reset_ref["sha256"])
        _, initial_hashes = complete_payload(initial_raw)
        initial = {
            "turn": start["turn"],
            "source_seed": 1005,
            "fingerprint": initial_meta["fingerprint"],
            "snapshot": initial_ref,
            **initial_hashes,
        }
        camps = sorted(int(v["camp"]) for v in start["villages"])
        action_map = {}
        background = {"mode": "native_teacher_for_unsubmitted_camps", "camps": camps}
        successor_step = producer.step(action_map)
        actual = producer.request({"op": "fingerprint_snapshot"})
        successor_raw = base64.b64decode(actual["snapshot"], validate=True)
        successor_path = tmp_path / "real-successor.save"
        successor_path.write_bytes(successor_raw)
        successor_ref = {
            "path": str(successor_path),
            "sha256": provenance.checkpoint_hash(successor_path),
        }
        outputs.add(successor_ref["sha256"])
        _, successor_hashes = complete_payload(successor_raw)
        successor = {
            "turn": successor_step["turn"],
            "source_seed": 1005,
            "fingerprint": actual["fingerprint"],
            "snapshot": successor_ref,
            **successor_hashes,
        }

    trace = {
        "schema": TRACE_SCHEMA,
        "kind": "simulator_semantic_trace",
        "normalization": NORMALIZATION,
        "fingerprint_contract": FINGERPRINT_CONTRACT,
        "action_contract": ACTION_CONTRACT,
        "simulator": identity,
        "flags": flags,
        "seed": 1005,
        "turns": 1,
        "bridge": binary_ref,
        "reset": {"seed": 1005, "players": 8, "withheld": 0, "movement": False},
        "reset_snapshot": reset_ref,
        "initial_state": initial,
        "transitions": [
            {
                "from_turn": start["turn"],
                "actions": {},
                "background": background,
                "successor": successor,
            }
        ],
    }
    row = {
        "seed": 1005,
        "turns": 1,
        "source_fingerprint": successor["fingerprint"],
        "live_fingerprint": successor["fingerprint"],
    }
    assert len(verify(trace, tmp_path, identity, flags, row, "source", outputs)) == 2
    assert len(verify(trace, tmp_path, identity, flags, row, "live", outputs)) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "same",
        "compensating-actions",
        "reset-players",
        "reset-movement",
        "ordinary-reset",
        "canonical-initial",
        "background",
    ],
)
def test_compatibility_requires_same_experiment_before_either_replay(
    tmp_path, monkeypatch, mutation
):
    source = {"commit": "a" * 40, "buildHash": "1" * 32, "clean": True}
    live = {"commit": "b" * 40, "buildHash": "2" * 32, "clean": True}
    calls = _fake_environment(
        monkeypatch,
        {"source.dll": source, "live.dll": live},
        {"live.dll": 3} if mutation == "compensating-actions" else None,
    )
    outputs = set()
    a, _ = _trace(tmp_path, "source", source, outputs)
    b, _ = _trace(tmp_path, "live", live, outputs)
    if mutation == "compensating-actions":
        b["transitions"][0]["actions"]["7"]["delta"] = 1
    elif mutation == "reset-players":
        b["reset"]["players"] = 999
    elif mutation == "reset-movement":
        b["reset"]["movement"] = True
    elif mutation == "ordinary-reset":
        _frame_unused, _raw, ref = _frame(tmp_path, "other-reset.save", 3, 11, 2000)
        b["reset_snapshot"] = ref
        outputs.add(ref["sha256"])
    elif mutation == "canonical-initial":
        frame, _, ref = _frame(tmp_path, "other-initial.save", 3, 11, 2000)
        b["initial_state"] = frame
        outputs.add(ref["sha256"])
    elif mutation == "background":
        b["transitions"][0]["background"]["camps"] = []
    row = {
        "seed": 2000,
        "turns": 1,
        "equal": True,
        "source_fingerprint": a["transitions"][0]["successor"]["fingerprint"],
        "live_fingerprint": b["transitions"][0]["successor"]["fingerprint"],
    }
    if mutation in ("same", "compensating-actions", "reset-players", "reset-movement"):
        # Each trace independently replays successfully, including compensating different
        # interventions.
        for trace, side in ((a, "source"), (b, "live")):
            verify(trace, tmp_path, trace["simulator"], FLAGS, row, side, outputs)
        calls.clear()
    for trace, side in ((a, "source"), (b, "live")):
        path = tmp_path / (side + ".json")
        path.write_text(json.dumps(trace))
        row[side + "_trace"] = {"path": str(path), "sha256": provenance.checkpoint_hash(path)}
        outputs.add(row[side + "_trace"]["sha256"])
    refs = [
        {"path": str(path), "sha256": provenance.checkpoint_hash(path)}
        for path in tmp_path.iterdir()
        if path.is_file() and provenance.checkpoint_hash(path) in outputs
    ]
    body = {
        "schema": 1,
        "status": "pass",
        "kind": "simulator_semantic_comparison",
        "source": source,
        "live": live,
        "flags": FLAGS,
        "outputs": refs,
        "comparisons": [row],
    }
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(body))
    ref = {"path": str(path), "sha256": provenance.checkpoint_hash(path)}
    mapping = {
        "source": source,
        "live": live,
        "reason": "pre-registered comparison",
        "flags": FLAGS,
        "evidence": ref,
        "evidence_digest": ref["sha256"],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"simulator_compatibility": [mapping]}))
    if mutation == "same":
        assert provenance.compatibility_mapping(manifest, FLAGS) == [mapping]
        assert calls == ["reset", "step", "reset", "step"]
    else:
        with pytest.raises(
            provenance.ProvenanceError, match=r"experiment|reset|initial|interventions"
        ):
            provenance.compatibility_mapping(manifest, FLAGS)
        assert calls == []
