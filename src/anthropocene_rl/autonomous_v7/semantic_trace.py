"""Complete-save comparator evidence; no partial hand-authored physical projection is accepted."""

import base64
import gzip
import hashlib
import io
import json
import os
import struct
from unittest.mock import patch

from .protocol import artifact, seeds
from .provenance import ProvenanceError, checkpoint_hash, simulator_identity

# Deliberately conservative: decompress only; remove/rewrite no serialized world field.
# Any future non-identity normalization needs its own reviewed version and exact allowlist.
NORMALIZATION = {"version": 1, "kind": "savecodec-complete-identity", "excluded_fields": []}
FINGERPRINT_CONTRACT = "savecodec-canonical-writepayload-sha256-v1"
TRACE_SCHEMA = 3
ACTION_CONTRACT = "model9-step-actions-and-native-unsubmitted-v1"


def complete_payload(raw):
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            payload = stream.read((1 << 30) + 1)
        if len(payload) > 1 << 30:
            raise ValueError("save too large")
        # BinaryWriter's 7-bit string length; modern SaveCodec magic is shorter than 128 bytes.
        length = payload[0]
        magic = payload[1 : 1 + length].decode("ascii")
        if magic not in {"ANTHROPOCENE-" + str(v) for v in range(11, 16)}:
            raise ValueError("unsupported SaveCodec envelope")
        offset = 1 + length + int(magic in ("ANTHROPOCENE-14", "ANTHROPOCENE-15"))
        size = struct.unpack_from("<i", payload, offset)[0]
        start = offset + 4
        if size <= 0 or size > 512 * 1024 * 1024 or start + size + 4 >= len(payload):
            raise ValueError("incomplete SaveCodec header/tail")
        header, tail = payload[start : start + size], payload[start + size :]
        if struct.unpack_from("<i", tail)[0] <= 0:
            raise ValueError("SaveCodec tile tail absent")
        hashes = {
            "normalized_payload_sha256": hashlib.sha256(payload).hexdigest(),
            "header_sha256": hashlib.sha256(header).hexdigest(),
            "binary_tail_sha256": hashlib.sha256(tail).hexdigest(),
        }
        return payload, hashes
    except (OSError, EOFError, ValueError, IndexError, struct.error) as error:
        raise ProvenanceError("invalid complete SaveCodec artifact: " + str(error)) from error


def same_experiment(source, source_base, live, live_base, outputs):
    """Different binaries must receive one identical, content-backed experiment before either replay."""

    def canonical(value):
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ProvenanceError("invalid shared comparator experiment") from error

    fields = (
        "schema",
        "kind",
        "normalization",
        "fingerprint_contract",
        "action_contract",
        "flags",
        "seed",
        "turns",
        "reset",
    )
    if any(
        key not in source or key not in live or canonical(source[key]) != canonical(live[key])
        for key in fields
    ):
        raise ProvenanceError("comparator traces do not share identical experiment/reset inputs")

    def payload(ref, base):
        path, sha = artifact(ref, base)
        if sha not in outputs:
            raise ProvenanceError("shared experiment artifact omitted from output inventory")
        return complete_payload(path.read_bytes())[0]

    if payload(source.get("reset_snapshot"), source_base) != payload(
        live.get("reset_snapshot"), live_base
    ):
        raise ProvenanceError("comparator traces have different ordinary reset snapshot semantics")
    a, b = source.get("initial_state", {}), live.get("initial_state", {})
    identity = (
        "turn",
        "source_seed",
        "fingerprint",
        "normalized_payload_sha256",
        "header_sha256",
        "binary_tail_sha256",
    )
    if any(
        key not in a or key not in b or canonical(a[key]) != canonical(b[key]) for key in identity
    ) or payload(a.get("snapshot"), source_base) != payload(b.get("snapshot"), live_base):
        raise ProvenanceError("comparator traces have different canonical initial state/root/turn")
    left, right = source.get("transitions"), live.get("transitions")
    if (
        not isinstance(left, list)
        or not left
        or not isinstance(right, list)
        or len(left) != len(right)
    ):
        raise ProvenanceError("comparator shared experiment transitions are incomplete")
    for a, b in zip(left, right):
        if (
            not isinstance(a, dict)
            or not isinstance(b, dict)
            or any(
                key not in a or key not in b or canonical(a[key]) != canonical(b[key])
                for key in ("from_turn", "actions", "background")
            )
        ):
            raise ProvenanceError(
                "comparator traces have different shared action/background interventions"
            )


def verify(trace, base, identity, flags, row, side, outputs):
    from .bridge import Model9Bridge, dotnet_command

    if (
        trace.get("schema") != TRACE_SCHEMA
        or trace.get("kind") != "simulator_semantic_trace"
        or trace.get("normalization") != NORMALIZATION
        or trace.get("simulator") != identity
        or trace.get("fingerprint_contract") != FINGERPRINT_CONTRACT
        or trace.get("action_contract") != ACTION_CONTRACT
        or trace.get("flags") != flags
        or trace.get("seed") != row["seed"]
        or trace.get("turns") != row["turns"]
        or not isinstance(trace.get("transitions"), list)
        or row["turns"] < 1
        or len(trace["transitions"]) != row["turns"]
    ):
        raise ProvenanceError("comparator needs a versioned replayable complete-save trace")
    if (
        not isinstance(flags, dict)
        or set(flags) != {"sim_rules", "sim_statics"}
        or not all(isinstance(v, str) for v in flags.values())
    ):
        raise ProvenanceError("comparator trace flags must pin simulator rules and statics")
    seeds([trace["seed"]], "comparator trace")
    binary, sha = artifact(trace.get("bridge"), base)
    if sha not in outputs:
        raise ProvenanceError("comparator binary omitted from output inventory")
    reset = trace.get("reset")
    if (
        not isinstance(reset, dict)
        or set(reset) != {"seed", "players", "withheld", "movement"}
        or reset.get("seed") != trace["seed"]
        or not isinstance(reset.get("players"), int)
        or not isinstance(reset.get("withheld"), int)
        or not isinstance(reset.get("movement"), bool)
    ):
        raise ProvenanceError("comparator trace reset input is incomplete or unsupported")
    initial = trace.get("initial_state")
    if not isinstance(initial, dict) or initial.get("turn") is None:
        raise ProvenanceError("comparator trace lacks a content-addressed initial state")
    try:
        first_turn = int(initial["turn"])
    except (KeyError, TypeError, ValueError) as error:
        raise ProvenanceError("invalid initial trace turn") from error
    paths = []

    def read_frame(frame, expected_turn, label):
        if not isinstance(frame, dict) or frame.get("turn") != expected_turn:
            raise ProvenanceError(label + " turn/root alignment mismatch")
        if frame.get("source_seed") != trace["seed"]:
            raise ProvenanceError(label + " root differs from trace seed")
        path, snapshot_sha = artifact(frame.get("snapshot"), base)
        if snapshot_sha not in outputs:
            raise ProvenanceError("comparator snapshot omitted from output inventory")
        raw = path.read_bytes()
        payload, hashes = complete_payload(raw)
        if any(frame.get(key) != value for key, value in hashes.items()):
            raise ProvenanceError(label + " complete header/binary-tail/payload hash mismatch")
        if not isinstance(frame.get("fingerprint"), str) or len(frame["fingerprint"]) != 64:
            raise ProvenanceError(label + " fingerprint is missing")
        if frame["fingerprint"].lower() != hashes["normalized_payload_sha256"]:
            raise ProvenanceError(label + " fingerprint does not bind its full payload")
        return path, raw, payload

    initial_path, initial_raw, initial_payload = read_frame(initial, first_turn, "initial state")
    if initial.get("source_seed") != trace["seed"]:
        raise ProvenanceError("comparator initial state root differs from trace seed")
    reset_path, reset_sha = artifact(trace.get("reset_snapshot"), base)
    if reset_sha not in outputs:
        raise ProvenanceError("comparator reset snapshot omitted from output inventory")
    reset_raw = reset_path.read_bytes()
    paths.append(initial_path)
    env = {
        "ANTHROPOCENE_SIM_RULES": flags["sim_rules"],
        "ANTHROPOCENE_SIM_STATICS": flags["sim_statics"],
    }
    with patch.dict(os.environ, env):
        with Model9Bridge([dotnet_command()[0], str(binary)]) as bridge:
            if simulator_identity(bridge.schema, production=True) != identity:
                raise ProvenanceError("comparator actual binary differs from trace simulator")
            start = bridge.reset(
                trace["seed"],
                players=reset["players"],
                movement=reset["movement"],
                withheld=reset["withheld"],
                snapshot=reset_raw,
            )
            if int(start.get("turn", -1)) != first_turn:
                raise ProvenanceError("native reset did not restore the trace initial turn")

            def compare_live(frame, expected_turn, label):
                live = bridge.request({"op": "fingerprint_snapshot"})
                try:
                    live_raw = base64.b64decode(live["snapshot"], validate=True)
                except (KeyError, ValueError) as error:
                    raise ProvenanceError(label + " native snapshot is invalid") from error
                live_payload, live_hashes = complete_payload(live_raw)
                checked = bridge.request(
                    {
                        "op": "validate_fingerprint_snapshot",
                        "snapshot": base64.b64encode(live_raw).decode("ascii"),
                    }
                )
                if (
                    live.get("turn") != expected_turn
                    or checked.get("turn") != expected_turn
                    or checked.get("sourceSeed") != trace["seed"]
                    or live.get("fingerprint") != frame.get("fingerprint")
                    or str(checked.get("payloadFingerprint", "")).lower()
                    != str(frame.get("fingerprint", "")).lower()
                    or live_hashes["normalized_payload_sha256"]
                    != frame.get("normalized_payload_sha256")
                    or live_payload != read_frame(frame, expected_turn, label)[2]
                ):
                    raise ProvenanceError(
                        label
                        + " replay successor differs in root, turn, fingerprint, or complete save payload"
                    )

            compare_live(initial, first_turn, "initial state")
            current = start
            for index, transition in enumerate(trace["transitions"]):
                expected_from = first_turn + index
                if not isinstance(transition, dict) or transition.get("from_turn") != expected_from:
                    raise ProvenanceError("comparator transition turn is not contiguous")
                actions = transition.get("actions")
                if not isinstance(actions, dict):
                    raise ProvenanceError("comparator transition action input is missing")
                try:
                    json.dumps(actions, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    numeric_actions = {int(key): value for key, value in actions.items()}
                    if (
                        len(numeric_actions) != len(actions)
                        or any(not isinstance(key, str) or str(int(key)) != key for key in actions)
                        or any(not isinstance(v, dict) for v in numeric_actions.values())
                    ):
                        raise ValueError("invalid action map")
                except (TypeError, ValueError) as error:
                    raise ProvenanceError("comparator action input is invalid") from error
                camp_ids = sorted(int(v["camp"]) for v in current.get("villages", []))
                if any(camp not in camp_ids for camp in numeric_actions):
                    raise ProvenanceError(
                        "comparator action references a camp absent at the decision boundary"
                    )
                background = transition.get("background")
                expected_background = {
                    "mode": "native_teacher_for_unsubmitted_camps",
                    "camps": [camp for camp in camp_ids if camp not in numeric_actions],
                }
                if background != expected_background:
                    raise ProvenanceError(
                        "comparator background input does not exactly identify native-teacher camps"
                    )
                next_step = bridge.step(numeric_actions)
                expected_successor = expected_from + 1
                successor = transition.get("successor")
                successor_path, _, _ = read_frame(successor, expected_successor, "successor")
                compare_live(successor, expected_successor, "successor")
                paths.append(successor_path)
                current = next_step
    if checkpoint_hash(binary) != sha:
        raise ProvenanceError("comparator producer binary changed during verification")
    if trace["transitions"][-1]["successor"]["fingerprint"] != row[side + "_fingerprint"]:
        raise ProvenanceError("comparator endpoint fingerprint mismatch")
    return paths
