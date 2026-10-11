import base64
import sys

from anthropocene_rl.env.transport import JsonLineProcessTransport


def test_snapshot_bytes_are_base64_encoded_for_native_json() -> None:
    script = """
import json
import sys
for line in sys.stdin:
    request = json.loads(line)
    if request["op"] == "close":
        print(json.dumps({"ok": True}), flush=True)
        break
    print(json.dumps({"ok": True, "request": request}), flush=True)
"""
    with JsonLineProcessTransport([sys.executable, "-u", "-c", script]) as transport:
        response = transport.reset({"Snapshot": b"\x00\xffnative"})
        replay = transport.verify_replay()
    encoded = response["request"]["request"]["Snapshot"]
    assert encoded == base64.b64encode(b"\x00\xffnative").decode("ascii")
    assert replay["request"] == {"op": "verify_replay"}
