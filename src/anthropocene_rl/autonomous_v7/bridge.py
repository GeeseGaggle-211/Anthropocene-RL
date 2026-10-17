"""JSON-lines client for the model9 native bridge (``integration/Anthropocene.Model9Bridge``)."""

from __future__ import annotations

import os
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import orjson

from . import contracts

REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_BRIDGE = REPOSITORY / "integration" / "bin" / "model9" / "Anthropocene.Model9Bridge.dll"


def dotnet_command() -> list[str]:
    dotnet = os.environ.get("ANTHROPOCENE_DOTNET", "dotnet")
    bridge = os.environ.get("ANTHROPOCENE_MODEL9_BRIDGE", str(DEFAULT_BRIDGE))
    return [dotnet, bridge]


class Model9Bridge:
    def __init__(self, command: Sequence[str] | None = None) -> None:
        self._process = subprocess.Popen(
            list(command or dotnet_command()),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._lock = threading.Lock()
        self._stderr: deque[str] = deque(maxlen=200)
        threading.Thread(target=self._drain, daemon=True).start()
        self.schema = contracts.verify_schema(self.request({"op": "schema"}))

    def request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError("model9 bridge exited: " + "\n".join(self._stderr))
            assert self._process.stdin is not None and self._process.stdout is not None
            self._process.stdin.write(orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY) + b"\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError("model9 bridge closed stdout: " + "\n".join(self._stderr))
            response = orjson.loads(line)
            if not response.get("ok", False):
                raise RuntimeError(f"model9 bridge error: {response.get('error')}")
            return response

    def reset(self, seed: int, players: int = 8, movement: bool = False) -> dict[str, Any]:
        return self.request({"op": "reset", "seed": seed, "players": players, "movement": movement})["step"]

    def step(self, actions: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
        return self.request({"op": "step", "actions": {str(k): v for k, v in actions.items()}})["step"]

    def verify_replay(self) -> int:
        return int(self.request({"op": "verify_replay"})["steps"])

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                self.request({"op": "close"})
            finally:
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()

    def _drain(self) -> None:
        assert self._process.stderr is not None
        for raw in self._process.stderr:
            self._stderr.append(raw.decode("utf-8", "replace").rstrip())

    def __enter__(self) -> "Model9Bridge":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def observation_batch(villages: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Stack bridge observations into the same arrays the NPY shards hold."""
    obs = [v["observation"] for v in villages]
    batch: dict[str, np.ndarray] = {
        "core": np.asarray([o["core"] for o in obs], dtype=np.float32),
        "core_present": np.asarray([o["corePresent"] for o in obs], dtype=np.bool_),
        "movement_mask": np.asarray([o["movementMask"] for o in obs], dtype=np.bool_),
        "clothing_route_mask": np.asarray([o["clothingRouteMask"] for o in obs], dtype=np.bool_),
        "shelter_route_mask": np.asarray([o["shelterRouteMask"] for o in obs], dtype=np.bool_),
    }
    for index, (name, rows, width) in enumerate(contracts.SETS):
        batch[name] = np.asarray([o["sets"][index] for o in obs], dtype=np.float32).reshape(len(obs), rows, width)
        batch[name + "_present"] = np.asarray([o["rowPresent"][index] for o in obs], dtype=np.bool_)
    return batch
