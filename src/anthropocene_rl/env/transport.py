"""Persistent JSON-lines process transport for the native Elder bridge."""

from __future__ import annotations

import base64
import json
import subprocess
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


class ElderTransport(Protocol):
    def schema(self) -> dict[str, Any]: ...

    def reset(self, payload: Mapping[str, Any]) -> dict[str, Any]: ...

    def step(self, payload: Mapping[str, Any]) -> dict[str, Any]: ...

    def option_a_step(self) -> dict[str, Any]: ...

    def verify_replay(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class JsonLineProcessTransport:
    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if not command:
            raise ValueError("Bridge command cannot be empty")
        self._process = subprocess.Popen(
            list(command),
            cwd=None if cwd is None else str(cwd),
            env=None if env is None else dict(env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()
        self._stderr_lines: deque[str] = deque(maxlen=200)
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="elder-bridge-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def schema(self) -> dict[str, Any]:
        return self._request({"op": "schema"})

    def reset(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._request({"op": "reset", "request": dict(payload)})

    def step(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._request({"op": "step", "directive": dict(payload)})

    def option_a_step(self) -> dict[str, Any]:
        return self._request({"op": "option_a_step"})

    def verify_replay(self) -> dict[str, Any]:
        return self._request({"op": "verify_replay"})

    def close(self) -> None:
        if self._process.poll() is not None:
            self._stderr_thread.join(timeout=1)
            return
        try:
            self._request({"op": "close"})
        finally:
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
            self._stderr_thread.join(timeout=1)

    def _request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError(f"Elder bridge exited early: {self._stderr_tail()}")
            if self._process.stdin is None or self._process.stdout is None:
                raise RuntimeError("Elder bridge pipes are unavailable")
            self._process.stdin.write(
                json.dumps(
                    payload,
                    separators=(",", ":"),
                    default=_json_default,
                )
                + "\n"
            )
            self._process.stdin.flush()
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError(f"Elder bridge closed stdout: {self._stderr_tail()}")
            response = json.loads(line)
            if not isinstance(response, dict):
                raise RuntimeError("Elder bridge returned a non-object response")
            if not response.get("ok", False):
                message = response.get("message", "unknown native bridge error")
                error_type = response.get("errorType", "NativeBridgeError")
                raise RuntimeError(f"{error_type}: {message}")
            return response

    def _drain_stderr(self) -> None:
        if self._process.stderr is None:
            return
        for line in self._process.stderr:
            self._stderr_lines.append(line.rstrip())

    def _stderr_tail(self) -> str:
        return "\n".join(self._stderr_lines).strip() or "no stderr output"

    def __enter__(self) -> JsonLineProcessTransport:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _json_default(value: object) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode("ascii")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
