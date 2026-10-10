"""Persistent JSON-lines process transport for the native Elder bridge."""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


class ElderTransport(Protocol):
    def schema(self) -> dict[str, Any]: ...

    def reset(self, payload: Mapping[str, Any]) -> dict[str, Any]: ...

    def step(self, payload: Mapping[str, Any]) -> dict[str, Any]: ...

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

    def schema(self) -> dict[str, Any]:
        return self._request({"op": "schema"})

    def reset(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._request({"op": "reset", "request": dict(payload)})

    def step(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._request({"op": "step", "directive": dict(payload)})

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            self._request({"op": "close"})
        finally:
            if self._process.poll() is None:
                self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)

    def _request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._process.poll() is not None:
                stderr = self._process.stderr.read() if self._process.stderr is not None else ""
                raise RuntimeError(f"Elder bridge exited early: {stderr.strip()}")
            if self._process.stdin is None or self._process.stdout is None:
                raise RuntimeError("Elder bridge pipes are unavailable")
            self._process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
            if not line:
                stderr = self._process.stderr.read() if self._process.stderr is not None else ""
                raise RuntimeError(f"Elder bridge closed stdout: {stderr.strip()}")
            response = json.loads(line)
            if not isinstance(response, dict):
                raise RuntimeError("Elder bridge returned a non-object response")
            if not response.get("ok", False):
                message = response.get("message", "unknown native bridge error")
                error_type = response.get("errorType", "NativeBridgeError")
                raise RuntimeError(f"{error_type}: {message}")
            return response

    def __enter__(self) -> "JsonLineProcessTransport":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
