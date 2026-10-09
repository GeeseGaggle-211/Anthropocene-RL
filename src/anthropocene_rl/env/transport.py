from __future__ import annotations

from typing import Protocol


class ElderTransport(Protocol):
    """Framework-neutral process boundary; concrete IPC is a section-34 decision."""

    def reset(self, payload: bytes) -> bytes: ...

    def step(self, payload: bytes) -> bytes: ...

    def close(self) -> None: ...
