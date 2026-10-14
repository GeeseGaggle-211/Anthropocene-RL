"""Durable reservations charge attempted native work, including failures and replay."""

import json
import threading
import time

from anthropocene_rl.autonomous_v5.causal import episode
from anthropocene_rl.autonomous_v5.collection import Collector, DeadlineHistoryTransport


class BudgetLedger:
    def __init__(
        self, path, caps, *, prior_resolves, prior_wall_seconds, deadline, native_version=5
    ):
        self.path, self.caps = path, dict(caps)
        self.prior_resolves, self.prior_wall_seconds = prior_resolves, prior_wall_seconds
        self.deadline = deadline
        self.native_version = native_version
        self.started = time.monotonic()
        self.counts = {k: 0 for k in caps}
        self.lock = threading.Lock()
        self.phase = "calibration"
        self.save()

    def save(self):
        data = dict(
            caps=self.caps,
            attempted=self.counts,
            prior_resolves=self.prior_resolves,
            total_resolves=self.prior_resolves + sum(self.counts.values()),
            prior_wall_seconds=self.prior_wall_seconds,
            active_wall_seconds=time.monotonic() - self.started,
            cumulative_wall_seconds=self.prior_wall_seconds + time.monotonic() - self.started,
        )
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(self.path)

    def reserve(self, category, count=1):
        with self.lock:
            if time.monotonic() >= self.deadline:
                raise TimeoutError("Cumulative causal campaign deadline")
            if (
                count < 0
                or category not in self.caps
                or self.counts[category] + count > self.caps[category]
            ):
                raise RuntimeError(f"Native budget exhausted: {category}")
            self.counts[category] += count
            self.save()


class BudgetTransport(DeadlineHistoryTransport):
    ledger = None

    def __init__(self, *args, **kwargs):
        self.episode_attempts = 0
        super().__init__(*args, **kwargs)

    def _request(self, payload):
        if self.ledger.native_version == 6 and payload["op"].startswith("autonomous_"):
            payload = dict(payload, op=payload["op"].replace("autonomous_", "autonomous6_", 1))
            if "actions" in payload:
                payload["actions"] = {
                    k: dict(v, SchemaVersion=4) for k, v in payload["actions"].items()
                }
        op = payload["op"]
        if op.endswith("_reset"):
            self.episode_attempts = 0
        elif op.endswith("_step"):
            self.ledger.reserve(self.ledger.phase)
            self.episode_attempts += 1
        elif op.endswith("_verify_replay"):
            self.ledger.reserve(self.ledger.phase + "_replay", self.episode_attempts)
        return super()._request(payload)


class CausalCollector(Collector):
    episode_runner = staticmethod(episode)

    def __init__(self, command, *, ledger, **kwargs):
        # Instance-bound class prevents collectors from sharing mutable class state.
        self.transport_class = type(
            "ReservedHistoryTransport", (BudgetTransport,), {"ledger": ledger}
        )
        super().__init__(command, **kwargs)
