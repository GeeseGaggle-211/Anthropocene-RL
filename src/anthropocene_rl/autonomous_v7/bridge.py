"""JSON-lines client for the model9 native bridge (``integration/Anthropocene.Model9Bridge``)."""

from __future__ import annotations

import base64
import copy
import os
import subprocess
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
    def __init__(
        self, command: Sequence[str] | None = None, *, economic_diagnostics: bool = False
    ) -> None:
        self.economic_diagnostics_enabled = economic_diagnostics
        self.economic_diagnostics: dict[int, dict] = {}
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
        schema_response = self.request({"op": "schema"})
        self.schema = contracts.verify_schema(schema_response)
        self.schema_response = schema_response
        # Critic-only privileged view layout (simulator PrivilegedSchemaVersion); None for
        # producers that predate it. Parsed strictly; it never enters the observation.
        self.privileged_layout = None
        if schema_response.get("privilegedLayout") is not None:
            from .labor_privileged import LaborPrivilegedLayout

            self.privileged_layout = LaborPrivilegedLayout.from_bridge(
                schema_response["privilegedLayout"]
            )
            if (
                schema_response.get("privilegedSchemaVersion")
                != self.privileged_layout.schema_version
                or schema_response.get("rowSchemaVersion") != 1
            ):
                raise ValueError("bridge privileged/row schema versions are inconsistent")
        self.decision_diagnostics: list[dict] = []
        self.raw_material_diagnostics: dict[str, dict] = {}
        self.diagnostic_schema_version = 0
        self.diagnostic_trajectory_id = None
        self._turn = None
        self._trajectory = None
        self._fork_diagnostics: dict[int, dict] = {}
        self._last_step = None

    def request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError("model9 bridge exited: " + "\n".join(self._stderr))
            assert self._process.stdin is not None and self._process.stdout is not None
            self._process.stdin.write(
                orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY) + b"\n"
            )
            self._process.stdin.flush()
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError("model9 bridge closed stdout: " + "\n".join(self._stderr))
            response = orjson.loads(line)
            if not response.get("ok", False):
                raise RuntimeError(f"model9 bridge error: {response.get('error')}")
            return response

    def reset(
        self,
        seed: int = 0,
        players: int = 8,
        movement: bool = False,
        withheld: int = 0,
        size: tuple[int, int, int] | None = None,
        snapshot: bytes | str | None = None,
        continuation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """withheld: training-only technology mask (1 << Technology) for capability fixtures; 0 is the game.
        size: optional (width, height, start separation) for scale probes; None keeps the Small training world.
        snapshot: a captured world (``capture``; raw bytes or base64) to start from instead of generating ``seed``'s
        opening. Per-village bridge context (active commitments, previous populations) is not part of a snapshot, so
        the first step after it reports no ``founded`` villages and every commitment starts free.
        continuation: collection resume only (``labor_collect_resume``), with ``snapshot``: {startTurn, trajectoryId}
        of the interrupted episode and optionally the live-only state from ``capture_transient`` (``transient``), so the successor of a replayed decision is byte-identical to the
        uninterrupted one. The reset step's own observation is not a faithful boundary."""
        from .protocol import seeds

        seeds([seed], "bridge reset")
        self._check_expected_producer()
        expected_world = getattr(self, "_expected_world_contract", None)
        if expected_world is not None:
            import hashlib

            from .provenance import ProvenanceError

            raw = (
                base64.b64decode(snapshot)
                if isinstance(snapshot, str)
                else bytes(snapshot)
                if snapshot is not None
                else None
            )
            actual_sha = hashlib.sha256(raw).hexdigest() if raw is not None else ""
            if (
                seed != expected_world["seed"]
                or withheld != expected_world["withheld"]
                or players != expected_world["players"]
                or movement != expected_world["movement"]
                or size is not None
                or actual_sha != expected_world["snapshot_sha256"]
            ):
                raise ProvenanceError(
                    "actual Reset arguments differ from immutable world/cohort contract"
                )
        payload: dict[str, Any] = {
            "op": "reset",
            "seed": seed,
            "players": players,
            "movement": movement,
            "withheld": withheld,
        }
        if getattr(self, "economic_diagnostics_enabled", False):
            if self.schema.economic_diagnostics_schema != 1:
                raise ValueError("bridge does not support economic diagnostics schema 1")
            payload["economicDiagnostics"] = True
        if snapshot is not None:
            payload["snapshot"] = (
                snapshot
                if isinstance(snapshot, str)
                else base64.b64encode(snapshot).decode("ascii")
            )
            checked = self.request({"op": "validate_snapshot", "snapshot": payload["snapshot"]})
            seeds([checked["sourceSeed"]], "snapshot root before reset")
            if checked["sourceSeed"] != seed:
                raise ValueError("snapshot root differs from declared seed before reset")
        if size is not None:
            payload.update(width=size[0], height=size[1], separation=size[2])
        if continuation is not None:
            if snapshot is None:
                raise ValueError("a continuation reset requires a snapshot")
            payload["continuation"] = {
                "startTurn": int(continuation["startTurn"]),
                "trajectoryId": str(continuation["trajectoryId"]),
            }
            if continuation.get("transient") is not None:
                payload["continuation"]["transient"] = continuation["transient"]
        response = self.request(payload)
        self.last_reset_identity = {
            k: response.get(k)
            for k in (
                "configurationHash",
                "fingerprint",
                "buildHash",
                "rawMaterialsEnabled",
                "sourceSeed",
            )
        }
        self._raw_materials_enabled = response.get("rawMaterialsEnabled")
        if response.get("sourceSeed") is not None:
            seeds([response["sourceSeed"]], "snapshot root")
            if response["sourceSeed"] != seed:
                raise ValueError("snapshot root differs from declared seed")
        step = response["step"]
        self._turn = int(step["turn"])
        self._trajectory = None
        self.decision_diagnostics = []
        self._fork_diagnostics.clear()
        self._last_step = step
        if any(v.get("decisionDiagnostics") is not None for v in step["villages"]):
            raise ValueError("initial reset must not carry preceding decision diagnostics")
        self._read_physical_diagnostics(step, initial=True)
        return step

    def capture(self) -> str:
        """The live world as a base64 snapshot (op capture), restartable with ``reset(snapshot=...)``."""
        self._check_expected_producer()
        return str(self.request({"op": "capture"})["snapshot"])

    def step(self, actions: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
        self._check_expected_producer()
        from .diagnostics import validate

        step = self.request({"op": "step", "actions": {str(k): v for k, v in actions.items()}})[
            "step"
        ]
        self.decision_diagnostics = validate(
            step,
            self._turn,
            actions,
            required=self.schema.decision_diagnostics_schema >= 1,
            trajectory=self._trajectory,
        )
        if self.decision_diagnostics:
            self._trajectory = self.decision_diagnostics[0]["trajectoryId"]
        self._read_physical_diagnostics(step)
        self._turn = int(step["turn"])
        self._last_step = step
        return step

    def fork(self) -> tuple[int, dict[str, Any]]:
        """A search branch point: its id and the decision step. Every restore replays from it exactly."""
        self._check_expected_producer()
        response = self.request({"op": "fork"})
        self._check_alignment(response["step"], self._last_step)
        self._fork_diagnostics[int(response["fork"])] = {
            "step": response["step"],
            "public": copy.deepcopy(self.decision_diagnostics),
        }
        # The returned sidecar belongs to the saved main path; the next action begins a branch trajectory.
        self._trajectory = None
        return int(response["fork"]), response["step"]

    def restore(self, fork: int) -> dict[str, Any]:
        self._check_expected_producer()
        step = self.request({"op": "restore", "fork": fork})["step"]
        self._check_alignment(step, self._fork_diagnostics[fork]["step"])
        self._turn = int(step["turn"])
        self._trajectory = None
        self._last_step = step
        # Public diagnostics refer to exactly the submitted action set, not implicit/native decisions at other camps.
        self.decision_diagnostics = copy.deepcopy(self._fork_diagnostics[fork]["public"])
        # Native-teacher actions can have physical receipts without any submitted-decision head sidecars.
        self._read_physical_diagnostics(step, initial=step.get("diagnosticSchemaVersion", 0) == 0)
        return step

    def _check_expected_producer(self):
        expected = getattr(self, "_expected_worker_contract", None)
        if expected is not None:
            from .provenance import ProvenanceError
            from .worker_contract import checked, observe

            if observe(self) != checked(expected):
                raise ProvenanceError(
                    "bound worker producer changed before Reset/Step/fork/restore"
                )

    def _read_physical_diagnostics(self, step: dict, *, initial: bool = False) -> None:
        from .diagnostics import validate_raw_materials

        self.raw_material_diagnostics = validate_raw_materials(
            step,
            int(step["turn"]) - 1,
            required=getattr(self.schema, "raw_material_diagnostics_schema", 0) >= 1,
            initial=initial,
            raw_enabled=getattr(self, "_raw_materials_enabled", None),
        )
        self.diagnostic_schema_version = int(step.get("diagnosticSchemaVersion", 0))
        self.diagnostic_trajectory_id = step.get("trajectoryId")
        from .economic_trace import validate_economic_step

        self.economic_diagnostics = validate_economic_step(
            step,
            required=getattr(self, "economic_diagnostics_enabled", False),
            initial=initial,
        )
        # The bridge's raw signed trade utility is defined by the simulator as
        # realised person-QoL-years.  Older diagnostic envelopes did not repeat
        # that unit in every row; enrich only the opt-in reporting sidecar and
        # reject a conflicting declaration rather than relabelling it.
        if getattr(self, "economic_diagnostics_enabled", False):
            from .gate1_diagnostics import complete_economic_units

            for row in self.economic_diagnostics.values():
                complete_economic_units(row)

    def _check_alignment(self, actual: dict, saved: dict) -> None:
        def sidecars(s):
            return {
                int(v["camp"]): v.get("decisionDiagnostics")
                for v in s["villages"]
                if not v.get("terminated")
            }

        if self.schema.decision_diagnostics_schema >= 1 and (
            int(actual["turn"]) != int(saved["turn"]) or sidecars(actual) != sidecars(saved)
        ):
            raise ValueError("fork/restore changed decision diagnostics alignment")
        if getattr(self.schema, "raw_material_diagnostics_schema", 0) >= 1 and any(
            actual.get(key) != saved.get(key)
            for key in ("diagnosticSchemaVersion", "trajectoryId", "rawMaterialDiagnostics")
        ):
            raise ValueError("fork/restore changed physical diagnostics alignment")
        if actual.get("economicDiagnostics", {}) != saved.get("economicDiagnostics", {}):
            raise ValueError("fork/restore changed economic diagnostics alignment")

    def drop(self, fork: int) -> None:
        self.request({"op": "drop", "fork": fork})
        self._fork_diagnostics.pop(fork, None)

    def verify_replay(self) -> int:
        self._check_expected_producer()
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

    def __enter__(self) -> Model9Bridge:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def privileged_batch(
    villages: Sequence[Mapping[str, Any]], layout: Any = None
) -> dict[str, np.ndarray]:
    """Critic-only batcher: the rows' ``privilegedView`` as raw per-family arrays.

    This is a separate dictionary (``privileged_<family>``, ``privileged_<family>_present``,
    ``privileged_schema_version``) and is never merged into :func:`observation_batch`'s
    actor batch. A row without the view or with another schema version is rejected.
    """
    from .labor_privileged import LABOR_PRIVILEGED_LAYOUT_V1, privileged_raw_batch

    return privileged_raw_batch(villages, LABOR_PRIVILEGED_LAYOUT_V1 if layout is None else layout)


def observation_batch(villages: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Stack bridge observations into the same arrays the NPY shards hold.

    Actor/observation inputs only: the critic-only ``privilegedView`` (and the legacy
    ``privileged`` vector) is deliberately excluded; see :func:`privileged_batch`."""
    obs = [v["observation"] for v in villages]
    batch: dict[str, np.ndarray] = {
        "core": np.asarray([o["core"] for o in obs], dtype=np.float32),
        "core_present": np.asarray([o["corePresent"] for o in obs], dtype=np.bool_),
        "movement_mask": np.asarray([o["movementMask"] for o in obs], dtype=np.bool_),
        "clothing_route_mask": np.asarray([o["clothingRouteMask"] for o in obs], dtype=np.bool_),
        "shelter_route_mask": np.asarray([o["shelterRouteMask"] for o in obs], dtype=np.bool_),
        "economy_mix_mask": np.asarray([o["economyMixMask"] for o in obs], dtype=np.bool_),
        "commit_hold_mask": np.asarray(
            [bool(o.get("commitHoldMask", False)) for o in obs], dtype=np.bool_
        ),
        # Contract 10: native, keep intent, region rows 0-7.
        "region_movement_mask": np.asarray(
            [
                o.get(
                    "regionMovementMask", [True] + [False] * (contracts.REGION_MOVEMENT_SLOTS - 1)
                )
                for o in obs
            ],
            dtype=np.bool_,
        ),
    }
    for index, (name, rows, width) in enumerate(contracts.SETS):
        batch[name] = np.asarray([o["sets"][index] for o in obs], dtype=np.float32).reshape(
            len(obs), rows, width
        )
        batch[name + "_present"] = np.asarray([o["rowPresent"][index] for o in obs], dtype=np.bool_)
    return batch
