"""Strict production adapters and inference comparator for Model9 Gate 1.

The historical contract-8 checkpoint and the actor-only behavioural-cloning
phase are real artifacts with different loading needs.  Neither is sent
through :mod:`upgrade`: contract 8 executes through the pinned implementation,
while the BC phase may borrow metadata and the unguarded normalizer only from
the recovered T3 initialization checkpoint after literal actor-state identity
has been proved. This composite makes no claim about the missing T2m guarded
clone or the original BC-time normalizer.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import platform
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .provenance import ProvenanceError, checkpoint_hash, digest


CONTRACT8_SOURCES = {"model.py", "contracts.py"}
CONTRACT8_OUTPUTS = (
    "strategic_mean",
    "strategic_log_std",
    "mix_logits",
    "mix_log_std",
    "service_mean",
    "service_log_std",
    "clothing",
    "shelter",
    "fission",
    "movement",
)
CONTRACT8_ACTIONS = (
    "strategic",
    "mix",
    "service",
    "clothing",
    "shelter",
    "fission",
    "movement",
)
GUARDED_BC_OUTPUTS = (
    "strategic_mean",
    "strategic_log_std",
    "mix_logits",
    "mix_log_std",
    "service_mean",
    "service_log_std",
    "clothing",
    "shelter",
    "fission",
    "movement",
    "span",
    "trade_value_mean",
    "trade_value_log_std",
    "trade_gate",
    "trade_share_mean",
    "trade_share_log_std",
)
GUARDED_BC_ACTIONS = (
    "strategic",
    "mix",
    "service",
    "clothing",
    "shelter",
    "fission",
    "movement",
    "span",
    "trade_value",
    "trade_gate",
    "trade_share",
)
INACTIVE_REGION_KEYS = {"region_flat", "region_group.weight", "region_group.bias"}


def _fail(message: str) -> None:
    raise ProvenanceError("Gate 1 production adapter: " + message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sealed_json(ref: dict[str, str], base: Path, *, kind: str) -> tuple[dict[str, Any], Path]:
    if (
        not isinstance(ref, dict)
        or set(ref) != {"path", "sha256"}
        or re.fullmatch(r"[0-9a-f]{64}", str(ref.get("sha256"))) is None
    ):
        _fail(f"{kind} reference is malformed")
    path = Path(ref["path"])
    if not path.is_absolute():
        path = base / path
    if checkpoint_hash(path) != ref["sha256"]:
        _fail(f"{kind} bytes do not match their reference")
    body = json.loads(path.read_text(encoding="utf-8"))
    if (
        body.get("schema") != 1
        or body.get("kind") != kind
        or body.get("digest")
        != digest({key: value for key, value in body.items() if key != "digest"})
    ):
        _fail(f"{kind} is not a canonical schema-1 sealed artifact")
    return body, path


def tensor_state_identity(state: dict[str, Any]) -> dict[str, Any]:
    """Return a literal-byte identity for an exact CPU tensor state mapping."""
    if not isinstance(state, dict) or not state or not all(isinstance(name, str) for name in state):
        _fail("tensor state must be a nonempty string-keyed mapping")
    stream = hashlib.sha256()
    rows = []
    for name in sorted(state):
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.layout != torch.strided
        ):
            _fail(f"tensor {name} is not a CPU strided tensor")
        contiguous = value.detach().contiguous()
        raw = contiguous.view(torch.uint8).reshape(-1).numpy().tobytes()
        row = {
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        encoded = _canonical(row)
        stream.update(len(encoded).to_bytes(8, "big"))
        stream.update(encoded)
        stream.update(len(raw).to_bytes(8, "big"))
        stream.update(raw)
        rows.append(row)
    return {"tensor_count": len(rows), "tensors": rows, "typed_content_sha256": stream.hexdigest()}


def inactive_region_state(core_width: int, *, experts: bool, token: bool) -> dict[str, Any]:
    """Construct and identify the only reviewed current-runtime appended state."""
    from .gate1_diagnostics import preserve_global_rng
    from .model import Actor

    with preserve_global_rng():
        expected = Actor(core_width, experts=experts, token=token, region_head="flat").state_dict()
    return {
        "values": {
            "region_flat": torch.tensor(True),
            "region_group.weight": torch.zeros_like(expected["region_group.weight"]),
            "region_group.bias": torch.zeros_like(expected["region_group.bias"]),
        },
        "identity": tensor_state_identity(
            {
                "region_flat": torch.tensor(True),
                "region_group.weight": torch.zeros_like(expected["region_group.weight"]),
                "region_group.bias": torch.zeros_like(expected["region_group.bias"]),
            }
        ),
    }


def _append_inactive_region(
    actor_state: dict[str, Any],
    core_width: int,
    *,
    experts: bool,
    token: bool,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    from .gate1_diagnostics import preserve_global_rng
    from .model import Actor

    with preserve_global_rng():
        expected = Actor(core_width, experts=experts, token=token, region_head="flat").state_dict()
    if set(actor_state) != set(expected) - INACTIVE_REGION_KEYS:
        _fail("recovered actor keys differ from the exact pre-region source schema")
    for name, value in actor_state.items():
        target = expected[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != target.dtype
            or value.shape != target.shape
            or value.layout != torch.strided
        ):
            _fail(f"recovered actor tensor {name} differs from its exact schema")
    appended = inactive_region_state(core_width, experts=experts, token=token)
    if appended["identity"] != expected_identity:
        _fail("reviewed inactive region state differs from the executing runtime")
    adapted = dict(actor_state)
    adapted.update(appended["values"])
    return adapted


def load_recovered_contract12(
    checkpoint: Path,
    loading: dict[str, Any],
    resolved_semantics: dict[str, Any],
    *,
    base: Path | None = None,
) -> dict[str, Any]:
    """Adapt an exact recovered contract-12 endpoint without using upgrade/hooks."""
    base = base or checkpoint.parent
    protocol, _ = _sealed_json(
        loading.get("protocol"),
        base,
        kind="gate1_recovered_contract12_adapter_protocol",
    )
    node_id = loading.get("node_id")
    allowed = protocol.get("allowed_checkpoints")
    checkpoint_sha = checkpoint_hash(checkpoint)
    if (
        not isinstance(node_id, str)
        or not isinstance(allowed, dict)
        or allowed.get(node_id) != checkpoint_sha
        or set(protocol.get("raw_missing_actor_keys", ())) != INACTIVE_REGION_KEYS
        or protocol.get("appended_region_scope") != "inactive-under-learned_movement=false"
        or protocol.get("historical_worker_contract_claim") is not False
        or protocol.get("current_upgrade_allowed") is not False
        or protocol.get("implicit_defaults") != []
    ):
        _fail("recovered contract-12 adapter is not bound to this exact node/checkpoint")
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or raw.get("contract") != 12:
        _fail("recovered contract-12 adapter input has another raw contract")
    core_width = raw.get("core_width")
    if type(core_width) is not int or core_width <= 0:
        _fail("recovered contract-12 checkpoint lacks its core width")
    experts = resolved_semantics.get("experts")
    token = resolved_semantics.get("token")
    if type(experts) is not bool or type(token) is not bool:
        _fail("recovered contract-12 adapter lacks reviewed actor semantics")
    identity_key = f"experts-{str(experts).lower()}-token-{str(token).lower()}"
    identities = protocol.get("appended_inactive_region_state")
    if not isinstance(identities, dict) or identity_key not in identities:
        _fail("recovered contract-12 adapter lacks this actor variant")
    adapted = dict(raw)
    adapted["actor"] = _append_inactive_region(
        raw.get("actor"),
        core_width,
        experts=experts,
        token=token,
        expected_identity=identities[identity_key],
    )
    return adapted


def assemble_bc_phase_state(
    bc_state: dict[str, Any], support_state: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    """Assemble a *current diagnostic composite* around the recovered BC actor.

    The historical ``.bc.pt`` is actor-only.  Its original fitted-normalizer
    bytes cannot be recovered from that artifact and the teacher dataset needed
    to reproduce them is unavailable.  The reviewed protocol therefore binds
    the recovered unguarded T3 initialization checkpoint only as the diagnostic
    metadata/normalizer carrier;
    it does not assert that those bytes were the BC-time normalizer.  The actor
    is always taken from the BC artifact and must be literally identical to the
    support actor.
    """
    if set(bc_state) != {"actor", "log"} or not isinstance(bc_state.get("log"), list):
        _fail("BC phase checkpoint differs from the recovered actor/log schema")
    required_support = {
        "actor",
        "contract",
        "core_width",
        "critic",
        "normalizer",
        "privileged_width",
        "seed",
        "token",
    }
    if not required_support <= set(support_state):
        _fail("T3 initialization support checkpoint lacks required metadata/normalizer fields")
    if support_state.get("contract") != 12 or support_state.get("token") is not True:
        _fail("BC support is not the recovered contract-12 token checkpoint")
    normalizer = support_state.get("normalizer")
    if (
        not isinstance(normalizer, dict)
        or "guard" in normalizer
        or any(name.endswith("_dead") for name in normalizer)
    ):
        _fail("BC support is not the proved unguarded T3 initialization normalizer")
    bc_identity = tensor_state_identity(bc_state["actor"])
    support_identity = tensor_state_identity(support_state["actor"])
    if bc_identity != support_identity:
        _fail("BC and T3 initialization actor tensors are not literally identical")
    if (
        protocol.get("bc_actor_identity") != bc_identity
        or protocol.get("support_actor_identity") != support_identity
        or protocol.get("equivalence_scope") != "actor-tensors-only"
        or protocol.get("behavioral_equivalence_claim") is not False
        or protocol.get("historical_bc_normalizer_claim") is not False
        or protocol.get("support_normalizer_semantics") != "unguarded"
        or protocol.get("missing_guarded_clone_claim") is not False
        or protocol.get("composite_scope") != "current-diagnostic-only"
        or protocol.get("appended_region_scope") != "inactive-under-learned_movement=false"
        or protocol.get("implicit_defaults") != []
    ):
        _fail("BC adapter protocol does not bind the exact actor-only relation")
    assembled = dict(support_state)
    assembled["actor"] = _append_inactive_region(
        bc_state["actor"],
        int(support_state["core_width"]),
        experts=False,
        token=True,
        expected_identity=protocol.get("appended_inactive_region_state"),
    )
    return assembled


def load_bc_phase(
    checkpoint: Path, loading: dict[str, Any], *, base: Path | None = None
) -> dict[str, Any]:
    """Read and assemble the reviewed BC phase, returning a native raw state."""
    base = base or checkpoint.parent
    protocol, _ = _sealed_json(
        loading.get("protocol"), base, kind="gate1_bc_phase_adapter_protocol"
    )
    support_ref = loading.get("support_checkpoint")
    if not isinstance(support_ref, dict) or set(support_ref) != {"path", "sha256"}:
        _fail("BC adapter lacks its exact T3 initialization support checkpoint")
    support = Path(support_ref["path"])
    if not support.is_absolute():
        support = base / support
    if (
        checkpoint_hash(checkpoint) != protocol.get("bc_checkpoint_sha256")
        or checkpoint_hash(support) != support_ref["sha256"]
        or support_ref["sha256"] != protocol.get("support_checkpoint_sha256")
    ):
        _fail("BC adapter checkpoint identities differ from the reviewed protocol")
    bc_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    support_state = torch.load(support, map_location="cpu", weights_only=True)
    if not isinstance(bc_state, dict) or not isinstance(support_state, dict):
        _fail("BC adapter inputs are not weights-only mappings")
    return assemble_bc_phase_state(bc_state, support_state, protocol)


def _historical_module(source_root: Path, package_name: str):
    package = types.ModuleType(package_name)
    package.__path__ = [str(source_root)]
    sys.modules[package_name] = package
    loaded = []
    try:
        for leaf in ("contracts", "model"):
            name = f"{package_name}.{leaf}"
            spec = importlib.util.spec_from_file_location(name, source_root / f"{leaf}.py")
            if spec is None or spec.loader is None:
                _fail(f"cannot load frozen historical {leaf}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            loaded.append(name)
            spec.loader.exec_module(module)
        return sys.modules[f"{package_name}.model"]
    except Exception:
        for name in reversed(loaded):
            sys.modules.pop(name, None)
        sys.modules.pop(package_name, None)
        raise


def _guarded_normalizer(state: dict[str, Any], core_width: int):
    from . import contracts
    from .model import Normalizer

    widths = {"core": core_width, **{name: width for name, _, width in contracts.SETS}}
    privileged_keys = {"privileged_mean", "privileged_std", "privileged_dead"}
    present_privileged = privileged_keys & set(state)
    if present_privileged:
        if present_privileged != privileged_keys:
            _fail("guarded BC privileged normalizer fields are partial")
        widths["privileged"] = len(state["privileged_mean"])
    expected = {f"{name}_{suffix}" for name in widths for suffix in ("mean", "std", "dead")} | {
        "guard"
    }
    if (
        set(state) != expected
        or type(state.get("guard")) is not float
        or not np.isfinite(state["guard"])
        or state["guard"] <= 0
    ):
        _fail("guarded BC normalizer schema/guard is invalid")
    for name, width in widths.items():
        for suffix in ("mean", "std"):
            raw = state[f"{name}_{suffix}"]
            if (
                not isinstance(raw, list)
                or len(raw) != width
                or any(
                    type(value) not in (int, float)
                    or isinstance(value, bool)
                    or not np.isfinite(value)
                    or float(np.float32(value)) != value
                    for value in raw
                )
                or suffix == "std"
                and any(value <= 0 for value in raw)
            ):
                _fail(f"guarded BC normalizer {name}_{suffix} is invalid")
        dead = state[f"{name}_dead"]
        if (
            not isinstance(dead, list)
            or len(dead) != width
            or any(type(value) is not bool for value in dead)
        ):
            _fail(f"guarded BC normalizer {name}_dead is invalid")
    return Normalizer.from_state(state)


def _guarded_bc_actor(state: dict[str, Any], protocol: dict[str, Any]):
    """Construct the explicit current-runtime adapter for one guarded BC actor."""
    from .model import Actor

    if set(state) != {"actor", "log", "normalizer", "bc_config"}:
        _fail("guarded BC checkpoint differs from the actor/log/normalizer/bc_config schema")
    if not isinstance(state["log"], list) or len(state["log"]) != 12:
        _fail("guarded BC checkpoint does not contain exactly 12 BC epochs")
    if state["bc_config"] != protocol.get("bc_config"):
        _fail("guarded BC configuration differs from the reviewed protocol")
    core_width = protocol.get("core_width")
    if type(core_width) is not int or core_width <= 0:
        _fail("guarded BC core width is invalid")
    actor = Actor(core_width, token=True, region_head="flat")
    expected = actor.state_dict()
    appended = {"region_flat", "region_group.weight", "region_group.bias"}
    if set(state["actor"]) != set(expected) - appended:
        _fail("guarded BC actor keys differ from the pinned source schema")
    for name, value in state["actor"].items():
        target = expected[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != target.dtype
            or value.shape != target.shape
        ):
            _fail(f"guarded BC actor tensor {name} differs from its exact schema")
    adapted = dict(state["actor"])
    adapted["region_flat"] = torch.tensor(True)
    adapted["region_group.weight"] = torch.zeros_like(expected["region_group.weight"])
    adapted["region_group.bias"] = torch.zeros_like(expected["region_group.bias"])
    append_identity = tensor_state_identity({name: adapted[name] for name in sorted(appended)})
    if (
        protocol.get("appended_inactive_region_state") != append_identity
        or protocol.get("appended_region_scope") != "inactive-under-learned_movement=false"
    ):
        _fail("guarded BC adapter does not bind its appended inactive region state")
    actor.load_state_dict(adapted, strict=True)
    actor.eval()
    actor.commitments = False
    actor.region_movement = False
    actor.learned_fission = False
    actor.learned_trade = False
    actor.grouped_region = False
    actor.region_selection = None
    return actor, _guarded_normalizer(state["normalizer"], core_width)


def load_guarded_bc(checkpoint: Path, loading: dict[str, Any], *, base: Path | None = None):
    """Load the recovered guarded actor-only BC checkpoint without a fake critic."""
    base = base or checkpoint.parent
    protocol, protocol_path = _sealed_json(
        loading.get("protocol"),
        base,
        kind="gate1_guarded_bc_adapter_protocol",
    )
    if checkpoint_hash(checkpoint) != protocol.get("checkpoint_sha256"):
        _fail("guarded BC checkpoint differs from the reviewed protocol")
    source_root = Path(protocol.get("historical_source_root", ""))
    if source_root.is_absolute() or source_root.as_posix() != "guarded-historical-source":
        _fail("guarded BC pinned source root is not package-local")
    source_path = protocol_path.parent / source_root
    sources = protocol.get("historical_sources")
    if (
        not isinstance(sources, dict)
        or set(sources) != CONTRACT8_SOURCES
        or not source_path.is_dir()
        or source_path.is_symlink()
        or any(checkpoint_hash(source_path / name) != sha for name, sha in sources.items())
    ):
        _fail("guarded BC pinned source inventory changed")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        _fail("guarded BC checkpoint is not a weights-only mapping")
    return _guarded_bc_actor(state, protocol)


def _normalizer_affine_identity(state: dict[str, Any]) -> dict[str, Any]:
    """Identify the exact affine fields without collapsing JSON scalar types."""
    if not isinstance(state, dict):
        _fail("normalizer state is not a mapping")
    affine = {
        name: value
        for name, value in state.items()
        if name != "guard" and not name.endswith("_dead")
    }
    if not affine or any(not (name.endswith("_mean") or name.endswith("_std")) for name in affine):
        _fail("normalizer affine field inventory is malformed")
    encoded = _canonical(affine)
    return {"fields": sorted(affine), "sha256": hashlib.sha256(encoded).hexdigest()}


def _guard_dead_identity(state: dict[str, Any]) -> dict[str, Any]:
    guard = state.get("guard")
    dead = {name: value for name, value in state.items() if name.endswith("_dead")}
    if type(guard) is not float or not dead:
        _fail("guarded normalizer lacks its exact guard/dead surface")
    encoded = _canonical({"guard": guard, "dead": dead})
    return {"guard": guard, "fields": sorted(dead), "sha256": hashlib.sha256(encoded).hexdigest()}


def guard_only_protocol_fields(
    base_state: dict[str, Any], guarded_state: dict[str, Any]
) -> dict[str, Any]:
    """Return exact identities needed to preregister the guard-only composition."""
    base_normalizer = base_state.get("normalizer")
    guarded_normalizer = guarded_state.get("normalizer")
    if (
        not isinstance(base_normalizer, dict)
        or "guard" in base_normalizer
        or any(name.endswith("_dead") for name in base_normalizer)
    ):
        _fail("guard-only base is not the proved unguarded T3 initialization")
    core_width = base_state.get("core_width")
    if type(core_width) is not int or core_width <= 0:
        _fail("guard-only base lacks its exact core width")
    _guarded_normalizer(guarded_normalizer, core_width)
    base_affine = _normalizer_affine_identity(base_normalizer)
    guarded_affine = _normalizer_affine_identity(guarded_normalizer)
    if base_affine != guarded_affine or {
        name: base_normalizer[name] for name in base_affine["fields"]
    } != {name: guarded_normalizer[name] for name in guarded_affine["fields"]}:
        _fail("T3 initialization and guarded seed-6 affine normalizers are not literally identical")
    appended = inactive_region_state(core_width, experts=False, token=True)
    evaluated_actor = dict(base_state.get("actor"))
    evaluated_actor.update(appended["values"])
    return {
        "base_actor_identity": tensor_state_identity(base_state.get("actor")),
        "evaluated_actor_identity": tensor_state_identity(evaluated_actor),
        "guard_source_actor_identity": tensor_state_identity(guarded_state.get("actor")),
        "affine_identity": base_affine,
        "guard_dead_identity": _guard_dead_identity(guarded_normalizer),
        "appended_inactive_region_state": appended["identity"],
        "core_width": core_width,
    }


def assemble_guard_only_state(
    base_state: dict[str, Any], guarded_state: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    """Compose the reviewed guard/dead intervention without rewriting an artifact.

    Actor tensors and all affine normalizer moments come from the unguarded T3
    initialization checkpoint.  Only ``guard`` and ``*_dead`` are copied from
    the independently recovered T2m seed-6 BC normalizer.  This is a prospective
    diagnostic intervention, never a historical checkpoint or ancestry claim.
    """
    identities = guard_only_protocol_fields(base_state, guarded_state)
    required = {
        "base_actor_identity",
        "evaluated_actor_identity",
        "guard_source_actor_identity",
        "affine_identity",
        "guard_dead_identity",
        "appended_inactive_region_state",
        "core_width",
    }
    if (
        any(protocol.get(name) != identities[name] for name in required)
        or protocol.get("composition_scope") != "prospective-current-diagnostic-only"
        or protocol.get("normalizer_change_only") is not True
        or protocol.get("historical_artifact_claim") is not False
        or protocol.get("historical_ancestry_claim") is not False
        or protocol.get("outcome_selection_used") is not False
        or protocol.get("implicit_defaults") != []
    ):
        _fail("guard-only intervention protocol does not bind the exact composition")
    base_normalizer = base_state["normalizer"]
    guard_normalizer = guarded_state["normalizer"]
    composed_normalizer = dict(base_normalizer)
    composed_normalizer["guard"] = guard_normalizer["guard"]
    for name in identities["guard_dead_identity"]["fields"]:
        composed_normalizer[name] = guard_normalizer[name]
    assembled = dict(base_state)
    assembled["actor"] = _append_inactive_region(
        base_state["actor"],
        identities["core_width"],
        experts=False,
        token=True,
        expected_identity=identities["appended_inactive_region_state"],
    )
    assembled["normalizer"] = composed_normalizer
    if tensor_state_identity(assembled["actor"]) != identities["evaluated_actor_identity"]:
        _fail("guard-only composition changed the T3 initialization actor")
    return assembled


def load_guard_only_intervention(
    checkpoint: Path, loading: dict[str, Any], *, base: Path | None = None
) -> dict[str, Any]:
    """Load the sealed guard-only composition as a native strict raw state."""
    base = base or checkpoint.parent
    protocol, _ = _sealed_json(
        loading.get("protocol"),
        base,
        kind="gate1_guard_only_intervention_protocol",
    )
    guard_ref = loading.get("guard_source_checkpoint")
    if not isinstance(guard_ref, dict) or set(guard_ref) != {"path", "sha256"}:
        _fail("guard-only intervention lacks its exact guard-source checkpoint")
    guard_path = Path(guard_ref["path"])
    if not guard_path.is_absolute():
        guard_path = base / guard_path
    if (
        checkpoint_hash(checkpoint) != protocol.get("base_checkpoint_sha256")
        or checkpoint_hash(guard_path) != guard_ref["sha256"]
        or guard_ref["sha256"] != protocol.get("guard_source_checkpoint_sha256")
    ):
        _fail("guard-only intervention checkpoint identities differ from the reviewed protocol")
    base_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    guarded_state = torch.load(guard_path, map_location="cpu", weights_only=True)
    if not isinstance(base_state, dict) or not isinstance(guarded_state, dict):
        _fail("guard-only intervention inputs are not weights-only mappings")
    return assemble_guard_only_state(base_state, guarded_state, protocol)


def _load_historical_contract8(source_root: Path):
    """Load the frozen historical model/contract blobs under an isolated name."""
    return _historical_module(source_root, "_gate1_historical_contract8")


def _batch(case: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays = {}
    for name, row in case["inputs"].items():
        dtype = row.get("dtype")
        if dtype not in {"float32", "bool"}:
            _fail(f"legacy corpus input {name} has unsupported dtype")
        values = np.asarray(row.get("values"), dtype=np.float32 if dtype == "float32" else bool)
        if list(values.shape) != row.get("shape"):
            _fail(f"legacy corpus input {name} differs from its exact shape")
        arrays[name] = values
    return arrays


def comparator_runtime_identity() -> dict[str, Any]:
    """Return the exact runtime surface that executes the frozen comparator."""
    executable = Path(sys.executable).resolve()
    adapter = Path(__file__).resolve().with_name("legacy_contract8.py")
    production_adapter = Path(__file__).resolve()
    return {
        "python": {
            "path": str(executable),
            "sha256": checkpoint_hash(executable),
            "version": sys.version,
        },
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
        },
        "adapter_source": {
            "repository_path": "src/anthropocene_rl/autonomous_v7/legacy_contract8.py",
            "sha256": checkpoint_hash(adapter),
        },
        "production_adapter_source": {
            "repository_path": "src/anthropocene_rl/autonomous_v7/gate1_production_adapter.py",
            "sha256": checkpoint_hash(production_adapter),
        },
    }


def _validate_contract8_corpus(
    corpus: dict[str, Any], *, core_width: int = 207
) -> list[dict[str, Any]]:
    from .legacy_contract8 import ECONOMY_MIX, SETS

    cases = corpus.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) < 4
        or len({row.get("id") for row in cases if isinstance(row, dict)}) != len(cases)
    ):
        _fail("legacy corpus must have at least four uniquely named cases")
    required = {
        "core",
        "core_present",
        "movement_mask",
        "clothing_route_mask",
        "shelter_route_mask",
        "economy_mix_mask",
    }
    for name, _, _ in SETS:
        required.update((name, name + "_present"))
    batch_sizes = []
    masks: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "movement_mask",
            "clothing_route_mask",
            "shelter_route_mask",
            "economy_mix_mask",
        )
    }
    for row in cases:
        if (
            not isinstance(row, dict)
            or set(row) != {"id", "inputs"}
            or not isinstance(row["id"], str)
            or not row["id"]
            or not isinstance(row["inputs"], dict)
            or set(row["inputs"]) != required
        ):
            _fail("legacy corpus case has an incomplete input/mask surface")
        arrays = _batch(row)
        batch = arrays["core"].shape[0]
        if batch <= 0 or arrays["core"].shape != (batch, core_width):
            _fail("legacy corpus core batch has the wrong shape")
        expected_shapes = {
            "core_present": (batch, core_width),
            "movement_mask": (batch, 8),
            "clothing_route_mask": (batch, 2),
            "shelter_route_mask": (batch, 2),
            "economy_mix_mask": (batch, ECONOMY_MIX),
        }
        expected_shapes.update({name: (batch, count, width) for name, count, width in SETS})
        expected_shapes.update({name + "_present": (batch, count) for name, count, _ in SETS})
        if any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
            _fail("legacy corpus input differs from the contract-8 shape surface")
        batch_sizes.append(batch)
        for name in masks:
            masks[name].append(arrays[name])
    if not any(size > 1 for size in batch_sizes):
        _fail("legacy corpus omits batched inference")
    if any(
        not any(not values.any() for values in rows) or not any(values.all() for values in rows)
        for rows in masks.values()
    ):
        _fail("legacy corpus must cover empty and full masks for every decoder mask")
    return cases


def validate_contract8_corpus(
    corpus: dict[str, Any], *, core_width: int = 207
) -> list[dict[str, Any]]:
    """Validate and return complete reviewed inference cases at an exact width."""
    return _validate_contract8_corpus(corpus, core_width=core_width)


def _surface(
    module: Any,
    actor: Any,
    normalizer: Any,
    cases: list[dict[str, Any]],
    mapped_outputs: tuple[str, ...],
    decoded_actions: tuple[str, ...],
) -> dict[str, Any]:
    rows = []
    for case in cases:
        tensors = module.to_tensors(_batch(case), normalizer)
        with torch.no_grad():
            output = actor(tensors)
            action = module.act_deterministic(output)
        encoded = {}
        if set(output) != set(CONTRACT8_OUTPUTS) or not set(mapped_outputs) <= set(output):
            _fail("contract-8 inference output differs from the registered head surface")
        if not set(decoded_actions) <= set(action):
            _fail("contract-8 decoder omits a registered action")
        for name in mapped_outputs:
            value = output[name]
            contiguous = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
            encoded[name] = {
                "dtype": "float32",
                "shape": list(contiguous.shape),
                "values": contiguous.tolist(),
            }
        rows.append(
            {
                "case_id": case["id"],
                "outputs": encoded,
                "decoded_action": {
                    name: np.asarray(action[name]).tolist() for name in decoded_actions
                },
            }
        )
    return {"cases": rows}


def _adapter_surface(
    state: dict[str, Any],
    cases: list[dict[str, Any]],
    mapped_outputs: tuple[str, ...],
    decoded_actions: tuple[str, ...],
) -> dict[str, Any]:
    from . import legacy_contract8

    actor, normalizer = legacy_contract8.load(state)
    # Give the pinned adapter the historical function name expected by _surface.
    module = types.SimpleNamespace(
        to_tensors=legacy_contract8.to_tensors,
        act_deterministic=legacy_contract8.deterministic_mode,
    )
    return _surface(module, actor, normalizer, cases, mapped_outputs, decoded_actions)


def run_contract8_comparator(
    checkpoint: Path, corpus_ref: dict[str, str], protocol_ref: dict[str, str], *, base: Path
) -> dict[str, Any]:
    """Execute the reviewed historical-source versus pinned-adapter corpus.

    This is intentionally separate from checkpoint evaluation.  It constructs
    only the actor and normalizer and never starts the bridge.  Production use
    is gated by the independently reviewed freeze candidate; unit tests use
    synthetic fixtures.
    """
    protocol, _ = _sealed_json(protocol_ref, base, kind="gate1_contract8_comparator_protocol")
    corpus, _ = _sealed_json(corpus_ref, base, kind="gate1_contract8_corpus")
    if checkpoint_hash(checkpoint) != protocol.get("checkpoint_sha256") or corpus_ref[
        "sha256"
    ] != protocol.get("corpus_sha256"):
        _fail("contract-8 comparator inputs differ from the reviewed protocol")
    if protocol.get("runtime") != comparator_runtime_identity():
        _fail("contract-8 comparator runtime/adapter source differs from the reviewed protocol")
    mapped_outputs = tuple(protocol.get("mapped_outputs", ()))
    decoded_actions = tuple(protocol.get("decoded_actions", ()))
    if mapped_outputs != CONTRACT8_OUTPUTS or decoded_actions != CONTRACT8_ACTIONS:
        _fail("contract-8 comparator output/action registry is incomplete or reordered")
    cases = _validate_contract8_corpus(corpus)
    source_root = Path(protocol["historical_source_root"])
    if source_root.is_absolute() or source_root.as_posix() != "historical-source":
        _fail("historical source root is not the frozen package-local directory")
    source_root = base / source_root
    sources = protocol.get("historical_sources")
    if not isinstance(sources, dict) or set(sources) != CONTRACT8_SOURCES:
        _fail("frozen historical source inventory must contain model.py and contracts.py exactly")
    if (
        not source_root.is_dir()
        or source_root.is_symlink()
        or {path.name for path in source_root.iterdir() if path.is_file()} != CONTRACT8_SOURCES
    ):
        _fail("frozen historical source directory is incomplete or contains substituted files")
    for name, expected in sources.items():
        if checkpoint_hash(source_root / name) != expected:
            _fail("frozen historical source bytes differ from comparator protocol")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        _fail("contract-8 checkpoint is not a weights-only mapping")
    from .gate1_diagnostics import preserve_global_rng

    with preserve_global_rng():
        historical = _load_historical_contract8(source_root)
        reference_actor = historical.Actor(int(state["core_width"]))
        if set(reference_actor.state_dict()) != set(state["actor"]):
            _fail("historical source actor schema differs from checkpoint")
        reference_actor.load_state_dict(state["actor"], strict=True)
        reference_actor.eval()
        reference_normalizer = historical.Normalizer.from_state(state["normalizer"])
        source = _surface(
            historical,
            reference_actor,
            reference_normalizer,
            cases,
            mapped_outputs,
            decoded_actions,
        )
        adapter = _adapter_surface(state, cases, mapped_outputs, decoded_actions)
    if source != adapter:
        _fail("pinned contract-8 adapter differs from historical source on the frozen corpus")
    result = {
        "schema": 1,
        "kind": "gate1_contract8_comparator_result",
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "protocol_sha256": protocol_ref["sha256"],
        "corpus_sha256": corpus_ref["sha256"],
        "case_count": len(cases),
        "mapped_outputs": protocol["mapped_outputs"],
        "status": "PASS",
        "claim_scope": "frozen-corpus-only",
    }
    result["digest"] = digest(result)
    return result


def _guarded_surface(
    module: Any, actor: Any, normalizer: Any, cases: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = []
    for case in cases:
        tensors = module.to_tensors(_batch(case), normalizer)
        with torch.no_grad():
            output = actor(tensors)
            action = module.act_deterministic(output)
        if not set(GUARDED_BC_OUTPUTS) <= set(output) or not set(GUARDED_BC_ACTIONS) <= set(action):
            _fail("guarded BC inference omits a registered mapped output/action")
        encoded = {}
        for name in GUARDED_BC_OUTPUTS:
            contiguous = output[name].detach().to(device="cpu", dtype=torch.float32).contiguous()
            encoded[name] = {
                "dtype": "float32",
                "shape": list(contiguous.shape),
                "values": contiguous.tolist(),
            }
        rows.append(
            {
                "case_id": case["id"],
                "outputs": encoded,
                "decoded_action": {
                    name: np.asarray(action[name]).tolist() for name in GUARDED_BC_ACTIONS
                },
            }
        )
    return {"cases": rows}


def run_guarded_bc_comparator(
    checkpoint: Path, corpus_ref: dict[str, str], protocol_ref: dict[str, str], *, base: Path
) -> dict[str, Any]:
    """Prove the explicit current adapter against the recovered T2m source."""
    protocol, _ = _sealed_json(protocol_ref, base, kind="gate1_guarded_bc_adapter_protocol")
    corpus, _ = _sealed_json(corpus_ref, base, kind="gate1_guarded_bc_corpus")
    if (
        checkpoint_hash(checkpoint) != protocol.get("checkpoint_sha256")
        or corpus_ref["sha256"] != protocol.get("corpus_sha256")
        or protocol.get("runtime") != comparator_runtime_identity()
        or tuple(protocol.get("mapped_outputs", ())) != GUARDED_BC_OUTPUTS
        or tuple(protocol.get("decoded_actions", ())) != GUARDED_BC_ACTIONS
    ):
        _fail("guarded BC comparator inputs/runtime/surface differ from the reviewed protocol")
    core_width = protocol.get("core_width")
    cases = _validate_contract8_corpus(corpus, core_width=core_width)
    source_root = Path(protocol.get("historical_source_root", ""))
    if source_root.is_absolute() or source_root.as_posix() != "guarded-historical-source":
        _fail("guarded historical source root is not package-local")
    source_root = base / source_root
    sources = protocol.get("historical_sources")
    if (
        not isinstance(sources, dict)
        or set(sources) != CONTRACT8_SOURCES
        or not source_root.is_dir()
        or source_root.is_symlink()
        or {path.name for path in source_root.iterdir() if path.is_file()} != CONTRACT8_SOURCES
        or any(checkpoint_hash(source_root / name) != sha for name, sha in sources.items())
    ):
        _fail("guarded historical source inventory is incomplete or changed")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        _fail("guarded BC checkpoint is not a weights-only mapping")
    from .gate1_diagnostics import preserve_global_rng

    with preserve_global_rng():
        historical = _historical_module(source_root, "_gate1_historical_guarded_bc")
        source_actor = historical.Actor(core_width, token=True)
        if set(source_actor.state_dict()) != set(state["actor"]):
            _fail("guarded historical actor schema differs from checkpoint")
        source_actor.load_state_dict(state["actor"], strict=True)
        source_actor.eval()
        source_normalizer = historical.Normalizer.from_state(state["normalizer"])
        source = _guarded_surface(historical, source_actor, source_normalizer, cases)
        adapter_actor, adapter_normalizer = _guarded_bc_actor(state, protocol)
        from . import model as current

        adapter = _guarded_surface(current, adapter_actor, adapter_normalizer, cases)
    if source != adapter:
        _fail("guarded BC current adapter differs from pinned source on the frozen corpus")
    result = {
        "schema": 1,
        "kind": "gate1_guarded_bc_comparator_result",
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "protocol_sha256": protocol_ref["sha256"],
        "corpus_sha256": corpus_ref["sha256"],
        "case_count": len(cases),
        "mapped_outputs": list(GUARDED_BC_OUTPUTS),
        "status": "PASS",
        "claim_scope": "frozen-corpus mapped active outputs/actions only; not T3 ancestry",
    }
    result["digest"] = digest(result)
    return result
