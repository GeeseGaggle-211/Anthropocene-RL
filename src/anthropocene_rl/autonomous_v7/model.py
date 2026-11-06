"""model9: typed entity-set encoder, service-level actor heads and a factored two-hot critic.

Actor and critic do not share parameters (the critic may later receive privileged inputs that must never reach
the actor). Inputs are normalized as symlog(x) followed by a fixed TRAIN-fitted affine; missing rows are excluded
from pooling, so missing and zero never look alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import numpy as np
import torch
from torch import nn

from . import contracts

LOGIT_EPS = 1e-3
MASKED = -30.0  # finite logit for actions the native mask rules out
BINS = 255
BIN_LOW, BIN_HIGH = -2.0, 9.0  # symlog(u): u in roughly [-6, 8100] welfare-years per resident

# New-run-only normalization contract (IMPLEMENTATION_CONTRACT_RETRY_V1.md, Stage 1). Historical checkpoints do
# not carry this identifier and continue through the original symlog/affine path below. In particular, the presence
# of the older ``guard`` field does not imply this contract.
BOUNDED_FEATURE_CONTRACT = "model9-bounded-physical-history-v1"
LEGACY_FEATURE_CONTRACT = "legacy-implicit-symlog-affine"
BOUNDED_NORMALIZATION_SCHEMA = 1
BOUNDED_GUARD = 10.0
BOUNDED_STD_FLOOR = 0.1
MIN_SUPPORT_ROWS = 256
MIN_SUPPORT_RANGE = 1e-3


@dataclass(frozen=True)
class PhysicalFeature:
    index: int
    name: str
    transform: str
    reference: float | None = None


# Core 40:55 are the previous-decision wire block. Core 144 is the current movement-enabled flag. Fractions are
# already in physical [0, 1] units; counts and durations use fixed declared scales rather than a fitted near-zero
# variance. Names describe wire fields, not a causal interpretation of compiler corrections.
PHYSICAL_CORE_FEATURES = (
    PhysicalFeature(40, "previous-decision-present", "bool"),
    PhysicalFeature(41, "requested-field-retention", "fraction"),
    PhysicalFeature(42, "accepted-field-retention", "fraction"),
    PhysicalFeature(43, "executed-field-retention", "fraction"),
    PhysicalFeature(44, "requested-herd-retention", "fraction"),
    PhysicalFeature(45, "accepted-herd-retention", "fraction"),
    PhysicalFeature(46, "executed-herd-retention", "fraction"),
    PhysicalFeature(47, "requested-development-budget", "fraction"),
    PhysicalFeature(48, "accepted-development-budget", "fraction"),
    PhysicalFeature(49, "executed-development-budget", "fraction"),
    PhysicalFeature(50, "requested-field-share", "fraction"),
    PhysicalFeature(51, "accepted-field-share", "fraction"),
    PhysicalFeature(52, "executed-field-share", "fraction"),
    PhysicalFeature(53, "compiler-intervention-count", "nonnegative-count", 32.0),
    PhysicalFeature(54, "elapsed-years", "nonnegative-duration", 20.0),
    PhysicalFeature(55, "movement-accepted", "bool"),
    PhysicalFeature(144, "movement-enabled", "bool"),
)
HISTORY_FEATURES = tuple(feature for feature in PHYSICAL_CORE_FEATURES if 40 <= feature.index <= 55)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def normalization_corpus_sha256(batch: dict[str, np.ndarray]) -> str:
    """Content identity of the actual arrays used to fit/support the normalizer.

    This hashes typed shapes and bytes instead of accepting a caller's asserted corpus digest. Object arrays are
    rejected because their in-memory pointer representation is not a portable evidence format.
    """
    digest = hashlib.sha256(b"model9-normalization-corpus-v1\0")
    for key in sorted(batch):
        value = batch[key]
        if not isinstance(value, np.ndarray):
            raise ValueError(f"normalization corpus {key!r} is not a numpy array")
        if value.dtype.hasobject:
            raise ValueError(f"normalization corpus {key!r} has unsupported object dtype")
        array = np.ascontiguousarray(value)
        descriptor = {"key": key, "dtype": array.dtype.str, "shape": list(array.shape)}
        encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        raw = memoryview(array).cast("B")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _physical_values(
    values: np.ndarray, present: np.ndarray, feature: PhysicalFeature
) -> np.ndarray:
    """Validate and transform one physical feature without coercing invalid present values."""
    source = np.asarray(values)
    mask = np.asarray(present, dtype=bool)
    if source.shape != mask.shape:
        raise ValueError(f"{feature.name} value/presence shape mismatch")
    selected = source[mask]
    if not np.issubdtype(source.dtype, np.number) or np.issubdtype(source.dtype, np.bool_):
        raise ValueError(f"{feature.name} must use a numeric, non-boolean array")
    if selected.size and not np.isfinite(selected).all():
        raise ValueError(f"{feature.name} contains a non-finite present value")
    # Zero missing positions before any arithmetic: IEEE NaN * 0 is still NaN and must not poison fitting/inference.
    numeric = np.where(mask, source, 0).astype(np.float64, copy=False)
    if feature.transform == "bool":
        if selected.size and not np.isin(selected, (0, 1)).all():
            raise ValueError(f"{feature.name} must be exactly 0 or 1")
        transformed = numeric
    elif feature.transform == "fraction":
        if selected.size and ((selected < 0).any() or (selected > 1).any()):
            raise ValueError(f"{feature.name} must be in [0, 1]")
        transformed = numeric
    elif feature.transform == "nonnegative-count":
        if selected.size and (
            (selected < 0).any() or not np.equal(selected, np.floor(selected)).all()
        ):
            raise ValueError(f"{feature.name} must be a nonnegative integer count")
        transformed = np.log1p(numeric) / np.log1p(float(feature.reference))
    elif feature.transform == "nonnegative-duration":
        if selected.size and (selected < 0).any():
            raise ValueError(f"{feature.name} must be a nonnegative duration")
        transformed = np.log1p(numeric) / np.log1p(float(feature.reference))
    else:  # pragma: no cover - registry is module-owned and exhaustively tested
        raise ValueError(f"unknown physical transform {feature.transform!r}")
    transformed = np.where(mask, transformed, 0.0)
    if not np.isfinite(transformed[mask]).all():
        raise ValueError(f"{feature.name} transform is non-finite")
    return np.clip(transformed, -BOUNDED_GUARD, BOUNDED_GUARD).astype(np.float32)


def _support_source(batch: dict[str, np.ndarray], provenance: dict[str, Any]) -> dict[str, Any]:
    """Validate row-level closed-loop provenance and bind it to the actual support arrays."""
    expected = {"schemaVersion", "kind", "producerDigest", "transitionIds", "policyVersions"}
    if not isinstance(provenance, dict) or set(provenance) != expected:
        raise ValueError(
            f"closed-loop support provenance fields must be exactly {sorted(expected)}"
        )
    if provenance["schemaVersion"] != 1 or provenance["kind"] != "closed-loop-policy-observations":
        raise ValueError("unsupported closed-loop support provenance schema/kind")
    producer = provenance["producerDigest"]
    if (
        not isinstance(producer, str)
        or len(producer) != 64
        or any(c not in "0123456789abcdef" for c in producer)
    ):
        raise ValueError("closed-loop support producerDigest must be lowercase SHA-256")
    transition_ids, versions = provenance["transitionIds"], provenance["policyVersions"]
    rows = len(batch["core"])
    if (
        not isinstance(transition_ids, list)
        or len(transition_ids) != rows
        or any(not isinstance(item, str) or not item for item in transition_ids)
        or len(set(transition_ids)) != rows
    ):
        raise ValueError("closed-loop support requires one unique nonempty transition id per row")
    if (
        not isinstance(versions, list)
        or len(versions) != rows
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in versions)
    ):
        raise ValueError("closed-loop support requires one nonnegative policy version per row")
    return {
        "kind": provenance["kind"],
        "producerDigest": producer,
        "corpusSha256": normalization_corpus_sha256(batch),
        "transitionIdsSha256": _canonical_sha256(transition_ids),
        "rowCount": rows,
        "policyVersionMin": min(versions) if versions else None,
        "policyVersionMax": max(versions) if versions else None,
    }


def _history_support(batch: dict[str, np.ndarray]) -> dict[int, dict[str, Any]]:
    core = np.asarray(batch["core"])
    present = np.asarray(batch["core_present"], dtype=bool)
    if (
        core.ndim != 2
        or present.shape != core.shape
        or core.shape[1] <= max(f.index for f in HISTORY_FEATURES)
    ):
        raise ValueError("bounded history support requires core/core_present with indices 40:55")
    support: dict[int, dict[str, Any]] = {}
    for feature in HISTORY_FEATURES:
        mask = present[:, feature.index]
        transformed = _physical_values(core[:, feature.index], mask, feature)
        values = transformed[mask]
        low = float(values.min()) if len(values) else None
        high = float(values.max()) if len(values) else None
        spread = float(high - low) if low is not None else None
        rows = int(mask.sum())
        support[feature.index] = {
            "presentFiniteRows": rows,
            "transformedMin": low,
            "transformedMax": high,
            "transformedRange": spread,
            "supported": bool(
                rows >= MIN_SUPPORT_ROWS and spread is not None and spread >= MIN_SUPPORT_RANGE
            ),
        }
    return support


@dataclass(frozen=True)
class BoundedNormalizationContract:
    support_source: dict[str, Any] | None
    history_support: dict[int, dict[str, Any]]

    @property
    def supported_history_indices(self) -> tuple[int, ...]:
        return tuple(
            sorted(index for index, row in self.history_support.items() if row["supported"])
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": BOUNDED_NORMALIZATION_SCHEMA,
            "contractId": BOUNDED_FEATURE_CONTRACT,
            "guard": BOUNDED_GUARD,
            "standardizedStdFloor": BOUNDED_STD_FLOOR,
            "minimumSupportRows": MIN_SUPPORT_ROWS,
            "minimumTransformedRange": MIN_SUPPORT_RANGE,
            "physicalFeatures": [
                {
                    "index": feature.index,
                    "name": feature.name,
                    "transform": feature.transform,
                    "reference": feature.reference,
                }
                for feature in PHYSICAL_CORE_FEATURES
            ],
            "supportSource": self.support_source,
            "historySupport": {
                str(index): dict(row) for index, row in sorted(self.history_support.items())
            },
        }

    @staticmethod
    def from_state(state: dict[str, Any]) -> "BoundedNormalizationContract":
        template = BoundedNormalizationContract(
            None,
            {
                feature.index: {
                    "presentFiniteRows": 0,
                    "transformedMin": None,
                    "transformedMax": None,
                    "transformedRange": None,
                    "supported": False,
                }
                for feature in HISTORY_FEATURES
            },
        ).state_dict()
        expected_keys = set(template)
        if not isinstance(state, dict) or set(state) != expected_keys:
            raise ValueError(
                f"normalization contract fields must be exactly {sorted(expected_keys)}"
            )
        for key in (
            "schemaVersion",
            "contractId",
            "guard",
            "standardizedStdFloor",
            "minimumSupportRows",
            "minimumTransformedRange",
            "physicalFeatures",
        ):
            if state[key] != template[key]:
                raise ValueError(
                    f"normalization contract {key} differs from {BOUNDED_FEATURE_CONTRACT}"
                )
        raw_support = state["historySupport"]
        expected_indices = {str(feature.index) for feature in HISTORY_FEATURES}
        if not isinstance(raw_support, dict) or set(raw_support) != expected_indices:
            raise ValueError("normalization contract history support inventory is incomplete")
        support: dict[int, dict[str, Any]] = {}
        fields = {
            "presentFiniteRows",
            "transformedMin",
            "transformedMax",
            "transformedRange",
            "supported",
        }
        for text, raw in raw_support.items():
            if not isinstance(raw, dict) or set(raw) != fields:
                raise ValueError(f"history support {text} fields are invalid")
            rows = raw["presentFiniteRows"]
            low, high, spread = (
                raw["transformedMin"],
                raw["transformedMax"],
                raw["transformedRange"],
            )
            if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
                raise ValueError(f"history support {text} row count is invalid")
            if rows == 0:
                if any(value is not None for value in (low, high, spread)):
                    raise ValueError(f"history support {text} empty range is inconsistent")
            else:
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    for value in (low, high, spread)
                ):
                    raise ValueError(f"history support {text} range is invalid")
                if (
                    high < low
                    or spread < 0
                    or not np.isclose(spread, high - low, rtol=0.0, atol=1e-7)
                ):
                    raise ValueError(f"history support {text} range is inconsistent")
            derived = bool(
                rows >= MIN_SUPPORT_ROWS and spread is not None and spread >= MIN_SUPPORT_RANGE
            )
            if raw["supported"] is not derived:
                raise ValueError(
                    f"history support {text} assertion is not derived from its measurements"
                )
            support[int(text)] = dict(raw)
        source = state["supportSource"]
        if source is None:
            if any(row["presentFiniteRows"] or row["supported"] for row in support.values()):
                raise ValueError("history support measurements require a closed-loop source")
        else:
            source_fields = {
                "kind",
                "producerDigest",
                "corpusSha256",
                "transitionIdsSha256",
                "rowCount",
                "policyVersionMin",
                "policyVersionMax",
            }
            if not isinstance(source, dict) or set(source) != source_fields:
                raise ValueError("closed-loop support source fields are invalid")
            for key in ("producerDigest", "corpusSha256", "transitionIdsSha256"):
                value = source[key]
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)
                ):
                    raise ValueError(f"closed-loop support {key} is not lowercase SHA-256")
            row_count = source["rowCount"]
            if (
                source["kind"] != "closed-loop-policy-observations"
                or isinstance(row_count, bool)
                or not isinstance(row_count, int)
                or row_count < 0
            ):
                raise ValueError("closed-loop support source identity is invalid")
            if any(row["presentFiniteRows"] > row_count for row in support.values()):
                raise ValueError("history support row count exceeds its source")
            low_version, high_version = source["policyVersionMin"], source["policyVersionMax"]
            if row_count == 0:
                if low_version is not None or high_version is not None:
                    raise ValueError("empty support source cannot have policy versions")
            elif (
                isinstance(low_version, bool)
                or not isinstance(low_version, int)
                or low_version < 0
                or isinstance(high_version, bool)
                or not isinstance(high_version, int)
                or high_version < low_version
            ):
                raise ValueError("closed-loop support policy version range is invalid")
            source = dict(source)
        return BoundedNormalizationContract(source, support)


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


@dataclass
class Normalizer:
    core_mean: np.ndarray
    core_std: np.ndarray
    set_mean: dict[str, np.ndarray] = field(default_factory=dict)
    set_std: dict[str, np.ndarray] = field(default_factory=dict)
    privileged_mean: np.ndarray | None = None
    privileged_std: np.ndarray | None = None
    # Input guard (docs/MODEL9_TOKENS.md, "Cloning stability"): None keeps the original affine exactly. Otherwise
    # inputs that were constant in the fitting data (``dead``: "core", set names, "privileged") are dropped, since the
    # 1e-3 std floor would turn any other value into a |z| of hundreds through weights that never received a gradient,
    # and every other normalized input is clipped to [-guard, guard].
    guard: float | None = None
    dead: dict[str, np.ndarray] = field(default_factory=dict)
    normalization_contract: BoundedNormalizationContract | None = None

    @staticmethod
    def fit(batch: dict[str, np.ndarray], guard: float | None = None) -> "Normalizer":
        dead: dict[str, np.ndarray] = {}

        def moments(x: np.ndarray, present: np.ndarray, key: str) -> tuple[np.ndarray, np.ndarray]:
            x = np.sign(x) * np.log1p(np.abs(x))
            w = present.astype(np.float64)
            n = np.maximum(1.0, w.sum(axis=0))
            mean = (x * w).sum(axis=0) / n
            var = (((x - mean) ** 2) * w).sum(axis=0) / n
            dead[key] = np.sqrt(var) < DEAD_STD
            return mean.astype(np.float32), np.maximum(np.sqrt(var), 1e-3).astype(np.float32)

        core_mean, core_std = moments(batch["core"], batch["core_present"], "core")
        normalizer = Normalizer(core_mean, core_std)
        for name, rows, width in contracts.SETS:
            if (
                name not in batch
            ):  # shards written before the set existed (contract 10 regions): identity moments
                normalizer.set_mean[name], normalizer.set_std[name] = (
                    np.zeros(width, np.float32),
                    np.ones(width, np.float32),
                )
                dead[name] = np.zeros(
                    width, bool
                )  # absent rows are masked anyway; a later set keeps its full input
                continue
            x = batch[name].reshape(-1, width)
            present = np.repeat(batch[name + "_present"].reshape(-1, 1), width, axis=1)
            normalizer.set_mean[name], normalizer.set_std[name] = moments(x, present, name)
        if "privileged" in batch:
            normalizer.privileged_mean, normalizer.privileged_std = moments(
                batch["privileged"], np.ones_like(batch["privileged"], bool), "privileged"
            )
        if guard is not None:
            normalizer.guard, normalizer.dead = float(guard), dead
        return normalizer

    @staticmethod
    def fit_bounded(
        batch: dict[str, np.ndarray], *, closed_loop_provenance: dict[str, Any] | None = None
    ) -> "Normalizer":
        """Fit the new-run bounded contract.

        With no provenance, history 40:55 is deliberately masked for clone/support collection. Passing typed
        row-level closed-loop provenance refits all ordinary moments on that closed-loop corpus and derives the
        supported history subset from actual values; callers cannot nominate support indices.
        """
        core = np.asarray(batch["core"])
        core_present = np.asarray(batch["core_present"])
        if core.ndim != 2 or core_present.shape != core.shape or core.shape[1] <= 144:
            raise ValueError("bounded normalization requires the Model9 core through index 144")
        dead: dict[str, np.ndarray] = {}

        def moments(x: np.ndarray, present: np.ndarray, key: str) -> tuple[np.ndarray, np.ndarray]:
            x = np.asarray(x)
            present = np.asarray(present, dtype=bool)
            if x.shape != present.shape or not np.issubdtype(x.dtype, np.number):
                raise ValueError(f"bounded normalization {key} values/presence are invalid")
            if not np.isfinite(x[present]).all():
                raise ValueError(f"bounded normalization {key} contains non-finite present values")
            safe = np.where(present, x, 0).astype(np.float64)
            transformed = np.sign(safe) * np.log1p(np.abs(safe))
            w = present.astype(np.float64)
            n = np.maximum(1.0, w.sum(axis=0))
            mean = (transformed * w).sum(axis=0) / n
            var = (((transformed - mean) ** 2) * w).sum(axis=0) / n
            raw_std = np.sqrt(var)
            dead[key] = raw_std < DEAD_STD
            return mean.astype(np.float32), np.maximum(raw_std, BOUNDED_STD_FLOOR).astype(
                np.float32
            )

        core_mean, core_std = moments(core, core_present, "core")
        normalizer = Normalizer(core_mean, core_std, guard=BOUNDED_GUARD)
        for name, rows, width in contracts.SETS:
            if name not in batch:
                normalizer.set_mean[name] = np.zeros(width, np.float32)
                normalizer.set_std[name] = np.ones(width, np.float32)
                dead[name] = np.zeros(width, bool)
                continue
            values = np.asarray(batch[name]).reshape(-1, width)
            present = np.repeat(np.asarray(batch[name + "_present"]).reshape(-1, 1), width, axis=1)
            normalizer.set_mean[name], normalizer.set_std[name] = moments(values, present, name)
        if "privileged" in batch:
            privileged = np.asarray(batch["privileged"])
            normalizer.privileged_mean, normalizer.privileged_std = moments(
                privileged, np.ones_like(privileged, bool), "privileged"
            )
        normalizer.dead = dead
        if closed_loop_provenance is None:
            empty = {
                feature.index: {
                    "presentFiniteRows": 0,
                    "transformedMin": None,
                    "transformedMax": None,
                    "transformedRange": None,
                    "supported": False,
                }
                for feature in HISTORY_FEATURES
            }
            normalizer.normalization_contract = BoundedNormalizationContract(None, empty)
        else:
            normalizer.normalization_contract = BoundedNormalizationContract(
                _support_source(batch, closed_loop_provenance), _history_support(batch)
            )
        return normalizer

    def state_dict(self) -> dict[str, Any]:
        out = {"core_mean": self.core_mean.tolist(), "core_std": self.core_std.tolist()}
        for name in self.set_mean:
            out[name + "_mean"] = self.set_mean[name].tolist()
            out[name + "_std"] = self.set_std[name].tolist()
        if self.privileged_mean is not None:
            out["privileged_mean"] = self.privileged_mean.tolist()
            out["privileged_std"] = self.privileged_std.tolist()
        if (
            self.guard is not None
        ):  # absent in unguarded checkpoints, so they load (and normalize) exactly as before
            out["guard"] = self.guard
            for key, value in self.dead.items():
                out[key + "_dead"] = value.tolist()
        if self.normalization_contract is not None:
            out["normalization_contract"] = self.normalization_contract.state_dict()
        return out

    @staticmethod
    def from_state(state: dict[str, Any]) -> "Normalizer":
        n = Normalizer(
            np.asarray(state["core_mean"], np.float32), np.asarray(state["core_std"], np.float32)
        )
        for name, _, _ in contracts.SETS:
            n.set_mean[name] = np.asarray(state[name + "_mean"], np.float32)
            n.set_std[name] = np.asarray(state[name + "_std"], np.float32)
        if "privileged_mean" in state:
            n.privileged_mean = np.asarray(state["privileged_mean"], np.float32)
            n.privileged_std = np.asarray(state["privileged_std"], np.float32)
        if state.get("guard") is not None:
            n.guard = float(state["guard"])
            n.dead = {
                key[: -len("_dead")]: np.asarray(value, bool)
                for key, value in state.items()
                if key.endswith("_dead")
            }
        if "normalization_contract" in state:
            n.normalization_contract = BoundedNormalizationContract.from_state(
                state["normalization_contract"]
            )
            if n.guard != BOUNDED_GUARD:
                raise ValueError("bounded normalization state requires guard 10")
            if (
                n.core_mean.ndim != 1
                or n.core_std.shape != n.core_mean.shape
                or len(n.core_mean) <= 144
            ):
                raise ValueError("bounded normalization core moment shapes are invalid")
            expected_dead = {"core", *(name for name, _, _ in contracts.SETS)}
            if n.privileged_mean is not None:
                if n.privileged_mean.ndim != 1 or n.privileged_std.shape != n.privileged_mean.shape:
                    raise ValueError("bounded normalization privileged moment shapes are invalid")
                expected_dead.add("privileged")
            for name, _, width in contracts.SETS:
                if n.set_mean[name].shape != (width,) or n.set_std[name].shape != (width,):
                    raise ValueError(f"bounded normalization {name} moment shapes are invalid")
            if set(n.dead) != expected_dead:
                raise ValueError("bounded normalization dead-mask inventory is invalid")
            expected_shapes = {
                "core": n.core_mean.shape,
                **{name: (width,) for name, _, width in contracts.SETS},
            }
            if n.privileged_mean is not None:
                expected_shapes["privileged"] = n.privileged_mean.shape
            if any(n.dead[key].shape != shape for key, shape in expected_shapes.items()):
                raise ValueError("bounded normalization dead-mask shapes are invalid")
            arrays = [n.core_mean, n.core_std, *n.set_mean.values(), *n.set_std.values()]
            if n.privileged_mean is not None:
                arrays.extend([n.privileged_mean, n.privileged_std])
            if any(not np.isfinite(value).all() for value in arrays):
                raise ValueError("bounded normalization moments must be finite")
            stds = [n.core_std, *n.set_std.values()]
            if n.privileged_std is not None:
                stds.append(n.privileged_std)
            if any((value < BOUNDED_STD_FLOOR).any() for value in stds):
                raise ValueError("bounded normalization std is below the declared floor")
        return n

    def dead_names(self) -> dict[str, list[int]]:
        """Indices of the dropped inputs per block (for the training log)."""
        return {
            key: np.flatnonzero(value).tolist() for key, value in self.dead.items() if value.any()
        }

    def require_closed_loop_support(self, indices: tuple[int, ...] | list[int]) -> None:
        requested = tuple(indices)
        known = {feature.index for feature in HISTORY_FEATURES}
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index not in known
            for index in requested
        ):
            raise ValueError("closed-loop support may only be required for history indices 40:55")
        supported = (
            set(self.normalization_contract.supported_history_indices)
            if self.normalization_contract is not None
            else set()
        )
        missing = sorted(set(requested) - supported)
        if missing:
            raise ValueError(f"closed-loop support is unavailable for history indices {missing}")


def normalizer_feature_contract(normalizer: Normalizer) -> str | None:
    """Checkpoint identifier for new normalizers; legacy checkpoints intentionally return None."""
    return BOUNDED_FEATURE_CONTRACT if normalizer.normalization_contract is not None else None


def load_checkpoint_normalizer(
    state: dict[str, Any],
    *,
    expected_feature_contract: str | None = None,
    required_history_indices: tuple[int, ...] | list[int] = (),
) -> Normalizer:
    """Strict checkpoint boundary for new-mode loaders without changing legacy loading semantics.

    New checkpoints must carry the same identifier both at top level (``feature_contract``) and inside their
    normalizer state. Historical checkpoints carry neither; an old ``guard`` alone is still legacy.
    """
    if not isinstance(state, dict) or not isinstance(state.get("normalizer"), dict):
        raise ValueError("checkpoint is missing its normalizer state")
    normalizer = Normalizer.from_state(state["normalizer"])
    nested = normalizer_feature_contract(normalizer)
    declared = state.get("feature_contract")
    if nested is None:
        if declared is not None:
            raise ValueError("legacy normalizer cannot declare a new feature contract")
    elif declared != nested:
        raise ValueError("checkpoint feature contract does not match its normalizer contract")
    actual = nested if nested is not None else LEGACY_FEATURE_CONTRACT
    if expected_feature_contract is not None and actual != expected_feature_contract:
        raise ValueError(
            f"checkpoint feature contract {actual!r} != expected {expected_feature_contract!r}"
        )
    normalizer.require_closed_loop_support(required_history_indices)
    return normalizer


# A normalized input whose fitting-data std (after symlog) is below this is constant there: the guard drops it.
DEAD_STD = 1e-3


def to_tensors(
    batch: dict[str, np.ndarray], normalizer: Normalizer, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    def norm(
        x: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        key: str,
        present_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        source = np.asarray(x)
        if normalizer.normalization_contract is not None:
            mask = (
                np.ones_like(source, dtype=bool)
                if present_mask is None
                else np.asarray(present_mask, dtype=bool)
            )
            if source.shape != mask.shape or not np.issubdtype(source.dtype, np.number):
                raise ValueError(f"bounded normalization {key} input/presence shapes are invalid")
            if not np.isfinite(source[mask]).all():
                raise ValueError(f"bounded normalization {key} contains non-finite present values")
            source = np.where(mask, source, 0)
        z = ((np.sign(source) * np.log1p(np.abs(source))) - mean) / std
        if normalizer.guard is not None:
            z = np.clip(z, -normalizer.guard, normalizer.guard)
            dead = normalizer.dead.get(key)
            if dead is not None and dead.any():
                z = np.where(dead, 0.0, z).astype(z.dtype)
        if normalizer.normalization_contract is not None and present_mask is not None:
            z = np.where(present_mask, z, 0.0)
        return z

    if normalizer.normalization_contract is None:
        # Keep the historical path exactly as saved, including its original presence dtype semantics.
        present = batch["core_present"].astype(np.float32)
        core = norm(batch["core"], normalizer.core_mean, normalizer.core_std, "core") * present
    else:
        present_mask = np.asarray(batch["core_present"], dtype=bool).copy()
        core = norm(batch["core"], normalizer.core_mean, normalizer.core_std, "core", present_mask)
        if core.shape[1] <= 144:
            raise ValueError("bounded normalization input lacks the physical core registry")
        supported = set(normalizer.normalization_contract.supported_history_indices)
        raw_core = np.asarray(batch["core"])
        for feature in PHYSICAL_CORE_FEATURES:
            values = _physical_values(
                raw_core[:, feature.index], present_mask[:, feature.index], feature
            )
            if feature in HISTORY_FEATURES and feature.index not in supported:
                core[:, feature.index] = 0.0
                present_mask[:, feature.index] = False
            else:
                core[:, feature.index] = values
        present = present_mask.astype(np.float32)
        core = core * present
    out = {
        "core": torch.as_tensor(
            np.concatenate([core, present], axis=1), dtype=torch.float32, device=device
        )
    }
    for name, rows, width in contracts.SETS:
        if name not in batch:  # a set the shard predates: every row absent
            out[name] = torch.zeros(
                len(batch["core"]), rows, width, dtype=torch.float32, device=device
            )
            out[name + "_present"] = torch.zeros(
                len(batch["core"]), rows, dtype=torch.bool, device=device
            )
            continue
        mask = batch[name + "_present"]
        expanded_mask = np.broadcast_to(
            np.asarray(mask, dtype=bool)[..., None], np.asarray(batch[name]).shape
        )
        x = norm(
            batch[name], normalizer.set_mean[name], normalizer.set_std[name], name, expanded_mask
        )
        x = x * mask[..., None]
        out[name] = torch.as_tensor(x, dtype=torch.float32, device=device)
        out[name + "_present"] = torch.as_tensor(mask, dtype=torch.bool, device=device)
    for key in ("movement_mask", "clothing_route_mask", "shelter_route_mask"):
        out[key] = torch.as_tensor(batch[key], dtype=torch.bool, device=device)
    mix_mask = batch.get("economy_mix_mask")
    if mix_mask is None:
        mix_mask = np.ones((len(batch["core"]), len(contracts.ECONOMY_MIX)), bool)
    out["economy_mix_mask"] = torch.as_tensor(mix_mask, dtype=torch.bool, device=device)
    hold = batch.get("commit_hold_mask")
    if hold is None:
        hold = np.zeros(len(batch["core"]), bool)
    out["commit_hold_mask"] = torch.as_tensor(hold, dtype=torch.bool, device=device)
    region = batch.get("region_movement_mask")
    if region is None:  # only native is legal
        region = np.zeros((len(batch["core"]), contracts.REGION_MOVEMENT_SLOTS), bool)
        region[:, contracts.REGION_NATIVE] = True
    out["region_movement_mask"] = torch.as_tensor(region, dtype=torch.bool, device=device)
    # Elder personality: per-row weights over contracts.OBJECTIVES (absent: the default personality).
    personality = batch.get("personality")
    if personality is None:
        personality = np.tile(
            np.asarray([contracts.OBJECTIVE_WEIGHTS[k] for k in contracts.OBJECTIVES], np.float32),
            (len(batch["core"]), 1),
        )
    out["personality"] = torch.as_tensor(personality, dtype=torch.float32, device=device)
    if "privileged" in batch and normalizer.privileged_mean is not None:
        out["privileged"] = torch.as_tensor(
            norm(
                batch["privileged"],
                normalizer.privileged_mean,
                normalizer.privileged_std,
                "privileged",
            ),
            dtype=torch.float32,
            device=device,
        )
    return out


class SetEncoder(nn.Module):
    """One small MLP applied to every row, then masked sum, mean and max pooling."""

    def __init__(self, width: int, hidden: int = 64) -> None:
        super().__init__()
        self.row = nn.Sequential(
            nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.hidden = hidden

    def forward(self, x: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = self.row(x)
        mask = present.unsqueeze(-1).float()
        rows = rows * mask
        count = mask.sum(dim=1).clamp(min=1.0)
        total = rows.sum(dim=1)
        mean = total / count
        maximum = rows.masked_fill(~present.unsqueeze(-1), -1e4).max(dim=1).values
        maximum = torch.where(present.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        any_row = present.any(dim=1, keepdim=True).float()
        return torch.cat([total / 8.0, mean, maximum, any_row], dim=1), rows


class Encoder(nn.Module):
    def __init__(
        self, core_width: int, width: int = 256, set_hidden: int = 64, privileged_width: int = 0
    ) -> None:
        super().__init__()
        self.sets = nn.ModuleDict(
            {name: SetEncoder(w, set_hidden) for name, _, w in contracts.SETS}
        )
        self.core = nn.Sequential(
            nn.Linear(core_width * 2 + privileged_width, width), nn.SiLU(), nn.LayerNorm(width)
        )
        self.privileged_width = privileged_width
        pooled = len(contracts.SETS) * (3 * set_hidden + 1)
        self.trunk = nn.Sequential(
            nn.Linear(width + pooled, width),
            nn.SiLU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )
        self.width, self.set_hidden = width, set_hidden

    def forward(self, t: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pooled, rows = [], {}
        for name, _, _ in contracts.SETS:
            p, r = self.sets[name](t[name], t[name + "_present"])
            pooled.append(p)
            rows[name] = r
        core = (
            t["core"]
            if self.privileged_width == 0
            else torch.cat([t["core"], t["privileged"]], dim=1)
        )
        h = self.trunk(torch.cat([self.core(core)] + pooled, dim=1))
        return h, rows


class TokenEncoder(nn.Module):
    """Contract 12 token encoder (docs/MODEL9_TOKENS.md): every present row of every entity set is a token (a per-set
    projection plus set-type and row-index embeddings), the core observation is the village token, and L self-attention
    layers mix them. The village token's output feeds a fixed-width trunk, so adding a set or a row kind never changes
    the trunk; pointer heads read their set's token outputs. Absent rows are masked out of attention, never zeros."""

    def __init__(
        self,
        core_width: int,
        width: int = 256,
        d: int = 128,
        layers: int = 2,
        heads: int = 4,
        privileged_width: int = 0,
    ) -> None:
        super().__init__()
        self.project = nn.ModuleDict({name: nn.Linear(w, d) for name, _, w in contracts.SETS})
        self.kind = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(d)) for name, _, _ in contracts.SETS}
        )
        self.position = nn.ModuleDict(
            {name: nn.Embedding(rows, d) for name, rows, _ in contracts.SETS}
        )
        for e in self.position.values():
            nn.init.normal_(e.weight, std=0.02)
        # The village token reads the core observation and the Elder's personality (docs/MODEL9_PERSONALITY.md).
        self.village = nn.Linear(core_width * 2 + privileged_width + contracts.PERSONALITY_WIDTH, d)
        layer = nn.TransformerEncoderLayer(
            d, heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True, norm_first=True
        )
        self.attention = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.trunk = nn.Sequential(
            nn.Linear(d, width),
            nn.SiLU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )
        self.width, self.set_hidden, self.privileged_width = width, d, privileged_width
        # FiLM conditioning (docs/MODEL9_PERSONALITY.md): the personality's offset from the default personality sets a
        # per-channel scale and shift on the tokens before every attention layer and after each trunk layer, so a
        # personality reshapes every computation instead of being one more input. Bias-free and zero-initialised: at
        # the default personality, and in any checkpoint saved before FiLM, it is exactly the identity. Later trait
        # vectors widen the conditioning input rather than adding tokens.
        self.film_tokens = nn.ModuleList(
            nn.Linear(contracts.PERSONALITY_WIDTH, 2 * d, bias=False) for _ in range(layers)
        )
        self.film_trunk = nn.ModuleList(
            nn.Linear(contracts.PERSONALITY_WIDTH, 2 * width, bias=False) for _ in range(2)
        )
        for film in [*self.film_tokens, *self.film_trunk]:
            nn.init.zeros_(film.weight)
        self.register_buffer(
            "default_personality",
            torch.tensor(
                [contracts.OBJECTIVE_WEIGHTS[k] for k in contracts.OBJECTIVES], dtype=torch.float32
            ),
            persistent=False,
        )
        self._register_load_state_dict_pre_hook(self._identity_film)

    def _identity_film(self, state_dict, prefix, *args) -> None:
        """Checkpoints saved before FiLM load with identity (zero) FiLM weights."""
        for name, p in self.named_parameters():
            if name.startswith("film_") and prefix + name not in state_dict:
                state_dict[prefix + name] = torch.zeros_like(p)

    # Optional bound on the FiLM scale and shift (bound x tanh(x / bound)). Off by default because it also dampens
    # FiLM; ppo --film-bound sets it only if film_max shows instability.
    film_bound: float | None = None

    def film_extent(self, t: dict[str, torch.Tensor]) -> float:
        """The largest |scale| or |shift| any FiLM layer applies on this batch (the instability monitor)."""
        c = t["personality"] - self.default_personality
        with torch.no_grad():
            return float(max(f(c).abs().max() for f in [*self.film_tokens, *self.film_trunk]))

    def _film(self, h: torch.Tensor, film: nn.Linear, c: torch.Tensor) -> torch.Tensor:
        scale, shift = film(c).chunk(2, dim=-1)
        if self.film_bound:
            scale = self.film_bound * torch.tanh(scale / self.film_bound)
            shift = self.film_bound * torch.tanh(shift / self.film_bound)
        if h.dim() == 3:
            scale, shift = scale.unsqueeze(1), shift.unsqueeze(1)
        return h * (1 + scale) + shift

    def forward(self, t: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        core = (
            t["core"]
            if self.privileged_width == 0
            else torch.cat([t["core"], t["privileged"]], dim=1)
        )
        condition = t["personality"] - self.default_personality
        core = torch.cat([core, t["personality"]], dim=1)
        tokens = [self.village(core).unsqueeze(1)]
        present = [torch.ones(len(core), 1, dtype=torch.bool, device=core.device)]
        spans = {}
        offset = 1
        for name, rows, _ in contracts.SETS:
            x = (
                self.project[name](t[name])
                + self.kind[name]
                + self.position[name].weight.unsqueeze(0)
            )
            tokens.append(x)
            present.append(t[name + "_present"])
            spans[name] = (offset, offset + rows)
            offset += rows
        x = torch.cat(tokens, dim=1)
        mask = torch.cat(present, dim=1)
        # The encoder's layers one by one (it has no final norm), each after its FiLM step.
        for layer, film in zip(self.attention.layers, self.film_tokens):
            x = layer(self._film(x, film, condition), src_key_padding_mask=~mask)
        out = self.norm(x)
        out = out * mask.unsqueeze(-1).float()
        rows_out = {name: out[:, a:b] for name, (a, b) in spans.items()}
        h = self._film(self.trunk[:3](out[:, 0]), self.film_trunk[0], condition)
        h = self._film(self.trunk[3:](h), self.film_trunk[1], condition)
        return h, rows_out


class EconomyExperts(nn.Module):
    """Capability-conditioned expert adapters (M7 pilot, option b): one residual MLP per economy (cultivation,
    herding, fishing). A village averages the adapters of the economies it can practise (its economy-mix mask), so a
    fishing-only village's gradient reaches only the fishing adapter. The adapters' output layers start at zero: a
    warm start from a shared-head checkpoint reproduces it exactly."""

    def __init__(self, width: int, hidden: int = 64) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(
            nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, width))
            for _ in contracts.ECONOMY_MIX
        )
        for adapter in self.adapters:
            nn.init.zeros_(adapter[2].weight)
            nn.init.zeros_(adapter[2].bias)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gate = mask.float() / mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        return h + sum(gate[:, e : e + 1] * adapter(h) for e, adapter in enumerate(self.adapters))


def init_span_head(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
        layer.bias[contracts.SPAN_SINGLE] = float(np.log(len(contracts.SPANS) - 1))


# Contract 10: the region movement head starts at NATIVE_START probability on native movement, the rest spread
# evenly, so a learned-movement policy begins as the heuristic (argmax native) and explores the alternatives.
NATIVE_START = 0.8


# Round 3 hierarchical region head (docs/MODEL9_M7.md, *Round 3 build: hierarchical region head*): one distribution
# for sampling, likelihood, search labels, selection and inference. First a 3-way decision (native, stay, go), then a
# pointer over the go slots (keep the intent, region rows 1-7). The group logits are the log-mass the flat logits give
# each group plus a learned offset (``Actor.region_group``, zero-initialised), so an untrained offset (and any checkpoint
# saved before the head) is exactly the flat distribution; only the deterministic choice differs (group first). The
# offset lets the policy move the go mass without moving any destination's relative odds.
REGION_GROUP_NATIVE, REGION_GROUP_STAY, REGION_GROUP_GO = 0, 1, 2
REGION_GROUPS = ("native", "stay", "go")
REGION_STAY = 2  # region row 0, the current tile
_FILL = (
    -1e4
)  # stands for an excluded slot inside the group reductions (finite, so gradients stay finite)


def region_group_index(slots: int = contracts.REGION_MOVEMENT_SLOTS) -> torch.Tensor:
    """Group of every region-movement slot: native, keep (go), stay (row 0), rows 1-7 (go)."""
    index = torch.full((slots,), REGION_GROUP_GO, dtype=torch.long)
    index[contracts.REGION_NATIVE] = REGION_GROUP_NATIVE
    index[REGION_STAY] = REGION_GROUP_STAY
    return index


def default_region_flat() -> bool:
    """The head a checkpoint saved before the hierarchical head loads with: hierarchical, unless the environment sets
    ANTHROPOCENE_REGION_HEAD=flat (to reproduce earlier flat-argmax evaluations of old checkpoints exactly)."""
    import os

    return os.environ.get("ANTHROPOCENE_REGION_HEAD", "hierarchical").strip().lower() == "flat"


def hierarchical_region(
    region: torch.Tensor, legal: torch.Tensor, offset: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group logits (B, 3) and the joint log-probability of every slot (B, S) of the hierarchical region head.
    ``region``: masked flat logits; ``legal``: the region-movement mask; ``offset``: the learned group offset. A group with
    no legal slot gets the MASKED logit; an illegal slot gets log-probability _FILL."""
    index = region_group_index(region.shape[1]).to(region.device)
    member = nn.functional.one_hot(index, len(REGION_GROUPS)).bool().unsqueeze(0) & legal.unsqueeze(
        2
    )  # B, S, 3
    lse = torch.logsumexp(region.unsqueeze(2).masked_fill(~member, _FILL), dim=1)  # B, 3
    any_legal = member.any(dim=1)
    group = torch.where(any_legal, lse + offset, torch.full_like(lse, MASKED))
    within = region - lse.gather(1, index.unsqueeze(0).expand(len(region), -1))
    joint = (
        torch.log_softmax(group, dim=1).gather(1, index.unsqueeze(0).expand(len(region), -1))
        + within
    )
    return group, torch.where(legal, joint, torch.full_like(joint, _FILL))


def region_mode(out: dict[str, torch.Tensor], grouped: bool = False) -> torch.Tensor:
    """The deterministic region slot. The hierarchical head (``region_group`` in ``out``): the most likely group, then
    its most likely slot, whatever ``grouped`` says. The flat head (``Actor.region_flat``): a plain argmax, or
    ``grouped_region_slot`` when ``grouped`` (the evaluation option that preceded the hierarchical head)."""
    if "region_group" in out:
        index = region_group_index(out["region"].shape[1]).to(out["region"].device)
        group = out["region_group"].argmax(dim=1)
        within = out["region_logp"].masked_fill(
            index.unsqueeze(0) != group.unsqueeze(1), -float("inf")
        )
        return within.argmax(dim=1)
    return grouped_region_slot(out["region"]) if grouped else out["region"].argmax(dim=1)


def init_region_heads(actor: "Actor") -> None:
    """Zero weights everywhere; the native logit's bias sets NATIVE_START. The mix offset starts at zero, so the mix
    given any destination equals the unconditioned mix."""
    others = contracts.REGION_MOVEMENT_SLOTS - 1
    with torch.no_grad():
        for layer in (
            actor.region_native,
            actor.region_keep,
            actor.region_query,
            actor.mix_region,
            actor.region_group,
        ):
            layer.weight.zero_()
            layer.bias.zero_()
        actor.region_native.bias.fill_(float(np.log(NATIVE_START / (1 - NATIVE_START) * others)))


# Contract 11 trade heads start at the native posture: log relative value 0, gates at Make with probability
# TRADE_GATE_START, and an export share whose mean sigmoid(TRADE_SHARE_START) snaps to 0 in deterministic evaluation
# (the shipped share is 0) while sampled shares explore small making capacity.
TRADE_GATE_START = 0.8
TRADE_SHARE_START = -4.6
TRADE_LOG_STD_START = -2.3


def init_trade_heads(actor: "Actor") -> None:
    with torch.no_grad():
        for layer in (actor.trade_value, actor.trade_gate, actor.trade_share):
            layer.weight.zero_()
            layer.bias.zero_()
        groups = len(contracts.TRADE_GROUPS)
        actor.trade_value.bias[groups:] = TRADE_LOG_STD_START
        gate = actor.trade_gate.bias.view(groups, contracts.TRADE_GATES)
        gate[:, contracts.TRADE_MAKE] = float(
            np.log(TRADE_GATE_START / (1 - TRADE_GATE_START) * (contracts.TRADE_GATES - 1))
        )
        actor.trade_share.bias[0] = TRADE_SHARE_START
        actor.trade_share.bias[1] = TRADE_LOG_STD_START


def condition_mix(out: dict[str, torch.Tensor], slot: torch.Tensor) -> dict[str, torch.Tensor]:
    """The economy mix given the chosen region-movement slot (contract 10: the mix is chosen after the site)."""
    if "mix_offset" not in out:
        return out
    out = dict(out)
    offset = (
        out["mix_offset"]
        .gather(1, slot.clamp(min=0).view(-1, 1, 1).expand(-1, 1, out["mix_offset"].shape[2]))
        .squeeze(1)
    )
    out["mix_logits"] = out["mix_logits"] + offset
    return out


def span_mask(t: dict[str, torch.Tensor]) -> torch.Tensor:
    """True where a span is illegal: holding needs an active, unbroken commitment."""
    rows = t["core"].shape[0]
    mask = torch.zeros(rows, len(contracts.SPANS), dtype=torch.bool, device=t["core"].device)
    hold = t.get("commit_hold_mask")
    mask[:, contracts.SPAN_HOLD] = True if hold is None else ~hold
    return mask


class Actor(nn.Module):
    """Logistic-normal service levels and unit strategic controls, a masked softmax economy mix over cultivation,
    herding and fishing, masked categorical routes and fission policy, and a pointer over site rows for movement
    (stay is slot 0). With ``experts`` the strategic, mix and service heads read capability-conditioned features."""

    def __init__(
        self,
        core_width: int,
        width: int = 256,
        experts: bool = False,
        token: bool = False,
        region_head: str | None = None,
    ) -> None:
        super().__init__()
        # Round 3: the hierarchical region head is the default; "flat" reproduces the earlier flat softmax and argmax.
        # Saved as a buffer, so every loader (evaluation included) restores the checkpoint's own head.
        if region_head not in (None, "hierarchical", "flat"):
            raise ValueError(f"unknown region head {region_head!r}")
        flat = default_region_flat() if region_head is None else region_head == "flat"
        self.register_buffer("region_flat", torch.tensor(flat))
        # Contract 12: the token model reads the economy mix from activity tokens and the trade posture from trade
        # tokens, so new activities and trade groups are rows, not heads.
        self.token = token
        self.encoder = TokenEncoder(core_width, width) if token else Encoder(core_width, width)
        self.experts = EconomyExperts(width) if experts else None
        self.strategic = nn.Linear(width, 2 * contracts.UNIT_STRATEGIC)
        self.mix = nn.Linear(width, 2 * len(contracts.ECONOMY_MIX))
        self.service = nn.Linear(width, 2 * len(contracts.SERVICES))
        self.clothing = nn.Linear(width, 2)
        self.shelter = nn.Linear(width, 2)
        self.fission = nn.Linear(width, contracts.FISSION_POLICIES)
        self.stay = nn.Linear(width, 1)
        self.site_query = nn.Linear(width, self.encoder.set_hidden)
        # Contract 9 commitment span over contracts.SPANS. It starts at about even odds for "this turn only" against
        # the rest, so the argmax (deterministic evaluation) is "this turn only", contract 8's behaviour.
        self.span = nn.Linear(width, len(contracts.SPANS))
        init_span_head(self.span)
        # Contract 10 movement: native and keep-intent logits plus a pointer over region rows, and the economy-mix
        # offset given the chosen row.
        hidden = self.encoder.set_hidden
        self.region_native = nn.Linear(width, 1)
        self.region_keep = nn.Linear(width, 1)
        self.region_query = nn.Linear(width, hidden)
        self.mix_region = nn.Linear(hidden, len(contracts.ECONOMY_MIX))
        # A dormant new head must not change any later head/critic initialization in flat compatibility mode.
        with torch.random.fork_rng():
            self.region_group = nn.Linear(width, len(REGION_GROUPS))
        init_region_heads(self)
        self._register_load_state_dict_pre_hook(self._region_defaults)
        # Contract 11 trade posture: relative value per group, offer gate per group, export labour share.
        groups = len(contracts.TRADE_GROUPS)
        self.trade_value = nn.Linear(width, 2 * groups)
        self.trade_gate = nn.Linear(width, groups * contracts.TRADE_GATES)
        self.trade_share = nn.Linear(width, 2)
        init_trade_heads(self)
        if token:
            self.activity = nn.Linear(hidden, 2)  # per activity token: mix logit, log std
            self.trade_token = nn.Linear(
                hidden, 2 + contracts.TRADE_GATES
            )  # per trade token: log value, log std, gates
            with torch.no_grad():
                self.trade_token.weight.zero_()
                self.trade_token.bias.zero_()
                self.trade_token.bias[1] = TRADE_LOG_STD_START
                self.trade_token.bias[2 + contracts.TRADE_MAKE] = float(
                    np.log(TRADE_GATE_START / (1 - TRADE_GATE_START) * (contracts.TRADE_GATES - 1))
                )

    def _region_defaults(self, state_dict, prefix, *args) -> None:
        """Checkpoints saved before the hierarchical head: a zero group offset (exactly the flat distribution) and the
        default head (default_region_flat)."""
        for name in ("region_group.weight", "region_group.bias"):
            if prefix + name not in state_dict:
                state_dict[prefix + name] = torch.zeros_like(dict(self.named_parameters())[name])
        if prefix + "region_flat" not in state_dict:
            state_dict[prefix + "region_flat"] = torch.tensor(default_region_flat())

    @property
    def region_head(self) -> str:
        return "flat" if bool(self.region_flat) else "hierarchical"

    def set_region_head(self, head: str) -> None:
        if head not in ("hierarchical", "flat"):
            raise ValueError(f"unknown region head {head!r}")
        self.region_flat.fill_(head == "flat")

    def forward(self, t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h, rows = self.encoder(t)
        e = self.experts(h, t["economy_mix_mask"]) if self.experts is not None else h
        strategic = self.strategic(e).view(-1, contracts.UNIT_STRATEGIC, 2)
        mix = self.mix(e).view(-1, len(contracts.ECONOMY_MIX), 2)
        service = self.service(e).view(-1, len(contracts.SERVICES), 2)
        site_scores = torch.einsum("bh,brh->br", self.site_query(h), rows["sites"]) / np.sqrt(
            self.encoder.set_hidden
        )
        movement = torch.cat([self.stay(h), site_scores], dim=1).masked_fill(
            ~t["movement_mask"], MASKED
        )
        region_rows = rows["regions"]
        region_scores = torch.einsum("bh,brh->br", self.region_query(h), region_rows) / np.sqrt(
            self.encoder.set_hidden
        )
        region = torch.cat([self.region_native(h), self.region_keep(h), region_scores], dim=1)
        mix_offset = torch.cat(
            [
                torch.zeros(len(h), 2, len(contracts.ECONOMY_MIX), device=h.device),
                self.mix_region(region_rows),
            ],
            dim=1,
        )
        groups = len(contracts.TRADE_GROUPS)
        trade_share = self.trade_share(e)
        if self.token:
            a = self.activity(rows["activities"])
            mix = torch.stack([a[..., 0], a[..., 1]], dim=-1)
            tt = self.trade_token(rows["trade"])
            trade_value = torch.stack([tt[..., 0], tt[..., 1]], dim=1)
            trade_gate = tt[..., 2:]
        else:
            trade_value = self.trade_value(e).view(-1, 2, groups)
            trade_gate = self.trade_gate(e).view(-1, groups, contracts.TRADE_GATES)
        legal = t["region_movement_mask"]
        region = region.masked_fill(~legal, MASKED)
        extra: dict[str, torch.Tensor] = {}
        if bool(self.region_flat):
            extra["region_logp"] = torch.log_softmax(region, dim=1)
            extra["region_probs"] = torch.softmax(region, dim=1)
        else:
            extra["region_group"], extra["region_logp"] = hierarchical_region(
                region, legal, self.region_group(h)
            )
            extra["region_probs"] = extra["region_logp"].exp()
        return {
            **extra,
            "trade_value_mean": trade_value[:, 0],
            "trade_value_log_std": trade_value[:, 1].clamp(-5, 1),
            "trade_gate": trade_gate,
            "trade_share_mean": trade_share[:, 0],
            "trade_share_log_std": trade_share[:, 1].clamp(-5, 1),
            "region": region,
            "mix_offset": mix_offset,
            "strategic_mean": strategic[..., 0],
            "strategic_log_std": strategic[..., 1].clamp(-5, 1),
            "mix_logits": mix[..., 0].masked_fill(~t["economy_mix_mask"], MASKED),
            "mix_log_std": mix[..., 1].clamp(-5, 1),
            "service_mean": service[..., 0],
            "service_log_std": service[..., 1].clamp(-5, 1),
            "clothing": self.clothing(h).masked_fill(~t["clothing_route_mask"], MASKED),
            "shelter": self.shelter(h).masked_fill(~t["shelter_route_mask"], MASKED),
            "fission": self.fission(h),
            "movement": movement,
            "span": self.span(h).masked_fill(span_mask(t), MASKED),
        }


class Critic(nn.Module):
    """Predicts u = return per opening resident as a two-hot distribution over symlog-spaced bins."""

    def __init__(
        self, core_width: int, width: int = 256, privileged_width: int = 0, token: bool = False
    ) -> None:
        super().__init__()
        # Privileged inputs reach only the critic's own encoder, never the actor graph.
        self.encoder = (
            TokenEncoder(core_width, width, privileged_width=privileged_width)
            if token
            else Encoder(core_width, width, privileged_width=privileged_width)
        )
        self.head = nn.Linear(width, BINS)
        self.register_buffer("bins", torch.linspace(BIN_LOW, BIN_HIGH, BINS))

    def forward(self, t: dict[str, torch.Tensor]) -> torch.Tensor:
        h, _ = self.encoder(t)
        return self.head(h)

    def expectation(self, logits: torch.Tensor) -> torch.Tensor:
        # Expectation in raw units avoids the downward bias of inverting a mean in symlog space.
        return (torch.softmax(logits, dim=-1) * symexp(self.bins)).sum(dim=-1)

    def two_hot(self, u: torch.Tensor) -> torch.Tensor:
        y = symlog(u).clamp(BIN_LOW, BIN_HIGH)
        step = (BIN_HIGH - BIN_LOW) / (BINS - 1)
        position = (y - BIN_LOW) / step
        low = position.floor().long().clamp(0, BINS - 2)
        upper_weight = (position - low.float()).clamp(0, 1)
        target = torch.zeros(u.shape[0], BINS, device=u.device)
        target.scatter_(1, low.unsqueeze(1), (1 - upper_weight).unsqueeze(1))
        target.scatter_add_(1, (low + 1).unsqueeze(1), upper_weight.unsqueeze(1))
        return target


class VectorCritic(nn.Module):
    """One two-hot value head per objective component (contracts.OBJECTIVES) on the critic's own encoder. With the
    default weights it reproduces the scalar critic: ``from_scalar`` copies a scalar critic into the welfare head."""

    def __init__(
        self,
        core_width: int,
        width: int = 256,
        privileged_width: int = 0,
        objectives: tuple[str, ...] = contracts.OBJECTIVES,
        token: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = (
            TokenEncoder(core_width, width, privileged_width=privileged_width)
            if token
            else Encoder(core_width, width, privileged_width=privileged_width)
        )
        self.heads = nn.ModuleDict({name: nn.Linear(width, BINS) for name in objectives})
        self.register_buffer("bins", torch.linspace(BIN_LOW, BIN_HIGH, BINS))

    def forward(self, t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h, _ = self.encoder(t)
        return {name: head(h) for name, head in self.heads.items()}

    expectation = Critic.expectation
    two_hot = Critic.two_hot

    @staticmethod
    def from_scalar(critic: "Critic", core_width: int, privileged_width: int = 0) -> "VectorCritic":
        vector = VectorCritic(
            core_width,
            critic.head.in_features,
            privileged_width,
            token=isinstance(critic.encoder, TokenEncoder),
        )
        vector.encoder.load_state_dict(critic.encoder.state_dict())
        vector.heads["welfare"].load_state_dict(critic.head.state_dict())
        return vector


def logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(LOGIT_EPS, 1 - LOGIT_EPS)
    return torch.log(p) - torch.log1p(-p)


def logistic_normal_nll(
    mean: torch.Tensor, log_std: torch.Tensor, label: torch.Tensor
) -> torch.Tensor:
    z = logit(label)
    return 0.5 * ((z - mean) / log_std.exp()) ** 2 + log_std


def mix_target(strategic: torch.Tensor) -> torch.Tensor:
    """Economy mix from the field share s and fishing share f (strategic columns 3 and 4)."""
    s, f = strategic[:, 3], strategic[:, 4]
    return torch.stack([(1 - f) * s, (1 - f) * (1 - s), f], dim=1)


def mix_shares(mix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of ``mix_target``: field share among fields and herds, and the fishing share."""
    rest = mix[:, 0] + mix[:, 1]
    s = np.where(rest > 1e-9, mix[:, 0] / np.maximum(rest, 1e-12), 0.5)
    return s, mix[:, 2]


def deterministic_level(mean: torch.Tensor) -> np.ndarray:
    """Decode unit-interval levels exactly as the deterministic actor does."""
    values = torch.sigmoid(mean).detach().cpu().numpy()
    values[values < 0.01] = 0.0
    values[values > 0.99] = 1.0
    return values


def deterministic_mix(logits: torch.Tensor) -> np.ndarray:
    """Decode the economy simplex using the actor's native float32 operations."""
    mix = torch.softmax(logits, dim=1).detach().cpu().numpy()
    mix[mix < 0.01] = 0.0
    return mix / np.maximum(mix.sum(axis=1, keepdims=True), 1e-12)


def validate_deterministic_level_image(
    values: list[float] | tuple[float, ...] | np.ndarray,
) -> None:
    """Require canonical float32 values in the deterministic level image."""
    supplied = np.asarray(values)
    if supplied.ndim != 1 or not np.issubdtype(supplied.dtype, np.number):
        raise ValueError("deterministic levels are malformed")
    decoded = supplied.astype(np.float32)
    if not np.isfinite(decoded).all() or any(
        float(item) != original for item, original in zip(decoded, supplied.tolist(), strict=True)
    ):
        raise ValueError("deterministic levels are not canonical float32 values")
    if np.any(
        ((decoded > np.float32(0.0)) & (decoded < np.float32(0.01)))
        | ((decoded > np.float32(0.99)) & (decoded < np.float32(1.0)))
    ):
        raise ValueError("deterministic levels are outside the actor decoder image")


def mix_soft_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy of the masked softmax against a simplex target restricted to the available coordinates.
    Returns per-row losses and the rows where the target carries mass on an available coordinate."""
    target = target * mask.float()
    mass = target.sum(dim=1)
    target = target / mass.clamp(min=1e-12).unsqueeze(1)
    loss = -(target * torch.log_softmax(logits, dim=1)).sum(dim=1)
    return loss, mass > 1e-9


def mix_log_prob(
    logits: torch.Tensor,
    log_std: torch.Tensor,
    mix: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Logistic-normal density of a sampled mix (for PPO): additive log-ratios against the last available
    coordinate are Gaussian with mean logits_i - logits_ref and std exp(log_std_i). Boundary points are pulled
    eps into the simplex; rows with one available coordinate have log-probability zero."""
    m = mask.float()
    p = mix.clamp(min=eps) * m
    p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-12)
    ref = (torch.arange(mask.shape[1], device=mask.device) * m).argmax(dim=1, keepdim=True)
    z = torch.log(p.clamp(min=1e-12)) - torch.log(p.gather(1, ref).clamp(min=1e-12))
    mu = logits - logits.gather(1, ref)
    others = m.clone()
    others.scatter_(1, ref, 0.0)
    std = log_std.exp()
    gauss = -0.5 * ((z - mu) / std) ** 2 - log_std - 0.5 * float(np.log(2 * np.pi))
    jacobian = -torch.log(p.clamp(min=1e-12))  # change of variables from log-ratios to the simplex
    return ((gauss + jacobian) * others).sum(dim=1) + (jacobian * (m - others)).sum(dim=1) * (
        others.sum(dim=1) > 0
    ).float()


def grouped_region_slot(logits: torch.Tensor) -> torch.Tensor:
    """Hierarchical choice over the region head: first the decision (native, stay, or go: keep the intent or a region
    row 1-7) with the most probability mass, then the most likely slot within it. A plain argmax picks native whenever
    no single destination outweighs it, even when moving somewhere is more likely than not."""
    p = torch.softmax(logits, dim=1)
    go = torch.zeros(p.shape[1], dtype=torch.bool, device=p.device)
    go[contracts.REGION_KEEP] = True
    go[3:] = True  # slot 2 is region row 0, the current tile (stay)
    masses = torch.stack([p[:, contracts.REGION_NATIVE], p[:, 2], (p * go).sum(dim=1)], dim=1)
    best_go = torch.where(go, p, torch.full_like(p, -1.0)).argmax(dim=1)
    group = masses.argmax(dim=1)
    return torch.where(
        group == 0,
        torch.full_like(best_go, contracts.REGION_NATIVE),
        torch.where(group == 1, torch.full_like(best_go, 2), best_go),
    )


def act_deterministic(
    out: dict[str, torch.Tensor],
    region: bool = False,
    grouped: bool = False,
    selection: str | None = None,
) -> dict[str, np.ndarray]:
    """The mode used for closed-loop evaluation of the cloned policy (levels snap to the unit-interval ends). With
    ``region`` (contract 10 learned movement) the region slot is chosen first and the mix given it: the hierarchical
    head's own choice (region_mode); on a flat head ``grouped`` makes it grouped_region_slot."""
    if region and "region" in out:
        if selection == "argmax":
            slot = out.get("region_logp", out["region"]).argmax(dim=1)
        elif selection == "stay":
            slot = torch.full(
                (len(out["region"]),), REGION_STAY, dtype=torch.long, device=out["region"].device
            )
        elif selection is None:
            slot = region_mode(out, grouped)
        else:
            raise ValueError(f"unknown diagnostic region selection {selection!r}")
    else:
        slot = torch.zeros(len(out["mix_logits"]), dtype=torch.long)
    if region:
        out = condition_mix(out, slot)

    mix = deterministic_mix(out["mix_logits"])
    s, f = mix_shares(mix)
    return {
        "strategic": np.concatenate(
            [
                deterministic_level(out["strategic_mean"]),
                s[:, None],
                f[:, None],
            ],
            axis=1,
        ),
        "mix": mix,
        "service": deterministic_level(out["service_mean"]),
        "clothing": out["clothing"].argmax(dim=1).cpu().numpy(),
        "shelter": out["shelter"].argmax(dim=1).cpu().numpy(),
        "fission": out["fission"].argmax(dim=1).cpu().numpy(),
        "movement": out["movement"].argmax(dim=1).cpu().numpy() - 1,
        "span": out["span"].argmax(dim=1).cpu().numpy()
        if "span" in out
        else np.full(len(mix), contracts.SPAN_SINGLE),
        "region": slot.cpu().numpy(),
        # Contract 11: relative values (clamped), the most likely gate, and the export share (snapped to 0 below 0.01).
        "trade_value": np.clip(
            np.exp(out["trade_value_mean"].detach().cpu().numpy()), *contracts.TRADE_VALUE_RANGE
        ),
        "trade_gate": out["trade_gate"].argmax(dim=2).cpu().numpy(),
        "trade_share": np.where(
            torch.sigmoid(out["trade_share_mean"]).detach().cpu().numpy() < 0.01,
            0.0,
            torch.sigmoid(out["trade_share_mean"]).detach().cpu().numpy(),
        ),
    }


def bridge_actions_from_mode(
    mode: dict[str, np.ndarray],
    *,
    movement: bool,
    region_movement: bool,
    commitments: bool,
    learned_fission: bool,
    learned_trade: bool,
) -> list[dict[str, Any]]:
    """Map an already decoded actor mode to the bridge schema.

    The Gate-1 decoder receipt replays this same helper.  Keeping the live and
    evidence paths on one implementation avoids attempting to invert the two
    rounded economy shares back into the actor's hidden three-simplex.
    """
    rows = len(mode["strategic"])
    if not commitments:
        spans = np.full(rows, contracts.SPAN_SINGLE)
    else:
        spans = mode["span"]
    actions: list[dict[str, Any]] = []
    for index in range(rows):
        code = contracts.NATIVE_MOVEMENT
        if movement:
            code = (
                region_code(int(mode["region"][index]))
                if region_movement
                else int(mode["movement"][index])
            )
        action: dict[str, Any] = {
            "strategic": mode["strategic"][index].tolist(),
            "service": mode["service"][index].tolist(),
            "clothingRoute": int(mode["clothing"][index]),
            "shelterRoute": int(mode["shelter"][index]),
            "movement": code,
            "fission": int(mode["fission"][index]) if learned_fission else -1,
            "commit": contracts.SPANS[int(spans[index])],
        }
        if learned_trade:
            action.update(
                {
                    "tradeValue": mode["trade_value"][index].tolist(),
                    "tradeGate": [int(gate) for gate in mode["trade_gate"][index]],
                    "tradeShare": float(mode["trade_share"][index]),
                }
            )
        actions.append(action)
    return actions


def region_code(slot: int) -> int:
    """Bridge movement code of a region-movement slot: native, keep intent, or region row."""
    if slot == contracts.REGION_NATIVE:
        return contracts.NATIVE_MOVEMENT
    if slot == contracts.REGION_KEEP:
        return contracts.KEEP_INTENT
    return contracts.REGION_BASE + slot - 2
