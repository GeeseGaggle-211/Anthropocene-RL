"""Critic-only privileged state for the labor critics (intervention A, learner side).

The simulator (``feature/r5-privileged``, ``ElderModel9Privileged``) emits a
versioned view of hidden economic state on every village row, captured at the
same decision boundary as the observation (row schema 1: reward/outcome fields
belong to the executed transition, observation and privileged view to the next
decision). This module is the only learner path that reads it:

* :class:`LaborPrivilegedLayout` is the persisted family layout (rows, widths,
  feature names, schema version); :data:`LABOR_PRIVILEGED_LAYOUT_V1` pins
  simulator ``PrivilegedSchemaVersion = 1``.
* :func:`privileged_raw_batch` parses bridge rows into raw per-family arrays,
  the same arrays teacher/DAgger/search shards store (``privileged_<family>``,
  ``privileged_<family>_present``, ``privileged_schema_version``).
* :class:`LaborPrivilegedNormalizer` holds frozen per-family moments fitted on
  fitting worlds only and turns raw arrays into the critic-only batch
  (``privileged__<family>`` / ``privileged__<family>_present``). That batch is
  its own dictionary; it is never merged into the actor batch, and
  :func:`assert_actor_batch` rejects any privileged key at the actor boundary.

Nothing here is an actor input. The actor encoder refuses a privileged layout
in its constructor (``labor_model.LaborObservationEncoder``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .labor_worlds import RESERVED_HELDOUT_WORLDS, require_training_worlds

PRIVILEGED_SCHEMA_VERSION = 1
PRIVILEGED_ROW_SCHEMA_VERSION = 1
PRIVILEGED_LAYOUT_SEMANTICS = "model9-critic-only-privileged-view-v1"
PRIVILEGED_NORMALIZER_KIND = "labor-privileged-normalizer-v1"
PRIVILEGED_NORMALIZER_TRANSFORM = "symlog-then-frozen-per-family-affine-guard-10-dead-zero"
# Raw shard arrays (unnormalized, as the bridge emits them).
SHARD_PREFIX = "privileged_"
SHARD_SCHEMA_KEY = "privileged_schema_version"
# Critic-only normalized batch keys.
CRITIC_PREFIX = "privileged__"
# Any tensor key with this prefix is privileged and may never reach the actor.
FORBIDDEN_ACTOR_PREFIX = "privileged"
# Same numeric guard as the bounded actor normalizer (model.BOUNDED_*).
GUARD = 10.0
STD_FLOOR = 0.1
DEAD_STD = 1e-3

# World partitions (G0 manifest section 4 and amendment r5-b). Reserved fitting
# worlds never train any network (single definition in labor_worlds);
# calibration worlds never fit moments.
G0_CALIBRATION_WORLDS = range(90180, 90200)


@dataclass(frozen=True)
class PrivilegedFamily:
    """One typed family: ``cap`` ordinary rows plus an optional overflow-aggregate row."""

    name: str
    cap: int
    overflow: bool
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.isidentifier()
            or self.name != self.name.lower()
        ):
            raise ValueError("privileged family name must be a lowercase identifier")
        if isinstance(self.cap, bool) or not isinstance(self.cap, int) or self.cap < 1:
            raise ValueError(f"privileged family {self.name} cap must be a positive integer")
        if not isinstance(self.overflow, bool):
            raise ValueError(f"privileged family {self.name} overflow must be boolean")
        features = tuple(self.features)
        if not features or any(not isinstance(f, str) or not f for f in features):
            raise ValueError(f"privileged family {self.name} needs named features")
        if len(set(features)) != len(features):
            raise ValueError(f"privileged family {self.name} repeats a feature name")
        object.__setattr__(self, "features", features)

    @property
    def rows(self) -> int:
        return self.cap + (1 if self.overflow else 0)

    @property
    def width(self) -> int:
        return len(self.features)

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cap": self.cap,
            "overflowRow": self.overflow,
            "rows": self.rows,
            "width": self.width,
            "features": list(self.features),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> PrivilegedFamily:
        if not isinstance(state, Mapping) or set(state) != {
            "name",
            "cap",
            "overflowRow",
            "rows",
            "width",
            "features",
        }:
            raise ValueError("privileged family fields are malformed")
        family = cls(
            str(state["name"]), state["cap"], state["overflowRow"], tuple(state["features"])
        )
        if family.rows != state["rows"] or family.width != state["width"]:
            raise ValueError(
                f"privileged family {family.name} rows/width disagree with its features"
            )
        return family


@dataclass(frozen=True)
class LaborPrivilegedLayout:
    """Versioned privileged block: per-family rows, widths, names and presence."""

    schema_version: int
    families: tuple[PrivilegedFamily, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version < 1
        ):
            raise ValueError("privileged schema version must be a positive integer")
        families = tuple(self.families)
        if not families or any(not isinstance(f, PrivilegedFamily) for f in families):
            raise ValueError("privileged layout needs at least one family")
        if len({f.name for f in families}) != len(families):
            raise ValueError("privileged layout repeats a family")
        object.__setattr__(self, "families", families)

    @property
    def total_rows(self) -> int:
        return sum(f.rows for f in self.families)

    @property
    def total_width(self) -> int:
        """Total privileged scalars (zero means no privileged block)."""
        return sum(f.rows * f.width for f in self.families)

    def family(self, name: str) -> PrivilegedFamily:
        for family in self.families:
            if family.name == name:
                return family
        raise KeyError(name)

    def state_dict(self) -> dict[str, Any]:
        return {
            "semantics": PRIVILEGED_LAYOUT_SEMANTICS,
            "schemaVersion": self.schema_version,
            "rowSchemaVersion": PRIVILEGED_ROW_SCHEMA_VERSION,
            "families": [f.state_dict() for f in self.families],
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> LaborPrivilegedLayout:
        if not isinstance(state, Mapping) or set(state) != {
            "semantics",
            "schemaVersion",
            "rowSchemaVersion",
            "families",
        }:
            raise ValueError("privileged layout fields are malformed")
        if state["semantics"] != PRIVILEGED_LAYOUT_SEMANTICS:
            raise ValueError("privileged layout semantics are unsupported")
        if state["rowSchemaVersion"] != PRIVILEGED_ROW_SCHEMA_VERSION:
            raise ValueError("privileged row schema version is unsupported")
        layout = cls(
            state["schemaVersion"], tuple(PrivilegedFamily.from_state(f) for f in state["families"])
        )
        layout.require_supported()
        return layout

    @classmethod
    def from_bridge(cls, layout: Mapping[str, Any]) -> LaborPrivilegedLayout:
        """The simulator's ``ElderModel9Privileged.Layout()`` (bridge schema or teacher meta)."""
        if not isinstance(layout, Mapping) or set(layout) != {"schemaVersion", "families"}:
            raise ValueError("bridge privileged layout is malformed")
        families = []
        for raw in layout["families"]:
            if not isinstance(raw, Mapping) or set(raw) != {
                "name",
                "cap",
                "overflowRow",
                "rows",
                "width",
                "features",
            }:
                raise ValueError("bridge privileged family is malformed")
            families.append(PrivilegedFamily.from_state(raw))
        return cls(int(layout["schemaVersion"]), tuple(families))

    def require_supported(self) -> None:
        """Only reviewed schema versions are accepted; a mismatch is never adapted."""
        if self.schema_version != PRIVILEGED_SCHEMA_VERSION:
            raise ValueError(
                f"privileged schema version {self.schema_version} != supported "
                f"{PRIVILEGED_SCHEMA_VERSION}"
            )
        if self != LABOR_PRIVILEGED_LAYOUT_V1:
            raise ValueError("privileged layout differs from the reviewed schema-1 layout")

    def digest(self) -> str:
        from .provenance import digest

        return digest(self.state_dict())


_DOMAINS = ("hunting", "gathering", "fishing")
_TECHNOLOGIES = ("husbandry", "cultivation", "watercraft")
_GOODS = (
    "HandTools",
    "Pots",
    "Cloth",
    "FarmTools",
    "Housing",
    "Tools",
    "AnimalFabric",
    "LightHousing",
    "HerdingTools",
    "HousingMaterial",
    "FishingTools",
)
_ROLES = (
    "Gatherer",
    "Farmer",
    "Herder",
    "Fisher",
    "Craftsperson",
    "Toolmaker",
    "Potter",
    "Weaver",
    "Carpenter",
    "Builder",
    "Woodcutter",
    "Stonecutter",
    "ClayMiner",
)

# Copied from simulator feature/r5-privileged 8bd480b (ElderModel9Privileged.cs and
# Docs/ElderModel9Contract7.md). ``test_labor_privileged`` checks it against the
# live bridge's advertised layout.
LABOR_PRIVILEGED_LAYOUT_V1 = LaborPrivilegedLayout(
    1,
    (
        PrivilegedFamily(
            "core",
            1,
            False,
            (
                "log-population",
                "episode-age-turns",
                "site-food-over-meals",
                "footprint-potential-food-over-meals",
                "footprint-food-over-meals",
                "sustainable-herd-over-animals",
                "absolute-herd-over-animals",
                "last-growth",
                "daughters",
                "fixed-housing-per-resident",
            ),
        ),
        PrivilegedFamily(
            "fields",
            16,
            True,
            (
                "row-count",
                "distance-hex",
                "operated-by-camp",
                "foreign-operated",
                "access",
                "capital-E",
                "intensity-q",
                "operating-fraction",
                "food-stock",
                "biomass-stock",
                "land-condition",
                "vegetation-tier",
                "mean-soil-moisture",
                "mean-max-food",
                "mean-food-recovery-per-day",
                "crop-suitability",
                "expansion-available",
                "maintenance-farmers",
                "farm-tool-demand",
            ),
        ),
        PrivilegedFamily(
            "herds",
            8,
            True,
            (
                "row-count",
                "animals",
                "species-index",
                "growth-rate-per-year",
                "food-per-animal-interval",
                "water-per-animal-interval",
                "starvation-mortality",
                "growth-remainder",
                "death-remainder",
                "food-nutrition",
                "water-nutrition",
                "nutrition",
                "unmet-forage",
                "unmet-water",
                "deficit-turns",
                "deficit-severity",
                "managed",
                "management-fulfilment",
                "breeding-intensity",
                "shared-zero-displacement-forage-capacity",
                "shared-absolute-forage-capacity",
                "shared-water-capacity",
                "animal-food-potential",
                "animal-food-produced",
            ),
        ),
        PrivilegedFamily(
            "unlocks",
            1,
            True,
            (
                "row-count",
                "era-neolithic",
                *(f"first-technology-{t}" for t in _TECHNOLOGIES),
                *(f"acquired-{t}" for t in _TECHNOLOGIES),
                *(f"lifetime-practice-{d}" for d in _DOMAINS),
                *(f"neolithic-practice-{d}" for d in _DOMAINS),
                *(f"exposure-{d}" for d in _DOMAINS),
                *(f"progress-over-threshold-{d}" for d in _DOMAINS),
                *(f"withheld-{t}" for t in _TECHNOLOGIES),
                "forced-transition",
                "secondary-domain-plus-one",
                "era-elapsed-years",
                "neolithic-technologies",
                "unlocked-specialists",
            ),
        ),
        PrivilegedFamily(
            "economy",
            1,
            True,
            (
                "row-count",
                "residents",
                "linked-population",
                "food",
                "plant-food",
                "animal-food",
                "aquatic-food",
                *(f"stock-{g}" for g in _GOODS),
                "stone",
                "clay",
                "wood",
                *(f"recipe-available-{g}" for g in _GOODS),
                "farm-tool-farmer-years",
                "meals-per-turn",
                "clothing-need-units",
                "housing-capacity-people",
                "cloth-per-capita",
                "housing-per-capita",
                *(f"workers-{r}" for r in _ROLES),
                "trade-partners",
                "partner-food",
                *(f"partner-stock-{g}" for g in _GOODS),
            ),
        ),
    ),
)


def shard_keys(layout: LaborPrivilegedLayout = LABOR_PRIVILEGED_LAYOUT_V1) -> tuple[str, ...]:
    """Raw privileged array names a shard must store (plus the schema column)."""
    keys = [SHARD_SCHEMA_KEY]
    for family in layout.families:
        keys.extend((SHARD_PREFIX + family.name, SHARD_PREFIX + family.name + "_present"))
    return tuple(keys)


def critic_keys(layout: LaborPrivilegedLayout = LABOR_PRIVILEGED_LAYOUT_V1) -> tuple[str, ...]:
    keys = []
    for family in layout.families:
        keys.extend((CRITIC_PREFIX + family.name, CRITIC_PREFIX + family.name + "_present"))
    return tuple(keys)


def is_privileged_key(name: str) -> bool:
    return isinstance(name, str) and name.startswith(FORBIDDEN_ACTOR_PREFIX)


def assert_actor_batch(tensors: Mapping[str, Any]) -> None:
    """Fail closed if any privileged tensor reaches an actor input."""
    leaked = sorted(key for key in tensors if is_privileged_key(key))
    if leaked:
        raise AssertionError(f"privileged critic-only inputs reached the actor batch: {leaked}")


def actor_inputs(tensors: Mapping[str, Any]) -> dict[str, Any]:
    """The actor/critic observation tensors of a stored row set, privileged arrays removed."""
    return {key: value for key, value in tensors.items() if not is_privileged_key(key)}


def has_privileged(
    arrays: Mapping[str, Any], layout: LaborPrivilegedLayout = LABOR_PRIVILEGED_LAYOUT_V1
) -> bool:
    return all(key in arrays for key in shard_keys(layout))


def privileged_raw_batch(
    villages: Sequence[Mapping[str, Any]],
    layout: LaborPrivilegedLayout = LABOR_PRIVILEGED_LAYOUT_V1,
) -> dict[str, np.ndarray]:
    """Strictly parse bridge village rows' ``privilegedView`` into raw shard arrays.

    A row without the view, with another schema version, or whose families
    differ from ``layout`` is rejected. Absent rows must be zero.
    """
    layout.require_supported()
    out: dict[str, list[np.ndarray]] = {key: [] for key in shard_keys(layout)}
    for village in villages:
        view = village.get("privilegedView") if isinstance(village, Mapping) else None
        version = village.get("privilegedSchemaVersion") if isinstance(village, Mapping) else None
        if not isinstance(view, Mapping):
            raise ValueError("bridge village row carries no privileged view")
        if version != layout.schema_version or view.get("schemaVersion") != layout.schema_version:
            raise ValueError(
                f"bridge privileged schema {version!r}/{view.get('schemaVersion')!r} "
                f"!= {layout.schema_version}"
            )
        rows, present = view.get("rows"), view.get("present")
        if (
            not isinstance(rows, list)
            or not isinstance(present, list)
            or len(rows) != len(layout.families)
            or len(present) != len(layout.families)
        ):
            raise ValueError("bridge privileged view family count differs from the layout")
        out[SHARD_SCHEMA_KEY].append(np.asarray(version, dtype=np.int64))
        for index, family in enumerate(layout.families):
            values = np.asarray(rows[index], dtype=np.float32)
            mask = np.asarray(present[index], dtype=np.bool_)
            if values.shape != (family.rows * family.width,) or mask.shape != (family.rows,):
                raise ValueError(f"bridge privileged family {family.name} has the wrong shape")
            values = values.reshape(family.rows, family.width)
            if not np.isfinite(values).all():
                raise ValueError(f"bridge privileged family {family.name} is not finite")
            if (values[~mask] != 0).any():
                raise ValueError(f"bridge privileged family {family.name} absent rows must be zero")
            out[SHARD_PREFIX + family.name].append(values)
            out[SHARD_PREFIX + family.name + "_present"].append(mask)
    return {key: np.stack(values) if values else np.zeros((0,)) for key, values in out.items()}


def privileged_row_tensors(
    villages: Sequence[Mapping[str, Any]],
    layout: LaborPrivilegedLayout = LABOR_PRIVILEGED_LAYOUT_V1,
) -> dict[str, torch.Tensor] | None:
    """Raw privileged arrays for stored rows, as tensors; ``None`` for a pre-privileged producer.

    Every row of one decision must carry the view or none may: a partial
    emission is a producer error, not missing data to zero-fill.
    """
    flags = [isinstance(v, Mapping) and v.get("privilegedView") is not None for v in villages]
    if not any(flags):
        return None
    if not all(flags):
        raise ValueError("some bridge village rows lack the privileged view")
    raw = privileged_raw_batch(villages, layout)
    return {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in raw.items()}


def validate_raw(
    arrays: Mapping[str, Any],
    layout: LaborPrivilegedLayout = LABOR_PRIVILEGED_LAYOUT_V1,
    *,
    context: str = "privileged arrays",
) -> int:
    """Check stored raw privileged arrays; returns the row count."""
    layout.require_supported()
    missing = [key for key in shard_keys(layout) if key not in arrays]
    if missing:
        raise ValueError(f"{context} lack privileged data {missing}")

    def numpy(value: Any) -> np.ndarray:
        return (
            value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        )

    versions = numpy(arrays[SHARD_SCHEMA_KEY]).reshape(-1)
    if not len(versions) or (versions != layout.schema_version).any():
        raise ValueError(
            f"{context} privileged schema versions {sorted(set(versions.tolist()))} "
            f"!= {layout.schema_version}"
        )
    for family in layout.families:
        values = numpy(arrays[SHARD_PREFIX + family.name])
        mask = numpy(arrays[SHARD_PREFIX + family.name + "_present"])
        if values.shape != (len(versions), family.rows, family.width) or mask.shape != (
            len(versions),
            family.rows,
        ):
            raise ValueError(f"{context} privileged family {family.name} has the wrong shape")
        if mask.dtype != np.bool_:
            raise ValueError(f"{context} privileged family {family.name} presence must be boolean")
        if not np.isfinite(values).all() or (values[~mask] != 0).any():
            raise ValueError(f"{context} privileged family {family.name} is malformed")
    return len(versions)


@dataclass
class LaborPrivilegedNormalizer:
    """Frozen per-family symlog moments for the critic-only privileged view.

    Fitted once, on fitting worlds only (never sealed, development, reserved
    90154-90159 or calibration worlds), and never updated by training. A
    feature whose transformed std is below ``DEAD_STD`` over the present rows
    is dead and normalizes to zero.
    """

    layout: LaborPrivilegedLayout
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]
    dead: dict[str, np.ndarray]
    present_rows: dict[str, int]
    fit_record: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def fit(
        arrays: Mapping[str, Any],
        *,
        worlds: Iterable[int],
        fit_record: Mapping[str, Any] | None = None,
        layout: LaborPrivilegedLayout = LABOR_PRIVILEGED_LAYOUT_V1,
        excluded_worlds: Iterable[int] = (),
    ) -> LaborPrivilegedNormalizer:
        fit_worlds = sorted({int(w) for w in worlds})
        require_privileged_fit_worlds(fit_worlds, excluded_worlds=excluded_worlds)
        validate_raw(arrays, layout, context="privileged normalizer fit data")
        mean, std, dead, counts = {}, {}, {}, {}
        for family in layout.families:
            values = (
                _numpy(arrays[SHARD_PREFIX + family.name])
                .astype(np.float64)
                .reshape(-1, family.width)
            )
            mask = _numpy(arrays[SHARD_PREFIX + family.name + "_present"]).reshape(-1).astype(bool)
            x = values[mask]
            counts[family.name] = int(mask.sum())
            if len(x):
                transformed = np.sign(x) * np.log1p(np.abs(x))
                m = transformed.mean(axis=0)
                raw_std = transformed.std(axis=0)
            else:
                m = np.zeros(family.width)
                raw_std = np.zeros(family.width)
            mean[family.name] = m.astype(np.float32)
            std[family.name] = np.maximum(raw_std, STD_FLOOR).astype(np.float32)
            dead[family.name] = raw_std < DEAD_STD
        record = {
            "worlds": fit_worlds,
            "rows": int(validate_raw(arrays, layout)),
            **dict(fit_record or {}),
        }
        return LaborPrivilegedNormalizer(layout, mean, std, dead, counts, record)

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": PRIVILEGED_NORMALIZER_KIND,
            "transform": PRIVILEGED_NORMALIZER_TRANSFORM,
            "schemaVersion": self.layout.schema_version,
            "layout": self.layout.state_dict(),
            "guard": GUARD,
            "stdFloor": STD_FLOOR,
            "deadStd": DEAD_STD,
            "frozen": True,
            "families": {
                family.name: {
                    "mean": self.mean[family.name].tolist(),
                    "std": self.std[family.name].tolist(),
                    "dead": self.dead[family.name].astype(bool).tolist(),
                    "presentRows": int(self.present_rows[family.name]),
                }
                for family in self.layout.families
            },
            "fit": dict(self.fit_record),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> LaborPrivilegedNormalizer:
        expected = {
            "kind",
            "transform",
            "schemaVersion",
            "layout",
            "guard",
            "stdFloor",
            "deadStd",
            "frozen",
            "families",
            "fit",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("privileged normalizer fields are malformed")
        if (
            state["kind"] != PRIVILEGED_NORMALIZER_KIND
            or state["transform"] != PRIVILEGED_NORMALIZER_TRANSFORM
        ):
            raise ValueError("privileged normalizer kind/transform is unsupported")
        if (state["guard"], state["stdFloor"], state["deadStd"], state["frozen"]) != (
            GUARD,
            STD_FLOOR,
            DEAD_STD,
            True,
        ):
            raise ValueError("privileged normalizer numeric contract differs")
        layout = LaborPrivilegedLayout.from_state(state["layout"])
        if state["schemaVersion"] != layout.schema_version:
            raise ValueError("privileged normalizer schema version differs from its layout")
        families = state["families"]
        if not isinstance(families, Mapping) or set(families) != {f.name for f in layout.families}:
            raise ValueError("privileged normalizer family inventory differs from its layout")
        mean, std, dead, counts = {}, {}, {}, {}
        for family in layout.families:
            entry = families[family.name]
            if not isinstance(entry, Mapping) or set(entry) != {
                "mean",
                "std",
                "dead",
                "presentRows",
            }:
                raise ValueError(f"privileged normalizer family {family.name} is malformed")
            mean[family.name] = np.asarray(entry["mean"], np.float32)
            std[family.name] = np.asarray(entry["std"], np.float32)
            dead[family.name] = np.asarray(entry["dead"], bool)
            counts[family.name] = int(entry["presentRows"])
            for array in (mean[family.name], std[family.name], dead[family.name]):
                if array.shape != (family.width,):
                    raise ValueError(
                        f"privileged normalizer family {family.name} moment shape differs"
                    )
            if not np.isfinite(mean[family.name]).all() or not np.isfinite(std[family.name]).all():
                raise ValueError("privileged normalizer moments must be finite")
            if (std[family.name] < STD_FLOOR).any():
                raise ValueError("privileged normalizer std is below the declared floor")
        fit = state["fit"]
        if not isinstance(fit, Mapping) or not isinstance(fit.get("worlds"), list):
            raise ValueError("privileged normalizer fit record is malformed")
        require_privileged_fit_worlds(fit["worlds"])
        return cls(layout, mean, std, dead, counts, dict(fit))

    def digest(self) -> str:
        from .provenance import digest

        return digest(
            {"semantics": "labor-privileged-normalizer-binding-v1", "state": self.state_dict()}
        )

    def critic_batch(
        self,
        arrays: Mapping[str, Any],
        rows: Any | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Normalize stored raw privileged arrays into the critic-only batch.

        The result is a separate dictionary keyed ``privileged__<family>`` and
        ``privileged__<family>_present``; callers pass it to a critic as its
        own argument and never merge it into the actor batch.
        """
        out: dict[str, torch.Tensor] = {}
        validate_raw(arrays, self.layout, context="critic privileged batch")
        for family in self.layout.families:
            values = _numpy(arrays[SHARD_PREFIX + family.name])
            mask = _numpy(arrays[SHARD_PREFIX + family.name + "_present"]).astype(bool)
            if rows is not None:
                index = _numpy(rows).astype(np.int64)
                values, mask = values[index], mask[index]
            x = values.astype(np.float64)
            z = (np.sign(x) * np.log1p(np.abs(x)) - self.mean[family.name]) / self.std[family.name]
            z = np.clip(z, -GUARD, GUARD)
            z = np.where(self.dead[family.name], 0.0, z)
            z = np.where(mask[..., None], z, 0.0)
            out[CRITIC_PREFIX + family.name] = torch.as_tensor(
                z, dtype=torch.float32, device=device
            )
            out[CRITIC_PREFIX + family.name + "_present"] = torch.as_tensor(
                mask, dtype=torch.bool, device=device
            )
        return out

    def dead_mask_report(self) -> dict[str, Any]:
        """Which named privileged fields the frozen normalizer silently zeroes."""
        families = {}
        total_dead = 0
        for family in self.layout.families:
            dead = self.dead[family.name]
            names = [
                name for name, flag in zip(family.features, dead.tolist(), strict=True) if flag
            ]
            total_dead += len(names)
            families[family.name] = {
                "present_rows": int(self.present_rows[family.name]),
                "never_present": self.present_rows[family.name] == 0,
                "width": family.width,
                "dead_count": len(names),
                "dead_fields": names,
                "live_fields": [n for n in family.features if n not in set(names)],
            }
        return {
            "schema_version": self.layout.schema_version,
            "fit_worlds": list(self.fit_record.get("worlds", [])),
            "fit_rows": self.fit_record.get("rows"),
            "dead_threshold_symlog_std": DEAD_STD,
            "dead_total": total_dead,
            "fields_total": sum(f.width for f in self.layout.families),
            "families": families,
        }


def dead_mask_audit(
    arrays: Mapping[str, Any], layout: LaborPrivilegedLayout = LABOR_PRIVILEGED_LAYOUT_V1
) -> dict[str, Any]:
    """Data-only audit (no fit-world policy): constant/dead named fields per family.

    Reports, per feature over present rows, whether it is constant in symlog
    space (dead), its constant value when it is, and its nonzero fraction, so
    a silently zeroed producer field is visible before any training.
    """
    count = validate_raw(arrays, layout, context="privileged dead-mask audit data")
    report: dict[str, Any] = {
        "rows": count,
        "schema_version": layout.schema_version,
        "families": {},
    }
    for family in layout.families:
        values = _numpy(arrays[SHARD_PREFIX + family.name]).astype(np.float64)
        mask = _numpy(arrays[SHARD_PREFIX + family.name + "_present"]).astype(bool)
        x = values[mask]
        per_row_presence = mask.mean(axis=0).tolist() if len(mask) else []
        fields = {}
        for column, name in enumerate(family.features):
            column_values = x[:, column] if len(x) else np.zeros(0)
            transformed = np.sign(column_values) * np.log1p(np.abs(column_values))
            symlog_std = float(transformed.std()) if len(column_values) else 0.0
            fields[name] = {
                "dead": bool(symlog_std < DEAD_STD),
                "constant_value": float(column_values[0])
                if len(column_values) and symlog_std < DEAD_STD
                else None,
                "nonzero_fraction": float((column_values != 0).mean())
                if len(column_values)
                else 0.0,
                "min": float(column_values.min()) if len(column_values) else None,
                "max": float(column_values.max()) if len(column_values) else None,
                "symlog_std": symlog_std,
                # Dead in symlog space although the raw values vary: a tiny-unit
                # feature (e.g. per-day rates near 1e-4) the dead mask would zero.
                "varying_but_dead": bool(
                    symlog_std < DEAD_STD
                    and len(column_values)
                    and column_values.min() != column_values.max()
                ),
            }
        report["families"][family.name] = {
            "present_rows": int(mask.sum()),
            "row_presence_fraction": per_row_presence,
            "dead_fields": [n for n, f in fields.items() if f["dead"]],
            "varying_but_dead_fields": [n for n, f in fields.items() if f["varying_but_dead"]],
            "fields": fields,
        }
    report["dead_total"] = sum(len(f["dead_fields"]) for f in report["families"].values())
    report["varying_but_dead_total"] = sum(
        len(f["varying_but_dead_fields"]) for f in report["families"].values()
    )
    return report


def require_privileged_fit_worlds(
    worlds: Iterable[int], *, excluded_worlds: Iterable[int] = ()
) -> None:
    """Privileged moments are fitted on fitting worlds only."""
    from .labor_bc import DEVELOPMENT_WORLDS, SEALED_WORLDS

    excluded = {int(w) for w in excluded_worlds}
    worlds = [int(w) for w in worlds]
    if not worlds:
        raise ValueError("privileged normalizer fit needs at least one fitting world")
    for world in worlds:
        if world in SEALED_WORLDS:
            raise ValueError(f"privileged normalizer refuses sealed world {world}")
        if world in DEVELOPMENT_WORLDS:
            raise ValueError(f"privileged normalizer refuses development world {world}")
        if world in RESERVED_HELDOUT_WORLDS:
            raise ValueError(
                f"privileged normalizer refuses reserved held-out world {world} (90154-90159)"
            )
        if world in G0_CALIBRATION_WORLDS or world in excluded:
            raise ValueError(f"privileged normalizer refuses calibration world {world}")


def require_critic_training_worlds(worlds: Iterable[int], context: str = "critic training") -> None:
    """Reserved held-out and G0 calibration worlds never train a value function."""
    worlds = {int(w) for w in worlds}
    require_training_worlds(worlds, context)
    calibration = sorted(worlds & set(G0_CALIBRATION_WORLDS))
    if calibration:
        raise ValueError(f"{context} refuses G0 calibration worlds {calibration[:5]} (90180-90199)")


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
