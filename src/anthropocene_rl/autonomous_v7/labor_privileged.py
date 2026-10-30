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

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .labor_worlds import RESERVED_HELDOUT_WORLDS, require_training_worlds

PRIVILEGED_SCHEMA_VERSION = 1
PRIVILEGED_ROW_SCHEMA_VERSION = 1
PRIVILEGED_LAYOUT_SEMANTICS = "model9-critic-only-privileged-view-v1"
# Normalizer schema. v1 (legacy, still loadable): symlog on every field and a
# dead rule of symlog std < DEAD_STD. v2 (fitted by default): declared
# per-feature transforms (PRIVILEGED_TRANSFORMS) and a dead rule of "constant
# within float tolerance after the transform".
PRIVILEGED_NORMALIZER_SCHEMA_VERSION = 2
LEGACY_PRIVILEGED_NORMALIZER_KIND = "labor-privileged-normalizer-v1"
PRIVILEGED_NORMALIZER_KIND = "labor-privileged-normalizer-v2"
PRIVILEGED_NORMALIZER_KINDS = (LEGACY_PRIVILEGED_NORMALIZER_KIND, PRIVILEGED_NORMALIZER_KIND)
PRIVILEGED_NORMALIZER_TRANSFORM = "symlog-then-frozen-per-family-affine-guard-10-dead-zero"
PRIVILEGED_NORMALIZER_TRANSFORM_V2 = (
    "declared-per-feature-transform-then-frozen-affine-guard-10-dead-if-constant-within-"
    "float-tolerance"
)
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
DEAD_STD = 1e-3  # v1 dead rule (symlog std); v2 uses DEAD_RELATIVE_TOLERANCE
# v2: a fitted feature is dead iff std(t) <= DEAD_RELATIVE_TOLERANCE * max(1, |mean(t)|)
# over its present rows (t the transformed value): constant up to float rounding.
DEAD_RELATIVE_TOLERANCE = 1e-6

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


# ---------------------------------------------------------------------------
# Declared per-feature transforms (normalizer schema v2)
# ---------------------------------------------------------------------------

# Transform kinds:
# * ``symlog``: sign(x) log1p|x|, fitted affine, std floor STD_FLOOR (the default).
# * ``identity``: x, fitted affine (raw z-score), no floor beyond float tolerance.
# * ``per-year-log1p:S``: log1p(365.25 x / S) for nonnegative per-day rates; S is the
#   per-year reference scale. Fitted affine, floor STD_FLOOR.
# * ``symlog>identity``: symlog when its spread reaches STD_FLOOR, else the raw
#   z-score (for large, nearly constant quantities whose symlog spread is below
#   rounding); the choice is made once at fit time and frozen in the state.
# * ``count-log1p``: log1p(max(x, 0)) with fixed moments (0, 1), never dead -- for
#   structural counts such as the overflow row-count, constant until overflow occurs.
TRANSFORM_KINDS = ("symlog", "identity", "per-year-log1p", "symlog>identity", "count-log1p")
DAYS_PER_YEAR = 365.25
_ROW_COUNT = "count-log1p"
_NEAR_CONSTANT = "symlog>identity"
PRIVILEGED_TRANSFORMS: dict[str, dict[str, str]] = {
    "fields": {
        "row-count": _ROW_COUNT,
        # Per-day regrowth rate (about 1e-4..4e-3 per day); 0.1 per year is the scale.
        "mean-food-recovery-per-day": "per-year-log1p:0.1",
    },
    "herds": {
        "row-count": _ROW_COUNT,
        "food-nutrition": _NEAR_CONSTANT,
        "water-nutrition": _NEAR_CONSTANT,
        "nutrition": _NEAR_CONSTANT,
        "management-fulfilment": _NEAR_CONSTANT,
        "shared-zero-displacement-forage-capacity": _NEAR_CONSTANT,
        "shared-absolute-forage-capacity": _NEAR_CONSTANT,
        "shared-water-capacity": _NEAR_CONSTANT,
    },
    "unlocks": {"row-count": _ROW_COUNT},
    "economy": {
        "row-count": _ROW_COUNT,
        "stone": _NEAR_CONSTANT,
        "clay": _NEAR_CONSTANT,
        "wood": _NEAR_CONSTANT,
    },
}


def _parse_transform(spec: str) -> tuple[str, float | None]:
    kind, _, argument = str(spec).partition(":")
    if kind not in TRANSFORM_KINDS:
        raise ValueError(f"unknown privileged transform {spec!r}")
    if kind == "per-year-log1p":
        scale = float(argument)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"privileged transform {spec!r} needs a positive scale")
        return kind, scale
    if argument:
        raise ValueError(f"privileged transform {spec!r} takes no argument")
    return kind, None


def transform_specs(
    layout: LaborPrivilegedLayout = LABOR_PRIVILEGED_LAYOUT_V1,
    declared: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Declared transform per feature, ``symlog`` unless :data:`PRIVILEGED_TRANSFORMS` says."""
    declared = PRIVILEGED_TRANSFORMS if declared is None else declared
    out = {}
    for family in layout.families:
        names = dict(declared.get(family.name, {}))
        unknown = sorted(set(names) - set(family.features))
        if unknown:
            raise ValueError(f"privileged transforms name unknown {family.name} fields {unknown}")
        specs = tuple(names.get(feature, "symlog") for feature in family.features)
        for spec in specs:
            _parse_transform(spec)
        out[family.name] = specs
    return out


def apply_transform(x: np.ndarray, spec: str) -> np.ndarray:
    """One resolved (non-adaptive) transform on raw float64 values."""
    kind, scale = _parse_transform(spec)
    if kind == "symlog":
        return np.sign(x) * np.log1p(np.abs(x))
    if kind == "identity":
        return x
    if kind == "per-year-log1p":
        return np.sign(x) * np.log1p(np.abs(x) * DAYS_PER_YEAR / scale)
    if kind == "count-log1p":
        return np.log1p(np.maximum(x, 0.0))
    raise ValueError(f"adaptive privileged transform {spec!r} must be resolved at fit time")


def _transform_columns(values: np.ndarray, specs: Sequence[str]) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    for column, spec in enumerate(specs):
        out[..., column] = apply_transform(values[..., column], spec)
    return out


def _floor(spec: str) -> float:
    kind, _ = _parse_transform(spec)
    return 0.0 if kind == "identity" else STD_FLOOR


def _constant_within_tolerance(std: float, mean: float) -> bool:
    return bool(std <= DEAD_RELATIVE_TOLERANCE * max(1.0, abs(mean)))


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
    """Frozen per-feature moments for the critic-only privileged view.

    Fitted once, on fitting worlds only (never sealed, development, reserved
    90154-90159 or calibration worlds), and never updated by training.

    Schema v2 (``version == 2``, the default fit): each feature has a declared
    transform (:data:`PRIVILEGED_TRANSFORMS`; ``resolved`` records the frozen
    choice of an adaptive one); a fitted feature is dead -- normalized to zero
    -- only when it is constant within float tolerance after its transform
    over the present rows. Schema v1 (legacy checkpoints) keeps symlog on
    every field and the symlog-std dead rule exactly.
    """

    layout: LaborPrivilegedLayout
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]
    dead: dict[str, np.ndarray]
    present_rows: dict[str, int]
    fit_record: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    declared: dict[str, tuple[str, ...]] | None = None
    resolved: dict[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        if self.version not in (1, PRIVILEGED_NORMALIZER_SCHEMA_VERSION):
            raise ValueError(f"privileged normalizer schema version {self.version} is unsupported")
        if self.version == 1:
            symlog = {f.name: ("symlog",) * f.width for f in self.layout.families}
            if self.declared is None:
                self.declared = dict(symlog)
            if self.resolved is None:
                self.resolved = dict(symlog)
        if self.declared is None or self.resolved is None:
            raise ValueError("privileged normalizer v2 needs declared and resolved transforms")

    @staticmethod
    def fit(
        arrays: Mapping[str, Any],
        *,
        worlds: Iterable[int],
        fit_record: Mapping[str, Any] | None = None,
        layout: LaborPrivilegedLayout = LABOR_PRIVILEGED_LAYOUT_V1,
        excluded_worlds: Iterable[int] = (),
        version: int = PRIVILEGED_NORMALIZER_SCHEMA_VERSION,
        transforms: Mapping[str, Mapping[str, str]] | None = None,
    ) -> LaborPrivilegedNormalizer:
        fit_worlds = sorted({int(w) for w in worlds})
        require_privileged_fit_worlds(fit_worlds, excluded_worlds=excluded_worlds)
        validate_raw(arrays, layout, context="privileged normalizer fit data")
        if version == 1:
            if transforms is not None:
                raise ValueError("the v1 privileged normalizer has no declared transforms")
            return LaborPrivilegedNormalizer._fit_v1(arrays, fit_worlds, fit_record, layout)
        if version != PRIVILEGED_NORMALIZER_SCHEMA_VERSION:
            raise ValueError(f"cannot fit privileged normalizer schema version {version}")
        declared = transform_specs(layout, transforms)
        mean, std, dead, counts, resolved = {}, {}, {}, {}, {}
        for family in layout.families:
            values = (
                _numpy(arrays[SHARD_PREFIX + family.name])
                .astype(np.float64)
                .reshape(-1, family.width)
            )
            mask = _numpy(arrays[SHARD_PREFIX + family.name + "_present"]).reshape(-1).astype(bool)
            x = values[mask]
            counts[family.name] = int(mask.sum())
            m = np.zeros(family.width)
            sd = np.ones(family.width)
            flags = np.zeros(family.width, dtype=bool)
            chosen = []
            for column, spec in enumerate(declared[family.name]):
                kind, _ = _parse_transform(spec)
                if kind == "symlog>identity":
                    sym = apply_transform(x[:, column], "symlog") if len(x) else np.zeros(0)
                    spec = "symlog" if len(sym) and float(sym.std()) >= STD_FLOOR else "identity"
                chosen.append(spec)
                if kind == "count-log1p":
                    continue  # fixed moments (0, 1); structural, never dead
                if not len(x):
                    flags[column] = True  # never present in the fit data
                    continue
                t = apply_transform(x[:, column], spec)
                column_mean, column_std = float(t.mean()), float(t.std())
                if _constant_within_tolerance(column_std, column_mean):
                    flags[column] = True
                    m[column] = column_mean
                    continue
                m[column] = column_mean
                sd[column] = max(column_std, _floor(spec))
            mean[family.name] = m
            std[family.name] = sd
            dead[family.name] = flags
            resolved[family.name] = tuple(chosen)
        record = {
            "worlds": fit_worlds,
            "rows": int(validate_raw(arrays, layout)),
            **dict(fit_record or {}),
        }
        return LaborPrivilegedNormalizer(
            layout, mean, std, dead, counts, record, PRIVILEGED_NORMALIZER_SCHEMA_VERSION,
            declared, resolved,
        )

    @staticmethod
    def _fit_v1(
        arrays: Mapping[str, Any],
        fit_worlds: list[int],
        fit_record: Mapping[str, Any] | None,
        layout: LaborPrivilegedLayout,
    ) -> LaborPrivilegedNormalizer:
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
        return LaborPrivilegedNormalizer(layout, mean, std, dead, counts, record, 1)

    def state_dict(self) -> dict[str, Any]:
        families = {
            family.name: {
                "mean": self.mean[family.name].tolist(),
                "std": self.std[family.name].tolist(),
                "dead": self.dead[family.name].astype(bool).tolist(),
                "presentRows": int(self.present_rows[family.name]),
            }
            for family in self.layout.families
        }
        if self.version == 1:
            return {
                "kind": LEGACY_PRIVILEGED_NORMALIZER_KIND,
                "transform": PRIVILEGED_NORMALIZER_TRANSFORM,
                "schemaVersion": self.layout.schema_version,
                "layout": self.layout.state_dict(),
                "guard": GUARD,
                "stdFloor": STD_FLOOR,
                "deadStd": DEAD_STD,
                "frozen": True,
                "families": families,
                "fit": dict(self.fit_record),
            }
        for family in self.layout.families:
            families[family.name]["declaredTransforms"] = list(self.declared[family.name])
            families[family.name]["resolvedTransforms"] = list(self.resolved[family.name])
        return {
            "kind": PRIVILEGED_NORMALIZER_KIND,
            "normalizerSchemaVersion": PRIVILEGED_NORMALIZER_SCHEMA_VERSION,
            "transform": PRIVILEGED_NORMALIZER_TRANSFORM_V2,
            "schemaVersion": self.layout.schema_version,
            "layout": self.layout.state_dict(),
            "guard": GUARD,
            "stdFloor": STD_FLOOR,
            "deadRelativeTolerance": DEAD_RELATIVE_TOLERANCE,
            "daysPerYear": DAYS_PER_YEAR,
            "frozen": True,
            "families": families,
            "fit": dict(self.fit_record),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> LaborPrivilegedNormalizer:
        if not isinstance(state, Mapping):
            raise ValueError("privileged normalizer fields are malformed")
        if state.get("kind") == LEGACY_PRIVILEGED_NORMALIZER_KIND:
            return cls._from_state_v1(state)
        expected = {
            "kind",
            "normalizerSchemaVersion",
            "transform",
            "schemaVersion",
            "layout",
            "guard",
            "stdFloor",
            "deadRelativeTolerance",
            "daysPerYear",
            "frozen",
            "families",
            "fit",
        }
        if set(state) != expected:
            raise ValueError("privileged normalizer fields are malformed")
        if (
            state["kind"] != PRIVILEGED_NORMALIZER_KIND
            or state["normalizerSchemaVersion"] != PRIVILEGED_NORMALIZER_SCHEMA_VERSION
            or state["transform"] != PRIVILEGED_NORMALIZER_TRANSFORM_V2
        ):
            raise ValueError("privileged normalizer kind/schema/transform is unsupported")
        numeric = (
            state["guard"],
            state["stdFloor"],
            state["deadRelativeTolerance"],
            state["daysPerYear"],
            state["frozen"],
        )
        if numeric != (GUARD, STD_FLOOR, DEAD_RELATIVE_TOLERANCE, DAYS_PER_YEAR, True):
            raise ValueError("privileged normalizer numeric contract differs")
        layout = LaborPrivilegedLayout.from_state(state["layout"])
        if state["schemaVersion"] != layout.schema_version:
            raise ValueError("privileged normalizer schema version differs from its layout")
        families = state["families"]
        if not isinstance(families, Mapping) or set(families) != {f.name for f in layout.families}:
            raise ValueError("privileged normalizer family inventory differs from its layout")
        mean, std, dead, counts, declared, resolved = {}, {}, {}, {}, {}, {}
        keys = {"mean", "std", "dead", "presentRows", "declaredTransforms", "resolvedTransforms"}
        for family in layout.families:
            entry = families[family.name]
            if not isinstance(entry, Mapping) or set(entry) != keys:
                raise ValueError(f"privileged normalizer family {family.name} is malformed")
            mean[family.name] = np.asarray(entry["mean"], np.float64)
            std[family.name] = np.asarray(entry["std"], np.float64)
            dead[family.name] = np.asarray(entry["dead"], bool)
            counts[family.name] = int(entry["presentRows"])
            declared[family.name] = tuple(str(x) for x in entry["declaredTransforms"])
            resolved[family.name] = tuple(str(x) for x in entry["resolvedTransforms"])
            for array in (mean[family.name], std[family.name], dead[family.name]):
                if array.shape != (family.width,):
                    raise ValueError(
                        f"privileged normalizer family {family.name} moment shape differs"
                    )
            if len(declared[family.name]) != family.width or len(resolved[family.name]) != (
                family.width
            ):
                raise ValueError(f"privileged normalizer family {family.name} transforms differ")
            for spec, chosen, sd, flag in zip(
                declared[family.name],
                resolved[family.name],
                std[family.name].tolist(),
                dead[family.name].tolist(),
                strict=True,
            ):
                kind, _ = _parse_transform(spec)
                resolved_kind, _ = _parse_transform(chosen)
                allowed = ("symlog", "identity") if kind == "symlog>identity" else (kind,)
                if resolved_kind not in allowed or (kind != "symlog>identity" and chosen != spec):
                    raise ValueError("privileged normalizer resolved transform differs")
                if not flag and (sd <= 0.0 or sd < _floor(chosen)):
                    raise ValueError("privileged normalizer std is below the declared floor")
            if not np.isfinite(mean[family.name]).all() or not np.isfinite(std[family.name]).all():
                raise ValueError("privileged normalizer moments must be finite")
        fit = state["fit"]
        if not isinstance(fit, Mapping) or not isinstance(fit.get("worlds"), list):
            raise ValueError("privileged normalizer fit record is malformed")
        require_privileged_fit_worlds(fit["worlds"])
        return cls(
            layout, mean, std, dead, counts, dict(fit), PRIVILEGED_NORMALIZER_SCHEMA_VERSION,
            declared, resolved,
        )

    @classmethod
    def _from_state_v1(cls, state: Mapping[str, Any]) -> LaborPrivilegedNormalizer:
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
        if set(state) != expected:
            raise ValueError("privileged normalizer fields are malformed")
        if state["transform"] != PRIVILEGED_NORMALIZER_TRANSFORM:
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
        return cls(layout, mean, std, dead, counts, dict(fit), 1)

    def digest(self) -> str:
        from .provenance import digest

        return digest(
            {"semantics": "labor-privileged-normalizer-binding-v1", "state": self.state_dict()}
        )

    def transformed(self, family: PrivilegedFamily, values: np.ndarray) -> np.ndarray:
        """The frozen (resolved) per-feature transform of raw values ``[..., width]``."""
        return _transform_columns(values.astype(np.float64), self.resolved[family.name])

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
            if self.version == 1:
                t = np.sign(x) * np.log1p(np.abs(x))
            else:
                t = self.transformed(family, x)
            z = (t - self.mean[family.name]) / self.std[family.name]
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
                "transforms": {
                    name: chosen
                    for name, spec, chosen in zip(
                        family.features,
                        self.declared[family.name],
                        self.resolved[family.name],
                        strict=True,
                    )
                    if spec != "symlog"
                },
            }
        return {
            "schema_version": self.layout.schema_version,
            "normalizer_schema_version": self.version,
            "fit_worlds": list(self.fit_record.get("worlds", [])),
            "fit_rows": self.fit_record.get("rows"),
            "dead_rule": (
                f"symlog std < {DEAD_STD}"
                if self.version == 1
                else "std(transformed) <= "
                f"{DEAD_RELATIVE_TOLERANCE} * max(1, |mean(transformed)|) (constant within float "
                "tolerance after the declared transform)"
            ),
            "dead_threshold_symlog_std": DEAD_STD if self.version == 1 else None,
            "dead_total": total_dead,
            "fields_total": sum(f.width for f in self.layout.families),
            "families": families,
        }


def dead_mask_audit(
    arrays: Mapping[str, Any],
    layout: LaborPrivilegedLayout = LABOR_PRIVILEGED_LAYOUT_V1,
    transforms: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Data-only audit (no fit-world policy): constant/dead named fields per family.

    Per feature over present rows it reports the declared and resolved
    transform, whether the field is dead under the v2 rule (constant within
    float tolerance after its transform) and under the legacy v1 symlog rule,
    a reason for every dead field, its constant value, range and nonzero
    fraction -- so a silently zeroed producer field is visible before training.
    """
    count = validate_raw(arrays, layout, context="privileged dead-mask audit data")
    declared = transform_specs(layout, transforms)
    report: dict[str, Any] = {
        "rows": count,
        "schema_version": layout.schema_version,
        "normalizer_schema_version": PRIVILEGED_NORMALIZER_SCHEMA_VERSION,
        "dead_rule": "std(transformed) <= "
        f"{DEAD_RELATIVE_TOLERANCE} * max(1, |mean(transformed)|)",
        "legacy_dead_rule": f"symlog std < {DEAD_STD}",
        "families": {},
    }
    for family in layout.families:
        values = _numpy(arrays[SHARD_PREFIX + family.name]).astype(np.float64)
        mask = _numpy(arrays[SHARD_PREFIX + family.name + "_present"]).astype(bool)
        x = values[mask]
        per_row_presence = mask.mean(axis=0).tolist() if len(mask) else []
        overflow_present = int(mask[:, -1].sum()) if family.overflow and len(mask) else 0
        fields = {}
        for column, name in enumerate(family.features):
            column_values = x[:, column] if len(x) else np.zeros(0)
            n = len(column_values)
            spec = declared[family.name][column]
            kind, _ = _parse_transform(spec)
            symlog = np.sign(column_values) * np.log1p(np.abs(column_values))
            symlog_std = float(symlog.std()) if n else 0.0
            chosen = spec
            if kind == "symlog>identity":
                chosen = "symlog" if n and symlog_std >= STD_FLOOR else "identity"
            t = apply_transform(column_values, chosen) if n else np.zeros(0)
            t_mean = float(t.mean()) if n else 0.0
            t_std = float(t.std()) if n else 0.0
            varies = bool(n and column_values.min() != column_values.max())
            if kind == "count-log1p":
                dead, reason = False, None
            elif not n:
                dead, reason = True, "never-present"
            elif _constant_within_tolerance(t_std, t_mean):
                dead = True
                reason = (
                    "varies-only-at-float-rounding" if varies else "constant-in-data"
                )
            else:
                dead, reason = False, None
            legacy_dead = bool(symlog_std < DEAD_STD)
            fields[name] = {
                "declared_transform": spec,
                "transform": chosen,
                "dead": dead,
                "reason": reason,
                "legacy_dead": legacy_dead,
                "constant_value": float(column_values[0]) if n and not varies else None,
                "nonzero_fraction": float((column_values != 0).mean()) if n else 0.0,
                "min": float(column_values.min()) if n else None,
                "max": float(column_values.max()) if n else None,
                "raw_std": float(column_values.std()) if n else 0.0,
                "symlog_std": symlog_std,
                "transformed_std": t_std,
                # Dead in symlog space although the raw values vary: a tiny-unit
                # feature (e.g. per-day rates near 1e-4) the v1 dead mask zeroed.
                "varying_but_dead": bool(legacy_dead and varies),
                "rescued_by_v2": bool(legacy_dead and not dead),
            }
        report["families"][family.name] = {
            "present_rows": int(mask.sum()),
            "row_presence_fraction": per_row_presence,
            "overflow_rows_present": overflow_present,
            "dead_fields": [n for n, f in fields.items() if f["dead"]],
            "legacy_dead_fields": [n for n, f in fields.items() if f["legacy_dead"]],
            "varying_but_dead_fields": [n for n, f in fields.items() if f["varying_but_dead"]],
            "rescued_fields": [n for n, f in fields.items() if f["rescued_by_v2"]],
            "fields": fields,
        }
    families = report["families"].values()
    report["dead_total"] = sum(len(f["dead_fields"]) for f in families)
    report["legacy_dead_total"] = sum(len(f["legacy_dead_fields"]) for f in families)
    report["varying_but_dead_total"] = sum(len(f["varying_but_dead_fields"]) for f in families)
    report["rescued_total"] = sum(len(f["rescued_fields"]) for f in families)
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


# ---------------------------------------------------------------------------
# Learner-derived lineage features (critic-only, ``initialize --privileged-lineage``)
# ---------------------------------------------------------------------------
#
# The settlement-guard cost exempts young daughters (parentSociety > 0 and at
# most YOUNG_DAUGHTER_TURNS after founding), but no critic input says "this camp
# is a daughter" or "founded k turns ago": fission-age resets for parent and
# daughter alike, episode-age is the world clock and ``daughters`` counts
# children. The lineage family supplies both, derived by the learner (the
# simulator's privileged schema-1 layout is unchanged):
#
# * ``is-daughter`` in {0, 1};
# * ``turns-since-founding-over-4`` = min(turn - founded_turn, 4) / 4 in [0, 1].
#
# Both are bounded by construction, so they bypass the privileged normalizer
# (identity, no fitted moments): the critic batch carries the raw values as one
# always-present token row ``privileged__lineage`` [N, 1, 2].
#
# Offline (stored rows, no parentSociety): a camp is a daughter iff its founding
# turn -- the stored ``founded_turn`` when a branch dataset has one, else its
# first row turn -- is after its world's first row turn. Camps present at the
# world's first window turn therefore count as originals (and their age starts
# at that turn); :func:`offline_lineage` reports how many rows that covers.
# Online (rollouts and the T4/G5 trace): ``is-daughter`` = parentSociety > 0 (the
# guard's own test) and the founding turn is the rollout's tracked
# ``founded_turns`` (first turn the camp was seen; originals at reset). On a
# fresh episode both derivations agree row for row (tested).

LINEAGE_FAMILY = "lineage"
LINEAGE_FEATURES = ("is-daughter", "turns-since-founding-over-4")
LINEAGE_CAP_TURNS = 4
LINEAGE_SEMANTICS = "labor-critic-lineage-v1"
# Raw (already bounded) per-row values [N, 2] float32; never an actor input.
LINEAGE_RAW_KEY = SHARD_PREFIX + LINEAGE_FAMILY
LINEAGE_CRITIC_KEYS = (CRITIC_PREFIX + LINEAGE_FAMILY, CRITIC_PREFIX + LINEAGE_FAMILY + "_present")
# A row r is a guard-exempt young daughter iff next_turn - founded <= 3, i.e. its
# own age (turn - founded) <= 2 (YOUNG_DAUGHTER_TURNS = 3 is applied at the outcome turn).
YOUNG_DAUGHTER_ROW_AGE = 2


def lineage_spec() -> dict[str, Any]:
    """The checkpoint record of the lineage family (stored only when enabled)."""
    return {
        "semantics": LINEAGE_SEMANTICS,
        "features": list(LINEAGE_FEATURES),
        "capTurns": LINEAGE_CAP_TURNS,
        "rows": 1,
        "normalization": "bounded-identity-bypasses-privileged-normalizer",
        "offline": (
            "daughter iff founding turn (stored founded_turn, else first row turn) > "
            "world's first row turn"
        ),
        "online": "daughter iff parentSociety > 0; founding turn from the rollout's founded_turns",
    }


def lineage_values(is_daughter: Any, turns_since: Any) -> np.ndarray:
    """[N, 2] float32 lineage features from daughter flags and turns since founding."""
    daughter = np.asarray(is_daughter, dtype=bool).reshape(-1)
    age = np.asarray(turns_since, dtype=np.int64).reshape(-1)
    if len(daughter) != len(age):
        raise ValueError("lineage flags and ages differ in length")
    capped = np.clip(age, 0, LINEAGE_CAP_TURNS).astype(np.float32) / float(LINEAGE_CAP_TURNS)
    return np.stack([daughter.astype(np.float32), capped], axis=1).astype(np.float32)


def offline_lineage(
    worlds: Any, camps: Any, turns: Any, founded_turn: Any | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Lineage features for stored rows, derived once per dataset (see the notes above)."""
    w = _numpy(worlds).reshape(-1).astype(np.int64)
    c = _numpy(camps).reshape(-1).astype(np.int64)
    t = _numpy(turns).reshape(-1).astype(np.int64)
    n = len(w)
    if len(c) != n or len(t) != n:
        raise ValueError("lineage derivation needs equal-length row_world/row_camp/row_turn")
    world_first: dict[int, int] = {}
    for world, turn in zip(w.tolist(), t.tolist(), strict=True):
        if turn < world_first.get(world, turn + 1):
            world_first[world] = turn
    if founded_turn is not None:
        founded = _numpy(founded_turn).reshape(-1).astype(np.int64)
        if len(founded) != n:
            raise ValueError("stored founded_turn length differs from the rows")
        source = "stored-founded_turn"
    else:
        first: dict[tuple[int, int], int] = {}
        for world, camp, turn in zip(w.tolist(), c.tolist(), t.tolist(), strict=True):
            key = (world, camp)
            if turn < first.get(key, turn + 1):
                first[key] = turn
        founded = np.asarray(
            [first[key] for key in zip(w.tolist(), c.tolist(), strict=True)], np.int64
        )
        source = "first-row-turn"
    start = np.asarray([world_first[world] for world in w.tolist()], np.int64)
    daughter = founded > start
    age = t - founded
    originals = ~daughter
    stats = {
        "rows": int(n),
        "founding_source": source,
        "daughter_rows": int(daughter.sum()),
        # Edge case: camps present at the world's first window turn count as originals.
        "window_original_rows": int(originals.sum()),
        "window_original_camps": len(
            set(zip(w[originals].tolist(), c[originals].tolist(), strict=True))
        ),
        "young_daughter_rows": int((daughter & (age <= YOUNG_DAUGHTER_ROW_AGE)).sum()),
        "negative_age_rows": int((age < 0).sum()),
    }
    return lineage_values(daughter, age), stats


def online_lineage(
    villages: Sequence[Mapping[str, Any]], founded_turns: Mapping[int, int], turn: int
) -> np.ndarray:
    """Lineage features for live bridge rows at decision ``turn`` (rollout/trace path).

    A camp not yet in ``founded_turns`` is founded now (age 0), the same rule the
    rollout's ``founded_turns.setdefault`` applies.
    """
    daughter, age = [], []
    for village in villages:
        camp = int(village["camp"])
        daughter.append(int(village.get("parentSociety", 0) or 0) > 0)
        age.append(int(turn) - int(founded_turns.get(camp, int(turn))))
    return lineage_values(daughter, age)


def ensure_lineage(arrays: Any) -> np.ndarray:
    """The dataset's lineage features, derived once and cached under :data:`LINEAGE_RAW_KEY`.

    ``arrays`` is a full stored dataset (the derivation needs every row of each
    world) or a live batch that already carries the key. The derivation stats are
    recorded in ``arrays["collection"]["lineage"]`` when there is a collection record.
    """
    if LINEAGE_RAW_KEY in arrays:
        values = _numpy(arrays[LINEAGE_RAW_KEY])
    else:
        missing = [k for k in ("row_world", "row_camp", "row_turn") if k not in arrays]
        if missing:
            raise ValueError(
                f"lineage critic needs {LINEAGE_RAW_KEY} or the row identity arrays {missing}"
            )
        values, stats = offline_lineage(
            arrays["row_world"], arrays["row_camp"], arrays["row_turn"],
            arrays.get("founded_turn"),
        )
        if not isinstance(arrays, dict):
            raise ValueError("lineage derivation needs a mutable dataset dictionary to cache into")
        arrays[LINEAGE_RAW_KEY] = torch.from_numpy(values)
        if isinstance(arrays.get("collection"), dict):
            arrays["collection"]["lineage"] = stats
    if values.ndim != 2 or values.shape[1] != len(LINEAGE_FEATURES):
        raise ValueError(f"{LINEAGE_RAW_KEY} must be [rows, {len(LINEAGE_FEATURES)}]")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError(f"{LINEAGE_RAW_KEY} must lie in [0, 1]")
    return values


def lineage_critic_batch(
    arrays: Any, rows: Any | None = None, *, device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor]:
    """The critic-only lineage token batch (identity: the features are bounded)."""
    values = ensure_lineage(arrays)
    if rows is not None:
        values = values[_numpy(rows).astype(np.int64)]
    out = torch.as_tensor(values, dtype=torch.float32, device=device).reshape(-1, 1, 2)
    return {
        LINEAGE_CRITIC_KEYS[0]: out,
        LINEAGE_CRITIC_KEYS[1]: torch.ones(len(out), 1, dtype=torch.bool, device=device),
    }
