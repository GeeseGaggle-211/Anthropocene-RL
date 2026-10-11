"""Stable semantic names for every checked-in schema-v2 tensor slot."""

from __future__ import annotations

from anthropocene_rl.contracts import TENSOR_FEATURE_COUNT

CAPABILITY_ROWS = (
    "field-retention",
    "herd-management",
    "resident-fishing",
    "fixed-construction",
    "local-trade",
)
MEMORY_ROWS = (
    "qol",
    "food-security",
    "water-security",
    "food-reserves",
    "population-pressure",
    "realized-migration",
    "field-pressure",
    "herd-pressure",
    "fishing-pressure",
    "realized-trade-utility",
    "realized-development",
)
COMMODITY_ROWS = (
    "hand-tools",
    "pots",
    "cloth",
    "farm-tools",
    "housing",
    "tools",
    "animal-fabric",
    "light-housing",
    "herding-tools",
    "housing-material",
    "fishing-tools",
)


def feature_names() -> tuple[str, ...]:
    names = [f"reserved.slot-{index}" for index in range(TENSOR_FEATURE_COUNT)]

    def put(offset: int, block: str, columns: tuple[str, ...], row: str | None = None) -> None:
        prefix = block if row is None else f"{block}.{row}"
        for column, name in enumerate(columns):
            names[offset + column] = f"{prefix}.{name}"

    put(
        0,
        "core",
        (
            "population",
            "connected-consumers",
            "previous-interval-years",
            "next-interval-years",
            "deficit-age-years",
            "qol",
            "food-needs-met",
            "water-needs-met",
            "clothing-coverage",
            "housing-coverage",
            "food-reserve-years",
            "movement-age-years",
            "fission-age-years",
            "realized-migration-per-capita-year",
            "is-village",
            "realized-trade-utility-per-capita-year",
        ),
    )
    put(
        16,
        "fields",
        (
            "retained-tiles",
            "current-operation",
            "maintenance-worker-years",
            "current-food-support",
            "future-food-support",
            "rebuild-burden",
            "retained-capital-service",
            "current-intensity",
        ),
    )
    put(
        24,
        "herd-summary",
        (
            "animals",
            "groups",
            "nutrition",
            "management-fulfilment",
            "zero-displacement-capacity",
            "absolute-capacity",
            "water-capacity",
            "current-management-worker-years",
            "current-animal-food",
            "human-food-displaced",
            "birth-trend",
            "mortality-trend",
            "slaughter-trend",
        ),
    )
    put(
        40,
        "previous-decision",
        (
            "present",
            "requested-field-retention",
            "accepted-field-retention",
            "realized-field-operation",
            "requested-herd-retention",
            "accepted-herd-retention",
            "realized-herd-retention",
            "requested-development-budget",
            "accepted-development-budget",
            "realized-development-budget",
            "requested-field-share",
            "accepted-field-share",
            "realized-field-share",
            "intervention-count",
            "elapsed-years",
            "movement-accepted",
        ),
    )
    put(
        56,
        "fission-history",
        (
            "candidate-present",
            "recent-split",
            "pressure",
            "expected-stayer-gain",
            "expected-daughter-qol",
            "exit-opportunity-pressure",
            "recent-daughter-population",
        ),
    )
    for row, capability in enumerate(CAPABILITY_ROWS):
        put(64 + 3 * row, "capabilities", ("available", "experience", "progress"), capability)
    put(
        80,
        "continuous-action-mask",
        ("field-retention", "herd-retention", "field-development", "herd-development"),
    )
    for row, signal in enumerate(MEMORY_ROWS):
        put(84 + 3 * row, "institutional-memory", ("20y", "60y", "180y"), signal)
    put(117, "institutional-memory-maturity", ("20y", "60y", "180y"))
    site_columns = (
        "current",
        "known",
        "human-food",
        "human-food-after-herd",
        "herd-zero-displacement",
        "herd-absolute",
        "herd-water",
        "fixed-shelter",
        "recurring-labor",
        "absolute-recurring-labor",
        "retained-field-service",
        "lost-rebuild-burden",
    )
    for row in range(7):
        put(128 + 12 * row, "sites", site_columns, str(row))
    fishing_columns = (
        "distance",
        "incumbent",
        "known",
        "yield-per-fisher-year",
        "food-potential",
        "biomass",
        "condition",
        "current-capital",
        "current-intensity",
        "operation-fraction",
        "maintenance-worker-years",
        "reserved",
    )
    for row in range(3):
        put(212 + 12 * row, "fishing", fishing_columns, str(row))
    herd_columns = (
        "animals",
        "nutrition",
        "management-fulfilment",
        "food-per-animal",
        "water-per-animal",
        "animal-food-per-animal",
        "birth-rate",
        "mortality-rate",
    )
    for row in range(8):
        put(248 + 8 * row, "herd-groups", herd_columns, str(row))
    commodity_columns = (
        "stock",
        "local-need",
        "production-capacity",
        "protected-floor",
        "recent-import",
        "recent-export",
        "can-produce",
        "can-trade",
    )
    for row, commodity in enumerate(COMMODITY_ROWS):
        put(312 + 8 * row, "commodities", commodity_columns, commodity)
    contact_columns = (
        "distance",
        "age-years",
        "affinity",
        "recent-trade-utility-per-capita-year",
        "recent-trade-flow-per-capita-year",
        "recent-migration-per-capita-year",
        "kin",
        "currently-visible",
    )
    for row in range(8):
        put(400 + 8 * row, "contacts", contact_columns, str(row))
    for row, block in enumerate(("sites", "fishing", "herds", "commodities", "contacts")):
        put(464 + 3 * row, "presence-overflow", ("present", "omitted", "overflow"), block)
    put(480, "movement-action-mask", ("enabled", "legal-candidate-count"))
    for candidate in range(8):
        names[482 + candidate] = f"movement-candidate-mask.{candidate}"
    if len(set(names)) != TENSOR_FEATURE_COUNT:
        raise AssertionError("Schema-v2 feature names must be unique and exhaustive")
    return tuple(names)


FEATURE_NAMES = feature_names()
