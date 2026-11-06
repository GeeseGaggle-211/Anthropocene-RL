from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7 import gate1_diagnostics as diagnostic
from anthropocene_rl.autonomous_v7.provenance import digest


def economic(*, transition: str, years: float, person_years: float, qol: float) -> dict:
    goods = [{"good": name, "unit": "good-units", "output": 0.0} for name in diagnostic.CRAFT_GOODS]
    materials = [
        {"material": name, "unit": "material-units", "received": 0.0}
        for name in diagnostic.MATERIALS
    ]
    trade = {"settledTrades": 0, "realisedUtility": 0.0}
    for name in diagnostic.TRADE_GOODS:
        trade["imported" + name] = 0.0
        trade["exported" + name] = 0.0
    return {
        "schemaVersion": 1,
        "transitionId": transition,
        "trajectoryId": "a" * 64,
        "decisionTurn": 1,
        "campId": 1,
        "societyId": 1,
        "years": years,
        "domains": {
            "population": {
                "personYears": person_years,
                "qol": qol,
                "openingResidents": 2,
                "openingParties": 0,
                "openingLinked": 2,
                "reportStartingLinked": 2,
                "reportEndingLinked": 2,
                "afterReintegrationResidents": 2,
                "afterReintegrationParties": 0,
                "splitOff": 0,
                "emigrated": 0,
                "immigrated": 0,
                "endingResidents": 2,
                "endingParties": 0,
                "endingLinked": 2,
            },
            "cultivation": {"foodDelivered": 0.0, "funnels": []},
            "herding": {"acceptedFood": 0.0, "funnels": []},
            "manufacturing": {"funnels": []},
            "trade": trade,
        },
        "units": diagnostic.economic_unit_registry(),
        "unavailable": {},
        "herds": [],
        "goods": goods,
        "materials": materials,
        "requests": [],
        "trades": [],
        "transfers": [],
        "fields": [],
        "conservation": {},
        "parentTransitionId": None,
        "forkId": None,
        "branchId": None,
    }


def observation(core_width: int = 1) -> dict:
    return {
        "core": [1.0] * core_width,
        "corePresent": [True] * core_width,
        "sets": [[0.0] * (rows * width) for _, rows, width in contracts.SETS],
        "rowPresent": [[False] * rows for _, rows, _ in contracts.SETS],
        "movementMask": [True] * contracts.MOVEMENT_SLOTS,
        "regionMovementMask": [True] * contracts.REGION_MOVEMENT_SLOTS,
        "clothingRouteMask": [True, True],
        "shelterRouteMask": [True, True],
        "economyMixMask": [True] * len(contracts.ECONOMY_MIX),
        "commitHoldMask": True,
    }


def action() -> dict:
    return {
        "strategic": [0.5] * len(contracts.STRATEGIC),
        "service": [float(np.float32(0.2))] * len(contracts.SERVICES),
        "clothingRoute": 0,
        "shelterRoute": 0,
        "movement": contracts.NATIVE_MOVEMENT,
        "fission": -1,
        "commit": 1,
    }


SEMANTICS = {
    "experts": False,
    "token": False,
    "commitments": True,
    "learned_movement": False,
    "region_movement": False,
    "learned_fission": False,
    "learned_trade": False,
    "region_head": "hierarchical",
    "grouped_region": False,
    "region_selection": None,
}


def _logit(value: float) -> float:
    if value <= 0:
        return -100.0
    if value >= 1:
        return 100.0
    return float(np.log(value / (1.0 - value)))


def decoder_receipt(
    value: dict,
    obs: dict | None = None,
    semantics: dict | None = None,
    *,
    mode: str = "deterministic",
    decoder: str = "native-contract-12",
    camp: int = 1,
    batch_rows: int = 1,
    row: int = 0,
    raw_override: dict[str, torch.Tensor] | None = None,
    return_raw: bool = False,
) -> dict | tuple[dict, dict, torch.Tensor | None]:
    """Build a sealed decoder-input fixture; validation still performs the forward replay."""
    obs = copy.deepcopy(obs or observation())
    semantics = copy.deepcopy(semantics or SEMANTICS)
    field, fishing = value["strategic"][-2:]
    rest = np.float32(1.0) - np.float32(fishing)
    mix = np.asarray(
        [
            [
                rest * np.float32(field),
                rest * (np.float32(1.0) - np.float32(field)),
                np.float32(fishing),
            ]
        ],
        dtype=np.float32,
    )
    if mode == "deterministic":

        def scores(width: int, selected: int) -> torch.Tensor:
            result = torch.full((1, width), -10.0, dtype=torch.float32)
            result[0, selected] = 10.0
            return result

        raw = {
            "strategic_mean": torch.tensor(
                [[*map(_logit, value["strategic"][:-2])]], dtype=torch.float32
            ),
            "mix_logits": torch.log(torch.tensor(mix).clamp_min(torch.finfo(torch.float32).tiny)),
            "service_mean": torch.tensor([[*map(_logit, value["service"])]], dtype=torch.float32),
            "clothing": scores(2, value["clothingRoute"]),
            "shelter": scores(2, value["shelterRoute"]),
            "fission": scores(
                contracts.FISSION_POLICIES, value["fission"] if semantics["learned_fission"] else 0
            ),
            "movement": scores(
                contracts.MOVEMENT_SLOTS,
                value["movement"] + 1
                if semantics["learned_movement"] and not semantics["region_movement"]
                else 0,
            ),
            "span": scores(len(contracts.SPANS), contracts.SPANS.index(value["commit"])),
            "trade_value_mean": torch.zeros((1, len(contracts.TRADE_GROUPS)), dtype=torch.float32),
            "trade_gate": torch.zeros(
                (1, len(contracts.TRADE_GROUPS), contracts.TRADE_GATES), dtype=torch.float32
            ),
            "trade_share_mean": torch.tensor(
                [_logit(value.get("tradeShare", 0.0))], dtype=torch.float32
            ),
        }
        if semantics["learned_trade"]:
            raw["trade_value_mean"] = torch.log(
                torch.tensor([value["tradeValue"]], dtype=torch.float32)
            )
            for index, gate in enumerate(value["tradeGate"]):
                raw["trade_gate"][0, index, gate] = 10.0
        if semantics["learned_movement"] and semantics["region_movement"]:
            if value["movement"] == contracts.NATIVE_MOVEMENT:
                slot = contracts.REGION_NATIVE
            elif value["movement"] == contracts.KEEP_INTENT:
                slot = contracts.REGION_KEEP
            else:
                slot = value["movement"] - contracts.REGION_BASE + 2
            raw.update(
                {
                    "region": scores(contracts.REGION_MOVEMENT_SLOTS, slot),
                    "region_logp": scores(contracts.REGION_MOVEMENT_SLOTS, slot),
                    "mix_offset": torch.zeros(
                        (1, contracts.REGION_MOVEMENT_SLOTS, len(contracts.ECONOMY_MIX))
                    ),
                }
            )
            if semantics["region_head"] == "hierarchical":
                from anthropocene_rl.autonomous_v7.model import region_group_index

                group = int(region_group_index()[slot])
                raw["region_group"] = scores(3, group)
        mask = None
    else:
        movement = (
            -1
            if not semantics["learned_movement"] or semantics["region_movement"]
            else value["movement"] + 1
        )
        if semantics["learned_movement"] and semantics["region_movement"]:
            if value["movement"] == contracts.NATIVE_MOVEMENT:
                region = contracts.REGION_NATIVE
            elif value["movement"] == contracts.KEEP_INTENT:
                region = contracts.REGION_KEEP
            else:
                region = value["movement"] - contracts.REGION_BASE + 2
        else:
            region = contracts.REGION_NATIVE
        raw = {
            "region": torch.tensor([region]),
            "strategic_z": torch.tensor(
                [[*map(_logit, value["strategic"][:-2])]], dtype=torch.float32
            ),
            "service_z": torch.tensor([[*map(_logit, value["service"])]], dtype=torch.float32),
            "mix_z": torch.log(torch.tensor(mix).clamp_min(torch.finfo(torch.float32).tiny)),
            "clothing": torch.tensor([value["clothingRoute"]]),
            "shelter": torch.tensor([value["shelterRoute"]]),
            "movement": torch.tensor([movement]),
            "span": torch.tensor([contracts.SPANS.index(value["commit"])]),
            "fission": torch.tensor([value["fission"] if semantics["learned_fission"] else -1]),
        }
        if semantics["learned_trade"]:
            raw.update(
                {
                    "trade_value_z": torch.log(
                        torch.tensor([value["tradeValue"]], dtype=torch.float32)
                    ),
                    "trade_gate": torch.tensor([value["tradeGate"]]),
                    "trade_share_z": torch.tensor([_logit(value["tradeShare"])]),
                }
            )
        mask = torch.tensor([obs["economyMixMask"]], dtype=torch.bool)
    if raw_override:
        raw.update(raw_override)
    if batch_rows != 1:
        raw = {
            name: tensor.repeat((batch_rows, *([1] * (tensor.ndim - 1))))
            for name, tensor in raw.items()
        }
        if mask is not None:
            mask = mask.repeat((batch_rows, 1))
    receipt = diagnostic.action_decoder_receipt(
        mode=mode,
        decoder=decoder,
        semantics=semantics,
        observation=obs,
        emitted_action=value,
        camp=camp,
        raw=raw,
        row=row,
        economy_mix_mask=mask,
    )
    return (receipt, raw, mask) if return_raw else receipt


def forward_action_and_receipt(
    value: dict, obs: dict, semantics: dict, camp: int = 1, batch_rows: int = 1, row: int = 0
) -> tuple[dict, dict]:
    """Use fixture logits only as inputs, then retain the production decoder's actual output."""
    _receipt, raw, _mask = decoder_receipt(
        value,
        obs,
        semantics,
        camp=camp,
        batch_rows=batch_rows,
        row=row,
        return_raw=True,
    )
    from anthropocene_rl.autonomous_v7.model import act_deterministic, bridge_actions_from_mode

    mode = act_deterministic(
        raw,
        region=semantics["learned_movement"] and semantics["region_movement"],
        grouped=semantics["grouped_region"],
        selection=semantics["region_selection"],
    )
    actual = bridge_actions_from_mode(
        mode,
        movement=semantics["learned_movement"],
        region_movement=semantics["region_movement"],
        commitments=semantics["commitments"],
        learned_fission=semantics["learned_fission"],
        learned_trade=semantics["learned_trade"],
    )[row]
    receipt = diagnostic.action_decoder_receipt(
        mode="deterministic",
        decoder="native-contract-12",
        semantics=semantics,
        observation=obs,
        emitted_action=actual,
        camp=camp,
        raw=raw,
        row=row,
    )
    return actual, receipt


def validation_kwargs() -> dict:
    return {"expected_opening_turn": 1, "observation_core_width": 1, "policy_semantics": SEMANTICS}


def test_integrals_are_hand_computable_across_fission_extinction_and_variable_duration():
    # Parent then parent+daughter then the daughter's terminal interval.  Rows
    # are authoritative committed transition records; no final-state estimate.
    rows = [
        economic(transition="parent-1", years=20, person_years=2000, qol=0.5),
        economic(transition="parent-2", years=5, person_years=350, qol=0.8),
        economic(transition="daughter-2", years=5, person_years=150, qol=0.8),
        economic(transition="daughter-terminal", years=2, person_years=20, qol=0.25),
    ]
    assert diagnostic.integrate_economic_rows(rows) == {
        "population_years": 2520.0,
        "qol_years": 20 * 0.5 + 5 * 0.8 + 5 * 0.8 + 2 * 0.25,
    }


def test_integrals_reject_non_authoritative_units_and_nonfinite_values():
    row = economic(transition="x", years=20, person_years=10, qol=0.5)
    row["units"]["domains.population.personYears"] = "people"
    with pytest.raises(ValueError, match="authoritative unit"):
        diagnostic.integrate_economic_rows([row])
    row = economic(transition="x", years=20, person_years=10, qol=0.5)
    row["domains"]["population"]["qol"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        diagnostic.integrate_economic_rows([row])


def test_complete_service_registry_preserves_identities_units_zero_and_missingness():
    row = economic(transition="x", years=20, person_years=10, qol=0.5)
    row["domains"]["cultivation"]["foodDelivered"] = 0.0
    observed = diagnostic.service_outcomes([row], diagnostic.service_outcome_registry())
    assert observed["cultivation.food-delivered"] == {"value": 0.0, "availability_reason": None}
    assert observed["manufacturing.HandTools.output"]["value"] == 0.0
    assert observed["materials.Wood.received"]["value"] == 0.0
    assert observed["trading.Wood.exported"]["value"] == 0.0
    del row["domains"]["cultivation"]["foodDelivered"]
    missing = diagnostic.service_outcomes([row], diagnostic.service_outcome_registry())
    assert missing["cultivation.food-delivered"]["value"] is None
    assert missing["cultivation.food-delivered"]["availability_reason"]


def test_compiler_interventions_keep_named_raw_rows_without_double_counting():
    row = economic(transition="x", years=20, person_years=10, qol=0.5)
    row["domains"]["cultivation"]["funnels"] = [
        {"metric": "fieldFood", "constraintReasons": ["labour"]},
    ]
    row["requests"] = [
        {"requestId": "r1", "rawMaterial": None, "unmetReason": "storage"},
        {"requestId": "r2", "rawMaterial": "Wood", "unmetReason": "source"},
    ]
    total, summary = diagnostic.compiler_interventions([row])
    assert total == 3
    assert summary["by_domain_reason"] == [
        {"domain": "cultivation", "reason": "labour", "count": 1},
        {"domain": "manufacturing", "reason": "storage", "count": 1},
        {"domain": "materials", "reason": "source", "count": 1},
    ]
    assert len(summary["raw_rows"]) == 3


def test_trajectory_digest_normalizes_unordered_economic_rows_but_is_action_sensitive():
    first = economic(transition="x", years=20, person_years=10, qol=0.5)
    first["goods"][0]["output"] = 2.0
    village = {
        "camp": 1,
        "society": 1,
        "parentSociety": 0,
        "priorPopulation": 2,
        "population": 2,
        "qol": 0.5,
        "years": 20.0,
        "welfare": 1.0,
        "tradeUtility": 0.0,
        "terminated": False,
        "founded": False,
        "observation": observation(),
    }
    village["observation_sha256"] = digest(village["observation"])
    opening = {"turn": 1, "fingerprint": "a" * 64, "villages": [village]}
    successor = copy.deepcopy(opening)
    successor["turn"] = 2
    successor["fingerprint"] = "b" * 64
    step = {
        "index": 0,
        "transition_id": digest({"transition": 1}),
        "source_turn": 1,
        "source_fingerprint": "a" * 64,
        "actions": [
            {
                "camp": 1,
                "action_request": action(),
                "decoded_action": action(),
                "decoder_receipt": decoder_receipt(action()),
            }
        ],
        "policy_rng_receipt": None,
        "successor": successor,
        "terminal": {"terminated": False, "truncated": True, "reason": "evaluation-horizon"},
        "economic": [first],
    }
    bounds = {
        "max_steps": 1,
        "max_villages_per_boundary": 8,
        "max_economic_rows_per_step": 8,
        "max_canonical_bytes": 1_000_000,
    }
    reordered = copy.deepcopy(step)
    reordered["economic"][0]["goods"].reverse()
    kwargs = {"world_id": "world", "snapshot_sha256": ""}
    assert diagnostic.trajectory_digest(
        opening, [step], bounds, **kwargs
    ) == diagnostic.trajectory_digest(
        opening,
        [reordered],
        bounds,
        **kwargs,
    )
    changed = copy.deepcopy(step)
    changed["actions"][0]["action_request"]["service"][0] = 0.3
    changed["actions"][0]["decoded_action"]["service"][0] = 0.3
    assert diagnostic.trajectory_digest(
        opening, [step], bounds, **kwargs
    ) != diagnostic.trajectory_digest(
        opening,
        [changed],
        bounds,
        **kwargs,
    )
    tiny = dict(bounds)
    tiny["max_canonical_bytes"] = 1
    evidence = diagnostic.trajectory_evidence(opening, [step], tiny, **kwargs)
    with pytest.raises(ValueError, match="capture bound"):
        diagnostic.validate_trajectory_evidence(
            evidence,
            horizon=1,
            bounds=tiny,
            action_mode="deterministic",
            rng_steps=None,
            service_fields=diagnostic.service_outcome_registry(),
            **validation_kwargs(),
            **kwargs,
        )


def test_complete_early_terminal_trajectory_recomputes_zero_population_boundary():
    opening_village = {
        "camp": 1,
        "society": 1,
        "parentSociety": 0,
        "priorPopulation": 2,
        "population": 2,
        "qol": 0.5,
        "years": 20.0,
        "welfare": 0.0,
        "tradeUtility": 0.0,
        "terminated": False,
        "founded": False,
        "observation": observation(),
    }
    opening_village["observation_sha256"] = digest(opening_village["observation"])
    opening = {"turn": 1, "fingerprint": "a" * 64, "villages": [opening_village]}
    terminal_village = copy.deepcopy(opening_village)
    terminal_village.update(
        population=0,
        priorPopulation=2,
        terminated=True,
        observation=None,
        observation_sha256=None,
        welfare=3.0,
        tradeUtility=-0.25,
    )
    successor = {"turn": 2, "fingerprint": "b" * 64, "villages": [terminal_village]}
    raw = economic(transition="terminal-transition", years=20.0, person_years=20.0, qol=0.5)
    raw["domains"]["population"].update(
        reportEndingLinked=0,
        afterReintegrationResidents=0,
        endingResidents=0,
        endingLinked=0,
    )
    raw["domains"]["trade"]["realisedUtility"] = -0.25
    transition_id = digest(
        {
            "index": 0,
            "source_turn": 1,
            "source_fingerprint": "a" * 64,
            "successor_turn": 2,
            "successor_fingerprint": "b" * 64,
        }
    )
    step = {
        "index": 0,
        "transition_id": transition_id,
        "source_turn": 1,
        "source_fingerprint": "a" * 64,
        "actions": [
            {
                "camp": 1,
                "action_request": action(),
                "decoded_action": action(),
                "decoder_receipt": decoder_receipt(action()),
            }
        ],
        "policy_rng_receipt": None,
        "successor": successor,
        "terminal": {"terminated": True, "truncated": False, "reason": "all-villages-terminated"},
        "economic": [raw],
    }
    bounds = {
        "max_steps": 2,
        "max_villages_per_boundary": 8,
        "max_economic_rows_per_step": 8,
        "max_canonical_bytes": 1_000_000,
    }
    evidence = diagnostic.trajectory_evidence(
        opening,
        [step],
        bounds,
        world_id="world",
        snapshot_sha256="",
    )
    result = diagnostic.validate_trajectory_evidence(
        evidence,
        world_id="world",
        snapshot_sha256="",
        horizon=2,
        bounds=bounds,
        action_mode="deterministic",
        rng_steps=None,
        service_fields=diagnostic.service_outcome_registry(),
        **validation_kwargs(),
    )
    assert result["evaluation_steps"] == 1
    assert result["ending_residents"] == result["settlements"] == 0
    assert result["extinctions"] == 1
    assert result["welfare"] == 3.0
    assert result["trade_utility"] == -0.25


def test_complete_fission_transition_requires_explicit_founded_identity_witness():
    parent = {
        "camp": 1,
        "society": 1,
        "parentSociety": 0,
        "priorPopulation": 2,
        "population": 2,
        "qol": 0.5,
        "years": 20.0,
        "welfare": 0.0,
        "tradeUtility": 0.0,
        "terminated": False,
        "founded": False,
        "observation": observation(),
    }
    parent["observation_sha256"] = digest(parent["observation"])
    opening = {"turn": 1, "fingerprint": "a" * 64, "villages": [parent]}
    successor_parent = copy.deepcopy(parent)
    successor_parent.update(priorPopulation=2, population=1, welfare=1.0)
    daughter = copy.deepcopy(parent)
    daughter.update(
        camp=2,
        society=2,
        parentSociety=1,
        priorPopulation=0,
        population=1,
        years=0.0,
        founded=True,
        welfare=0.0,
    )
    successor = {"turn": 2, "fingerprint": "b" * 64, "villages": [successor_parent, daughter]}
    raw = economic(transition="parent-fission", years=20.0, person_years=20.0, qol=0.5)
    raw["domains"]["population"]["endingResidents"] = 1
    raw["domains"]["population"]["splitOff"] = 1
    raw["domains"]["population"]["reportEndingLinked"] = 2
    raw["domains"]["population"].update(
        afterReintegrationResidents=2,
        endingParties=0,
        endingLinked=1,
    )
    raw["transfers"] = [
        {
            "kind": "fission",
            "phase": "close-after-demography",
            "sourceCamp": 1,
            "destinationCamp": 2,
            "people": 1,
            "animals": 0,
        }
    ]
    transition_id = digest(
        {
            "index": 0,
            "source_turn": 1,
            "source_fingerprint": "a" * 64,
            "successor_turn": 2,
            "successor_fingerprint": "b" * 64,
        }
    )
    steps = [
        {
            "index": 0,
            "transition_id": transition_id,
            "source_turn": 1,
            "source_fingerprint": "a" * 64,
            "actions": [
                {
                    "camp": 1,
                    "action_request": action(),
                    "decoded_action": action(),
                    "decoder_receipt": decoder_receipt(action()),
                }
            ],
            "policy_rng_receipt": None,
            "successor": successor,
            "terminal": {"terminated": False, "truncated": True, "reason": "evaluation-horizon"},
            "economic": [raw],
        }
    ]
    bounds = {
        "max_steps": 1,
        "max_villages_per_boundary": 8,
        "max_economic_rows_per_step": 8,
        "max_canonical_bytes": 1_000_000,
    }
    evidence = diagnostic.trajectory_evidence(
        opening,
        steps,
        bounds,
        world_id="world",
        snapshot_sha256="",
    )
    result = diagnostic.validate_trajectory_evidence(
        evidence,
        world_id="world",
        snapshot_sha256="",
        horizon=1,
        bounds=bounds,
        action_mode="deterministic",
        rng_steps=None,
        service_fields=diagnostic.service_outcome_registry(),
        **validation_kwargs(),
    )
    assert result["ending_residents"] == 2
    assert result["settlements"] == 2


def test_import_validation_does_not_mutate_units_and_rejects_missing_or_extra_declarations():
    source = {
        "camp": 1,
        "society": 1,
        "parentSociety": 0,
        "priorPopulation": 0,
        "population": 2,
        "qol": 0.5,
        "years": 0.0,
        "welfare": 0.0,
        "tradeUtility": 0.0,
        "terminated": False,
        "founded": False,
        "observation": observation(),
    }
    target = copy.deepcopy(source)
    target.update(priorPopulation=2, years=20.0)
    row = economic(transition="unit-immutability", years=20.0, person_years=40.0, qol=0.5)
    before = copy.deepcopy(row)
    diagnostic.validate_economic_rows(
        [row],
        trajectory_id="a" * 64,
        decision_turn=1,
        source_villages=[source],
        successor_villages=[target],
    )
    assert row == before
    missing = copy.deepcopy(row)
    missing["units"] = {}
    with pytest.raises(ValueError, match="unit registry"):
        diagnostic.validate_economic_rows(
            [missing],
            trajectory_id="a" * 64,
            decision_turn=1,
            source_villages=[source],
            successor_villages=[target],
        )
    extra = copy.deepcopy(row)
    extra["units"]["domains.trade.unregistered"] = "qol"
    with pytest.raises(ValueError, match="absent field"):
        diagnostic.validate_economic_rows(
            [extra],
            trajectory_id="a" * 64,
            decision_turn=1,
            source_villages=[source],
            successor_villages=[target],
        )


@pytest.mark.parametrize("case", ["clothing", "shelter", "movement", "region", "mix", "commit"])
def test_action_legality_is_bound_to_source_masks_and_active_semantics(case):
    obs = observation()
    semantics = copy.deepcopy(SEMANTICS)
    value = action()
    if case == "clothing":
        obs["clothingRouteMask"] = [False, True]
    elif case == "shelter":
        obs["shelterRouteMask"] = [False, True]
    elif case == "movement":
        semantics["learned_movement"] = True
        value["movement"] = -1
        obs["movementMask"][0] = False
    elif case == "region":
        semantics.update(learned_movement=True, region_movement=True)
        value["movement"] = contracts.REGION_BASE
        obs["regionMovementMask"][2] = False
    elif case == "mix":
        value["strategic"][-1] = 1.0
        obs["economyMixMask"][-1] = False
    else:
        value["commit"] = contracts.SPAN_HOLD
        obs["commitHoldMask"] = False
    with pytest.raises(ValueError, match="illegal|unavailable"):
        diagnostic._validate_action(
            value,
            semantics,
            f"{case} action",
            obs,
            decoder_receipt=decoder_receipt(value, obs, semantics),
        )


def test_action_legality_accepts_legal_masked_choices_for_deterministic_and_stochastic_rows():
    obs = observation()
    obs["clothingRouteMask"] = [False, True]
    obs["shelterRouteMask"] = [False, True]
    obs["economyMixMask"] = [True, False, False]
    value = action()
    value.update(clothingRoute=1, shelterRoute=1)
    value["strategic"][-2:] = [1.0, 0.0]
    diagnostic._validate_action(
        value,
        SEMANTICS,
        "deterministic action",
        obs,
        decoder_receipt=decoder_receipt(value, obs),
    )
    diagnostic._validate_action(
        value,
        SEMANTICS,
        "stochastic decoded action",
        obs,
        action_mode="seeded-stochastic",
        decoder_receipt=decoder_receipt(value, obs, mode="seeded-stochastic"),
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("service", 0.005),
        ("service", 0.995),
        ("strategic", 0.005),
        ("strategic", 0.995),
    ],
)
def test_deterministic_action_requires_exact_snap_decoder_image(name, value):
    decoded = action()
    decoded[name][0] = value
    with pytest.raises(ValueError, match="forward decoder replay"):
        diagnostic._validate_action(
            decoded,
            SEMANTICS,
            "deterministic action",
            observation(),
            decoder_receipt=decoder_receipt(decoded),
        )
    decoded[name][0] = float(torch.sigmoid(torch.tensor(_logit(value), dtype=torch.float32)))
    diagnostic._validate_action(
        decoded,
        SEMANTICS,
        "stochastic action",
        observation(),
        action_mode="seeded-stochastic",
        decoder_receipt=decoder_receipt(decoded, mode="seeded-stochastic"),
    )


@pytest.mark.parametrize("value", [float(np.float32(item)) for item in (0.0, 0.5, 1.0)])
def test_deterministic_snap_boundary_controls_are_accepted(value):
    decoded = action()
    decoded["service"][0] = value
    decoded["strategic"][0] = value
    diagnostic._validate_action(
        decoded,
        SEMANTICS,
        "deterministic action",
        observation(),
        decoder_receipt=decoder_receipt(decoded),
    )


def test_deterministic_decoder_round_trip_uses_native_float32_mix_arithmetic():
    from anthropocene_rl.autonomous_v7.model import deterministic_mix, mix_shares

    logits = torch.tensor(
        [[-4.605170249938965, -4.604200839996338, -0.020212601870298386]], dtype=torch.float32
    )
    mix = deterministic_mix(logits)
    field, fishing = mix_shares(mix)
    assert mix.tolist()[0] == [0.010000000707805157, 0.010009698569774628, 0.9799903035163879]
    assert [float(field[0]), float(fishing[0])] == [0.49975764751434326, 0.9799903035163879]
    decoded = action()
    decoded["strategic"][-2:] = [float(field[0]), float(fishing[0])]
    diagnostic._validate_action(
        decoded,
        SEMANTICS,
        "float32 decoder action",
        observation(),
        decoder_receipt=decoder_receipt(decoded, raw_override={"mix_logits": logits}),
    )


def test_zero_rest_mix_requires_the_native_half_field_fallback():
    decoded = action()
    decoded["strategic"][-2:] = [float(np.float32(0.123)), 1.0]
    with pytest.raises(ValueError, match="forward decoder replay"):
        diagnostic._validate_action(
            decoded,
            SEMANTICS,
            "fabricated pure fishing",
            observation(),
            decoder_receipt=decoder_receipt(decoded),
        )
    decoded["strategic"][-2:] = [0.5, 1.0]
    diagnostic._validate_action(
        decoded,
        SEMANTICS,
        "native pure fishing",
        observation(),
        decoder_receipt=decoder_receipt(decoded),
    )


def test_actual_level_decoder_outputs_round_trip_at_snap_thresholds():
    from anthropocene_rl.autonomous_v7.model import deterministic_level

    probabilities = torch.tensor(
        [[0.0099, 0.01, 0.0101, 0.9899, 0.99, 0.9901]], dtype=torch.float32
    )
    decoded_levels = deterministic_level(torch.logit(probabilities))[0].tolist()
    assert decoded_levels[0] == 0.0 and decoded_levels[-1] == 1.0
    decoded = action()
    decoded["service"] = decoded_levels
    diagnostic._validate_action(
        decoded,
        SEMANTICS,
        "threshold action",
        observation(),
        decoder_receipt=decoder_receipt(
            decoded,
            raw_override={"service_mean": torch.logit(probabilities)},
        ),
    )


def test_forced_regional_stay_rejects_another_legal_region_slot():
    semantics = {
        **SEMANTICS,
        "learned_movement": True,
        "region_movement": True,
        "region_selection": "stay",
    }
    decoded = action()
    decoded["movement"] = contracts.REGION_BASE + 3
    with pytest.raises(ValueError, match="forward decoder replay"):
        diagnostic._validate_action(
            decoded,
            semantics,
            "deterministic action",
            observation(),
            decoder_receipt=decoder_receipt(decoded, semantics=semantics),
        )
    decoded["movement"] = contracts.REGION_BASE
    diagnostic._validate_action(
        decoded,
        semantics,
        "deterministic action",
        observation(),
        decoder_receipt=decoder_receipt(decoded, semantics=semantics),
    )
    decoded["movement"] = contracts.REGION_BASE + 3
    diagnostic._validate_action(
        decoded,
        semantics,
        "stochastic action",
        observation(),
        action_mode="seeded-stochastic",
        decoder_receipt=decoder_receipt(
            decoded,
            semantics=semantics,
            mode="seeded-stochastic",
        ),
    )


def test_randomized_native_forward_receipts_never_invert_rounded_mix_shares():
    """Cover more than Astra's 20k/10k failing distributions with exact forward replay."""
    from anthropocene_rl.autonomous_v7.model import act_deterministic, bridge_actions_from_mode
    from anthropocene_rl.autonomous_v7.ppo import to_bridge_actions

    generator = torch.Generator().manual_seed(613)
    obs = {
        "movementMask": [True] * contracts.MOVEMENT_SLOTS,
        "regionMovementMask": [True] * contracts.REGION_MOVEMENT_SLOTS,
        "clothingRouteMask": [True, True],
        "shelterRouteMask": [True, True],
        "economyMixMask": [True] * len(contracts.ECONOMY_MIX),
        "commitHoldMask": True,
    }

    deterministic_rows = 20_001
    output = {
        "strategic_mean": torch.randn(
            (deterministic_rows, contracts.UNIT_STRATEGIC), generator=generator
        ),
        "mix_logits": torch.randn(
            (deterministic_rows, len(contracts.ECONOMY_MIX)), generator=generator
        )
        * 5,
        "service_mean": torch.randn(
            (deterministic_rows, len(contracts.SERVICES)), generator=generator
        ),
        "clothing": torch.randn((deterministic_rows, 2), generator=generator),
        "shelter": torch.randn((deterministic_rows, 2), generator=generator),
        "fission": torch.randn(
            (deterministic_rows, contracts.FISSION_POLICIES), generator=generator
        ),
        "movement": torch.randn(
            (deterministic_rows, contracts.MOVEMENT_SLOTS), generator=generator
        ),
        "span": torch.randn((deterministic_rows, len(contracts.SPANS)), generator=generator),
        "trade_value_mean": torch.randn(
            (deterministic_rows, len(contracts.TRADE_GROUPS)), generator=generator
        ),
        "trade_gate": torch.randn(
            (deterministic_rows, len(contracts.TRADE_GROUPS), contracts.TRADE_GATES),
            generator=generator,
        ),
        "trade_share_mean": torch.randn(deterministic_rows, generator=generator),
    }
    mode = act_deterministic(output)
    actions = bridge_actions_from_mode(
        mode,
        movement=False,
        region_movement=False,
        commitments=True,
        learned_fission=False,
        learned_trade=False,
    )

    def lossy_inverse_fails(emitted: dict, *, deterministic: bool) -> bool:
        from anthropocene_rl.autonomous_v7.model import mix_shares

        field, fishing = np.asarray(emitted["strategic"][-2:], dtype=np.float32)
        rest = np.float32(1.0) - fishing
        reconstructed = np.asarray(
            [[rest * field, rest * (np.float32(1.0) - field), fishing]],
            dtype=np.float32,
        )
        roundtrip = mix_shares(reconstructed)
        mismatch = not np.array_equal(
            roundtrip[0], np.asarray([field], np.float32)
        ) or not np.array_equal(roundtrip[1], np.asarray([fishing], np.float32))
        below = ((reconstructed > 0) & (reconstructed < np.float32(0.01))).any()
        return mismatch or (deterministic and below)

    formerly_rejected = [
        index
        for index, emitted in enumerate(actions)
        if lossy_inverse_fails(emitted, deterministic=True)
    ]
    assert formerly_rejected
    selected = sorted(
        set(formerly_rejected[:64])
        | set(range(0, deterministic_rows, max(1, deterministic_rows // 64)))
        | {deterministic_rows - 1}
    )
    for row in selected:
        emitted = actions[row]
        receipt = diagnostic.action_decoder_receipt(
            mode="deterministic",
            decoder="native-contract-12",
            semantics=SEMANTICS,
            observation=obs,
            emitted_action=emitted,
            camp=row + 1,
            raw=output,
            row=row,
        )
        diagnostic._validate_action(
            emitted,
            SEMANTICS,
            f"random deterministic row {row}",
            obs,
            decoder_receipt=receipt,
            expected_camp=row + 1,
        )

    stochastic_rows = 10_001
    draw = {
        "region": torch.full((stochastic_rows,), contracts.REGION_NATIVE, dtype=torch.int64),
        "strategic_z": torch.randn(
            (stochastic_rows, contracts.UNIT_STRATEGIC), generator=generator
        ),
        "service_z": torch.randn((stochastic_rows, len(contracts.SERVICES)), generator=generator),
        "mix_z": torch.randn((stochastic_rows, len(contracts.ECONOMY_MIX)), generator=generator)
        * 5,
        "clothing": torch.randint(0, 2, (stochastic_rows,), generator=generator),
        "shelter": torch.randint(0, 2, (stochastic_rows,), generator=generator),
        "movement": torch.full((stochastic_rows,), -1, dtype=torch.int64),
        "span": torch.randint(0, len(contracts.SPANS), (stochastic_rows,), generator=generator),
        "fission": torch.full((stochastic_rows,), -1, dtype=torch.int64),
    }
    mask = torch.ones((stochastic_rows, len(contracts.ECONOMY_MIX)), dtype=torch.bool)
    actions = to_bridge_actions(draw, {"economy_mix_mask": mask})
    formerly_rejected = [
        index
        for index, emitted in enumerate(actions)
        if lossy_inverse_fails(emitted, deterministic=False)
    ]
    assert formerly_rejected
    selected = sorted(
        set(formerly_rejected[:64])
        | set(range(0, stochastic_rows, max(1, stochastic_rows // 64)))
        | {stochastic_rows - 1}
    )
    for row in selected:
        emitted = actions[row]
        receipt = diagnostic.action_decoder_receipt(
            mode="seeded-stochastic",
            decoder="native-contract-12",
            semantics=SEMANTICS,
            observation=obs,
            emitted_action=emitted,
            camp=row + 1,
            raw=draw,
            row=row,
            economy_mix_mask=mask,
        )
        diagnostic._validate_action(
            emitted,
            SEMANTICS,
            f"random stochastic row {row}",
            obs,
            action_mode="seeded-stochastic",
            decoder_receipt=receipt,
            expected_camp=row + 1,
        )


def _encoded_tensor(dtype: str, shape: tuple[int, ...], values) -> dict:
    torch_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
        "int64": torch.int64,
        "bool": torch.bool,
    }[dtype]
    stride = list(torch.empty((1, *shape), dtype=torch_dtype).stride())
    return {
        "dtype": dtype,
        "shape": list(shape),
        "stride": stride,
        "storage_offset": 0,
        "values": values,
    }


@pytest.mark.parametrize(
    "dtype,shape,values",
    [
        ("float32", (3,), [False, 0.0, 0.0]),
        ("float32", (3,), [0.0, 2**60 + 1, 0.0]),
        ("float32", (3,), [0.0, -(2**60 + 1), 0.0]),
        ("float64", (1,), [2**53 + 1]),
        ("float64", (1,), [-(2**53 + 1)]),
        ("int64", (2,), [0, False]),
        ("int64", (1,), [0.0]),
        ("int64", (1,), [2**63]),
        ("float32", (3,), [0.0, "0", 0.0]),
        ("float32", (3,), [0.0, None, 0.0]),
        ("float32", (3,), [0.0, [0.0], 0.0]),
        ("float32", (2, 2), [[0.0, 0.0], [0.0]]),
        ("float32", (1,), [1e39]),
        ("float32", (1,), [1e-50]),
        ("float32", (1,), [float("inf")]),
        ("float32", (1,), [float("nan")]),
        ("float64", (1,), [10**400]),
    ],
)
def test_decoder_tensor_json_leaves_reject_lossy_or_coercive_values(dtype, shape, values):
    with pytest.raises(ValueError, match="tensor"):
        diagnostic._tensor_from_row(
            _encoded_tensor(dtype, shape, values),
            dtype=dtype,
            shape=shape,
            batch_rows=1,
            row=0,
            what="hostile JSON",
        )


@pytest.mark.parametrize(
    "dtype,values",
    [
        ("float32", [2**24, -(2**24), 2**60, float(np.float32(0.1)), -0.0]),
        ("float64", [2**53, -(2**53), 2**60, 0.1, -0.0]),
        ("int64", [-(2**63), 0, 2**63 - 1]),
        ("bool", [False, True]),
    ],
)
def test_decoder_tensor_json_leaves_accept_exact_controls(dtype, values):
    shape = (len(values),)
    restored = diagnostic._tensor_from_row(
        _encoded_tensor(dtype, shape, values),
        dtype=dtype,
        shape=shape,
        batch_rows=1,
        row=0,
        what="exact JSON",
    )
    assert restored[0].tolist() == values
    if dtype.startswith("float"):
        assert bool(torch.signbit(restored[0, -1]))


def test_more_than_six_thousand_actual_forward_decoder_receipts_round_trip():
    from anthropocene_rl.autonomous_v7.model import act_deterministic, bridge_actions_from_mode

    generator = torch.Generator().manual_seed(701)
    obs = observation()
    rows, batches = 32, 192
    validated = 0
    for batch_index in range(batches):

        def random(shape):
            return torch.randn((rows, *shape), generator=generator, dtype=torch.float32)

        raw = {
            "strategic_mean": random((contracts.UNIT_STRATEGIC,)),
            "mix_logits": random((len(contracts.ECONOMY_MIX),)),
            "service_mean": random((len(contracts.SERVICES),)),
            "clothing": random((2,)),
            "shelter": random((2,)),
            "fission": random((contracts.FISSION_POLICIES,)),
            "movement": random((contracts.MOVEMENT_SLOTS,)),
            "span": random((len(contracts.SPANS),)),
            "trade_value_mean": random((len(contracts.TRADE_GROUPS),)),
            "trade_gate": random((len(contracts.TRADE_GROUPS), contracts.TRADE_GATES)),
            "trade_share_mean": random(()),
        }
        mode = act_deterministic(raw)
        actions = bridge_actions_from_mode(
            mode,
            movement=False,
            region_movement=False,
            commitments=True,
            learned_fission=False,
            learned_trade=False,
        )
        for row, emitted in enumerate(actions):
            camp = batch_index * rows + row + 1
            receipt = diagnostic.action_decoder_receipt(
                mode="deterministic",
                decoder="native-contract-12",
                semantics=SEMANTICS,
                observation=obs,
                emitted_action=emitted,
                camp=camp,
                raw=raw,
                row=row,
            )
            diagnostic._validate_action(
                emitted,
                SEMANTICS,
                f"sample row {camp}",
                obs,
                decoder_receipt=receipt,
                expected_camp=camp,
            )
            validated += 1
    assert validated == 6_144


@pytest.mark.parametrize("attack", ["absent", "raw", "stride", "mode", "mask"])
def test_decoder_receipt_rejects_absent_or_fabricated_inputs(attack):
    obs, emitted = observation(), action()
    receipt = decoder_receipt(emitted, obs)
    if attack == "absent":
        receipt = None
    elif attack == "raw":
        receipt["raw"]["mix_logits"]["values"][0] += 1.0
        receipt["digest"] = digest(
            {key: value for key, value in receipt.items() if key != "digest"}
        )
    elif attack == "stride":
        receipt["raw"]["mix_logits"]["stride"] = [0, 1]
        receipt["digest"] = digest(
            {key: value for key, value in receipt.items() if key != "digest"}
        )
    elif attack == "mode":
        receipt["mode"] = "seeded-stochastic"
        receipt["digest"] = digest(
            {key: value for key, value in receipt.items() if key != "digest"}
        )
    else:
        receipt["observation_masks"]["economyMixMask"][0] = False
        receipt["digest"] = digest(
            {key: value for key, value in receipt.items() if key != "digest"}
        )
    with pytest.raises(
        ValueError, match="decoder receipt|forward decoder replay|tensor declaration"
    ):
        diagnostic._validate_action(
            emitted,
            SEMANTICS,
            "forged decoder action",
            obs,
            decoder_receipt=receipt,
        )


def _migration_evidence() -> tuple[dict, dict]:
    def village(camp: int, society: int, population: int) -> dict:
        value = {
            "camp": camp,
            "society": society,
            "parentSociety": 0,
            "priorPopulation": population,
            "population": population,
            "qol": 0.5,
            "years": 20.0,
            "welfare": 0.0,
            "tradeUtility": 0.0,
            "terminated": False,
            "founded": False,
            "observation": observation(),
        }
        value["observation_sha256"] = digest(value["observation"])
        return value

    opening = {
        "turn": 1,
        "fingerprint": "a" * 64,
        "villages": [village(1, 1, 10), village(2, 2, 20)],
    }
    successor = copy.deepcopy(opening)
    successor.update(turn=2, fingerprint="b" * 64)
    successor["villages"][0].update(priorPopulation=10, population=8)
    successor["villages"][1].update(priorPopulation=20, population=22)
    transfer = {
        "kind": "migration",
        "phase": "close-after-demography",
        "sourceCamp": 1,
        "destinationCamp": 2,
        "people": 2,
        "plantFood": 0.0,
        "animalFood": 0.0,
        "aquaticFood": 0.0,
        "carriedFoodPersonEquivalents": 0.0,
    }
    rows = []
    for camp, society, opening_people, linked, emigrated, immigrated, ending in (
        (1, 1, 10, 10, 2, 0, 8),
        (2, 2, 20, 20, 0, 2, 22),
    ):
        row = economic(
            transition=f"migration-{camp}",
            years=20.0,
            person_years=float(opening_people * 20),
            qol=0.5,
        )
        row.update(campId=camp, societyId=society)
        row["domains"]["population"].update(
            openingResidents=opening_people,
            openingParties=0,
            openingLinked=opening_people,
            reportStartingLinked=opening_people,
            reportEndingLinked=linked,
            splitOff=0,
            emigrated=emigrated,
            immigrated=immigrated,
            endingResidents=ending,
            endingParties=0,
            endingLinked=ending,
            afterReintegrationResidents=linked,
            afterReintegrationParties=0,
        )
        row["transfers"] = [copy.deepcopy(transfer)]
        rows.append(row)
    transition_id = digest(
        {
            "index": 0,
            "source_turn": 1,
            "source_fingerprint": "a" * 64,
            "successor_turn": 2,
            "successor_fingerprint": "b" * 64,
        }
    )
    step = {
        "index": 0,
        "transition_id": transition_id,
        "source_turn": 1,
        "source_fingerprint": "a" * 64,
        "actions": [
            {
                "camp": camp,
                "action_request": action(),
                "decoded_action": action(),
                "decoder_receipt": decoder_receipt(
                    action(),
                    camp=camp,
                    batch_rows=2,
                    row=camp - 1,
                ),
            }
            for camp in (1, 2)
        ],
        "policy_rng_receipt": None,
        "successor": successor,
        "terminal": {"terminated": False, "truncated": True, "reason": "evaluation-horizon"},
        "economic": rows,
    }
    bounds = {
        "max_steps": 1,
        "max_villages_per_boundary": 8,
        "max_economic_rows_per_step": 8,
        "max_canonical_bytes": 1_000_000,
    }
    evidence = diagnostic.trajectory_evidence(
        opening,
        [step],
        bounds,
        world_id="migration-world",
        snapshot_sha256="",
    )
    return evidence, bounds


def _validate_migration(evidence: dict, bounds: dict) -> dict:
    return diagnostic.validate_trajectory_evidence(
        evidence,
        world_id="migration-world",
        snapshot_sha256="",
        horizon=1,
        bounds=bounds,
        action_mode="deterministic",
        rng_steps=None,
        service_fields=diagnostic.service_outcome_registry(),
        expected_opening_turn=evidence["opening"]["turn"],
        expected_reset_turn=evidence["reset"]["turn"],
        observation_core_width=1,
        policy_semantics=SEMANTICS,
    )


def _reseal_evidence(evidence: dict) -> None:
    evidence["capture_receipt"] = diagnostic._capture_receipt(
        evidence["reset"],
        evidence["opening"],
        evidence["steps"],
    )
    evidence["digest"] = digest({key: value for key, value in evidence.items() if key != "digest"})


def test_migration_accepts_and_deduplicates_exact_native_endpoint_copies():
    evidence, bounds = _migration_evidence()
    result = _validate_migration(evidence, bounds)
    assert result["ending_residents"] == 30


@pytest.mark.parametrize(
    "attack",
    [
        "one-sided",
        "inconsistent-copy",
        "nonexistent-endpoint",
        "alias",
        "negative-people",
        "unreconciled-source",
        "invented-destination",
        "delete-both",
        "rename-both",
        "inflate-destination",
    ],
)
def test_migration_rejects_incomplete_or_unreconciled_endpoint_evidence(attack):
    evidence, bounds = _migration_evidence()
    steps = evidence["steps"]
    left = steps[0]["economic"][0]["transfers"][0]
    right = steps[0]["economic"][1]["transfers"][0]
    if attack == "one-sided":
        steps[0]["economic"][1]["transfers"] = []
    elif attack == "inconsistent-copy":
        right["carriedFoodPersonEquivalents"] = 1.0
    elif attack == "nonexistent-endpoint":
        left["destinationCamp"] = right["destinationCamp"] = 99
    elif attack == "alias":
        left["destinationCamp"] = right["destinationCamp"] = 1
    elif attack == "negative-people":
        left["people"] = right["people"] = -2
    elif attack == "unreconciled-source":
        steps[0]["economic"][0]["domains"]["population"]["emigrated"] = 0
    elif attack == "delete-both":
        steps[0]["economic"][0]["transfers"] = []
        steps[0]["economic"][1]["transfers"] = []
    elif attack == "rename-both":
        left["kind"] = right["kind"] = "unreviewed-movement"
    elif attack == "inflate-destination":
        steps[0]["successor"]["villages"][1]["population"] += 999999
        steps[0]["economic"][1]["domains"]["population"]["endingResidents"] += 999999
    else:
        daughter = copy.deepcopy(steps[0]["successor"]["villages"][1])
        daughter.update(
            camp=3,
            society=3,
            parentSociety=1,
            priorPopulation=0,
            population=999999,
            years=0.0,
            founded=True,
        )
        steps[0]["successor"]["villages"].append(daughter)
        left["destinationCamp"] = right["destinationCamp"] = 3
        left["people"] = right["people"] = 999999
    _reseal_evidence(evidence)
    with pytest.raises(ValueError, match="migration|founding|transfer"):
        _validate_migration(evidence, bounds)


def test_capture_bytes_include_reset_and_each_warmup_link_with_exact_bound_controls():
    evidence, original_bounds = _migration_evidence()
    reset = evidence["reset"]
    fingerprint = reset["fingerprint"]
    turn = reset["turn"]
    warmup = []
    for index in range(100):
        successor = digest({"warmup": index})
        warmup.append(
            {
                "index": index,
                "source_turn": turn,
                "source_fingerprint": fingerprint,
                "successor_turn": turn + 1,
                "successor_fingerprint": successor,
                "trajectory_id": reset["trajectory_id"],
            }
        )
        turn += 1
        fingerprint = successor
    evidence["reset"]["warmup_chain"] = warmup
    evidence["opening"]["turn"] = turn
    evidence["opening"]["fingerprint"] = fingerprint
    step = evidence["steps"][0]
    step["source_turn"] = turn
    step["source_fingerprint"] = fingerprint
    step["successor"]["turn"] = turn + 1
    step["successor"]["fingerprint"] = digest({"successor": turn + 1})
    for row in step["economic"]:
        row["decisionTurn"] = turn
    step["transition_id"] = digest(
        {
            "index": 0,
            "source_turn": turn,
            "source_fingerprint": fingerprint,
            "successor_turn": turn + 1,
            "successor_fingerprint": step["successor"]["fingerprint"],
        }
    )
    _reseal_evidence(evidence)
    full = evidence["capture_receipt"]["canonical_bytes"]
    legacy_partial = diagnostic.capture_envelope_bytes(
        diagnostic.capture_component_bytes({}),
        diagnostic.capture_component_bytes(evidence["opening"]),
        [diagnostic.capture_component_bytes(step)],
    )
    assert full > legacy_partial

    partial_bounds = {**original_bounds, "max_canonical_bytes": legacy_partial}
    formerly_underreported = diagnostic.trajectory_evidence(
        evidence["opening"],
        evidence["steps"],
        partial_bounds,
        world_id="migration-world",
        snapshot_sha256="",
        reset_turn=1,
        reset_fingerprint="a" * 64,
        warmup_chain=warmup,
    )
    with pytest.raises(ValueError, match="capture bound"):
        _validate_migration(formerly_underreported, partial_bounds)

    exact_bounds = {**original_bounds, "max_canonical_bytes": full}
    exact = diagnostic.trajectory_evidence(
        evidence["opening"],
        evidence["steps"],
        exact_bounds,
        world_id="migration-world",
        snapshot_sha256="",
        reset_turn=1,
        reset_fingerprint="a" * 64,
        warmup_chain=warmup,
    )
    assert _validate_migration(exact, exact_bounds)["capture_receipt"]["canonical_bytes"] == full
    too_small = {**exact_bounds, "max_canonical_bytes": full - 1}
    over = diagnostic.trajectory_evidence(
        evidence["opening"],
        evidence["steps"],
        too_small,
        world_id="migration-world",
        snapshot_sha256="",
        reset_turn=1,
        reset_fingerprint="a" * 64,
        warmup_chain=warmup,
    )
    with pytest.raises(ValueError, match="capture bound"):
        _validate_migration(over, too_small)


def test_rng_schedule_binds_seed_state_runtime_and_preserves_global_streams():
    generator = torch.Generator(device="cpu").manual_seed(17)
    schedule = {
        "schema": 1,
        "kind": "gate1_policy_rng_schedule",
        "algorithm": "torch.Generator.cpu-default",
        "torch_version": torch.__version__,
        "device": "cpu",
        "worlds": [
            {
                "world_id": "world",
                "nodes": [
                    {
                        "node_id": "prior",
                        "draw_order": ["strategic"],
                        "steps": [
                            {
                                "step": 0,
                                "seed": 17,
                                "state_sha256": diagnostic.generator_state_sha256(generator),
                            },
                        ],
                    }
                ],
            }
        ],
    }
    schedule["digest"] = digest(schedule)
    assert (
        diagnostic.validate_rng_schedule(
            schedule,
            "world",
            1,
            node_id="prior",
            expected_draw_order=["strategic"],
        )[0]["seed"]
        == 17
    )
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    with diagnostic.preserve_global_rng():
        random.random()
        np.random.random()
        torch.rand(4)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    bad = copy.deepcopy(schedule)
    bad["worlds"][0]["nodes"][0]["steps"][0]["state_sha256"] = "0" * 64
    bad["digest"] = digest({key: value for key, value in bad.items() if key != "digest"})
    with pytest.raises(ValueError, match="state"):
        diagnostic.validate_rng_schedule(
            bad,
            "world",
            1,
            node_id="prior",
            expected_draw_order=["strategic"],
        )
