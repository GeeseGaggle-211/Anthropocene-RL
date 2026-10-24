"""Live heuristic labor behavioral cloning (BC) pretraining for Model 9 actor.

Reconstructs native heuristic specialty and activity simplexes from shadow
simulation steps with economic diagnostics, and trains the LaborActor model
via masked supervised behavioral cloning before RL rollouts.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import contracts
from .bridge import Model9Bridge
from .labor_bridge import (
    LaborAttempt,
    LaborCatalog,
    LaborModel9Bridge,
    LaborPolicySample,
    _simplex,
)
from .labor_model import (
    LABOR_COSTS,
    LABOR_OBJECTIVES,
    LaborActor,
    LaborCostCritics,
    LaborPersonality,
    LaborTaskCritic,
    _masked_softmax,
    _masks,
    labor_policy_tensors,
)
from .labor_ppo import (
    RetainedLaborAction,
    execute_labor_decision,
    labor_cost_value_loss,
    labor_task_value_loss,
)
from .protocol import scan
from .protocol import seeds as checked_seeds

LABOR_BC_CONTRACT = "model9-heuristic-labor-bc-v1"


def _get(mapping: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    """Case-tolerant field lookup for bridge and diagnostic mappings."""
    if not isinstance(mapping, Mapping):
        return default
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
        for alt in (k.lower(), k[:1].upper() + k[1:], k[:1].lower() + k[1:]):
            if alt in mapping and mapping[alt] is not None:
                return mapping[alt]
    return default


def _one_row(values: Mapping[str, torch.Tensor], row: int) -> dict[str, torch.Tensor]:
    return {
        key: tensor[row : row + 1].detach().cpu()
        for key, tensor in values.items()
        if isinstance(tensor, torch.Tensor)
    }


def _concat(rows: Sequence[Mapping[str, torch.Tensor]], label: str) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError(f"{label} has no rows to concatenate")
    keys = set(rows[0])
    for item in rows[1:]:
        if set(item) != keys:
            raise ValueError(f"{label} row keys differ")
    return {key: torch.cat([row[key] for row in rows], dim=0) for key in sorted(keys)}


def extract_heuristic_labor_action(
    catalog: LaborCatalog,
    village_heur: Mapping[str, Any],
    econ_diag: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reconstruct native heuristic specialty/activity allocations and action dict.

    Returns:
        (specialty_simplex, activity_simplex, schema12_action_dict)
    """
    P = max(1, int(catalog.population))
    Y = max(1e-9, float(catalog.turn_years))
    spec_mask = np.asarray(catalog.masks.specialty, dtype=bool)
    act_mask = np.asarray(catalog.masks.activity, dtype=bool)

    spec_id_to_spec_idx: dict[str, int] = {
        row["id"]: row["row"] for row in catalog.specialty_rows
    }
    spec_id_to_role_idx: dict[str, int] = {
        row["id"]: row["row"] - 1 for row in catalog.specialty_rows if row["row"] > 0
    }
    idle_spec = spec_id_to_spec_idx.get("idle", 0)
    gatherer_role = spec_id_to_role_idx.get("Gatherer", 0)
    gatherer_spec = spec_id_to_spec_idx.get("Gatherer", 1)
    farmer_role = spec_id_to_role_idx.get("Farmer", 1)
    farmer_spec = spec_id_to_spec_idx.get("Farmer", 2)
    herder_role = spec_id_to_role_idx.get("Herder", 2)
    herder_spec = spec_id_to_spec_idx.get("Herder", 3)
    fisher_role = spec_id_to_role_idx.get("Fisher", 3)
    fisher_spec = spec_id_to_spec_idx.get("Fisher", 4)

    farmer_op_act = next(
        (
            r["activityRow"]
            for r in catalog.activity_rows
            if r["specialtyRow"] == farmer_role
            and r.get("isOperationOrHarvest")
            and act_mask[farmer_role, r["activityRow"]]
        ),
        5,
    )
    farmer_exp_act = next(
        (
            r["activityRow"]
            for r in catalog.activity_rows
            if r["specialtyRow"] == farmer_role
            and r.get("isDevelopmentOrExpansion")
            and act_mask[farmer_role, r["activityRow"]]
        ),
        6,
    )

    herder_op_act = next(
        (
            r["activityRow"]
            for r in catalog.activity_rows
            if r["specialtyRow"] == herder_role
            and r.get("isOperationOrHarvest")
            and act_mask[herder_role, r["activityRow"]]
        ),
        7,
    )
    herder_exp_act = next(
        (
            r["activityRow"]
            for r in catalog.activity_rows
            if r["specialtyRow"] == herder_role
            and r.get("isDevelopmentOrExpansion")
            and act_mask[herder_role, r["activityRow"]]
        ),
        8,
    )

    fisher_op_act = next(
        (
            r["activityRow"]
            for r in catalog.activity_rows
            if r["specialtyRow"] == fisher_role
            and r.get("isOperationOrHarvest")
            and act_mask[fisher_role, r["activityRow"]]
        ),
        9,
    )
    fisher_exp_act = next(
        (
            r["activityRow"]
            for r in catalog.activity_rows
            if r["specialtyRow"] == fisher_role
            and r.get("isDevelopmentOrExpansion")
            and act_mask[fisher_role, r["activityRow"]]
        ),
        10,
    )

    domains = _get(econ_diag, "domains", default={})

    # 1. Farmer
    cult = _get(domains, "cultivation", default={})
    assigned_farmer_wy = float(_get(cult, "assignedFarmerYears", default=0.0))
    field_exp_wy = float(_get(cult, "expansionWorkerYears", default=0.0))
    if field_exp_wy <= 0.0:
        funnels = _get(cult, "funnels", default=[])
        for f in funnels:
            if _get(f, "metric") == "fieldDevelopment":
                work = float(_get(f, "executedWork") or _get(f, "compilerGrant") or 0.0)
                field_exp_wy = work * Y
                break
    field_op_wy = max(0.0, assigned_farmer_wy - field_exp_wy)
    farmers_people = (
        max(0, round(assigned_farmer_wy / Y)) if spec_mask[farmer_spec] else 0
    )
    farmer_act_wy = np.zeros(22, dtype=np.float64)
    if assigned_farmer_wy > 1e-8:
        farmer_act_wy[farmer_op_act] = field_op_wy
        farmer_act_wy[farmer_exp_act] = field_exp_wy

    # 2. Herder
    herd = _get(domains, "herding", default={})
    herd_mgmt_wy = float(
        _get(herd, "plannedHerderYears") or _get(herd, "coveredHerderYears") or 0.0
    )
    herd_exp_wy = 0.0
    funnels = _get(herd, "funnels", default=[])
    for f in funnels:
        if _get(f, "metric") == "herdDevelopment":
            dev_people = float(_get(f, "executedWork") or _get(f, "compilerGrant") or 0.0)
            herd_exp_wy = max(0.0, dev_people * Y)
            break
    herders_people = (
        max(0, round((herd_mgmt_wy + herd_exp_wy) / Y)) if spec_mask[herder_spec] else 0
    )
    herder_act_wy = np.zeros(22, dtype=np.float64)
    if (herd_mgmt_wy + herd_exp_wy) > 1e-8:
        herder_act_wy[herder_op_act] = herd_mgmt_wy
        herder_act_wy[herder_exp_act] = herd_exp_wy

    # 3. Fisher
    fisher_harv_people = 0
    fisher_dev_people = 0
    obs = village_heur.get("observation")
    if obs is not None:
        pop_after = max(1, int(village_heur.get("population", catalog.population)))
        core = obs.get("core", [])
        core_present = obs.get("corePresent", [])
        if len(core) > 202 and (len(core_present) <= 202 or core_present[202]):
            fisher_harv_people = max(0, round(float(core[202]) * pop_after))
        if len(core) > 206 and (len(core_present) <= 206 or core_present[206]):
            fisher_dev_people = max(0, round(float(core[206]) * pop_after))
    else:
        reqs = _get(econ_diag, "requests", default=[])
        sum_grant = sum(
            float(_get(r, "compilerGrant", default=0.0))
            for r in reqs
            if _get(r, "role") == "Fisher"
        )
        fisher_harv_people = max(0, round(sum_grant))
    fishers_people = (fisher_harv_people + fisher_dev_people) if spec_mask[fisher_spec] else 0
    fisher_act_wy = np.zeros(22, dtype=np.float64)
    fisher_act_wy[fisher_op_act] = fisher_harv_people * Y
    fisher_act_wy[fisher_exp_act] = fisher_dev_people * Y

    # 4. Craft Goods
    goods = _get(econ_diag, "goods", default=[])
    craft_wy = np.zeros((13, 22), dtype=np.float64)
    craft_people: dict[int, int] = {}
    for item in goods:
        good_name = _get(item, "good")
        if not good_name or good_name == "none":
            continue
        norm_name = str(good_name).lower().replace("-", "").replace("_", "").replace(" ", "")
        matching_rows = [
            row
            for row in catalog.activity_rows
            if str(row.get("outputGood") or "")
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
            == norm_name
            and spec_mask[row["specialtyRow"] + 1]
            and act_mask[row["specialtyRow"], row["activityRow"]]
        ]
        if not matching_rows:
            continue
        specialist_matches = [r for r in matching_rows if r.get("specialistOnly")]
        if specialist_matches:
            target_row = specialist_matches[0]
        else:
            craft_specialists = [
                r for r in matching_rows if r.get("specialtyId") != "Craftsperson"
            ]
            target_row = craft_specialists[0] if craft_specialists else matching_rows[0]
        target_role = target_row["specialtyRow"]
        target_act = target_row["activityRow"]

        wy = float(_get(item, "workerYears", default=0.0))
        if wy <= 0.0:
            ceiling = float(_get(item, "plannedOutputCeiling", default=0.0))
            features = target_row.get("features", [])
            rate = float(features[13]) if len(features) > 13 else 0.0
            if ceiling > 0.0 and rate > 1e-6:
                wy = ceiling / rate
        craft_wy[target_role, target_act] += max(0.0, wy)

    for r in range(13):
        total_r = craft_wy[r].sum()
        if total_r > 1e-8 and spec_mask[r + 1]:
            craft_people[r] = math.ceil(total_r / Y - 1e-8)

    # 5. Raw Materials
    materials = _get(econ_diag, "materials", default=[])
    ext_wy = np.zeros((13, 22), dtype=np.float64)
    ext_people: dict[int, int] = {}
    gatherer_mat_wy: dict[int, float] = {}
    for item in materials:
        mat_name = _get(item, "material")
        mat_wy = max(0.0, float(_get(item, "workerYears", default=0.0)))
        if not mat_name or mat_wy <= 0.0:
            continue
        if mat_name == "Stone":
            flag = "outputsStone"
        elif mat_name == "Clay":
            flag = "outputsClay"
        elif mat_name == "Wood":
            flag = "outputsWood"
        else:
            continue

        matching_rows = [
            row
            for row in catalog.activity_rows
            if row.get(flag)
            and spec_mask[row["specialtyRow"] + 1]
            and act_mask[row["specialtyRow"], row["activityRow"]]
        ]
        if not matching_rows:
            continue
        specialist_matches = [r for r in matching_rows if r.get("specialistOnly")]
        if specialist_matches:
            target_row = specialist_matches[0]
            target_role = target_row["specialtyRow"]
            target_act = target_row["activityRow"]
            ext_wy[target_role, target_act] += mat_wy
            ext_people[target_role] = math.ceil(ext_wy[target_role].sum() / Y - 1e-8)
        else:
            material_specialists = [
                r for r in matching_rows if r.get("specialtyRow") != gatherer_role
            ]
            if material_specialists:
                target_row = material_specialists[0]
                target_role = target_row["specialtyRow"]
                target_act = target_row["activityRow"]
                ext_wy[target_role, target_act] += mat_wy
                ext_people[target_role] = math.ceil(ext_wy[target_role].sum() / Y - 1e-8)
            else:
                target_row = matching_rows[0]
                target_role = target_row["specialtyRow"]
                target_act = target_row["activityRow"]
                if target_role == gatherer_role:
                    gatherer_mat_wy[target_act] = gatherer_mat_wy.get(target_act, 0.0) + mat_wy
                else:
                    ext_wy[target_role, target_act] += mat_wy
                    ext_people[target_role] = math.ceil(ext_wy[target_role].sum() / Y - 1e-8)

    # 6. Gatherer & Population Budget
    non_gatherer_people = (
        farmers_people
        + herders_people
        + fishers_people
        + sum(craft_people.values())
        + sum(ext_people.values())
    )
    if non_gatherer_people > P:
        scale = P / float(non_gatherer_people)
        farmers_people = math.floor(farmers_people * scale)
        herders_people = math.floor(herders_people * scale)
        fishers_people = math.floor(fishers_people * scale)
        for r in craft_people:
            craft_people[r] = math.floor(craft_people[r] * scale)
        for r in ext_people:
            ext_people[r] = math.floor(ext_people[r] * scale)
        non_gatherer_people = (
            farmers_people
            + herders_people
            + fishers_people
            + sum(craft_people.values())
            + sum(ext_people.values())
        )

    gatherer_people = max(0, P - non_gatherer_people) if spec_mask[gatherer_spec] else 0
    idle_people = max(0, P - non_gatherer_people - gatherer_people)

    gather_food_act = next(
        (
            r["activityRow"]
            for r in catalog.activity_rows
            if r["specialtyRow"] == gatherer_role
            and r.get("isOperationOrHarvest")
            and not (r.get("outputsStone") or r.get("outputsClay") or r.get("outputsWood"))
            and act_mask[gatherer_role, r["activityRow"]]
        ),
        1,
    )
    gatherer_wy = max(0.0, gatherer_people * Y - sum(gatherer_mat_wy.values()))
    gather_cond = (
        act_mask[gatherer_role, gather_food_act]
        and gatherer_wy <= 0.0
        and sum(gatherer_mat_wy.values()) <= 0.0
    )
    if gather_cond:
        gatherer_wy = 1.0

    gatherer_act_wy = np.zeros(22, dtype=np.float64)
    gatherer_act_wy[gather_food_act] = gatherer_wy
    for act_idx, m_wy in gatherer_mat_wy.items():
        gatherer_act_wy[act_idx] = m_wy

    # Combine role activities
    role_acts = [np.zeros(22, dtype=np.float64) for _ in range(13)]
    role_acts[gatherer_role] = gatherer_act_wy
    role_acts[farmer_role] = farmer_act_wy
    role_acts[herder_role] = herder_act_wy
    role_acts[fisher_role] = fisher_act_wy
    for r in range(13):
        if craft_wy[r].sum() > 0:
            role_acts[r] += craft_wy[r]
        if ext_wy[r].sum() > 0:
            role_acts[r] += ext_wy[r]

    activity_simplex_list: list[tuple[float, ...]] = []
    for r in range(13):
        vec = role_acts[r].copy()
        vec[~act_mask[r]] = 0.0
        if spec_mask[r + 1]:
            if vec.sum() <= 1e-8:
                legal_act_rows = [
                    row
                    for row in catalog.activity_rows
                    if row["specialtyRow"] == r
                    and row["activityRow"] > 0
                    and act_mask[r, row["activityRow"]]
                ]
                op_harvest_acts = [
                    row["activityRow"]
                    for row in legal_act_rows
                    if row.get("isOperationOrHarvest")
                ]
                if op_harvest_acts:
                    chosen_acts = op_harvest_acts
                elif legal_act_rows:
                    chosen_acts = [row["activityRow"] for row in legal_act_rows]
                else:
                    chosen_acts = [0] if act_mask[r, 0] else []

                if chosen_acts:
                    for a in chosen_acts:
                        vec[a] = 1.0 / len(chosen_acts)
                else:
                    vec[0] = 1.0
        else:
            vec[0] = 1.0
        vec = vec / vec.sum()
        simplex_r = _simplex(tuple(vec), tuple(act_mask[r]), f"activity[{r}]")
        activity_simplex_list.append(simplex_r)

    # 7. Specialty simplex (length 14)
    spec_counts = np.zeros(len(catalog.specialty_rows), dtype=np.float64)
    spec_counts[idle_spec] += idle_people
    spec_counts[gatherer_spec] += gatherer_people
    spec_counts[farmer_spec] += farmers_people
    spec_counts[herder_spec] += herders_people
    spec_counts[fisher_spec] += fishers_people
    for r, count in craft_people.items():
        spec_counts[r + 1] += count
    for r, count in ext_people.items():
        spec_counts[r + 1] += count
    spec_counts[~spec_mask] = 0.0

    if spec_counts.sum() > 0:
        spec_vec = spec_counts / spec_counts.sum()
    else:
        spec_vec = np.zeros(len(catalog.specialty_rows), dtype=np.float64)
        if spec_mask[gatherer_spec]:
            spec_vec[gatherer_spec] = 1.0
        else:
            spec_vec[idle_spec] = 1.0
    specialty_simplex_tuple = _simplex(tuple(spec_vec), tuple(spec_mask), "specialty")

    # 8. Non-labor teacher fields
    teacher = village_heur.get("teacher") or {}
    raw_movement = int(teacher.get("movement", -1))
    obs_heur = village_heur.get("observation") or {}
    m_mask = obs_heur.get("movementMask")
    r_mask = obs_heur.get("regionMovementMask")
    if raw_movement >= 0 and raw_movement < 7:
        if m_mask and (raw_movement + 1 < len(m_mask)) and m_mask[raw_movement + 1]:
            movement = raw_movement
        else:
            movement = -1
    elif raw_movement == -3:
        movement = raw_movement if r_mask and len(r_mask) > 1 and r_mask[1] else -1
    elif raw_movement >= 100:
        r_idx = 2 + (raw_movement - 100)
        movement = raw_movement if r_mask and r_idx < len(r_mask) and r_mask[r_idx] else -1
    else:
        movement = -1
    fission = int(np.clip(int(teacher.get("fission", 0)), 0, 2))
    tv_raw = teacher.get("tradeValue")
    if tv_raw is not None and len(tv_raw) == 7:
        trade_value = [
            float(np.clip(float(x), contracts.TRADE_VALUE_RANGE[0], contracts.TRADE_VALUE_RANGE[1]))
            for x in tv_raw
        ]
    else:
        trade_value = [1.0] * 7
    tg_raw = teacher.get("tradeGate")
    if tg_raw is not None and len(tg_raw) == 7:
        trade_gate = [int(np.clip(int(x), 0, 2)) for x in tg_raw]
    else:
        trade_gate = [1] * 7
    trade_share = float(np.clip(float(teacher.get("tradeShare", 0.5)), 1e-4, 1.0 - 1e-4))

    # Assemble schema12 action
    policy_sample = LaborPolicySample(
        specialty_simplex_tuple, tuple(activity_simplex_list)
    )
    payload = policy_sample.payload()
    schema12_action = {
        "schemaVersion": 12,
        "movement": movement,
        "fission": fission,
        "tradeValue": trade_value,
        "tradeGate": trade_gate,
        "tradeShare": trade_share,
        **payload,
    }

    return (
        np.asarray(specialty_simplex_tuple, dtype=np.float64),
        np.asarray(activity_simplex_list, dtype=np.float64),
        schema12_action,
    )


def collect_heuristic_bc_dataset(
    *,
    normalizer: Any,
    world_seeds: Sequence[int],
    turns: int,
    personality: LaborPersonality | None = None,
    labor_bridge_factory: Any = LaborModel9Bridge,
    heuristic_bridge_factory: Any = None,
    economic_trace: Any | None = None,
    allow_vm_scale: bool = False,
) -> dict[str, Any]:
    """Collect closed-loop heuristic labor trajectories for behavioral cloning."""
    worlds = checked_seeds(list(world_seeds), "labor heuristic bc worlds")
    scan({"worldSeeds": worlds}, "labor heuristic bc worlds")
    if not allow_vm_scale and len(worlds) > 3:
        raise ValueError("local labor heuristic bc requires 1..3 world seeds")
    if not worlds:
        raise ValueError("labor heuristic bc requires at least one world seed")
    if turns < 1:
        raise ValueError("turns must be >= 1")

    declared_personality = LaborPersonality() if personality is None else personality
    heur_factory = heuristic_bridge_factory or (
        lambda: Model9Bridge(economic_diagnostics=True)
    )

    all_rows: list[dict[str, torch.Tensor]] = []
    total_decisions = 0
    initial_accepted = 0
    initial_rejected = 0
    surviving_camps_by_world: dict[str, int] = {}
    specialty_targets_accum: list[np.ndarray] = []

    for seed in worlds:
        with labor_bridge_factory() as labor_bridge, heur_factory() as heur_bridge:
            step = labor_bridge.reset(seed)
            for _ in range(turns):
                live = [
                    v
                    for v in step["villages"]
                    if v.get("observation") is not None and not v.get("terminated")
                ]
                if not live:
                    break
                contexts, tensors = labor_policy_tensors(
                    live, normalizer, personality=declared_personality
                )
                catalogs = {context[0].camp_id: context[0] for context in contexts}

                # Capture pre-step snapshot and advance shadow heuristic
                snap = labor_bridge._bridge.capture()
                heur_bridge.reset(seed, snapshot=snap)
                h_step = heur_bridge.step({})
                h_villages = {int(v["camp"]): v for v in h_step.get("villages", [])}
                econ_by_camp = h_step.get("economicDiagnostics") or {}

                if economic_trace is not None and getattr(
                    heur_bridge, "economic_diagnostics", None
                ):
                    for camp_id, diag in heur_bridge.economic_diagnostics.items():
                        turn_m1 = int(h_step["turn"]) - 1
                        trans_id = diag.get("transitionId") or f"{seed}:{camp_id}:{turn_m1}"
                        if trans_id not in economic_trace.transition_ids:
                            economic_trace.transition_ids.append(trans_id)
                        economic_trace.write(
                            "transitions",
                            {
                                "campId": int(camp_id),
                                "turn": int(h_step["turn"]) - 1,
                                "worldSeed": seed,
                                "transitionId": trans_id,
                                "economic": diag,
                            },
                        )

                retained: dict[int, RetainedLaborAction] = {}
                for row, (village, context) in enumerate(zip(live, contexts, strict=True)):
                    camp = int(village["camp"])
                    catalog = context[0]
                    h_v = h_villages.get(camp, {})
                    econ = econ_by_camp.get(str(camp))

                    spec_target, act_target, schema12_action = (
                        extract_heuristic_labor_action(catalog, h_v, econ)
                    )
                    specialty_targets_accum.append(spec_target)

                    row_tensors = _one_row(tensors, row)
                    row_tensors["target_specialty"] = torch.tensor(
                        spec_target, dtype=torch.float32
                    ).unsqueeze(0)
                    row_tensors["target_activity"] = torch.tensor(
                        act_target, dtype=torch.float32
                    ).unsqueeze(0)
                    row_tensors["target_movement_code"] = torch.tensor(
                        [int(schema12_action["movement"])], dtype=torch.long
                    )
                    row_tensors["target_fission"] = torch.tensor(
                        [int(schema12_action["fission"])], dtype=torch.long
                    )
                    tv = [
                        math.log(np.clip(float(x), 0.25, 4.0))
                        for x in schema12_action["tradeValue"]
                    ]
                    row_tensors["target_trade_value_latent"] = torch.tensor(
                        tv, dtype=torch.float32
                    ).unsqueeze(0)
                    row_tensors["target_trade_gate"] = torch.tensor(
                        [int(x) for x in schema12_action["tradeGate"]],
                        dtype=torch.long,
                    ).unsqueeze(0)
                    ts = float(schema12_action["tradeShare"])
                    ts_clamped = np.clip(ts, 1e-4, 1.0 - 1e-4)
                    row_tensors["target_trade_share_latent"] = torch.tensor(
                        [math.log(ts_clamped / (1.0 - ts_clamped))],
                        dtype=torch.float32,
                    )
                    all_rows.append(row_tensors)

                    retained[camp] = RetainedLaborAction(
                        payload=schema12_action,
                        labor_sample={
                            "specialty_allocation": torch.as_tensor(
                                spec_target, dtype=torch.float64
                            ).unsqueeze(0),
                            "activity_allocation": torch.as_tensor(
                                act_target, dtype=torch.float64
                            ).unsqueeze(0),
                            "specialty_latent": torch.zeros((1, 14), dtype=torch.float32),
                            "activity_latent": torch.zeros(
                                (1, 13, 22), dtype=torch.float32
                            ),
                            "specialty_mask": torch.tensor(
                                [catalog.masks.specialty], dtype=torch.bool
                            ),
                            "activity_mask": torch.tensor(
                                [catalog.masks.activity], dtype=torch.bool
                            ),
                        },
                        behavior_labor_logp=0.0,
                        nonlabor_sample={
                            "movement_index": torch.tensor([0], dtype=torch.long),
                            "movement": torch.tensor(
                                [int(schema12_action["movement"])], dtype=torch.long
                            ),
                            "fission": torch.tensor(
                                [int(schema12_action["fission"])], dtype=torch.long
                            ),
                            "trade_value_latent": torch.tensor(
                                tv, dtype=torch.float32
                            ).unsqueeze(0),
                            "trade_gate": torch.tensor(
                                [int(x) for x in schema12_action["tradeGate"]],
                                dtype=torch.long,
                            ).unsqueeze(0),
                            "trade_share_latent": torch.tensor(
                                [math.log(ts_clamped / (1.0 - ts_clamped))],
                                dtype=torch.float32,
                            ),
                            "movement_mask": row_tensors["movement_mask"],
                            "movement_codes": torch.tensor([-1], dtype=torch.long),
                        },
                        behavior_nonlabor_logp=0.0,
                    )

                def retry_factory(
                    camp: int, catalog: LaborCatalog, attempt: LaborAttempt
                ) -> RetainedLaborAction:
                    idle_spec = np.zeros(14, dtype=np.float64)
                    idle_spec[0] = 1.0
                    idle_act = np.zeros((13, 22), dtype=np.float64)
                    idle_act[:, 0] = 1.0
                    safe_sample = LaborPolicySample(
                        tuple(idle_spec), tuple(tuple(r) for r in idle_act)
                    )
                    return RetainedLaborAction(
                        payload=safe_sample.payload(),
                        labor_sample={
                            "specialty_allocation": torch.as_tensor(
                                idle_spec, dtype=torch.float64
                            ).unsqueeze(0),
                            "activity_allocation": torch.as_tensor(
                                idle_act, dtype=torch.float64
                            ).unsqueeze(0),
                            "specialty_latent": torch.zeros((1, 14), dtype=torch.float32),
                            "activity_latent": torch.zeros(
                                (1, 13, 22), dtype=torch.float32
                            ),
                            "specialty_mask": torch.tensor(
                                [catalog.masks.specialty], dtype=torch.bool
                            ),
                            "activity_mask": torch.tensor(
                                [catalog.masks.activity], dtype=torch.bool
                            ),
                        },
                        behavior_labor_logp=0.0,
                        nonlabor_sample=None,
                        behavior_nonlabor_logp=None,
                    )

                execution = execute_labor_decision(
                    labor_bridge, catalogs, retained, retry_factory
                )
                step = execution.step
                for att in execution.initial_attempts.values():
                    total_decisions += 1
                    if att.valid:
                        initial_accepted += 1
                    else:
                        initial_rejected += 1

            surviving = len([v for v in step.get("villages", []) if not v.get("terminated")])
            surviving_camps_by_world[str(seed)] = surviving

    if not all_rows:
        raise ValueError("heuristic BC collection produced zero training rows")

    dataset = _concat(all_rows, "heuristic BC dataset")
    mean_spec = (
        np.mean(specialty_targets_accum, axis=0).tolist()
        if specialty_targets_accum
        else []
    )
    dataset["collection"] = {
        "total_decisions": int(total_decisions),
        "initial_accepted": int(initial_accepted),
        "initial_rejected": int(initial_rejected),
        "surviving_camps_by_world": surviving_camps_by_world,
        "mean_specialty_target": mean_spec,
        "world_seeds": list(worlds),
        "turns": int(turns),
    }
    if economic_trace is not None:
        economic_trace.close()
    return dataset


def _batch_loss_and_l1(
    actor: LaborActor,
    batch_tensors: Mapping[str, torch.Tensor],
    *,
    target_log_std: float,
    inactive_role_weight: float = 0.0,
    label_smoothing: float = 1e-6,
    spec_weights: torch.Tensor | None = None,
    act_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, float]:
    B = len(batch_tensors["target_specialty"])
    output = actor(batch_tensors)
    spec_mask, act_mask = _masks(batch_tensors, B)

    # 1. Specialty loss
    pred_spec = _masked_softmax(output["specialty_logits"], spec_mask)
    uniform_spec = spec_mask.float() / spec_mask.float().sum(
        dim=-1, keepdim=True
    ).clamp(min=1.0)
    target_spec = batch_tensors["target_specialty"]
    smoothed_target_spec = (1.0 - label_smoothing) * target_spec + label_smoothing * uniform_spec
    spec_ce = -(smoothed_target_spec * torch.log(pred_spec.clamp(min=1e-12))).sum(
        dim=-1
    )
    if spec_weights is not None:
        sw = spec_weights.to(pred_spec.device)
        row_spec_w = (target_spec * sw).sum(dim=-1).detach()
        spec_loss = (spec_ce * row_spec_w).sum() / row_spec_w.sum().clamp(min=1e-6)
    else:
        spec_loss = spec_ce.mean()
    gauge_anchor_spec = 0.01 * (output["specialty_logits"][:, 0] ** 2).mean()
    total_spec_loss = spec_loss + gauge_anchor_spec
    spec_l1 = (
        (torch.abs(pred_spec - target_spec) * spec_mask.float())
        .sum(dim=-1)
        .mean()
        .item()
    )

    # 2. Activity loss
    pred_act = _masked_softmax(output["activity_logits"], act_mask)
    uniform_act = act_mask.float() / act_mask.float().sum(dim=-1, keepdim=True).clamp(
        min=1.0
    )
    target_act = batch_tensors["target_activity"]
    smoothed_target_act = (1.0 - label_smoothing) * target_act + label_smoothing * uniform_act
    act_role_ce = -(smoothed_target_act * torch.log(pred_act.clamp(min=1e-12))).sum(
        dim=-1
    )  # (B, 13)
    legal_roles = act_mask.any(dim=-1)  # (B, 13)
    if act_weights is not None:
        aw = act_weights.to(pred_act.device)
        role_act_w = (target_act * aw).sum(dim=-1).detach()
    else:
        role_act_w = torch.ones_like(act_role_ce)
    base_role_weights = (target_spec[:, 1:] + inactive_role_weight) * legal_roles.float()
    role_weights = base_role_weights * role_act_w
    act_loss = (act_role_ce * role_weights).sum() / role_weights.sum().clamp(min=1e-6)
    gauge_anchor_act = 0.01 * (output["activity_logits"][..., 0] ** 2).mean()
    total_act_loss = act_loss + gauge_anchor_act
    act_role_l1 = (torch.abs(pred_act - target_act) * act_mask.float()).sum(dim=-1)
    act_l1 = (
        ((act_role_l1 * base_role_weights).sum() / base_role_weights.sum().clamp(min=1e-6)).item()
    )

    # 3. Log std regularization
    target_std_val = float(target_log_std)
    spec_std_loss = ((output["specialty_log_std"] - target_std_val) ** 2)[spec_mask].mean()
    act_std_loss = ((output["activity_log_std"] - target_std_val) ** 2)[act_mask].mean()
    std_reg_loss = spec_std_loss + act_std_loss

    # 4. Movement loss
    codes = output["movement_codes"]
    target_codes = batch_tensors["target_movement_code"]
    union_mask = output["movement_union_mask"]
    target_idx = torch.zeros(B, dtype=torch.long, device=codes.device)
    for b in range(B):
        matches = (codes == target_codes[b]).nonzero(as_tuple=True)[0]
        if len(matches) > 0 and union_mask[b, matches[0]]:
            target_idx[b] = matches[0]
        else:
            target_idx[b] = 0
    masked_movement_logits = output["movement_union_logits"].masked_fill(
        ~union_mask, -1e9
    )
    move_loss = F.cross_entropy(masked_movement_logits, target_idx)

    # 5. Fission loss
    fission_loss = F.cross_entropy(output["fission"], batch_tensors["target_fission"])

    # 6. Trade gate loss
    gate_logits = output["trade_gate"].view(-1, contracts.TRADE_GATES)
    gate_targets = batch_tensors["target_trade_gate"].view(-1)
    trade_gate_loss = F.cross_entropy(gate_logits, gate_targets)

    # 7. Trade value loss
    tv_mean_loss = (
        (output["trade_value_mean"] - batch_tensors["target_trade_value_latent"]) ** 2
    ).mean()
    tv_std_loss = ((output["trade_value_log_std"] - (-1.5)) ** 2).mean()
    trade_value_loss = tv_mean_loss + tv_std_loss

    # 8. Trade share loss
    ts_mean_loss = (
        (output["trade_share_mean"] - batch_tensors["target_trade_share_latent"]) ** 2
    ).mean()
    ts_std_loss = ((output["trade_share_log_std"] - (-1.5)) ** 2).mean()
    trade_share_loss = ts_mean_loss + ts_std_loss

    total_loss = (
        total_spec_loss
        + total_act_loss
        + std_reg_loss
        + move_loss
        + fission_loss
        + trade_gate_loss
        + trade_value_loss
        + trade_share_loss
    )
    return total_loss, spec_l1, act_l1


def _evaluate_dataset(
    actor: LaborActor,
    dataset: Mapping[str, Any],
    *,
    batch_size: int,
    target_log_std: float,
    inactive_role_weight: float = 0.0,
    label_smoothing: float = 1e-6,
    spec_weights: torch.Tensor | None = None,
    act_weights: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    actor.eval()
    tensor_keys = [
        k for k, v in dataset.items() if isinstance(v, torch.Tensor)
    ]
    N = dataset["target_specialty"].shape[0]
    total_loss_sum = 0.0
    spec_l1_sum = 0.0
    act_l1_sum = 0.0
    total_count = 0

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            batch = {k: dataset[k][start:end] for k in tensor_keys}
            count = end - start
            loss, spec_l1, act_l1 = _batch_loss_and_l1(
                actor,
                batch,
                target_log_std=target_log_std,
                inactive_role_weight=inactive_role_weight,
                label_smoothing=label_smoothing,
                spec_weights=spec_weights,
                act_weights=act_weights,
            )
            total_loss_sum += float(loss.item()) * count
            spec_l1_sum += spec_l1 * count
            act_l1_sum += act_l1 * count
            total_count += count

    return (
        total_loss_sum / max(1, total_count),
        spec_l1_sum / max(1, total_count),
        act_l1_sum / max(1, total_count),
    )


def pretrain_labor_actor_bc(
    *,
    actor: LaborActor,
    dataset: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    lr: float = 1e-3,
    epochs: int = 25,
    batch_size: int = 64,
    target_log_std: float = -2.0,
    max_grad_norm: float = 1.0,
    rng: np.random.Generator | None = None,
    inactive_role_weight: float = 0.0,
    label_smoothing: float = 1e-6,
    inverse_freq_power: float = 0.5,
) -> dict[str, Any]:
    """Supervised behavioral cloning on LaborActor using heuristic dataset."""
    if not isinstance(dataset, Mapping) or "target_specialty" not in dataset:
        raise ValueError("dataset must contain target_specialty and input tensors")
    N = int(dataset["target_specialty"].shape[0])
    if N == 0:
        raise ValueError("cannot pretrain on empty dataset")

    opt = (
        optimizer
        if optimizer is not None
        else torch.optim.Adam(actor.parameters(), lr=lr)
    )
    generator = rng if rng is not None else np.random.default_rng(0)
    tensor_keys = [k for k, v in dataset.items() if isinstance(v, torch.Tensor)]

    spec_weights: torch.Tensor | None = None
    act_weights: torch.Tensor | None = None
    if inverse_freq_power > 0.0:
        mean_spec = dataset["target_specialty"].mean(dim=0)
        spec_weights = (mean_spec + 0.02).pow(-inverse_freq_power)
        spec_weights = spec_weights / spec_weights.mean().clamp(min=1e-8)

        mean_act = dataset["target_activity"].mean(dim=0)
        act_weights = (mean_act + 0.02).pow(-inverse_freq_power)
        act_weights = act_weights / act_weights.mean().clamp(min=1e-8)

    # Compute baseline metrics before training
    init_loss, spec_l1_before, act_l1_before = _evaluate_dataset(
        actor,
        dataset,
        batch_size=batch_size,
        target_log_std=target_log_std,
        inactive_role_weight=inactive_role_weight,
        label_smoothing=label_smoothing,
        spec_weights=spec_weights,
        act_weights=act_weights,
    )

    actor.train()
    for _ in range(epochs):
        indices = generator.permutation(N)
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            batch_idx = torch.as_tensor(indices[start:end], dtype=torch.long)
            batch = {k: dataset[k][batch_idx] for k in tensor_keys}

            loss, _, _ = _batch_loss_and_l1(
                actor,
                batch,
                target_log_std=target_log_std,
                inactive_role_weight=inactive_role_weight,
                label_smoothing=label_smoothing,
                spec_weights=spec_weights,
                act_weights=act_weights,
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            opt.step()

    final_loss, spec_l1_after, act_l1_after = _evaluate_dataset(
        actor,
        dataset,
        batch_size=batch_size,
        target_log_std=target_log_std,
        inactive_role_weight=inactive_role_weight,
        label_smoothing=label_smoothing,
        spec_weights=spec_weights,
        act_weights=act_weights,
    )

    return {
        "contract": LABOR_BC_CONTRACT,
        "epochs": int(epochs),
        "rows": int(N),
        "initial_loss": float(init_loss),
        "final_loss": float(final_loss),
        "specialty_l1_before": float(spec_l1_before),
        "specialty_l1_after": float(spec_l1_after),
        "activity_l1_before": float(act_l1_before),
        "activity_l1_after": float(act_l1_after),
        "inactive_role_weight": float(inactive_role_weight),
        "label_smoothing": float(label_smoothing),
        "inverse_freq_power": float(inverse_freq_power),
        "collection": copy.deepcopy(dataset.get("collection", {})),
    }


def pretrain_critic_on_heuristic_next_states(
    *,
    task_critic: LaborTaskCritic,
    cost_critics: LaborCostCritics,
    normalizer: Any,
    world_seeds: Sequence[int],
    turns: int = 25,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_grad_norm: float = 1.0,
    personality: LaborPersonality | None = None,
    rng: np.random.Generator | None = None,
    labor_bridge_factory: Any = LaborModel9Bridge,
    heuristic_bridge_factory: Any = None,
    allow_vm_scale: bool = False,
) -> dict[str, Any]:
    """Train value critics directly on heuristic next-state discounted returns."""
    worlds = checked_seeds(list(world_seeds), "labor heuristic critic worlds")
    scan({"worldSeeds": worlds}, "labor heuristic critic worlds")
    if not allow_vm_scale and len(worlds) > 3:
        raise ValueError("local labor heuristic critic pretraining requires 1..3 world seeds")
    if not worlds:
        raise ValueError("world seeds cannot be empty")
    if turns < 1:
        raise ValueError("turns must be >= 1")

    from .labor_training import _settlement_guard, _transition_rewards

    declared_personality = LaborPersonality() if personality is None else personality
    heur_factory = heuristic_bridge_factory or (
        lambda: Model9Bridge(economic_diagnostics=True)
    )

    all_tensors_by_camp: dict[tuple[int, int], list[dict[str, torch.Tensor]]] = defaultdict(list)
    all_rewards_by_camp: dict[tuple[int, int], list[dict[str, float]]] = defaultdict(list)
    all_elapsed_by_camp: dict[tuple[int, int], list[float]] = defaultdict(list)
    all_terminal_by_camp: dict[tuple[int, int], list[bool]] = defaultdict(list)

    for seed in worlds:
        with labor_bridge_factory() as labor_bridge, heur_factory() as heur_bridge:
            step = labor_bridge.reset(seed)
            founded = {int(row["camp"]): int(step["turn"]) for row in step["villages"]}
            for _ in range(turns):
                live = [
                    v
                    for v in step["villages"]
                    if v.get("observation") is not None and not v.get("terminated")
                ]
                if not live:
                    break
                contexts, tensors = labor_policy_tensors(
                    live, normalizer, personality=declared_personality
                )
                catalogs = {context[0].camp_id: context[0] for context in contexts}

                # Shadow heuristic step to extract heuristic action
                snap = labor_bridge._bridge.capture()
                heur_bridge.reset(seed, snapshot=snap)
                h_step = heur_bridge.step({})
                h_villages = {int(v["camp"]): v for v in h_step.get("villages", [])}
                econ_by_camp = h_step.get("economicDiagnostics") or {}

                retained: dict[int, RetainedLaborAction] = {}
                row_by_camp: dict[int, int] = {}
                for row, (village, context) in enumerate(zip(live, contexts, strict=True)):
                    camp = int(village["camp"])
                    row_by_camp[camp] = row
                    catalog = context[0]
                    h_v = h_villages.get(camp, {})
                    econ = econ_by_camp.get(str(camp))
                    spec_target, act_target, schema12_action = (
                        extract_heuristic_labor_action(catalog, h_v, econ)
                    )
                    tv = [math.log(max(1e-4, float(x))) for x in schema12_action["tradeValue"]]
                    ts = float(schema12_action["tradeShare"])
                    ts_clamped = float(np.clip(ts, 1e-4, 1.0 - 1e-4))
                    retained[camp] = RetainedLaborAction(
                        payload=schema12_action,
                        labor_sample={
                            "specialty_allocation": torch.as_tensor(
                                spec_target, dtype=torch.float64
                            ).unsqueeze(0),
                            "activity_allocation": torch.as_tensor(
                                act_target, dtype=torch.float64
                            ).unsqueeze(0),
                            "specialty_latent": torch.zeros((1, 14), dtype=torch.float32),
                            "activity_latent": torch.zeros((1, 13, 22), dtype=torch.float32),
                            "specialty_mask": torch.tensor(
                                [catalog.masks.specialty], dtype=torch.bool
                            ),
                            "activity_mask": torch.tensor(
                                [catalog.masks.activity], dtype=torch.bool
                            ),
                        },
                        behavior_labor_logp=0.0,
                        nonlabor_sample={
                            "movement_index": torch.tensor([0], dtype=torch.long),
                            "movement": torch.tensor(
                                [int(schema12_action["movement"])], dtype=torch.long
                            ),
                            "fission": torch.tensor(
                                [int(schema12_action["fission"])], dtype=torch.long
                            ),
                            "trade_value_latent": torch.tensor(
                                tv, dtype=torch.float32
                            ).unsqueeze(0),
                            "trade_gate": torch.tensor(
                                [int(x) for x in schema12_action["tradeGate"]],
                                dtype=torch.long,
                            ).unsqueeze(0),
                            "trade_share_latent": torch.tensor(
                                [math.log(ts_clamped / (1.0 - ts_clamped))],
                                dtype=torch.float32,
                            ),
                            "movement_mask": tensors["movement_mask"][row : row + 1],
                            "movement_codes": torch.tensor([-1], dtype=torch.long),
                        },
                        behavior_nonlabor_logp=0.0,
                    )

                def retry_factory(
                    camp: int, catalog: LaborCatalog, attempt: LaborAttempt
                ) -> RetainedLaborAction:
                    idle_spec = np.zeros(14, dtype=np.float64)
                    idle_spec[0] = 1.0
                    idle_act = np.zeros((13, 22), dtype=np.float64)
                    idle_act[:, 0] = 1.0
                    safe_sample = LaborPolicySample(
                        tuple(idle_spec), tuple(tuple(r) for r in idle_act)
                    )
                    return RetainedLaborAction(
                        payload=safe_sample.payload(),
                        labor_sample={
                            "specialty_allocation": torch.as_tensor(
                                idle_spec, dtype=torch.float64
                            ).unsqueeze(0),
                            "activity_allocation": torch.as_tensor(
                                idle_act, dtype=torch.float64
                            ).unsqueeze(0),
                            "specialty_latent": torch.zeros((1, 14), dtype=torch.float32),
                            "activity_latent": torch.zeros(
                                (1, 13, 22), dtype=torch.float32
                            ),
                            "specialty_mask": torch.tensor(
                                [catalog.masks.specialty], dtype=torch.bool
                            ),
                            "activity_mask": torch.tensor(
                                [catalog.masks.activity], dtype=torch.bool
                            ),
                        },
                        behavior_labor_logp=0.0,
                        nonlabor_sample=None,
                        behavior_nonlabor_logp=None,
                    )

                execution = execute_labor_decision(
                    labor_bridge, catalogs, retained, retry_factory
                )
                next_step = execution.step
                rewards, terminal, elapsed = _transition_rewards(
                    next_step,
                    tuple(catalogs),
                    founded,
                    world_index=0,
                    decision_turn=int(step["turn"]),
                )
                outcomes = {int(village["camp"]): village for village in next_step["villages"]}

                for camp in catalogs:
                    row = row_by_camp[camp]
                    camp_key = (seed, camp)
                    all_tensors_by_camp[camp_key].append(_one_row(tensors, row))
                    outcome = outcomes[camp]
                    feas = float(execution.initial_attempts[camp].feasibility_cost)
                    guard = _settlement_guard(
                        outcome, founded.get(camp, int(step["turn"])), int(next_step["turn"])
                    )
                    all_rewards_by_camp[camp_key].append(
                        {
                            "qol_years": rewards[camp]["qol_years"],
                            "population_years": rewards[camp]["population_years"],
                            "feasibility": feas,
                            "settlement_guard": guard,
                        }
                    )
                    all_elapsed_by_camp[camp_key].append(elapsed[camp])
                    all_terminal_by_camp[camp_key].append(terminal[camp])

                step = next_step

    dataset_tensor_rows: list[dict[str, torch.Tensor]] = []
    task_targets_list: dict[str, list[float]] = {"qol_years": [], "population_years": []}
    cost_targets_list: dict[str, list[float]] = {"feasibility": [], "settlement_guard": []}

    for camp_key, camp_tensors in all_tensors_by_camp.items():
        T = len(camp_tensors)
        camp_rewards = all_rewards_by_camp[camp_key]
        camp_elapsed = all_elapsed_by_camp[camp_key]
        camp_terminal = all_terminal_by_camp[camp_key]

        returns = {name: [0.0] * T for name in (*LABOR_OBJECTIVES, *LABOR_COSTS)}
        running = {name: 0.0 for name in (*LABOR_OBJECTIVES, *LABOR_COSTS)}

        for t in reversed(range(T)):
            gamma = math.exp(-camp_elapsed[t] / 400.0)
            if camp_terminal[t]:
                running = {name: 0.0 for name in (*LABOR_OBJECTIVES, *LABOR_COSTS)}
            for name in (*LABOR_OBJECTIVES, *LABOR_COSTS):
                running[name] = camp_rewards[t][name] + gamma * running[name]
                returns[name][t] = running[name]

        for t in range(T):
            dataset_tensor_rows.append(camp_tensors[t])
            task_targets_list["qol_years"].append(returns["qol_years"][t])
            task_targets_list["population_years"].append(returns["population_years"][t])
            cost_targets_list["feasibility"].append(returns["feasibility"][t])
            cost_targets_list["settlement_guard"].append(returns["settlement_guard"][t])

    if not dataset_tensor_rows:
        raise ValueError("heuristic critic collection produced zero training rows")

    dataset_tensors = _concat(dataset_tensor_rows, "heuristic critic dataset")
    task_targets = {
        name: torch.tensor(vals, dtype=torch.float32)
        for name, vals in task_targets_list.items()
    }
    cost_targets = {
        name: torch.tensor(vals, dtype=torch.float32)
        for name, vals in cost_targets_list.items()
    }
    N = len(task_targets["qol_years"])

    critic_params = list(task_critic.parameters()) + list(cost_critics.parameters())
    optimizer = torch.optim.Adam(critic_params, lr=lr)

    with torch.no_grad():
        init_td_task = float(
            labor_task_value_loss(task_critic, dataset_tensors, task_targets).detach()
        )
        init_td_cost = float(
            labor_cost_value_loss(cost_critics, dataset_tensors, cost_targets).detach()
        )
        init_loss = init_td_task + init_td_cost

    generator = rng if rng is not None else np.random.default_rng(0)
    for _ in range(epochs):
        perm = generator.permutation(N)
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            batch_idx = torch.as_tensor(perm[start:end], dtype=torch.long)
            b_tensors = {k: v[batch_idx] for k, v in dataset_tensors.items()}
            b_task = {k: v[batch_idx] for k, v in task_targets.items()}
            b_cost = {k: v[batch_idx] for k, v in cost_targets.items()}

            optimizer.zero_grad()
            td_t = labor_task_value_loss(task_critic, b_tensors, b_task)
            td_c = labor_cost_value_loss(cost_critics, b_tensors, b_cost)
            loss = td_t + td_c
            loss.backward()
            nn.utils.clip_grad_norm_(critic_params, max_grad_norm)
            optimizer.step()

    with torch.no_grad():
        final_td_task = float(
            labor_task_value_loss(task_critic, dataset_tensors, task_targets).detach()
        )
        final_td_cost = float(
            labor_cost_value_loss(cost_critics, dataset_tensors, cost_targets).detach()
        )
        final_loss = final_td_task + final_td_cost

    return {
        "contract": "model9-heuristic-critic-pretraining-v1",
        "epochs": int(epochs),
        "rows": int(N),
        "initial_loss": float(init_loss),
        "final_loss": float(final_loss),
        "world_seeds": list(worlds),
        "turns": int(turns),
    }
