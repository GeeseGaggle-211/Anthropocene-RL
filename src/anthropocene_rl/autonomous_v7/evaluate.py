"""Closed-loop evaluation of a model9 checkpoint against the native heuristic (stage-1 gate).

    python -m anthropocene_rl.autonomous_v7.evaluate --checkpoint <pt> --seeds 2000-2019 --turns 25 --out <json>

Each held-out world runs twice from the same opening: the native heuristic for every village (teacher) and the
cloned policy for every village. Stages 1-3 keep movement and fission native. Welfare is realized population x
QoL x years summed over every settlement (lineage welfare summed over lineages), zero after extinction.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import contracts
from .bridge import Model9Bridge, observation_batch
from .gate1_diagnostics import (boundary, capture_component_bytes, capture_envelope_bytes,
                                action_decoder_receipt, generator_state_sha256, preserve_global_rng,
                                recompute_rng_receipt, trajectory_evidence,
                                validate_capture_bounds, validate_trajectory_evidence)
from .provenance import digest
from .model import (Actor, Normalizer, act_deterministic, bridge_actions_from_mode,
                    to_tensors)
from .upgrade import upgrade


def decoder_identity(actor: Actor) -> str:
    """Return the receipt/validator decoder identity from the constructed actor."""
    from .legacy_contract8 import Contract8Actor
    return "pinned-contract-8" if isinstance(actor, Contract8Actor) else "native-contract-12"


def load_strict_state(raw: dict[str, Any], *, resolved_semantics: dict[str, Any] | None,
                      diagnostic_loading: dict[str, Any] | None) -> tuple[Actor, Normalizer]:
    """Construct only when raw bytes exactly satisfy their declared Gate-1 loader."""
    if raw.get("contract", 8) == 8:
        if (not isinstance(diagnostic_loading, dict)
                or diagnostic_loading.get("mode") != "reviewed-adapter"
                or diagnostic_loading.get("adapter") != "pinned-contract-8"):
            raise ValueError("contract-8 diagnostic inference requires its reviewed pinned adapter")
        from .legacy_contract8 import load as load_contract8
        actor, normalizer = load_contract8(raw)
        from .worker_contract import from_state
        actor.worker_contract = from_state(raw)
        return actor, normalizer
    if raw.get("contract") != contracts.NATIVE_CONTRACT:
        raise ValueError("strict diagnostic evaluation requires the native raw checkpoint contract")
    required = {"experts", "token", "commitments", "learned_movement", "region_movement",
                "learned_fission", "learned_trade", "region_head", "grouped_region", "region_selection"}
    if not isinstance(resolved_semantics, dict) or set(resolved_semantics) != required:
        raise ValueError("strict diagnostic evaluation requires complete resolved checkpoint semantics")
    for key in ("experts", "token", "commitments", "learned_movement", "region_movement",
                "learned_fission", "learned_trade"):
        if key in raw and (type(raw[key]) is not bool or raw[key] != resolved_semantics[key]):
            raise ValueError("strict diagnostic semantics differ from explicit checkpoint metadata")
    actor_state = raw.get("actor", raw)
    if not isinstance(actor_state, dict):
        raise ValueError("strict diagnostic checkpoint lacks a raw actor state dict")
    if "region_flat" in actor_state:
        saved_head = "flat" if bool(actor_state["region_flat"]) else "hierarchical"
        if saved_head != resolved_semantics["region_head"]:
            raise ValueError("strict diagnostic region head differs from raw checkpoint metadata")
    core_width = raw.get("core_width")
    if isinstance(core_width, bool) or not isinstance(core_width, int) or core_width <= 0:
        raise ValueError("strict diagnostic checkpoint lacks an exact core width")
    actor = Actor(core_width, experts=resolved_semantics["experts"],
                  token=resolved_semantics["token"], region_head=resolved_semantics["region_head"])
    expected_state = actor.state_dict()
    if set(actor_state) != set(expected_state):
        missing = sorted(set(expected_state) - set(actor_state))
        unexpected = sorted(set(actor_state) - set(expected_state))
        raise ValueError(
            "strict diagnostic actor requires exact raw tensor keys "
            f"(missing={missing}, unexpected={unexpected})"
        )
    for name, reference in expected_state.items():
        value = actor_state[name]
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != reference.dtype or value.shape != reference.shape
                or value.layout != torch.strided):
            raise ValueError(
                f"strict diagnostic raw actor tensor {name} differs in dtype/device/shape/layout"
            )
    normalizer_state = raw.get("normalizer")
    required_normalizer = {"core_mean", "core_std"} | {
        f"{name}_{suffix}" for name, _, _ in contracts.SETS for suffix in ("mean", "std")
    }
    privileged_width = raw.get("privileged_width", 0)
    if (isinstance(privileged_width, bool) or not isinstance(privileged_width, int)
            or privileged_width < 0):
        raise ValueError("strict diagnostic checkpoint has an invalid privileged width")
    privileged = {"privileged_mean", "privileged_std"} if privileged_width else set()
    moment_keys = required_normalizer | privileged
    block_widths = {"core": core_width,
                    **{name: width for name, _, width in contracts.SETS}}
    if privileged_width:
        block_widths["privileged"] = privileged_width
    guarded_keys = moment_keys | {"guard"} | {f"{name}_dead" for name in block_widths}
    if (not isinstance(normalizer_state, dict)
            or set(normalizer_state) not in (moment_keys, guarded_keys)):
        raise ValueError("strict diagnostic normalizer keys differ from the native schema")
    for name, width in block_widths.items():
        for suffix in ("mean", "std"):
            raw_values = normalizer_state[f"{name}_{suffix}"]
            values = np.asarray(raw_values)
            if (not isinstance(raw_values, list) or values.shape != (width,)
                    or any(type(value) is not float
                           or not np.isfinite(value)
                           or float(np.float32(value)) != value for value in raw_values)
                    or not np.isfinite(values).all()
                    or suffix == "std" and not np.all(values > 0)):
                raise ValueError(f"strict diagnostic normalizer {name}_{suffix} is invalid")
    if "guard" in normalizer_state:
        guard = normalizer_state["guard"]
        if (type(guard) is not float
                or not np.isfinite(guard) or guard < 0):
            raise ValueError("strict diagnostic normalizer guard is invalid")
        for name, width in block_widths.items():
            dead = normalizer_state[f"{name}_dead"]
            values = np.asarray(dead)
            if (not isinstance(dead, list) or values.shape != (width,)
                    or values.dtype != np.dtype(bool)
                    or any(type(value) is not bool for value in dead)):
                raise ValueError(f"strict diagnostic normalizer {name}_dead is invalid")
    from .checkpoint import restore_actor
    restore_actor(actor, raw, resolved_semantics["region_head"]); actor.eval()
    actor.commitments = resolved_semantics["commitments"]
    actor.region_movement = resolved_semantics["region_movement"]
    actor.learned_fission = resolved_semantics["learned_fission"]
    actor.learned_trade = resolved_semantics["learned_trade"]
    actor.grouped_region = resolved_semantics["grouped_region"]
    actor.region_selection = resolved_semantics["region_selection"]
    from .worker_contract import from_state
    actor.worker_contract = from_state(raw)
    return actor, Normalizer.from_state(normalizer_state)


def load(checkpoint: Path, region_head: str | None = None, *, strict: bool = False,
         resolved_semantics: dict[str, Any] | None = None,
         diagnostic_loading: dict[str, Any] | None = None) -> tuple[Actor, Normalizer]:
    """Load an ordinary checkpoint, or a Gate-1 checkpoint without any upgrade/default inference.

    The compatibility loader remains available to existing callers.  ``strict`` is a separate diagnostic surface:
    it uses the weights-only deserializer, requires the current raw contract, and receives every constructor and
    decoder default from a content-bound resolution record.  It never calls :func:`upgrade`.
    """
    if (strict and isinstance(diagnostic_loading, dict)
            and diagnostic_loading.get("mode") == "reviewed-adapter"
            and diagnostic_loading.get("adapter") == "t3-initialization-bc-composite"):
        from .gate1_production_adapter import load_bc_phase
        raw = load_bc_phase(checkpoint, diagnostic_loading, base=checkpoint.parent)
        # The adapter has already proved and assembled an exact native raw
        # state.  The ordinary strict contract-12 loader performs the remaining
        # key/dtype/shape/normalizer checks without upgrade or defaults.
        return load_strict_state(raw, resolved_semantics=resolved_semantics,
                                 diagnostic_loading=None)
    if (strict and isinstance(diagnostic_loading, dict)
            and diagnostic_loading.get("mode") == "reviewed-adapter"
            and diagnostic_loading.get("adapter") == "t2m-guarded-bc-actor-only"):
        from .gate1_production_adapter import load_guarded_bc
        return load_guarded_bc(checkpoint, diagnostic_loading, base=checkpoint.parent)
    if (strict and isinstance(diagnostic_loading, dict)
            and diagnostic_loading.get("mode") == "reviewed-adapter"
            and diagnostic_loading.get("adapter") == "t3-init-guard-only-intervention"):
        from .gate1_production_adapter import load_guard_only_intervention
        raw = load_guard_only_intervention(checkpoint, diagnostic_loading,
                                           base=checkpoint.parent)
        return load_strict_state(raw, resolved_semantics=resolved_semantics,
                                 diagnostic_loading=None)
    if (strict and isinstance(diagnostic_loading, dict)
            and diagnostic_loading.get("mode") == "reviewed-adapter"
            and diagnostic_loading.get("adapter")
            == "recovered-contract12-inactive-region"):
        from .gate1_production_adapter import load_recovered_contract12
        raw = load_recovered_contract12(
            checkpoint, diagnostic_loading, resolved_semantics or {},
            base=checkpoint.parent,
        )
        return load_strict_state(raw, resolved_semantics=resolved_semantics,
                                 diagnostic_loading=None)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=strict)
    if strict:
        return load_strict_state(raw, resolved_semantics=resolved_semantics,
                                 diagnostic_loading=diagnostic_loading)
    state = upgrade(raw)
    experts, token = bool(state.get("experts", False)), bool(state.get("token", False))
    actor = Actor(int(state["core_width"]), experts=experts, token=token, region_head=region_head)
    from .checkpoint import restore_actor
    restore_actor(actor, state, region_head); actor.eval()
    # Contract 9: a policy trained without commitments decides every turn at evaluation too, whatever span head it
    # inherited. Checkpoints saved before the flag existed keep the span argmax, so recorded results reproduce.
    actor.commitments = bool(state.get("commitments", True))
    # Contract 10: a policy trained with learned movement moves over region rows; others keep the contract-8 pointer.
    actor.region_movement = bool(state.get("region_movement", False))
    actor.learned_fission = bool(state.get("learned_fission", False))
    actor.learned_trade = bool(state.get("learned_trade", False))
    from .worker_contract import from_state
    actor.worker_contract = from_state(state)
    from .model import load_checkpoint_normalizer
    return actor, load_checkpoint_normalizer(state)


def policy_actions(actor: Actor, normalizer: Normalizer, villages: list[dict[str, Any]],
                   movement: bool = False, mixes: dict[int, np.ndarray] | None = None,
                   personality: list[float] | dict[int, Any] | None = None,
                   decoder_receipts: dict[int, dict] | None = None,
                   decoder_semantics: dict[str, Any] | None = None) -> dict[int, dict[str, Any]]:
    """Deterministic actions; movement stays native unless ``movement`` (curriculum stage 4). ``mixes`` receives each
    village's economy mix. ``personality``: one vector for every village (a sweep point), or a map from society id to
    vector (societies not in it take the default)."""
    live = [v for v in villages if v.get("observation") is not None and not v.get("terminated")]
    if not live:
        return {}
    region = movement and getattr(actor, "region_movement", False)
    batch = observation_batch(live)
    if isinstance(personality, dict):
        default = [contracts.OBJECTIVE_WEIGHTS[k] for k in contracts.OBJECTIVES]
        batch["personality"] = np.asarray([personality.get(int(v["society"]), default) for v in live], np.float32)
    elif personality is not None:  # every village of the rollout at this personality (a sweep point)
        batch["personality"] = np.tile(np.asarray(personality, np.float32), (len(live), 1))
    from .legacy_contract8 import Contract8Actor
    with torch.no_grad():
        if isinstance(actor, Contract8Actor):
            from .legacy_contract8 import deterministic_mode, to_tensors as legacy_tensors
            output = actor(legacy_tensors(batch, normalizer))
            mode = deterministic_mode(output)
        else:
            output = actor(to_tensors(batch, normalizer))
            mode = act_deterministic(output, region=region,
                                     grouped=getattr(actor, "grouped_region", False),
                                     selection=getattr(actor, "region_selection", None))
    decoded = bridge_actions_from_mode(
        mode, movement=movement, region_movement=region,
        commitments=getattr(actor, "commitments", True),
        learned_fission=getattr(actor, "learned_fission", False),
        learned_trade=getattr(actor, "learned_trade", False),
    )
    actions = {int(village["camp"]): action
               for village, action in zip(live, decoded, strict=True)}
    if mixes is not None:
        for i, v in enumerate(live):
            mixes[int(v["camp"])] = np.asarray(mode["mix"][i], dtype=np.float64)
    if decoder_receipts is not None:
        if decoder_semantics is None:
            raise ValueError("Gate-1 decoder capture requires exact policy semantics")
        decoder = decoder_identity(actor)
        for index, village in enumerate(live):
            camp = int(village["camp"])
            decoder_receipts[camp] = action_decoder_receipt(
                mode="deterministic", decoder=decoder, semantics=decoder_semantics,
                observation=village["observation"], emitted_action=actions[camp], camp=camp,
                raw=output, row=index,
            )
    return actions


def stochastic_draw_order(actor: Actor, movement: bool) -> list[str]:
    """The exact local-generator draw order used by :func:`policy_actions_stochastic`."""
    region = movement and getattr(actor, "region_movement", False)
    order = (["region"] if region else []) + ["mix", "strategic", "service", "clothing", "shelter"]
    if movement and not region:
        order.append("movement")
    if getattr(actor, "commitments", True):
        order.append("commitment-span")
    if getattr(actor, "learned_fission", False):
        order.append("fission")
    if getattr(actor, "learned_trade", False):
        order.extend(("trade-value", "trade-gate", "trade-share"))
    return order


def policy_actions_stochastic(actor: Actor, normalizer: Normalizer, villages: list[dict[str, Any]],
                              movement: bool, personality: list[float] | dict[int, Any] | None,
                              schedule: dict, decoder_receipts: dict[int, dict] | None = None,
                              decoder_semantics: dict[str, Any] | None = None,
                              ) -> tuple[dict[int, dict[str, Any]], dict]:
    """Sample only through a fresh, schedule-bound CPU generator.

    The process-global Python, NumPy and Torch generators are neither consulted
    nor advanced.  State hashes before and after the named draw sequence are
    returned as execution evidence.
    """
    live = [v for v in villages if v.get("observation") is not None and not v.get("terminated")]
    expected_order = stochastic_draw_order(actor, movement)
    if schedule.get("draw_order") != expected_order:
        raise ValueError("policy RNG draw schedule differs from active policy heads")
    generator = torch.Generator(device="cpu").manual_seed(schedule["seed"])
    before = generator_state_sha256(generator)
    if before != schedule["state_sha256"]:
        raise ValueError("policy RNG initial state differs from frozen schedule")
    if not live:
        return {}, {"step": schedule["step"], "seed": schedule["seed"], "draw_order": expected_order,
                    "state_before_sha256": before, "state_after_sha256": before, "rows": 0}
    batch = observation_batch(live)
    if isinstance(personality, dict):
        default = [contracts.OBJECTIVE_WEIGHTS[k] for k in contracts.OBJECTIVES]
        batch["personality"] = np.asarray(
            [personality.get(int(v["society"]), default) for v in live], np.float32,
        )
    elif personality is not None:
        batch["personality"] = np.tile(np.asarray(personality, np.float32), (len(live), 1))
    from .ppo import Stage, sample, to_bridge_actions
    stage = Stage("gate1-diagnostic", movement=movement, routes=True,
                  description="diagnostic-only frozen stochastic execution",
                  commitments=getattr(actor, "commitments", True),
                  region_movement=movement and getattr(actor, "region_movement", False),
                  fission=getattr(actor, "learned_fission", False),
                  trade=getattr(actor, "learned_trade", False))
    with torch.no_grad():
        from .legacy_contract8 import Contract8Actor, to_tensors as legacy_tensors
        tensors = legacy_tensors(batch, normalizer) if isinstance(actor, Contract8Actor) else to_tensors(batch, normalizer)
        draw = sample(actor(tensors), tensors, stage, generator=generator)
        decoded = to_bridge_actions(draw, tensors)
    actions = {int(v["camp"]): action for v, action in zip(live, decoded, strict=True)}
    if decoder_receipts is not None:
        if decoder_semantics is None:
            raise ValueError("Gate-1 decoder capture requires exact policy semantics")
        decoder = decoder_identity(actor)
        for index, village in enumerate(live):
            camp = int(village["camp"])
            decoder_receipts[camp] = action_decoder_receipt(
                mode="seeded-stochastic", decoder=decoder, semantics=decoder_semantics,
                observation=village["observation"], emitted_action=actions[camp], camp=camp,
                raw=draw, row=index, economy_mix_mask=tensors["economy_mix_mask"],
            )
    after = generator_state_sha256(generator)
    receipt = recompute_rng_receipt(schedule, len(live))
    if receipt["state_after_sha256"] != after:
        raise ValueError("policy RNG execution differs from the frozen draw plan")
    return actions, receipt


def specialist_share(villages: list[dict[str, Any]]) -> float:
    """Population-weighted mean of the bridge's per-village ``specialistShare`` (0-1: goods output per resident;
    herding, farming and fishing count as food, not specialist work). 0 for bridges built before the field existed."""
    people = sum(v["population"] for v in villages)
    return float(sum(float(v.get("specialistShare", 0.0)) * v["population"] for v in villages) / max(1, people))


def producer_identity(bridge):
    from .teacher_data import binary_identity
    from .provenance import simulator_identity
    return {"simulator": simulator_identity(bridge.schema), "bridge_sha256": binary_identity()["sha256"],
            "flags": {"sim_rules": os.environ.get("ANTHROPOCENE_SIM_RULES", ""),
                      "sim_statics": os.environ.get("ANTHROPOCENE_SIM_STATICS", "")},
            "raw_materials": bridge.last_reset_identity["rawMaterialsEnabled"]}


def _validated_reset_turn(step: dict[str, Any], expected_reset_turn: int | None, *,
                          bind_observed_reset_turn: bool, world_contract: dict[str, Any],
                          snapshot: bytes | None) -> int:
    """Validate and return the authoritative post-reset turn.

    Fresh seeded generation has no turn before ``reset``: its reviewed world
    descriptor therefore carries ``turn=None`` and the native bridge supplies
    the boundary turn.  Snapshot-backed worlds remain pre-materialized and
    must match their reviewed turn exactly.
    """
    if type(bind_observed_reset_turn) is not bool or not isinstance(step, dict):
        raise ValueError("reset-turn binding declaration is malformed")
    actual = step.get("turn")
    if type(actual) is not int or actual < 0:
        raise ValueError("bridge reset lacks an exact nonnegative native turn")
    if bind_observed_reset_turn:
        if (expected_reset_turn is not None or snapshot is not None
                or world_contract.get("snapshot_sha256") not in {None, ""}
                or world_contract.get("turn") is not None):
            raise ValueError("observed reset-turn binding is only valid for a fresh generated world")
        return actual
    if type(expected_reset_turn) is not int or expected_reset_turn < 0:
        raise ValueError("reviewed snapshot/reset boundary lacks an exact turn")
    if actual != expected_reset_turn:
        raise ValueError("bridge reset turn differs from the reviewed world boundary")
    return actual


def rollout(seed: int, turns: int, checkpoint: str | None, withheld: int = 0, movement: bool = False,
            commitments: bool | None = None, region: bool | None = None, warmup: int = 0,
            fission: bool | None = None, trade: bool | None = None,
            personality: list[float] | dict[int, Any] | None = None,
            grouped: bool = False, focal: bool = False, expected_producer=None, world_contract=None,
            strict_checkpoint: bool = False, resolved_semantics: dict[str, Any] | None = None,
            evaluation_semantics: dict[str, Any] | None = None,
            snapshot: bytes | None = None, action_mode: str = "deterministic",
            policy_rng_steps: list[dict] | None = None,
            requested_metrics: list[str] | None = None,
            service_fields: list[dict] | None = None,
            diagnostic_loading: dict[str, Any] | None = None,
            capture_bounds: dict[str, int] | None = None,
            evaluation_world_id: str | None = None,
            expected_reset_turn: int | None = None,
            bind_observed_reset_turn: bool = False,
            observation_core_width: int | None = None) -> dict[str, Any]:
    """``movement`` enables learned rollout movement; ``region`` selects its contract-10 regional decoder.
    ``commitments`` and ``region`` override the checkpoint's own settings.
    ``grouped``: the hierarchical region choice (model.grouped_region_slot) instead of a plain argmax.
    ``focal`` (with ``personality``): only alternate founding lineages (and their daughters) take ``personality``; the
    others draw Dirichlet(1) personalities, as in training. The focal lineages' outcomes are reported separately.
    ``warmup``: the heuristic plays that many turns first (selected fixture states, e.g. valley fixtures); they count
    toward neither welfare nor decisions."""
    torch.set_num_threads(1)
    actor = normalizer = None
    if action_mode not in {"deterministic", "seeded-stochastic"}:
        raise ValueError("evaluation action mode is unsupported")
    if action_mode == "seeded-stochastic" and (not isinstance(policy_rng_steps, list)
                                                 or len(policy_rng_steps) != turns):
        raise ValueError("seeded-stochastic evaluation needs an exact per-step RNG schedule")
    if action_mode == "deterministic" and policy_rng_steps is not None:
        raise ValueError("deterministic evaluation cannot carry a policy RNG schedule")
    if checkpoint is not None:
        # Actor constructors initialise parameters before the strict restore.
        # Preserve the caller's process-global RNG streams around that work.
        with preserve_global_rng():
            if strict_checkpoint:
                actor, normalizer = load(Path(checkpoint), strict=True, resolved_semantics=resolved_semantics,
                                         diagnostic_loading=diagnostic_loading)
                if evaluation_semantics is not None:
                    if hasattr(actor, "set_region_head"):
                        actor.set_region_head(evaluation_semantics["region_head"])
                    actor.region_selection = evaluation_semantics["region_selection"]
            else:
                actor, normalizer = load(Path(checkpoint))
        if commitments is not None:
            actor.commitments = commitments
        if region is not None:
            actor.region_movement = region
        if fission is not None:
            actor.learned_fission = fission
        if trade is not None:
            actor.learned_trade = trade
        actor.grouped_region = grouped
    welfare = 0.0; camp_turns = hungry = interventions = decisions = 0
    previous: dict[int, np.ndarray] = {}; changes: list[float] = []
    spans: dict[int, int] = {}
    moves = 0
    policies: dict[int, int] = {}
    trade_utility = 0.0; trades = 0
    diagnostic_records = []
    traits: dict[int, Any] = {}; focal_set: set[int] = set()
    focal_rng = np.random.default_rng([seed, 31])
    focal_welfare = 0.0; focal_roots = set()
    requested_metrics = list(requested_metrics or [])
    economic_required = bool(requested_metrics)
    if economic_required:
        capture_bounds = validate_capture_bounds(capture_bounds, turns)
        if not service_fields:
            raise ValueError("Gate-1 diagnostic capture requires its registered service fields")
        if (type(bind_observed_reset_turn) is not bool
                or (bind_observed_reset_turn and expected_reset_turn is not None)
                or (not bind_observed_reset_turn
                    and (type(expected_reset_turn) is not int or expected_reset_turn < 0))
                or type(observation_core_width) is not int or observation_core_width <= 0
                or not isinstance(evaluation_semantics, dict)):
            raise ValueError("Gate-1 diagnostic capture lacks reviewed reset/observation/policy semantics")
    trajectory_steps: list[dict] = []
    retained_step_bytes: list[int] = []
    policy_rng_receipts: list[dict] = []

    def assign(villages: list[dict[str, Any]]) -> None:
        founders = sorted(int(v["society"]) for v in villages
                          if int(v["society"]) not in traits and int(v.get("parentSociety", 0)) not in traits)
        for v in villages:  # daughters inherit their parent's personality and focal membership
            society, parent = int(v["society"]), int(v.get("parentSociety", 0))
            if society not in traits and parent in traits:
                traits[society] = traits[parent]
                if parent in focal_set:
                    focal_set.add(society)
        for i, society in enumerate(founders):
            if i % 2 == 0:
                traits[society] = list(personality); focal_set.add(society); focal_roots.add(society)
            else:
                traits[society] = focal_rng.dirichlet(np.ones(contracts.PERSONALITY_WIDTH)).tolist()

    bridge_context = Model9Bridge(economic_diagnostics=True) if economic_required else Model9Bridge()
    with bridge_context as bridge:
        from .worker_contract import validate, world
        expected_producer = expected_producer or getattr(actor, "worker_contract", None)
        world_contract = world_contract or world(seed, withheld, movement=movement, head=getattr(actor, "region_head", None))
        worker = validate(bridge, expected_producer, world_contract, snapshot=snapshot,
                          head=getattr(actor, "region_head", None))
        step = bridge.reset(seed, players=world_contract["players"], withheld=withheld,
                            movement=movement, snapshot=snapshot)
        reset_turn = (_validated_reset_turn(
            step, expected_reset_turn,
            bind_observed_reset_turn=bind_observed_reset_turn,
            world_contract=world_contract, snapshot=snapshot,
        ) if economic_required else int(step["turn"]))
        reset_fingerprint = str(step["fingerprint"])
        diagnostic_world_id = evaluation_world_id or digest(world_contract)
        warmup_chain = []
        reset_record = ({"world_id": diagnostic_world_id,
                         "snapshot_sha256": world_contract["snapshot_sha256"],
                         "turn": reset_turn, "fingerprint": reset_fingerprint,
                         "trajectory_id": reset_fingerprint, "warmup_chain": []}
                        if economic_required else None)
        if (economic_required
                and capture_envelope_bytes(capture_component_bytes(reset_record),
                                           capture_component_bytes(None), [])
                > capture_bounds["max_canonical_bytes"]):
            raise ValueError("trajectory reset exceeds its reviewed canonical-byte capture bound")
        from .teacher_data import binary_identity
        producer = producer_identity(bridge)
        for warmup_index in range(warmup):
            warmup_source = step
            step = bridge.step({})
            if economic_required:
                link = {
                    "index": warmup_index,
                    "source_turn": int(warmup_source["turn"]),
                    "source_fingerprint": warmup_source["fingerprint"],
                    "successor_turn": int(step["turn"]),
                    "successor_fingerprint": step["fingerprint"],
                    "trajectory_id": step.get("trajectoryId"),
                }
                candidate_reset = {**reset_record, "warmup_chain": [*warmup_chain, link]}
                if (capture_envelope_bytes(capture_component_bytes(candidate_reset),
                                           capture_component_bytes(None), [])
                        > capture_bounds["max_canonical_bytes"]):
                    raise ValueError("trajectory warm-up exceeds its reviewed canonical-byte capture bound")
                reset_record = candidate_reset
                warmup_chain = candidate_reset["warmup_chain"]
        opening_boundary = boundary(step) if economic_required else None
        opening_bytes = capture_component_bytes(opening_boundary) if economic_required else 0
        reset_bytes = capture_component_bytes(reset_record) if economic_required else 0
        if (economic_required
                and capture_envelope_bytes(reset_bytes, opening_bytes, retained_step_bytes)
                > capture_bounds["max_canonical_bytes"]):
            raise ValueError("trajectory opening exceeds its reviewed canonical-byte capture bound")
        for evaluation_step in range(turns):
            mixes: dict[int, np.ndarray] = {}
            decoder_receipts: dict[int, dict] = {}
            if focal and personality is not None:
                assign(step["villages"])
            chosen = traits if focal and personality is not None else personality
            pre_step = step
            if actor is None:
                actions = {}
            elif action_mode == "seeded-stochastic":
                actions, receipt = policy_actions_stochastic(
                    actor, normalizer, step["villages"], movement, chosen,
                    policy_rng_steps[evaluation_step],
                    decoder_receipts=(decoder_receipts if economic_required else None),
                    decoder_semantics=evaluation_semantics,
                )
                policy_rng_receipts.append(receipt)
            else:
                actions = policy_actions(
                    actor, normalizer, step["villages"], movement, mixes, chosen,
                    decoder_receipts=(decoder_receipts if economic_required else None),
                    decoder_semantics=evaluation_semantics,
                )
            for action in actions.values():
                spans[action.get("commit", 1)] = spans.get(action.get("commit", 1), 0) + 1
                code = action.get("movement", contracts.NATIVE_MOVEMENT)
                moves += int(code == contracts.KEEP_INTENT or code > contracts.REGION_BASE)
                policies[action.get("fission", -1)] = policies.get(action.get("fission", -1), 0) + 1
            for camp, mix in mixes.items():
                if camp in previous:
                    changes.append(0.5 * float(np.abs(mix - previous[camp]).sum()))
                previous[camp] = mix
            decisions += len(actions)
            step = bridge.step(actions)
            terminal = not any(not village["terminated"] for village in step["villages"])
            if economic_required:
                economic = [copy.deepcopy(bridge.economic_diagnostics[camp])
                            for camp in sorted(bridge.economic_diagnostics)]
                successor = boundary(step)
                transition_id = digest({"index": evaluation_step, "source_turn": int(pre_step["turn"]),
                                        "source_fingerprint": pre_step["fingerprint"],
                                        "successor_turn": int(step["turn"]),
                                        "successor_fingerprint": step["fingerprint"]})
                captured_step = {
                    "index": evaluation_step, "transition_id": transition_id,
                    "source_turn": int(pre_step["turn"]),
                    "source_fingerprint": pre_step["fingerprint"],
                    "actions": [{"camp": camp, "action_request": copy.deepcopy(actions[camp]),
                                 "decoded_action": copy.deepcopy(actions[camp]),
                                 "decoder_receipt": copy.deepcopy(decoder_receipts[camp])}
                                for camp in sorted(actions)],
                    "policy_rng_receipt": (copy.deepcopy(receipt)
                                           if action_mode == "seeded-stochastic" else None),
                    "successor": successor,
                    "terminal": ({"terminated": True, "truncated": False,
                                  "reason": "all-villages-terminated"}
                                 if terminal else
                                 {"terminated": False, "truncated": evaluation_step == turns - 1,
                                  "reason": ("evaluation-horizon" if evaluation_step == turns - 1
                                             else "continuing")}),
                    "economic": economic,
                }
                if len(successor["villages"]) > capture_bounds["max_villages_per_boundary"]:
                    raise ValueError("trajectory successor exceeds its reviewed village capture bound")
                if len(economic) > capture_bounds["max_economic_rows_per_step"]:
                    raise ValueError("trajectory transition exceeds its reviewed economic-row capture bound")
                captured_bytes = capture_component_bytes(captured_step)
                if (capture_envelope_bytes(reset_bytes, opening_bytes,
                                           [*retained_step_bytes, captured_bytes])
                        > capture_bounds["max_canonical_bytes"]):
                    raise ValueError("trajectory transition exceeds its reviewed canonical-byte capture bound")
                trajectory_steps.append(captured_step)
                retained_step_bytes.append(captured_bytes)
            if focal and personality is not None: assign(step["villages"])
            diagnostic_records.extend(getattr(bridge, "decision_diagnostics", []))
            for v in step["villages"]:
                if v["priorPopulation"] <= 0:
                    continue
                welfare += float(v["welfare"]); camp_turns += 1
                if int(v["society"]) in focal_set: focal_welfare += float(v["welfare"])
                trade_utility += float(v.get("tradeUtility", 0.0)); trades += int(abs(float(v.get("tradeUtility", 0.0))) > 0)
                hungry += int(v["nutrition"] < 0.8 and not v["terminated"])
                interventions += int(v["interventions"])
            if terminal:
                break
        replay = bridge.verify_replay()
        final = [v for v in step["villages"] if not v["terminated"] and v["population"] > 0]
        if binary_identity()["sha256"] != producer["bridge_sha256"]:
            raise ValueError("bridge binary changed during evaluation rollout")
    population = sum(v["population"] for v in final)
    weight = max(1, population)
    focal_final = [v for v in final if int(v["society"]) in focal_set]
    focal_people = sum(v["population"] for v in focal_final)
    focal_stats = {"focal_population": focal_people, "focal_settlements": len(focal_final),
                   "focal_qol": sum(v["qol"] * v["population"] for v in focal_final) / max(1, focal_people),
                   "focal_below_30": sum(1 for v in focal_final if v["population"] < 30),
                   "focal_welfare": focal_welfare, "focal_root_lineages": len(focal_roots),
                   "focal_population_denominator": focal_people,
                   "focal_extinctions": sum(1 for v in step["villages"] if v["terminated"] and int(v["society"]) in focal_set),
                   "focal_societies": sorted(focal_set),
                   "focal_specialist_share": specialist_share(focal_final)} if focal else {}
    from .diagnostics import aggregate
    result = {
        "producer": producer,
        "worker_contract": worker,
        "world_contract": world_contract,
        "personality": personality,
        "decision_diagnostics": aggregate(diagnostic_records),
        "region_head": actor.region_head if actor is not None else None,
        **focal_stats,
        "seed": seed, "withheld": withheld, "policy": checkpoint is not None, "welfare": welfare, "population": population,
        "warmup": warmup,
        "settlements": len(final), "villages_600": sum(1 for v in final if v["population"] >= 600),
        "nutrition": sum(v["nutrition"] * v["population"] for v in final) / weight,
        "clothing": sum(v["clothing"] * v["population"] for v in final) / weight,
        "housing": sum(v["housing"] * v["population"] for v in final) / weight,
        "qol": sum(v["qol"] * v["population"] for v in final) / weight,
        # Round 3: the population-weighted share of goods (specialist) output per resident in final villages.
        "specialist_share": specialist_share(final),
        "hungry_share": hungry / max(1, camp_turns),
        "extinctions": sum(1 for v in step["villages"] if v["terminated"]),
        "below_30": sum(1 for v in final if v["population"] < 30),
        "interventions_per_decision": interventions / max(1, decisions), "replay_steps": replay,
        # Turn-to-turn total-variation change of each village's economy mix, and the share of abrupt switches.
        "mix_change": float(np.mean(changes)) if changes else 0.0,
        "mix_switches": float(np.mean([c > 0.5 for c in changes])) if changes else 0.0,
        # Contract 9: share of decisions holding (0), deciding one turn (1) or committing (2-6).
        "hold_share": spans.get(0, 0) / max(1, sum(spans.values())),
        "commit_share": sum(v for k, v in spans.items() if k >= 2) / max(1, sum(spans.values())),
        # Contract 10: share of decisions that chose a region row other than the current tile, or kept an intent.
        "region_move_share": moves / max(1, sum(spans.values())),
        # Learned fission: share of decisions setting Discourage (1) and Encourage (2); -1 is native.
        "fission_discourage_share": policies.get(1, 0) / max(1, sum(policies.values())),
        "fission_encourage_share": policies.get(2, 0) / max(1, sum(policies.values())),
        # Contract 11: realised trade utility summed over villages and turns, and village-turns that traded.
        "trade_utility": trade_utility, "trades": trades,
        "snapshot_sha256": world_contract["snapshot_sha256"],
        "action_mode": action_mode,
        "policy_rng_receipts": policy_rng_receipts,
        "evaluation_steps": len(trajectory_steps),
    }
    if economic_required:
        result["trajectory_evidence"] = trajectory_evidence(
            opening_boundary, trajectory_steps, capture_bounds,
            world_id=diagnostic_world_id, snapshot_sha256=world_contract["snapshot_sha256"],
            reset_turn=reset_turn, reset_fingerprint=reset_fingerprint,
            warmup_chain=warmup_chain,
        )
        recomputed = validate_trajectory_evidence(
            result["trajectory_evidence"], world_id=diagnostic_world_id,
            snapshot_sha256=world_contract["snapshot_sha256"], horizon=turns,
            bounds=capture_bounds, action_mode=action_mode,
            rng_steps=policy_rng_steps, service_fields=service_fields,
            expected_opening_turn=reset_turn + warmup,
            expected_reset_turn=reset_turn,
            observation_core_width=observation_core_width,
            policy_semantics=evaluation_semantics,
            expected_decoder=decoder_identity(actor),
        )
        result.update({key: recomputed[key] for key in (
            "welfare", "population_years", "qol_years", "trade_utility",
            "extinctions", "settlements", "service_outcomes", "compiler_interventions",
            "trajectory_digest", "policy_rng_receipts", "evaluation_steps", "capture_receipt",
        )})
        result["population"] = recomputed["ending_residents"]
        result["compiler_intervention_summary"] = recomputed["compiler_intervention_summary"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", default="2000-2019")
    parser.add_argument("--turns", type=int, default=25)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    from .checkpoint import parse
    args = parse(parser)
    a, b = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(a, b + 1))
    clock = time.time()
    from functools import partial
    from .worker_contract import from_state, world
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected = from_state(saved)
    bound = partial(rollout, expected_producer=expected)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        teacher = list(pool.map(bound, seeds, [args.turns] * len(seeds), [None] * len(seeds)))
        policy = list(pool.map(bound, seeds, [args.turns] * len(seeds), [str(args.checkpoint)] * len(seeds)))
    if expected is not None and any(row.get("worker_contract") != expected for row in teacher + policy):
        from .provenance import ProvenanceError
        raise ProvenanceError("evaluation worker differs from frozen checkpoint producer")
    from .checkpoint import resolve_head
    descriptors = [world(seed) for seed in seeds] + [world(seed, head=resolve_head(saved)) for seed in seeds]
    if (len(teacher) != len(seeds) or len(policy) != len(seeds)
            or any(row.get("world_contract") != descriptor for row, descriptor in zip(teacher + policy, descriptors))):
        from .provenance import ProvenanceError
        raise ProvenanceError("evaluation worker differs from requested world contract")
    ratio = [p["welfare"] / max(1e-9, t["welfare"]) for p, t in zip(policy, teacher)]

    def mean(rows: list[dict[str, Any]], key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    keys = ["welfare", "population", "settlements", "villages_600", "nutrition", "clothing", "housing", "qol", "specialist_share",
            "hungry_share", "extinctions", "below_30", "interventions_per_decision"]
    report = {
        "seeds": args.seeds, "turns": args.turns, "checkpoint": str(args.checkpoint),
        "welfare_ratio_total": sum(p["welfare"] for p in policy) / max(1e-9, sum(t["welfare"] for t in teacher)),
        "welfare_ratio_median": float(np.median(ratio)), "welfare_ratio_min": float(np.min(ratio)),
        "teacher": {k: mean(teacher, k) for k in keys}, "policy": {k: mean(policy, k) for k in keys},
        "per_seed": [{"seed": t["seed"], "teacher": t["welfare"], "policy": p["welfare"], "ratio": r}
                     for t, p, r in zip(teacher, policy, ratio)],
        "seconds": time.time() - clock,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("welfare_ratio_total", "welfare_ratio_median", "welfare_ratio_min", "teacher", "policy", "seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
