"""Labor task-critic value support: target audit and checkpoint conversion.

The task critics encode ``y = clip(symlog(G / s), low, high)`` two-hot over
``bins`` evenly spaced values and decode ``s * sum_j p_j symexp(b_j)``. The
historical support ``[-2, 9] x 255`` can decode up to about 8102 reference
scales, so a little stray mass in the upper bins dominates the decoded value
(brief section 17.2). Changing the support is a versioned representation
change declared in the checkpoint's labor model contract (see
``LaborValueSupport``); this module provides the evidence and the migration:

``audit``
    Distribution of the normalized task targets ``symlog(G / s)`` (and the raw
    cost targets) on a teacher dataset, with clipping fractions for candidate
    supports and a recommended narrower support.

``convert``
    Re-declare an existing trainer checkpoint's task-critic support. Only the
    two task-head output layers (and the ``bins`` buffer) are re-initialised;
    state, the cost critics, actor, normalizer and trainer RNG
    state are kept. The critic optimizer is reset (its head moments no longer
    match), and the conversion is recorded in the checkpoint.

``convert-privileged``
    Migrate a checkpoint without a privileged block to the critic-only
    privileged architecture (intervention A). Every existing tensor (actor,
    critic encoders and heads) is loaded strictly; only the new privileged
    family projectors and embeddings are initialised, from a declared seed,
    through the critic encoders' strict-load migration hook. The frozen
    privileged normalizer is bound, the critic optimizer is reset and the
    migration is recorded.

``convert-qol-encoder``
    Give a shared-encoder checkpoint's task critic a separate QoL encoder
    (contract ``qolEncoder: "separate"``). Every existing tensor is kept; the
    new encoder is either a copy of the trained task encoder (``--init copy``:
    a warm start whose outputs equal the source critic's exactly) or a fresh
    initialization seeded with ``labor_qol_encoder_seed(seed)`` (``--init
    fresh``: for an untrained ``initialize`` checkpoint and the default seed it
    is the same tensors ``initialize --qol-encoder separate`` builds). The
    critic optimizer is reset and the conversion is recorded.

Example::

    python -m anthropocene_rl.autonomous_v7.labor_critic_support audit \\
        --dataset fit-data --out audit.json
    python -m anthropocene_rl.autonomous_v7.labor_critic_support convert \\
        --checkpoint critic2.pt --critic-value-support=-1,4,255 --out converted.pt
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .labor_actor_optim import rebuild_actor_optimizer
from .labor_model import (
    LABOR_COSTS,
    LABOR_OBJECTIVE_REFERENCE_SCALES,
    LABOR_OBJECTIVES,
    LaborCostCritics,
    LaborCriticEncoder,
    LaborEncoderLayout,
    LaborTaskCritic,
    LaborValueSupport,
    bind_privileged_normalizer,
    labor_checkpoint_contract,
    labor_checkpoint_personality,
    labor_qol_encoder_seed,
    set_labor_film_bound,
)

CONVERSION_CONTRACT = "labor-critic-value-support-conversion-v1"
PRIVILEGED_CONVERSION_CONTRACT = "labor-critic-privileged-migration-v1"
QOL_ENCODER_CONVERSION_CONTRACT = "labor-critic-separate-qol-encoder-conversion-v1"
QOL_ENCODER_INITS = ("copy", "fresh")
AUDIT_SCHEMA = "labor-critic-target-audit-v1"
QUANTILES = (0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0)
DEFAULT_CANDIDATES = ((-2.0, 9.0), (-2.0, 6.0), (-2.0, 5.0), (-1.0, 4.0), (-1.0, 3.0), (-0.5, 2.5))


def _support_record(support: LaborValueSupport) -> dict[str, Any]:
    return {
        "low": support.low,
        "high": support.high,
        "bins": support.bins,
        "default": support.is_default,
    }


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {f"p{q * 100:g}": None for q in QUANTILES}
    out = np.quantile(values, QUANTILES)
    names = {0.0: "min", 1.0: "max"}
    return {names.get(q, f"p{q * 100:g}"): float(v) for q, v in zip(QUANTILES, out, strict=True)}


def _symlog(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def target_audit(
    dataset: Mapping[str, Any],
    candidates: Sequence[tuple[float, float]] = DEFAULT_CANDIDATES,
    *,
    margin: float = 0.5,
) -> dict[str, Any]:
    """Audit fitting targets exactly as ``pretrain_critic_on_dataset`` builds them."""
    from .labor_bc import critic_target_available, discounted_returns

    returns = {k: v.double().numpy() for k, v in discounted_returns(dataset).items()}
    n = int(dataset["row_world"].shape[0])
    keep = (
        ~dataset["row_quarantined"].reshape(n).bool() & critic_target_available(dataset)
    ).numpy()
    task: dict[str, Any] = {}
    for name in LABOR_OBJECTIVES:
        scale = float(LABOR_OBJECTIVE_REFERENCE_SCALES[name])
        normalized = returns[name][keep] / scale
        y = _symlog(normalized)
        clipping = {}
        for low, high in candidates:
            below, above = y < low, y > high
            clipping[f"{low:g},{high:g}"] = {
                "below_fraction": float(below.mean()) if y.size else 0.0,
                "above_fraction": float(above.mean()) if y.size else 0.0,
                # Normalized-value error from clamping, mean over all rows.
                "mean_abs_clip_error": float(
                    np.where(above, normalized - math.expm1(high), 0.0).mean()
                    + np.where(below, -math.expm1(-low) - normalized, 0.0).mean()
                )
                if y.size
                else 0.0,
            }
        task[name] = {
            "reference_scale": scale,
            "rows": int(y.size),
            "symlog_normalized": _quantiles(y),
            "normalized": _quantiles(normalized),
            "fraction_zero": float((normalized == 0.0).mean()) if y.size else 0.0,
            "clipping": clipping,
        }
    costs = {}
    for name in LABOR_COSTS:
        raw = returns[name][keep]
        costs[name] = {
            "raw": _quantiles(raw),
            "mean_square": float(np.square(raw).mean()) if raw.size else 0.0,
            "fraction_zero": float((raw == 0.0).mean()) if raw.size else 0.0,
        }
    return {
        "schema": AUDIT_SCHEMA,
        "rows": int(n),
        "admitted_rows": int(keep.sum()),
        "worlds": len(np.unique(dataset["row_world"].reshape(-1).numpy()[keep])),
        "targets": "per (world, camp) truncated discounted returns, quarantined rows excluded",
        "task_heads": task,
        "cost_heads": costs,
        "recommended_support": recommend_support(task, margin=margin),
    }


def recommend_support(task_audit: Mapping[str, Any], *, margin: float = 0.5) -> dict[str, Any]:
    """Smallest half-unit support covering every head's observed range plus ``margin``.

    The observed max (not P99.9) bounds ``high`` because clipped targets bias
    the fit; the margin leaves room for unseen states. ``low`` keeps at least
    -0.5 so zero targets stay interior.
    """
    highs = [head["symlog_normalized"]["max"] for head in task_audit.values()]
    lows = [head["symlog_normalized"]["min"] for head in task_audit.values()]
    if any(value is None for value in highs + lows):
        return {"low": None, "high": None, "bins": None}
    high = math.ceil((max(highs) + margin) * 2.0) / 2.0
    low = min(-0.5, math.floor((min(lows) - margin) * 2.0) / 2.0)
    support = LaborValueSupport(max(low, -2.0), min(high, 9.0), 255)
    return {
        **_support_record(support),
        "decoded_max_reference_scales": float(math.expm1(support.high)),
        "margin": float(margin),
        "flag": f"{support.low:g},{support.high:g},{support.bins}",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_task_critic_support(
    state: Mapping[str, Any],
    support: LaborValueSupport,
    *,
    seed: int | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a trainer checkpoint whose task heads use ``support``.

    Keeps the task encoder, cost critics, actor, normalizer and trainer RNG
    states; re-initialises only the two task-head ``nn.Linear`` layers (and
    the ``bins`` buffer) from ``seed``; resets the critic optimizer.
    """
    from .labor_training import (
        _models_from_checkpoint,
        _require_trainer_checkpoint,
        _trainer_checkpoint,
    )

    _require_trainer_checkpoint(state)
    layout, actor, old_task, costs, normalizer = _models_from_checkpoint(state)
    contract = labor_checkpoint_contract(state, expected_layout=layout)
    if support == old_task.support:
        raise ValueError("checkpoint already declares this value support")
    if old_task.qol_head is not None:
        raise ValueError("value-support conversion does not apply to a bounded QoL head checkpoint")
    # Same (default) widths as _models_from_checkpoint, which just loaded old_task strictly.
    width = old_task.heads[LABOR_OBJECTIVES[0]].in_features
    head_seed = int(state["optimizer_seed"] if seed is None else seed)
    task = LaborTaskCritic(layout, support=support, qol_encoder=old_task.qol_encoder_mode)
    task.encoder.load_state_dict(old_task.encoder.state_dict(), strict=True)
    if old_task.qol_encoder is not None:
        task.qol_encoder.load_state_dict(old_task.qol_encoder.state_dict(), strict=True)
    bind_privileged_normalizer(getattr(old_task, "privileged_normalizer", None), task)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(head_seed)
        task.heads = nn.ModuleDict(
            {name: nn.Linear(width, support.bins) for name in LABOR_OBJECTIVES}
        )
    set_labor_film_bound(contract.conditioning.film_bound, task)
    task.check_support()

    # Pass-through: the saved actor moments and rates (two named groups).
    actor_optimizer = rebuild_actor_optimizer(actor, state["actor_optimizer"])
    critic_lr = float(state["critic_optimizer"]["param_groups"][0]["lr"])
    critic_optimizer = torch.optim.Adam(
        list(task.parameters()) + list(costs.parameters()), lr=critic_lr
    )
    sampling = torch.Generator()
    sampling.set_state(state["sampling_generator_state"])
    shuffle = np.random.default_rng(int(state["optimizer_seed"]))
    shuffle.bit_generator.state = copy.deepcopy(state["shuffle_rng_state"])
    converted = _trainer_checkpoint(
        actor,
        task,
        costs,
        normalizer,
        layout,
        actor_optimizer,
        critic_optimizer,
        personality=labor_checkpoint_personality(state),
        optimizer_seed=int(state["optimizer_seed"]),
        iteration=int(state["iteration"]),
        sampling_generator=sampling,
        shuffle_rng=shuffle,
        critic_training_world_seeds=state["critic_training_world_seeds"],
        critic_training_independent_unit_ids=state["critic_training_independent_unit_ids"],
    )
    for key, value in state.items():
        converted.setdefault(key, value)
    history = list(state.get("critic_value_support_conversions", []))
    history.append(
        {
            "contract": CONVERSION_CONTRACT,
            "from": _support_record(old_task.support),
            "to": _support_record(support),
            "head_seed": head_seed,
            "source_sha256": source_sha256,
            "reinitialized": [f"task_critic.heads.{name}" for name in LABOR_OBJECTIVES]
            + ["task_critic.bins"],
            "kept": [
                "task_critic.encoder",
                *(["task_critic.qol_encoder"] if old_task.qol_encoder is not None else []),
                "cost_critics",
                "actor",
                "normalizer",
                "trainer_rng",
            ],
            "critic_optimizer": "reset-fresh-adam-same-lr",
        }
    )
    converted["critic_value_support_conversions"] = history
    return converted


def _set_migration(enabled: bool, *modules: nn.Module) -> None:
    for module in modules:
        for child in module.modules():
            if isinstance(child, LaborCriticEncoder):
                child.privileged_migration = enabled


def convert_privileged_critics(
    state: Mapping[str, Any],
    privileged_normalizer: Any,
    *,
    seed: int | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a trainer checkpoint whose three critics carry the privileged block.

    Every tensor of the source checkpoint is kept and loaded strictly; the
    missing ``privileged_*`` parameters are filled from a seeded fresh
    initialization by the strict-load migration hook, which is enabled only
    for this call. The actor is untouched (its encoder rejects privileged
    state). The critic optimizer is reset.
    """
    from .labor_privileged import LaborPrivilegedNormalizer
    from .labor_training import (
        _models_from_checkpoint,
        _require_trainer_checkpoint,
        _trainer_checkpoint,
    )

    if not isinstance(privileged_normalizer, LaborPrivilegedNormalizer):
        raise ValueError("privileged migration requires a fitted LaborPrivilegedNormalizer")
    _require_trainer_checkpoint(state)
    layout, actor, old_task, old_costs, normalizer = _models_from_checkpoint(state)
    if layout.privileged is not None:
        raise ValueError("checkpoint already carries a privileged critic block")
    contract = labor_checkpoint_contract(state, expected_layout=layout)
    new_layout = LaborEncoderLayout(
        layout.core_input_width,
        layout.specialty_feature_width,
        layout.activity_feature_width,
        layout.resource_feature_width,
        privileged=privileged_normalizer.layout,
    )
    init_seed = int(state["optimizer_seed"] if seed is None else seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        task = LaborTaskCritic(
            new_layout,
            support=old_task.support,
            qol_head=old_task.qol_head,
            qol_encoder=old_task.qol_encoder_mode,
        )
        costs = LaborCostCritics(new_layout)
    _set_migration(True, task, costs)
    try:
        task.load_state_dict(old_task.state_dict(), strict=True)
        costs.load_state_dict(old_costs.state_dict(), strict=True)
    finally:
        _set_migration(False, task, costs)
    task.check_support()
    bind_privileged_normalizer(privileged_normalizer, task, costs)
    set_labor_film_bound(contract.conditioning.film_bound, task, costs)

    # Pass-through: the saved actor moments and rates (two named groups).
    actor_optimizer = rebuild_actor_optimizer(actor, state["actor_optimizer"])
    critic_lr = float(state["critic_optimizer"]["param_groups"][0]["lr"])
    critic_optimizer = torch.optim.Adam(
        list(task.parameters()) + list(costs.parameters()), lr=critic_lr
    )
    sampling = torch.Generator()
    sampling.set_state(state["sampling_generator_state"])
    shuffle = np.random.default_rng(int(state["optimizer_seed"]))
    shuffle.bit_generator.state = copy.deepcopy(state["shuffle_rng_state"])
    converted = _trainer_checkpoint(
        actor,
        task,
        costs,
        normalizer,
        new_layout,
        actor_optimizer,
        critic_optimizer,
        personality=labor_checkpoint_personality(state),
        optimizer_seed=int(state["optimizer_seed"]),
        iteration=int(state["iteration"]),
        sampling_generator=sampling,
        shuffle_rng=shuffle,
        critic_training_world_seeds=state["critic_training_world_seeds"],
        critic_training_independent_unit_ids=state["critic_training_independent_unit_ids"],
    )
    for key, value in state.items():
        converted.setdefault(key, value)
    initialized = sorted(
        f"{prefix}.{name}"
        for prefix, module in (("task_critic", task), ("cost_critics", costs))
        for name, _ in module.named_parameters()
        if ".privileged_" in f".{name}"
    )
    history = list(state.get("critic_privileged_conversions", []))
    history.append(
        {
            "contract": PRIVILEGED_CONVERSION_CONTRACT,
            "privileged_schema_version": int(privileged_normalizer.layout.schema_version),
            "privileged_normalizer_digest": privileged_normalizer.digest(),
            "init_seed": init_seed,
            "source_sha256": source_sha256,
            "initialized": initialized,
            "kept": ["actor", "task_critic", "cost_critics", "normalizer", "trainer_rng"],
            "critic_optimizer": "reset-fresh-adam-same-lr",
        }
    )
    converted["critic_privileged_conversions"] = history
    return converted


def convert_qol_encoder(
    state: Mapping[str, Any],
    init: str,
    *,
    seed: int | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a trainer checkpoint whose task critic has a separate QoL encoder.

    ``init="copy"`` copies the source task encoder (privileged block included);
    ``init="fresh"`` builds it from ``labor_qol_encoder_seed(seed)`` (default:
    the checkpoint's optimizer seed). Every source tensor is kept and loaded
    strictly; the critic optimizer is reset (the new parameters have no moments).
    """
    from .labor_training import (
        _models_from_checkpoint,
        _require_trainer_checkpoint,
        _trainer_checkpoint,
    )

    if init not in QOL_ENCODER_INITS:
        raise ValueError(f"QoL encoder init must be one of {QOL_ENCODER_INITS}")
    _require_trainer_checkpoint(state)
    layout, actor, old_task, costs, normalizer = _models_from_checkpoint(state)
    if old_task.separate_qol_encoder:
        raise ValueError("checkpoint already declares a separate QoL encoder")
    contract = labor_checkpoint_contract(state, expected_layout=layout)
    init_seed = int(state["optimizer_seed"] if seed is None else seed)
    task = LaborTaskCritic(
        layout,
        int(old_task.encoder.width),
        int(old_task.encoder.token_width),
        support=old_task.support,
        qol_head=old_task.qol_head,
        qol_encoder="separate",
        qol_encoder_seed=labor_qol_encoder_seed(init_seed),
    )
    source = dict(old_task.state_dict())
    new_encoder = old_task.encoder if init == "copy" else task.qol_encoder
    source.update({f"qol_encoder.{k}": v for k, v in new_encoder.state_dict().items()})
    task.load_state_dict(source, strict=True)
    task.check_support()
    bind_privileged_normalizer(getattr(old_task, "privileged_normalizer", None), task)
    set_labor_film_bound(contract.conditioning.film_bound, task)

    # Pass-through: the saved actor moments and rates (two named groups).
    actor_optimizer = rebuild_actor_optimizer(actor, state["actor_optimizer"])
    critic_lr = float(state["critic_optimizer"]["param_groups"][0]["lr"])
    critic_optimizer = torch.optim.Adam(
        list(task.parameters()) + list(costs.parameters()), lr=critic_lr
    )
    sampling = torch.Generator()
    sampling.set_state(state["sampling_generator_state"])
    shuffle = np.random.default_rng(int(state["optimizer_seed"]))
    shuffle.bit_generator.state = copy.deepcopy(state["shuffle_rng_state"])
    converted = _trainer_checkpoint(
        actor,
        task,
        costs,
        normalizer,
        layout,
        actor_optimizer,
        critic_optimizer,
        personality=labor_checkpoint_personality(state),
        optimizer_seed=int(state["optimizer_seed"]),
        iteration=int(state["iteration"]),
        sampling_generator=sampling,
        shuffle_rng=shuffle,
        critic_training_world_seeds=state["critic_training_world_seeds"],
        critic_training_independent_unit_ids=state["critic_training_independent_unit_ids"],
    )
    for key, value in state.items():
        converted.setdefault(key, value)
    history = list(state.get("critic_qol_encoder_conversions", []))
    history.append(
        {
            "contract": QOL_ENCODER_CONVERSION_CONTRACT,
            "init": init,
            "init_seed": None if init == "copy" else init_seed,
            "source_sha256": source_sha256,
            "initialized": ["task_critic.qol_encoder"],
            "kept": ["actor", "task_critic", "cost_critics", "normalizer", "trainer_rng"],
            "critic_optimizer": "reset-fresh-adam-same-lr",
        }
    )
    converted["critic_qol_encoder_conversions"] = history
    return converted


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser(
        "audit", help="normalized task-target distribution of a teacher dataset"
    )
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--margin", type=float, default=0.5)
    audit.add_argument("--out", type=Path, default=None)
    convert = commands.add_parser(
        "convert", help="re-declare a checkpoint's task-critic value support"
    )
    convert.add_argument("--checkpoint", type=Path, required=True)
    convert.add_argument("--critic-value-support", required=True, metavar="LOW,HIGH,BINS")
    convert.add_argument(
        "--seed", type=int, default=None, help="head init seed (default: optimizer seed)"
    )
    convert.add_argument("--out", type=Path, required=True)
    privileged = commands.add_parser(
        "convert-privileged",
        help="migrate a checkpoint to critic-only privileged critics (intervention A)",
    )
    privileged.add_argument("--checkpoint", type=Path, required=True)
    privileged.add_argument("--privileged-normalizer", type=Path, required=True)
    privileged.add_argument(
        "--seed", type=int, default=None, help="privileged init seed (default: optimizer seed)"
    )
    privileged.add_argument("--out", type=Path, required=True)
    qol = commands.add_parser(
        "convert-qol-encoder",
        help="give a shared-encoder checkpoint's task critic a separate QoL encoder",
    )
    qol.add_argument("--checkpoint", type=Path, required=True)
    qol.add_argument(
        "--init",
        choices=QOL_ENCODER_INITS,
        required=True,
        help="copy: warm start from the task encoder (outputs unchanged); fresh: seeded init",
    )
    qol.add_argument(
        "--seed", type=int, default=None, help="fresh init seed (default: optimizer seed)"
    )
    qol.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    if args.command == "audit":
        from .labor_teacher_data import load_labor_dataset

        report = target_audit(load_labor_dataset(args.dataset, training=False), margin=args.margin)
        text = json.dumps(report, indent=2, sort_keys=True)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n")
        print(text)
        return 0
    from .labor_training import _save_checkpoint

    if args.out.exists():
        raise ValueError("output checkpoint exists; choose a new path")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.command == "convert-qol-encoder":
        converted = convert_qol_encoder(
            state, args.init, seed=args.seed, source_sha256=_sha256(args.checkpoint)
        )
        _save_checkpoint(args.out, converted)
        record = converted["critic_qol_encoder_conversions"][-1]
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    if args.command == "convert-privileged":
        from .labor_training import _load_privileged_normalizer

        converted = convert_privileged_critics(
            state,
            _load_privileged_normalizer(args.privileged_normalizer),
            seed=args.seed,
            source_sha256=_sha256(args.checkpoint),
        )
        _save_checkpoint(args.out, converted)
        print(json.dumps(converted["critic_privileged_conversions"][-1], indent=2, sort_keys=True))
        return 0
    support = LaborValueSupport.parse(args.critic_value_support)
    converted = convert_task_critic_support(
        state, support, seed=args.seed, source_sha256=_sha256(args.checkpoint)
    )
    _save_checkpoint(args.out, converted)
    print(json.dumps(converted["critic_value_support_conversions"][-1], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
