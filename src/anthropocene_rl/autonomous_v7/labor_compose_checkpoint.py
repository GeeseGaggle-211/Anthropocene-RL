"""Compose a labor trainer checkpoint: one checkpoint's critics and state, another's actor.

The PPO start checkpoint for the r5 A(s,a) measurement is the stage-1 critic's trainer
state (``critic-privileged-e4-seed0/critic.pt``: critics, normalizers, RNG, optimizer
seed, critic-data history) with its actor replaced by the best BC actor's weights
(``actor-bc-f1.pt``). Everything except ``actor`` and ``actor_optimizer`` is copied
from the base unchanged; ``actor_optimizer`` is a fresh legacy one-group Adam state
(``torch.optim.Adam(actor.parameters(), lr=...)``, no moments). ``train`` re-indexes it
into its two named groups (``labor_actor_optim``) and replaces the saved lr with its own
configured rates, so ``--actor-lr`` here is only the recorded value.

Refusals (never forced): the actor source must be a labor checkpoint whose model
contracts (artifact kind, feature/model-input contracts, native producer contract,
encoder layout, ``labor_model_contract``) and observation normalizer equal the base's
exactly, and whose ``actor`` tensors have the base actor's exact keys, shapes and
dtypes. The composed state must pass ``labor_training``'s own trainer-checkpoint
loader and keep the base's ``labor_critic_digest`` and normalizer binding.

Usage::

    python -m anthropocene_rl.autonomous_v7.labor_compose_checkpoint \
        --base critic.pt --actor-from actor-bc-f1.pt --out checkpoint-start.pt \
        [--actor-lr 3e-4] [--expect-critic-digest HEX] [--expect-normalizer-digest HEX] \
        [--a-warm-start a_warm_start.pt] [--check-worlds 91000,91001] [--report out.json]
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

COMPOSE_SCHEMA = "labor-composed-trainer-checkpoint-v1"
COMPOSE_KEY = "composed_checkpoint"
# Model-level fields the actor depends on; all must be equal in base and actor source.
CONTRACT_KEYS = (
    "artifact_kind",
    "feature_contract",
    "model_input_contract",
    "native_producer_contract",
    "labor_encoder_layout",
    "labor_model_contract",
    "normalizer",
)


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.shape == b.shape
            and a.dtype == b.dtype
            and torch.equal(a, b)
        )
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        return (
            isinstance(a, Mapping)
            and isinstance(b, Mapping)
            and set(a) == set(b)
            and all(_equal(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(_equal(x, y) for x, y in zip(a, b, strict=True))
        )
    return a == b


def _differing(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    return sorted(k for k in set(a) | set(b) if not _equal(a.get(k), b.get(k)))


def compose_trainer_checkpoint(
    base: Mapping[str, Any],
    actor_source: Mapping[str, Any],
    *,
    actor_lr: float = 3e-4,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return ``base`` with ``actor_source``'s actor and a fresh actor Adam state.

    Raises ``ValueError`` (never forces) when the actor source's contracts, normalizer
    or actor tensor layout differ from the base's.
    """
    from .labor_training import _models_from_checkpoint, _require_trainer_checkpoint

    _require_trainer_checkpoint(base)
    if not isinstance(actor_source, Mapping) or not isinstance(actor_source.get("actor"), Mapping):
        raise ValueError("actor source is not a labor checkpoint with an actor state")
    for key in CONTRACT_KEYS:
        if key not in base or key not in actor_source:
            raise ValueError(f"{key} is missing from the base or the actor source")
        if not _equal(base[key], actor_source[key]):
            detail = ""
            if isinstance(base[key], Mapping) and isinstance(actor_source[key], Mapping):
                detail = f" (fields {_differing(base[key], actor_source[key])})"
            raise ValueError(f"actor source {key} differs from the base{detail}; refusing")
    base_actor, source_actor = base["actor"], actor_source["actor"]
    if set(base_actor) != set(source_actor):
        raise ValueError(
            "actor source tensors differ from the base actor's keys: "
            f"missing {sorted(set(base_actor) - set(source_actor))[:5]}, "
            f"extra {sorted(set(source_actor) - set(base_actor))[:5]}"
        )
    for name, value in base_actor.items():
        other = source_actor[name]
        if (
            not isinstance(other, torch.Tensor)
            or other.shape != value.shape
            or other.dtype != value.dtype
        ):
            raise ValueError(f"actor tensor {name} differs in shape/dtype from the base actor")
    # The actor source must itself load strictly (its own contracts with its own actor).
    _models_from_checkpoint(actor_source)

    composed = dict(base)
    composed["actor"] = {name: value.detach().clone() for name, value in source_actor.items()}
    _, actor, _, _, _ = _models_from_checkpoint(composed)
    for name, value in actor.state_dict().items():
        if not torch.equal(value, source_actor[name]):
            raise AssertionError(f"composed actor tensor {name} does not equal the source's")
    if isinstance(actor_lr, bool) or not float(actor_lr) > 0:
        raise ValueError("actor_lr must be positive")
    composed["actor_optimizer"] = torch.optim.Adam(
        actor.parameters(), lr=float(actor_lr)
    ).state_dict()
    composed[COMPOSE_KEY] = {
        "schema": COMPOSE_SCHEMA,
        "replaced": ["actor", "actor_optimizer"],
        "actorOptimizer": f"fresh torch.optim.Adam(actor.parameters(), lr={float(actor_lr)!r})",
        **({} if provenance is None else dict(provenance)),
    }
    _require_trainer_checkpoint(composed)
    return composed


def verify_composed(
    composed: Mapping[str, Any],
    base: Mapping[str, Any],
    actor_source: Mapping[str, Any],
    *,
    expect_critic_digest: str | None = None,
    expect_normalizer_digest: str | None = None,
    a_warm_start: Mapping[str, Any] | None = None,
    check_worlds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """The checks a collect/train run would make, without a native bridge.

    Exercises ``_command_collect``'s preamble (trainer loader, models, personality,
    sampling generator, world parsing with ``--allow-vm-scale`` and the reserved-world
    guard) and, with ``a_warm_start``, ``labor_a_ppo.load_frozen_a`` against both the base
    critic checkpoint (what ``train --a-critic`` passes) and the composed checkpoint.
    """
    from . import labor_a_ppo as ap
    from .labor_model import labor_checkpoint_personality
    from .labor_training import (
        _models_from_checkpoint,
        _normalization_digest,
        _parse_worlds,
        _require_trainer_checkpoint,
        labor_behavior_configuration,
    )
    from .labor_value_calibration import labor_critic_digest
    from .labor_worlds import require_training_worlds
    from .value_calibration import behavior_policy_digest

    report: dict[str, Any] = {}
    _require_trainer_checkpoint(composed)
    _, actor, task, costs, normalizer = _models_from_checkpoint(composed)
    _, _, base_task, base_costs, base_normalizer = _models_from_checkpoint(base)
    _, source_actor, _, _, source_normalizer = _models_from_checkpoint(actor_source)
    critic = labor_critic_digest(task, costs)
    norm = _normalization_digest(normalizer)
    report["criticDigest"] = critic
    report["criticDigestUnchanged"] = critic == labor_critic_digest(base_task, base_costs)
    report["normalizerDigest"] = norm
    report["normalizerDigestMatches"] = (
        norm == _normalization_digest(base_normalizer) == _normalization_digest(source_normalizer)
    )
    report["actorEqualsSource"] = _equal(
        dict(actor.state_dict()), dict(source_actor.state_dict())
    )
    personality = labor_checkpoint_personality(composed)
    behavior = labor_behavior_configuration(personality)
    report["behaviorPolicyDigest"] = behavior_policy_digest(actor, behavior)
    source_behavior = labor_behavior_configuration(labor_checkpoint_personality(actor_source))
    report["behaviorPolicyDigestEqualsSource"] = report[
        "behaviorPolicyDigest"
    ] == behavior_policy_digest(source_actor, source_behavior)
    unchanged = [k for k in base if k not in ("actor", "actor_optimizer")]
    report["otherKeysUnchanged"] = all(_equal(composed[k], base[k]) for k in unchanged)
    report["actorOptimizer"] = {
        "paramGroups": [
            {k: v for k, v in group.items() if k != "params"}
            for group in composed["actor_optimizer"]["param_groups"]
        ],
        "params": len(composed["actor_optimizer"]["param_groups"][0]["params"]),
        "actorParameters": len(list(actor.parameters())),
        "stateEntries": len(composed["actor_optimizer"]["state"]),
    }
    from .labor_actor_optim import ActorLrConfig, build_actor_optimizer, load_actor_optimizer_state

    lr = ActorLrConfig(3e-4, 3e-4)
    optimizer = build_actor_optimizer(actor, lr)
    load_actor_optimizer_state(optimizer, actor, composed["actor_optimizer"], lr)  # train's load
    sampling = torch.Generator()
    sampling.set_state(composed["sampling_generator_state"])  # what collect does
    report["iteration"] = int(composed["iteration"])
    report["criticTrainingWorldSeeds"] = list(composed["critic_training_world_seeds"])
    if expect_critic_digest is not None and not critic.startswith(expect_critic_digest):
        raise ValueError(f"critic digest {critic} != expected {expect_critic_digest}")
    if expect_normalizer_digest is not None and not norm.startswith(expect_normalizer_digest):
        raise ValueError(f"normalizer digest {norm} != expected {expect_normalizer_digest}")
    for key in (
        "criticDigestUnchanged",
        "normalizerDigestMatches",
        "actorEqualsSource",
        "behaviorPolicyDigestEqualsSource",
        "otherKeysUnchanged",
    ):
        if report[key] is not True:
            raise AssertionError(f"composed checkpoint check failed: {key}")
    if check_worlds is not None:
        text = ",".join(str(int(w)) for w in check_worlds)
        worlds = require_training_worlds(
            _parse_worlds(text, allow_vm_scale=True), "labor PPO rollout collection"
        )
        overlap = sorted(set(worlds) & set(composed["critic_training_world_seeds"]))
        if overlap:
            raise ValueError(f"collect worlds {overlap} are critic-training worlds")
        report["collectWorlds"] = {"worlds": worlds, "parseAndReservedGuard": "pass"}
    if a_warm_start is not None:
        fit = a_warm_start.get("fit", {}) if isinstance(a_warm_start, Mapping) else {}
        a_worlds = sorted(
            {int(w) for key in ("fit_worlds", "heldout_worlds") for w in fit.get(key, [])}
            | {int(w) for w in a_warm_start.get("warm_start", {}).get("eval_worlds", [])}
        )
        report["aWorlds"] = a_worlds
        if check_worlds is not None:
            overlap = sorted(set(int(w) for w in check_worlds) & set(a_worlds))
            if overlap:
                raise ValueError(f"collect worlds {overlap} are A fit/eval worlds")
        frozen_base = ap.load_frozen_a(a_warm_start, base, normalization_binding_digest=norm)
        frozen_composed = ap.load_frozen_a(
            a_warm_start, composed, normalization_binding_digest=norm
        )
        frozen_base.assert_unchanged()
        report["frozenA"] = {
            "againstBase": frozen_base.record(),
            "againstComposedCriticDigest": frozen_composed.critic_digest,
            "headDigest": frozen_base.head_digest,
        }
        if frozen_composed.critic_digest != frozen_base.critic_digest:
            raise AssertionError("the composed critic differs from the A critic")
    return report


def _sha256(path: Path) -> str:
    from .provenance import checkpoint_hash

    return checkpoint_hash(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", type=Path, required=True, help="trainer checkpoint (critics)")
    parser.add_argument("--actor-from", type=Path, required=True, help="checkpoint with the actor")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--expect-critic-digest", default=None, help="hex prefix")
    parser.add_argument("--expect-normalizer-digest", default=None, help="hex prefix")
    parser.add_argument("--a-warm-start", type=Path, default=None)
    parser.add_argument("--check-worlds", default=None, help="comma-separated collect worlds")
    parser.add_argument("--report", type=Path, default=None, help="verification JSON")
    args = parser.parse_args(argv)

    base = torch.load(args.base, map_location="cpu", weights_only=True)
    source = torch.load(args.actor_from, map_location="cpu", weights_only=True)
    provenance = {
        "baseSha256": _sha256(args.base),
        "baseName": args.base.name,
        "actorSourceSha256": _sha256(args.actor_from),
        "actorSourceName": args.actor_from.name,
    }
    for key in ("g0_partial", "iteration", "optimizer_seed"):
        if key in source:
            provenance[f"actorSource_{key}"] = copy.deepcopy(source[key])
    composed = compose_trainer_checkpoint(
        base, source, actor_lr=args.actor_lr, provenance=provenance
    )
    warm = (
        None
        if args.a_warm_start is None
        else torch.load(args.a_warm_start, map_location="cpu", weights_only=True)
    )
    worlds = (
        None if args.check_worlds is None else [int(w) for w in args.check_worlds.split(",") if w]
    )
    report = verify_composed(
        composed,
        base,
        source,
        expect_critic_digest=args.expect_critic_digest,
        expect_normalizer_digest=args.expect_normalizer_digest,
        a_warm_start=warm,
        check_worlds=worlds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    torch.save(composed, temporary)
    temporary.replace(args.out)
    # Reload what was written: the collect/train loader reads the file, not the dict.
    reloaded = torch.load(args.out, map_location="cpu", weights_only=True)
    verify_composed(reloaded, base, source, a_warm_start=warm, check_worlds=worlds)
    report["provenance"] = provenance
    if args.a_warm_start is not None:
        report["aWarmStartSha256"] = _sha256(args.a_warm_start)
    report["outSha256"] = _sha256(args.out)
    text = json.dumps(report, indent=1, sort_keys=True, default=str)
    if args.report is not None:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
