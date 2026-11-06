"""One region-head resolution and resume policy for every model9 consumer."""

from __future__ import annotations

import os
import re
from pathlib import Path


def arguments(parser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--region-head", choices=("hierarchical", "flat"), default=None)
    group.add_argument(
        "--flat-region-head",
        dest="region_head",
        action="store_const",
        const="flat",
        help="exact pre-hierarchy action and optimizer compatibility",
    )


def parse(parser):
    arguments(parser)
    args = parser.parse_args()
    import sys

    from .protocol import command, scan

    command(sys.argv[1:])
    scan(vars(args), "CLI")
    if args.region_head is not None:
        os.environ["ANTHROPOCENE_REGION_HEAD"] = args.region_head
    return args


def saved_head(state: dict, old: str = "hierarchical") -> str:
    net = state.get("actor", state)
    if "region_flat" in net:
        return "flat" if bool(net["region_flat"]) else "hierarchical"
    return state.get("region_head", old)


def resolve_head(state: dict, override: str | None = None, resume: bool = False) -> str:
    saved = saved_head(state, "flat" if resume else "hierarchical")
    explicit = override or os.environ.get("ANTHROPOCENE_REGION_HEAD")
    if explicit not in (None, "flat", "hierarchical"):
        raise ValueError(f"unknown region head {explicit!r}")
    if resume and explicit is not None and explicit != saved:
        raise ValueError(
            "region head differs from frozen resume; start a new named run from actor weights"
        )
    return explicit or saved


def restore_actor(
    actor, state: dict, override: str | None = None, *, resume: bool = False, strict: bool = True
):
    from .protocol import scan

    scan(state)
    if resume and (state.get("artifact_kind") == "promotion" or state.get("resumable") is False):
        raise ValueError("promoted artifact is non-resumable; start a new run from weights")
    result = actor.load_state_dict(state.get("actor", state), strict=strict)
    actor.set_region_head(resolve_head(state, override, resume))
    return result


def actor_parameters(actor, state: dict | None = None):
    """Keep pre-head Adam parameter ordering exactly; dormant offsets have no flat-mode optimizer slots."""  # noqa: E501 - docstring/prose line kept verbatim
    legacy = state is not None and (
        state.get("legacy_actor_optimizer", False)
        or "region_group.weight" not in state.get("actor", state)
    )
    if legacy and actor.region_head != "flat":
        raise ValueError(
            "pre-head optimizer can resume only flat; start a new named hierarchical run"
        )
    return [
        p
        for name, p in actor.named_parameters()
        if not (legacy and name.startswith("region_group."))
    ]


def resume_setting(name: str, saved: dict | None, override: bool | None, new_default: bool) -> bool:
    value = new_default if saved is None else bool(saved.get(name, False))
    if saved is not None and override is not None and override != value:
        raise ValueError(f"{name} differs from frozen resume; start a new named run")
    return value if override is None else override


def run_manifest(
    args, actor, resume_state=None, *, require_init: bool = True, identity=None
) -> dict:
    """Freeze resolved flags and file contents, not just filenames, for same-round continuation."""
    from . import contracts
    from .bridge import Model9Bridge
    from .provenance import (
        ProvenanceError,
        checkpoint_hash,
        code_commit,
        digest,
        simulator_identity,
    )

    # Schema only: no Reset/Step, episode, seeded draw or actor RNG is consumed.
    cached_probe = getattr(args, "_qualified_probe", None)
    if cached_probe:
        from .teacher_data import binary_identity

        if binary_identity()["sha256"] != cached_probe["bridge_sha256"] or cached_probe[
            "flags"
        ] != {
            "sim_rules": os.environ.get("ANTHROPOCENE_SIM_RULES", ""),
            "sim_statics": os.environ.get("ANTHROPOCENE_SIM_STATICS", ""),
        }:
            raise ProvenanceError("qualified bridge/flags changed before run freeze")
        simulator = cached_probe["simulator"]
        worker = cached_probe["worker_contract"]
    else:
        with Model9Bridge() as probe:
            simulator = simulator_identity(probe.schema, production=True)
            from .worker_contract import observe

            worker = observe(probe)
    learner = identity["learner_commit"] if identity is not None else code_commit()
    if not re.fullmatch(r"[0-9a-f]{40}", learner):
        raise ProvenanceError("production requires a full clean learner commit")
    options = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k != "resume" and not k.startswith("_")
    }
    if resume_state and getattr(args, "curriculum", None) == "legacy":
        for k in ("curriculum", "bank_image", "prior_use_inventory"):
            if k not in resume_state.get("run_manifest", {}).get("options", {}):
                options.pop(k, None)
    options["region_head"] = actor.region_head
    if hasattr(args, "cost_bootstrap"):
        options["cost_bootstrap"] = resume_setting(
            "cost_bootstrap", resume_state, args.cost_bootstrap, True
        )
    inputs = {}
    for name in (
        "init",
        "teacher",
        "calibration_manifest",
        "source_manifest",
        "grassland_seeds",
        "valley_seeds",
        "bank_image",
        "prior_use_inventory",
        "clone_protocol",
        "execution_manifest",
        "local_execution_design",
        "preregistration",
        "protocol_approval",
    ):
        value = getattr(args, name, None)
        if value:
            inputs[name] = checkpoint_hash(value)
    for name in ("search_labels", "branch_shards", "train", "dev"):
        paths = getattr(args, name, None)
        if paths:
            for value in str(paths).split(","):
                path = Path(value)
                if path.is_file():
                    inputs[str(path)] = checkpoint_hash(path)
                else:
                    patterns = (
                        ("*.npy", "meta.json") if name in ("train", "dev") else ("meta.json",)
                    )
                    for pattern in patterns:
                        for meta in sorted(path.rglob(pattern)):
                            inputs[str(meta)] = checkpoint_hash(meta)
    if require_init and not re.fullmatch(r"[0-9a-f]{64}", inputs.get("init", "")):
        raise ProvenanceError("production run lacks a full init checkpoint SHA-256")
    result = {
        "options": options,
        "inputs": inputs,
        "region_head": actor.region_head,
        "code": learner,
        "simulator": simulator,
        "contract": contracts.NATIVE_CONTRACT,
        "utility_version": 2,
        "config_digest": digest(
            {
                "options": options,
                "sim_rules": os.environ.get("ANTHROPOCENE_SIM_RULES", ""),
                "sim_statics": os.environ.get("ANTHROPOCENE_SIM_STATICS", ""),
            }
        ),
        "sim_rules": os.environ.get("ANTHROPOCENE_SIM_RULES", ""),
        "sim_statics": os.environ.get("ANTHROPOCENE_SIM_STATICS", ""),
    }
    from .calibration import manifest

    held_path = getattr(args, "calibration_manifest", None)
    held = manifest(held_path) if held_path else {}
    result["calibration_manifest_digest"] = held.get("manifest_digest")
    result["protocol_digest"] = held.get("protocol_digest")
    result["initialization"] = "checkpoint" if require_init else "random"
    from .teacher_data import binary_identity

    result["bridge_sha256"] = (
        identity["bridge_sha256"] if identity is not None else binary_identity()["sha256"]
    )
    result["worker_contract"] = worker
    from .protocol import scan, verify_inventory

    scan(options)
    if held_path:
        inventory_path = getattr(args, "prior_use_inventory", None)
        if inventory_path is None:
            raise ProvenanceError("qualification requires verified prior-use inventory")
        result["prior_use_inventory"] = verify_inventory(inventory_path)
    from .provenance import compatibility_mapping

    result["simulator_compatibility"] = compatibility_mapping(
        getattr(args, "source_manifest", None),
        {"sim_rules": result["sim_rules"], "sim_statics": result["sim_statics"]},
    )
    if getattr(args, "curriculum", None) == "r3-balanced-v1":
        from .curriculum import specification
        from .ppo import STAGES

        result["curriculum_spec"] = specification(STAGES[args.stage])
    previous = {
        k: v
        for k, v in (resume_state or {}).get("run_manifest", {}).items()
        if k not in ("resolved_hyper", "teacher_inputs", "resolved_stage", "effective_iterations")
    }
    if resume_state and "run_manifest" in resume_state and previous != result:
        raise ValueError("frozen run inputs or flags changed; start a new named round")
    return result
