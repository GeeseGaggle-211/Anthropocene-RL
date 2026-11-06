"""Read-only, content-verifying final submission preflight. Never starts a VM or submits a job."""

from __future__ import annotations

import argparse
import json
import subprocess
import shlex
import os
import re
from pathlib import Path

import torch

from .calibration import manifest
from .lineage import checkpoint_evidence
from .promotion import semantic_hash
from .protocol import artifact, execution_design, receipt, scan, verify_inventory, ROLES
from .provenance import ProvenanceError, checkpoint_hash, code_commit, digest, simulator_identity
from .teacher_data import binary_identity

GATES = (
    "protocol",
    "source",
    "clone",
    "search",
    "calibration",
    "movement-integration",
    "submission",
)


def queue_receipt(body, expected, base):
    from .gate_validation import read
    from .protocol import command

    proof = body.get("queue", {})
    job, _ = read(proof.get("job"), base)
    status, _ = read(proof.get("status"), base)
    phases = [p for p in job.get("phases", []) if p.get("name") == proof.get("phase")]
    states = [p for p in status.get("phases", []) if p.get("name") == proof.get("phase")]
    if (
        len(phases) != 1
        or len(states) != 1
        or states[0].get("state") != "done"
        or states[0].get("exit_code") != 0
        or job.get("rl_sha") != expected["learner_commit"]
        or job.get("sim_sha") != expected["simulator"]["commit"]
        or status.get("name") != job.get("name")
    ):
        raise ProvenanceError(
            "qualification queue phase is missing/failed or uses different actual code"
        )
    command(shlex.split(phases[0]["run"]))
    done, path = read(proof.get("done"), base)
    if (
        path.name != Path(phases[0]["done"]).name
        or done.get("kind") != "qualification_phase_done"
        or done.get("job_sha256") != proof["job"]["sha256"]
        or done.get("phase") != proof["phase"]
        or done.get("outputs") != body.get("outputs")
        or any(done.get(k) != v for k, v in expected.items())
    ):
        raise ProvenanceError(
            "queue done marker does not bind complete output inventory and frozen identities"
        )
    return proof["job"]["sha256"]


def validate_runs(spec, base, state, expected, frozen_runs=None):
    """Parse the actual PPO argv; a free resolved-config dict is not executable evidence."""
    from . import ppo, contracts
    from .checkpoint import arguments, resolve_head
    from .protocol import command, seed_list
    from .gate_validation import read
    import dataclasses

    job, _ = read(spec["queue_job"], base)
    if (
        job.get("rl_sha") != expected["learner_commit"]
        or job.get("sim_sha") != expected["simulator"]["commit"]
    ):
        raise ProvenanceError("actual final queue manifest uses different commits")
    phases = {phase["name"]: phase for phase in job.get("phases", [])}
    runs = spec.get("runs", [])
    if len(phases) != 15 or len(job.get("phases", [])) != 15 or len(runs) != 15:
        raise ProvenanceError("final queue must contain the exact fifteen bound training phases")
    common, bank_digests, cohorts, used, outputs, banks = [], set(), {}, set(), set(), set()
    for run in runs:
        argv = run.get("argv")
        if (
            not isinstance(argv, list)
            or len(argv) < 4
            or argv[1:3] != ["-m", "anthropocene_rl.autonomous_v7.ppo"]
        ):
            raise ProvenanceError(
                "launch requires direct fully expanded PPO argv, not a shell/script self-attestation"
            )
        phase = phases.get(run.get("phase"))
        if phase is None or run["phase"] in used or phase["run"] != shlex.join(argv):
            raise ProvenanceError("queue run string differs from actual frozen argv")
        used.add(run["phase"])
        command(argv[3:])
        parser = ppo.argument_parser()
        arguments(parser)
        args = parser.parse_args(argv[3:])
        options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        if args.execution_manifest is None:
            raise ProvenanceError("actual registered argv lacks its execution manifest")
        execution = json.loads(args.execution_manifest.read_text())
        if execution.get("digest") != spec.get("digest"):
            raise ProvenanceError(
                "actual registered argv names a different frozen execution manifest"
            )
        if (
            args.arm != run["arm"]
            or args.seed != run["seed"]
            or not args.resume
            or args.teacher is not None
            or args.nonqualifying
            or args.local_execution_design is not None
            or args.stage != "mixed"
            or args.iterations != 50
            or args.worlds != 12
            or args.turns != 25
            or args.commitments
            or not all(
                (
                    args.learned_movement,
                    args.learned_fission,
                    args.learned_trade,
                    args.experts,
                    args.vector_critic,
                )
            )
            or args.personality != ("default" if args.arm == "R3def" else "dirichlet:1")
            or args.teacher_kl != (0.002 if args.arm == "R3lo" else 0.01)
            or args.cost_bootstrap is not True
            or args.kl_refit is not True
            or args.curriculum != "r3-balanced-v1"
            or resolve_head(state, args.region_head) != "hierarchical"
            or args.personality_scales != "reference"
            or args.personality_clip != 5
            or args.film_bound is not None
            or args.eval_turns != 25
            or args.eval_every <= 0
            or seed_list(args.eval_seeds) != ROLES["ppo_selection"]
            or args.provenance != "reject"
            or args.branch_share != 0.25
        ):
            raise ProvenanceError(
                "actual PPO argv differs from preregistered treatment/budget/selection roots"
            )
        weights = dict(contracts.OBJECTIVE_WEIGHTS)
        weights["trade_utility"] = contracts.TRADE_UTILITY_WEIGHT
        if args.objective_weights:
            for item in args.objective_weights.split(","):
                name, value = item.split("=")
                weights[name] = float(value)
        if weights != {
            "welfare": 1.0,
            "population_years": 0.0,
            "trade_utility": 0.0,
            "qol_years": 0.0,
        }:
            raise ProvenanceError(
                "actual base objective weights differ from frozen welfare-only objective"
            )
        hyper, stage, effective = ppo.resolve_configuration(args, state)
        if hyper.objective_weights != {
            "welfare": 1.0,
            "population_years": 0.0,
            "trade_utility": 0.0,
            "qol_years": 0.0,
        }:
            raise ProvenanceError("actual PPO resolver changed the base objective weights")
        if effective != {"critic_only": 5, "actor": 50, "total": 55}:
            raise ProvenanceError("actual effective warmup/actor budget is not 5+50")
        init_ref = {"path": str(args.init), "sha256": expected["initialization_sha256"]}
        artifact(init_ref, base)
        if (
            args.bank_image is None
            or args.daughter_bank is None
            or not args.daughter_bank.resolve().is_relative_to(args.out.resolve())
        ):
            raise ProvenanceError("actual run lacks its private initialized bank")
        if str(args.out.resolve()) in outputs or str(args.daughter_bank.resolve()) in banks:
            raise ProvenanceError("launch runs alias one output/private bank")
        outputs.add(str(args.out.resolve()))
        banks.add(str(args.daughter_bank.resolve()))
        if phase["done"] != str(Path(args.out.name) / "done").replace("\\", "/"):
            raise ProvenanceError(
                "final queue done marker does not name this run's validated completion marker"
            )
        image_path, _ = artifact(
            {"path": str(args.bank_image), "sha256": run["bank_image_sha256"]}, base
        )
        from .curriculum import verify_image

        image = verify_image(image_path)
        bank_digests.add(image["digest"])
        config, _ = read(run["resolved_config"], base)
        # Exact complete argparse expansion and effective counts are reproducible, not handwritten summaries.
        if config != {
            "schema": 1,
            "argv": argv,
            "options": options,
            "effective_iterations": effective,
            "resolved_hyper": dataclasses.asdict(hyper),
            "resolved_stage": dataclasses.asdict(stage),
            "initialization_sha256": expected["initialization_sha256"],
            "bank_image_digest": image["digest"],
        }:
            raise ProvenanceError(
                "resolved config differs from actual PPO parser/default expansion"
            )
        if frozen_runs is not None:
            from .checkpoint import run_manifest

            args._qualified_probe = expected["live_probe"]
            frozen = run_manifest(
                args,
                type("Head", (), {"region_head": "hierarchical"})(),
                identity=expected.get("identity"),
            )
            frozen.update(
                resolved_hyper=dataclasses.asdict(hyper),
                resolved_stage=dataclasses.asdict(stage),
                effective_iterations=effective,
            )
            frozen_runs[run["phase"]] = {
                "arm": run["arm"],
                "seed": run["seed"],
                "run_manifest": frozen,
                "run_manifest_digest": digest(frozen),
            }
        cohorts.setdefault(args.arm, set()).add(run["search_cohort_digest"])
        if not args.search_labels or not args.branch_shards:
            raise ProvenanceError("actual run omits frozen search labels/branches")
        # Treatment/identity/output/input paths may differ; every other expanded default must be identical.
        excluded = {
            "arm",
            "seed",
            "out",
            "daughter_bank",
            "personality",
            "teacher_kl",
            "search_labels",
            "branch_shards",
        }
        common.append({k: v for k, v in options.items() if k not in excluded})
    if len(bank_digests) != 1 or len({digest(c) for c in common}) != 1:
        raise ProvenanceError("arms do not share identical initial bank and non-treatment defaults")
    if any(len(v) != 1 for v in cohorts.values()) or cohorts.get("R3per") != cohorts.get("R3lo"):
        raise ProvenanceError("personality arms do not share the identical frozen search cohort")
    return cohorts


def validate_search_bindings(spec, search_receipt, receipt_base):
    from . import ppo, branch_shards
    from .checkpoint import arguments

    for run in spec["runs"]:
        parser = ppo.argument_parser()
        arguments(parser)
        args = parser.parse_args(run["argv"][3:])
        item = search_receipt["evidence"][
            "training-default" if args.arm == "R3def" else "training-dirichlet"
        ]
        expected_labels = artifact(item["labels"], receipt_base)[1]
        labels = args.search_labels.split(",")
        if len(labels) != 1 or checkpoint_hash(labels[0]) != expected_labels:
            raise ProvenanceError("actual PPO argv loads labels other than the qualified cohort")
        expected_meta = sorted(ref["sha256"] for ref in item["branch_indexes"])
        actual_meta = sorted(
            checkpoint_hash(d / "meta.json")
            for d in branch_shards.directories([Path(p) for p in args.branch_shards.split(",")])
        )
        if actual_meta != expected_meta:
            raise ProvenanceError("actual PPO argv loads a different/subset branch cohort")


def validate_qualified_submission(spec: dict, base: Path, *, check_live=True) -> dict:
    """One shared non-training truth validator for initial qualification and later formal reporting."""
    scan(spec)
    if spec.get("schema") != 1 or spec.get("digest") != digest(
        {k: v for k, v in spec.items() if k != "digest"}
    ):
        raise ProvenanceError("submission specification is not frozen")
    design = execution_design(spec.get("execution", {}))
    if not design["readiness_bearing"]:
        raise ProvenanceError("final submission requires the reviewed VM cohort")
    held_path, _ = artifact(spec["calibration_manifest"], base)
    held = manifest(held_path)
    if held.get("waivers", []) != []:
        raise ProvenanceError("Round-3 waivers must be frozen empty")
    inventory_path, _ = artifact(spec["prior_use_inventory"], base)
    inventory = verify_inventory(inventory_path)
    init_path, init_sha = artifact(spec["initialization"], base)
    state = torch.load(init_path, map_location="cpu", weights_only=False)
    if (
        state.get("artifact_kind") != "promotion"
        or not checkpoint_evidence(state, held)["verified"]
        or state.get("model_semantics") != semantic_hash(state)
    ):
        raise ProvenanceError("submission initialization is not the verified promoted clone")
    from .bridge import Model9Bridge

    with Model9Bridge() as bridge:
        live = simulator_identity(bridge.schema, production=True)
        from .worker_contract import observe

        worker = observe(bridge)
    learner, binary = code_commit() if check_live else spec["learner_commit"], binary_identity()
    if (
        learner != spec.get("learner_commit")
        or live != spec.get("simulator")
        or binary["sha256"] != spec.get("bridge_sha256")
    ):
        raise ProvenanceError("submission live clean code/binary differs from frozen identities")
    flags = {
        "sim_rules": os.environ.get("ANTHROPOCENE_SIM_RULES", ""),
        "sim_statics": os.environ.get("ANTHROPOCENE_SIM_STATICS", ""),
    }
    if (
        spec.get("flags") != flags
        or {k: state["run_manifest"].get(k) for k in flags} != flags
        or re.search(
            r"ElderRawMaterialsEnabled\s*=\s*(?:true|1)\b", flags["sim_rules"], re.IGNORECASE
        )
    ):
        raise ProvenanceError(
            "actual flags differ from frozen promoted producer or enable raw materials"
        )
    # Read-only reachability check: clean is insufficient if this commit was never pushed.
    if check_live:
        result = subprocess.run(
            ["git", "rev-list", "-1", learner, "--not", "--remotes"], capture_output=True, text=True
        )
        if result.returncode or result.stdout.strip():
            raise ProvenanceError("learner commit is not reachable from a fetched remote ref")
        sim_root = spec.get("simulator_repository")
        if not sim_root:
            raise ProvenanceError(
                "submission requires the exact simulator repository for clean/pushed verification"
            )
        for args, target in (
            (["rev-parse", "HEAD"], live["commit"]),
            (["status", "--porcelain"], ""),
            (["rev-list", "-1", live["commit"], "--not", "--remotes"], ""),
        ):
            result = subprocess.run(["git", "-C", sim_root, *args], capture_output=True, text=True)
            if result.returncode or result.stdout.strip() != target:
                raise ProvenanceError("simulator repository is not the exact clean pushed source")
    expected = {
        "protocol_digest": spec["protocol_digest"],
        "initialization_sha256": init_sha,
        "learner_commit": learner,
        "simulator": live,
        "bridge_sha256": binary["sha256"],
        "calibration_manifest_digest": held["manifest_digest"],
    }
    expected["flags"] = flags
    if held.get("protocol_digest") != spec["protocol_digest"]:
        raise ProvenanceError("submission protocol differs from pre-clone exclusion manifest")
    if any(
        c["frozen_run"].get("clone_protocol", {}).get("sha256") != spec.get("clone_protocol_sha256")
        for c in state["promotion"]["candidates"]
    ):
        raise ProvenanceError("submission clones were not fit under the reviewed pre-frozen cohort")
    expected["live_probe"] = {
        "simulator": live,
        "bridge_sha256": binary["sha256"],
        "flags": flags,
        "worker_contract": worker,
    }
    frozen_runs = {}
    # live_probe is local resolver context, not an additional gate receipt identity field.
    context = dict(expected)
    context["identity"] = {"learner_commit": learner, "bridge_sha256": binary["sha256"]}
    cohorts = validate_runs(spec, base, state, context, frozen_runs)
    expected.pop("live_probe")
    receipts = {}
    if set(spec.get("gates", {})) != set(GATES):
        raise ProvenanceError("submission gate receipts incomplete")
    for gate in GATES:
        receipts[gate] = receipt(spec["gates"][gate], dict(expected, kind=gate), base=base)
        path, _ = artifact(spec["gates"][gate], base)
        if gate not in ("protocol", "submission"):
            queue_receipt(receipts[gate], expected, path.parent)
        from .gate_validation import gate as validate_gate

        try:
            value = validate_gate(gate, receipts[gate], expected, held, state, path.parent, spec)
        except (KeyError, TypeError, IndexError, OSError) as error:
            raise ProvenanceError(f"{gate} gate incomplete semantic evidence: {error}") from error
        if gate == "search":
            if cohorts["R3def"] != {value["training-default"]} or cohorts["R3per"] != {
                value["training-dirichlet"]
            }:
                raise ProvenanceError("actual launch runs name different qualified search cohorts")
            validate_search_bindings(spec, receipts[gate], path.parent)
    from .gate_validation import comparator

    for ref in spec.get("comparators", []):
        comparator(ref, base)
    if not spec.get("comparators"):
        raise ProvenanceError("submission lacks verified historical comparator evidence")
    runs = spec.get("runs", [])
    identities = {(r.get("arm"), r.get("seed")) for r in runs}
    if len(runs) != 15 or identities != {(a, s) for a in design["arms"] for s in design["arms"][a]}:
        raise ProvenanceError("submission does not contain the exact fifteen paired runs")
    return {
        "schema": 1,
        "kind": "qualified_submission",
        "status": "pass",
        "submission_digest": spec["digest"],
        "execution": design,
        "initialization": spec["initialization"],
        "protocol_digest": spec["protocol_digest"],
        "frozen_runs": frozen_runs,
        "live_probe": {
            "simulator": live,
            "bridge_sha256": binary["sha256"],
            "flags": flags,
            "worker_contract": worker,
        },
        "gates": {k: digest(v) for k, v in receipts.items()},
        "inventory_digest": inventory["digest"],
    }


def preflight(spec, base):
    return validate_qualified_submission(spec, base, check_live=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = preflight(json.loads(args.manifest.read_text()), args.manifest.parent)
    result["submission"] = {
        "path": str(args.manifest.resolve()),
        "sha256": checkpoint_hash(args.manifest),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
