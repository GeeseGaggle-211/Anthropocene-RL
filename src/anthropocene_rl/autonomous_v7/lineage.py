"""Auditable closure of initialization, training and selection families for calibration checkpoints."""  # noqa: E501 - docstring/prose line kept verbatim

from __future__ import annotations

import re

from . import contracts
from .calibration import assert_training, manifest
from .provenance import ProvenanceError, digest


def families(seeds=(), snapshots=()) -> dict:
    return {
        "seeds": sorted({int(x) for x in seeds}),
        "snapshots": sorted({str(x) for x in snapshots if x}),
    }


def rows_families(rows: dict) -> dict:
    if "source_seed" not in rows:
        raise ProvenanceError("dataset lacks source-family evidence")
    return families(rows["source_seed"], rows.get("source_snapshot", []))


def merge(*records) -> dict:
    return families(
        [s for r in records for s in r.get("seeds", [])],
        [s for r in records for s in r.get("snapshots", [])],
    )


def proof(state: dict) -> dict:
    return {
        k: state.get(k)
        for k in (
            "contract",
            "calibration_manifest",
            "run_manifest",
            "family_lineage",
            "artifact_kind",
            "resumable",
            "nonqualifying",
            "promotion",
            "model_semantics",
            "bc_config",
            "token",
            "vector_critic",
            "region_head",
            "core_width",
            "privileged_width",
        )
    }


def teacher_proof(record: dict) -> bool:
    from .teacher_data import validate_source

    if (
        not isinstance(record, dict)
        or record.get("schema") != 1
        or record.get("kind") != "native_teacher"
        or record.get("verified") is not True
        or not record.get("datasets")
        or record.get("digest") != digest({k: v for k, v in record.items() if k != "digest"})
    ):
        return False
    try:
        from .protocol import artifact
        from .teacher_data import verify_dataset

        refs = record.get("dataset_refs", [])
        if len(refs) != len(record["datasets"]):
            return False
        for ref, declared in zip(refs, record["datasets"], strict=False):
            path, _ = artifact(ref)
            if verify_dataset(path.parent) != declared:
                return False
        for dataset in record["datasets"]:
            validate_source(dataset["source_record"])
            if (
                not dataset.get("shards")
                or dataset.get("dataset_digest")
                != digest({k: v for k, v in dataset.items() if k != "dataset_digest"})
                or dataset["source_run"] != dataset["source_record"]["source_run"]
            ):
                return False
        return {d["source_run"] for d in record["datasets"]} == {
            r["source_run"] for r in record["records"]
        }
    except (KeyError, TypeError, ValueError, OSError):
        return False


def teacher_families(record: dict) -> dict:
    return families(
        [
            w["seed"]
            for d in record.get("datasets", [])
            for w in d["source_record"]["options"]["worlds"]
        ],
        [
            w.get("snapshot_sha256", "")
            for d in record.get("datasets", [])
            for w in d["source_record"]["options"]["worlds"]
        ],
    )


def checkpoint_evidence(state: dict, held: dict) -> dict:
    """A clean shard manifest cannot establish that the evaluated model (or its ancestors) was held out."""  # noqa: E501 - docstring/prose line kept verbatim
    failures = []
    if state.get("nonqualifying") or state.get("run_manifest", {}).get("options", {}).get(
        "nonqualifying"
    ):
        failures.append("explicit nonqualifying diagnostic artifact cannot establish qualification")
    if "actor" in state and state.get("model_semantics"):
        from .promotion import semantic_hash

        if state["model_semantics"] != semantic_hash(state):
            failures.append("checkpoint model/normalizer semantic seal mismatch")
        if (
            state.get("run_manifest", {}).get("clone_protocol")
            and state.get("artifact_kind") != "promotion"
        ):
            from .promotion import validate_cohort

            try:
                validate_cohort(state)
            except (ValueError, OSError, KeyError, TypeError):
                failures.append("checkpoint differs from pre-frozen clone cohort")
    from .protocol import scan

    scan(state)
    expected = held["manifest_digest"]
    own = state.get("calibration_manifest")
    try:
        own_valid = isinstance(own, dict) and manifest(own)["manifest_digest"] == expected
    except (ValueError, TypeError, KeyError):
        own_valid = False
    if not own_valid:
        failures.append("checkpoint exclusion manifest missing or mismatched")
    run = state.get("run_manifest") or {}
    sim = run.get("simulator", {})
    config = {
        "options": run.get("options", {}),
        "sim_rules": run.get("sim_rules", ""),
        "sim_statics": run.get("sim_statics", ""),
    }
    if (
        state.get("contract") != contracts.NATIVE_CONTRACT
        or run.get("contract") != contracts.NATIVE_CONTRACT
        or run.get("utility_version") != 2
        or run.get("calibration_manifest_digest") != expected
        or not re.fullmatch(r"[0-9a-f]{40}", str(run.get("code", "")))
        or not re.fullmatch(r"[0-9a-f]{40}", str(sim.get("commit", "")))
        or sim.get("clean") is not True
        or not sim.get("buildHash")
        or run.get("config_digest") != digest(config)
    ):
        failures.append("checkpoint frozen provenance is missing, dirty or inconsistent")
    record = state.get("family_lineage") or {}
    if (
        record.get("schema") != 2
        or not record.get("verified")
        or record.get("calibration_manifest_digest") != expected
        or any(
            not isinstance(record.get(role), dict) or set(record[role]) != {"seeds", "snapshots"}
            for role in ("training", "selection")
        )
        or record.get("digest") != digest({k: v for k, v in record.items() if k != "digest"})
    ):
        failures.append(
            "checkpoint initialization/training lineage evidence is missing or unverified"
        )
    else:
        for role in ("training", "selection"):
            group = record.get(role, {})
            try:
                assert_training(held, group.get("seeds", []), group.get("snapshots", []))
            except ProvenanceError:
                failures.append(f"checkpoint {role} lineage includes a calibration family")
        for parent in record.get("parents", []):
            if not re.fullmatch(r"[0-9a-f]{64}", str(parent.get("checkpoint", ""))):
                failures.append("initialization checkpoint identity missing")
            try:
                import torch

                from .protocol import artifact

                path, sha = artifact(parent.get("artifact", {}))
                actual = torch.load(path, map_location="cpu", weights_only=False)
                if sha != parent["checkpoint"] or proof(actual) != parent.get("proof"):
                    raise ProvenanceError(
                        "parent SHA/embedded proof differs from actual parent artifact"
                    )
                check = checkpoint_evidence(actual, held)
            except (OSError, ValueError, KeyError, TypeError) as error:
                check = {"failures": ["parent content evidence invalid: " + str(error)]}
            failures.extend("inherited " + f for f in check["failures"])
        init = run.get("inputs", {}).get("init")
        parents = record.get("parents", [])
        if not parents and (
            not teacher_proof(record.get("teacher_inputs", {}).get("training"))
            or not teacher_proof(record.get("teacher_inputs", {}).get("selection"))
        ):
            failures.append("random root lacks verified teacher source proofs")
        for role, inputs in record.get("teacher_inputs", {}).items():
            if not teacher_proof(inputs) or run.get("teacher_inputs", {}).get(role) != inputs.get(
                "digest"
            ):
                failures.append("teacher source proof differs from frozen run inputs")
            else:
                family = teacher_families(inputs)
                try:
                    assert_training(held, family["seeds"], family["snapshots"])
                    for source in inputs["records"]:
                        from .teacher_data import validate_source

                        validate_source(source, live=sim, binary_sha256=run.get("bridge_sha256"))
                        if source["flags"] != {k: run[k] for k in ("sim_rules", "sim_statics")}:
                            raise ProvenanceError("teacher flags differ")
                    if not set(family["seeds"]) <= set(record[role]["seeds"]) or not set(
                        family["snapshots"]
                    ) <= set(record[role]["snapshots"]):
                        raise ProvenanceError("teacher family omitted from lineage closure")
                except ProvenanceError as error:
                    failures.append(str(error))
        for name in ("init", "teacher"):
            weights = run.get("inputs", {}).get(name)
            if weights and not any(parent.get("checkpoint") == weights for parent in parents):
                failures.append(f"{name} weights are not bound to inherited family evidence")
        if not init and run.get("initialization") != "random":
            failures.append("checkpoint has no attributable initialization")
        if state.get("artifact_kind") == "promotion":
            from .promotion import validate_embedded

            try:
                validate_embedded(state)
            except ProvenanceError as error:
                failures.append(str(error))
    return {"verified": not failures, "failures": failures, "lineage_digest": record.get("digest")}


def build(
    held: dict | None, training: dict, selection: dict, parents=(), *, teacher_inputs=None
) -> dict:
    """Parents are (full checkpoint hash, saved state) pairs; missing history remains explicitly unverified."""  # noqa: E501 - docstring/prose line kept verbatim
    parent_proofs, reasons = [], []
    teacher_inputs = teacher_inputs or {}
    if not parents and (
        not teacher_proof(teacher_inputs.get("training"))
        or not teacher_proof(teacher_inputs.get("selection"))
    ):
        reasons.append("random root lacks verified teacher source proofs")
    inherited_train, inherited_selection = [], []
    training = merge(training, teacher_families(teacher_inputs.get("training", {})))
    selection = merge(selection, teacher_families(teacher_inputs.get("selection", {})))
    for item in parents:
        checkpoint_hash, state = item[:2]
        artifact_ref = item[2] if len(item) == 3 else {}
        try:
            import torch

            from .protocol import artifact

            path, sha = artifact(artifact_ref)
            if sha != checkpoint_hash or proof(
                torch.load(path, map_location="cpu", weights_only=False)
            ) != proof(state):
                raise ProvenanceError("parent artifact/proof mismatch")
        except (ValueError, OSError, KeyError, TypeError):
            reasons.append("parent lacks matching content-backed artifact")
        parent = state.get("family_lineage") or {}
        if held:
            for role in ("training", "selection"):
                family = parent.get(role, {})
                assert_training(held, family.get("seeds", []), family.get("snapshots", []))
            check = checkpoint_evidence(state, held)
            reasons.extend("initialization: " + f for f in check["failures"])
        else:
            reasons.append("no frozen exclusion manifest")
        parent_proofs.append(
            {"checkpoint": checkpoint_hash, "artifact": artifact_ref, "proof": proof(state)}
        )
        inherited_train.append(parent.get("training", {}))
        inherited_selection.append(parent.get("selection", {}))
    training = merge(training, *inherited_train)
    selection = merge(selection, *inherited_selection)
    if held:
        assert_training(
            held,
            training["seeds"] + selection["seeds"],
            training["snapshots"] + selection["snapshots"],
        )
    record = {
        "schema": 2,
        "verified": bool(held) and not reasons,
        "reasons": reasons,
        "teacher_inputs": teacher_inputs,
        "calibration_manifest_digest": (held or {}).get("manifest_digest"),
        "training": training,
        "selection": selection,
        "parents": parent_proofs,
    }
    record["digest"] = digest(record)
    return record


def source_families(report: dict) -> dict:
    """All input families, including search rows filtered out by winner/margin selection."""
    if "families" in report:
        return report["families"]
    return families(
        [m["seed"] for m in report.get("branches", [])],
        [m.get("snapshot_hash", "") for m in report.get("branches", [])],
    )
