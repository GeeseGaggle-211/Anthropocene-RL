"""Fail-closed evidence, family inventory and seed boundaries for the reviewed Round-3 protocol."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .provenance import ProvenanceError, checkpoint_hash, digest

ROLES = {
    "teacher_train": list(range(1000, 1600)),
    "teacher_dev": list(range(2100, 2160)),
    "clone_selection": list(range(2160, 2180)),
    "ppo_selection": list(range(2060, 2100)),
    "development": list(range(2000, 2060)),
    "calibration": list(range(10000, 10100)),
}
DENIED = set(range(3000, 3060))


def protocol_digest(document):
    """Version-1 projection: only the explicitly late-bound clone-plan SHA and digest slot are excluded.

    Approval is external, never embedded (embedding its SHA would create a cryptographic cycle).
    Every other field, including unknown future fields, participates in the immutable decision-rule identity.
    """
    if document.get("protocol_digest_version") != 1:
        raise ProvenanceError("unsupported protocol digest projection")
    return digest(
        {k: v for k, v in document.items() if k not in ("digest", "clone_protocol_sha256")}
    )


def verify_preregistration(protocol_ref, approval_ref, base=None):
    path, sha = artifact(protocol_ref, base)
    approval_path, _ = artifact(approval_ref, base)
    document = json.loads(path.read_text())
    approval = json.loads(approval_path.read_text())
    scan(document)
    scan(approval)
    projected = protocol_digest(document)
    if (
        document.get("digest") != projected
        or approval.get("schema") != 1
        or approval.get("kind") != "protocol_approval"
        or approval.get("reviewer") != "Astra"
        or approval.get("decision") != "APPROVE"
        or approval.get("protocol_digest") != projected
        or approval.get("preregistration_sha256") != sha
    ):
        raise ProvenanceError(
            "external immutable approval does not bind the complete preregistration bytes"
        )
    return document


def seeds(values, context="input") -> list[int]:
    result = []
    for value in values:
        if isinstance(value, bool) or int(value) != value:
            raise ProvenanceError(f"invalid seed in {context}")
        seed = int(value)
        if seed in DENIED:
            raise ProvenanceError(f"confirmation seed denied in {context}: {seed}")
        result.append(seed)
    return result


def seed_list(value: str) -> list[int]:
    out = []
    for part in value.split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", part.strip()):
            raise ProvenanceError("seed list must be explicit integer/range values")
        a, _, b = part.strip().partition("-")
        lo, hi = int(a), int(b or a)
        if hi < lo or hi - lo > 1000000:
            raise ProvenanceError("invalid seed range")
        if any(lo <= seed <= hi for seed in DENIED):
            raise ProvenanceError("confirmation seed denied in seed range")
        out.extend(range(lo, hi + 1))
    return seeds(out)


def scan(value, context="evidence") -> None:
    """Inspect semantic seed-bearing fields, including recursively inherited checkpoint proofs."""
    if isinstance(value, dict):
        for key, item in value.items():
            lower = key.lower()
            if (lower == "seed" or lower.endswith("_seed") or lower == "seed_value") and isinstance(
                item, (int, float)
            ):
                seeds([item], context + "/" + key)
            elif (
                (lower == "seeds" or lower.endswith("_seeds"))
                and isinstance(item, list)
                and all(isinstance(x, (int, float)) for x in item)
            ):
                seeds(item, context + "/" + key)
            elif (
                (lower == "seeds" or lower.endswith("_seeds"))
                and isinstance(item, str)
                and re.fullmatch(r"[\d, -]+", item)
            ):
                seed_list(item)
            scan(item, context + "/" + key)
    elif isinstance(value, list):
        for item in value:
            scan(item, context)


def command(argv: list[str]) -> None:
    for i, token in enumerate(argv):
        name, sep, inline = str(token).partition("=")
        if name.startswith("--") and (
            name.endswith("-seed") or name.endswith("-seeds") or name in ("--seed", "--seeds")
        ):
            selected = inline if sep else argv[i + 1] if i + 1 < len(argv) else ""
            if name in ("--grassland-seeds", "--valley-seeds"):
                content = Path(selected).read_text()
                if name == "--grassland-seeds":
                    seed_list(content.strip())
                else:
                    seeds([int(row.split(",")[0]) for row in content.split()])
            else:
                seed_list(selected)


def artifact(ref: dict, base: Path | None = None) -> tuple[Path, str]:
    """A digest is evidence only when the named content exists and is rehashed."""
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise ProvenanceError("evidence requires a content path and SHA-256")
    path = Path(ref["path"])
    if not path.is_absolute():
        path = (base or Path.cwd()) / path
    actual = checkpoint_hash(path)
    if not re.fullmatch(r"[0-9a-f]{64}", actual) or actual != ref.get("sha256"):
        raise ProvenanceError(f"evidence content missing or hash mismatch: {path}")
    return path.resolve(), actual


def receipt(ref: dict, expected: dict, *, base: Path | None = None) -> dict:
    path, _ = artifact(ref, base)
    body = json.loads(path.read_text())
    scan(body)
    if (
        body.get("schema") != 1
        or body.get("status") != "pass"
        or any(body.get(k) != v for k, v in expected.items())
    ):
        raise ProvenanceError("receipt identity/status differs from expected evidence")
    outputs = body.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ProvenanceError("receipt needs hash-verified outputs, not a bare success marker")
    for output in outputs:
        artifact(output, path.parent)
    return body


def execution_design(design: dict) -> dict:
    """Local probes cannot select arms or prove readiness; readiness is a reviewed VM cohort."""
    if design.get("tier") not in ("local", "vm") or design.get("purpose") not in (
        "validation",
        "stability",
        "readiness",
    ):
        raise ProvenanceError("execution tier and purpose must be explicit")
    arms = design.get("arms", {})
    if not isinstance(arms, dict) or not arms:
        raise ProvenanceError("execution design needs explicit matched arm seed lists")
    for group in arms.values():
        if not isinstance(group, list) or len(group) != len(set(seeds(group, "optimizer seeds"))):
            raise ProvenanceError("execution design has duplicate/invalid optimizer seeds")
    if design["tier"] == "local":
        if (
            design["purpose"] == "readiness"
            or design.get("selects_arm", False)
            or len(set(s for group in arms.values() for s in group)) > 3
        ):
            raise ProvenanceError(
                "local training is limited to three diagnostic seeds and cannot select or establish readiness"
            )
    elif design["purpose"] == "readiness":
        if set(arms) != {"R3def", "R3per", "R3lo"} or any(
            group != list(range(170, 175)) for group in arms.values()
        ):
            raise ProvenanceError(
                "readiness requires three matched arms and five paired seeds; alternate designs require reviewed preregistration"
            )
    return {
        **design,
        "readiness_bearing": design["tier"] == "vm" and design["purpose"] == "readiness",
    }


def enforce_execution(args):
    """Registered arms may optimize only inside their exact qualified immutable queue job."""
    if getattr(args, "arm", None) is None:
        if not getattr(args, "nonqualifying", False):
            raise ProvenanceError(
                "training requires frozen VM execution authority or explicit --nonqualifying diagnostic mode"
            )
        if any(
            getattr(args, name, None)
            for name in (
                "calibration_manifest",
                "clone_protocol",
                "preregistration",
                "protocol_approval",
                "execution_manifest",
            )
        ):
            raise ProvenanceError(
                "nonqualifying local probes cannot carry protocol/qualification inputs"
            )
        path = getattr(args, "local_execution_design", None)
        if path is None:
            raise ProvenanceError("nonqualifying probes require frozen --local-execution-design")
        frozen = json.loads(Path(path).read_text())
        if frozen.get("digest") != digest({k: v for k, v in frozen.items() if k != "digest"}):
            raise ProvenanceError("local execution design is not frozen")
        design = execution_design(frozen.get("execution", {}))
        if (
            design["tier"] != "local"
            or set(design["arms"]) != {"diagnostic"}
            or args.seed not in design["arms"]["diagnostic"]
        ):
            raise ProvenanceError(
                "local execution must name at most three diagnostic seeds and this exact seed"
            )
        return {"nonqualifying": True, "selection_allowed": False, "execution": design}
    if getattr(args, "nonqualifying", False) or getattr(args, "local_execution_design", None):
        raise ProvenanceError("registered arms cannot use nonqualifying/local execution mode")
    import os
    import sys

    path = getattr(args, "execution_manifest", None)
    if path is None:
        raise ProvenanceError(
            "registered arms require --execution-manifest; local probes must remain diagnostic"
        )
    spec = json.loads(Path(path).read_text())
    scan(spec)
    design = execution_design(spec.get("execution", {}))
    if not design["readiness_bearing"]:
        raise ProvenanceError("registered training arms require the reviewed VM execution design")
    job_path, _ = artifact(spec["queue_job"], Path(path).parent)
    job = json.loads(job_path.read_text())
    require_vm_authority(spec["execution"])
    rows = [
        r for r in spec.get("runs", []) if r.get("arm") == args.arm and r.get("seed") == args.seed
    ]
    if (
        os.environ.get("JOB") != job.get("name")
        or not os.environ.get("CODE")
        or not os.environ.get("RESULTS")
        or len(rows) != 1
        or rows[0].get("argv", [])[3:] != sys.argv[1:]
    ):
        raise ProvenanceError("actual launch is not the exact expanded argv of its VM queue job")
    from .qualification import preflight

    return preflight(spec, Path(path).parent)


def enforce_clone_execution(args):
    """All-ten clone qualification fits need a frozen VM queue plan; local probes never qualify."""
    if getattr(args, "nonqualifying", False):
        return enforce_execution(args)
    if getattr(args, "local_execution_design", None):
        raise ProvenanceError("qualifying clones cannot carry local execution design")
    import os
    import shlex
    import sys

    path = getattr(args, "execution_manifest", None)
    if path is None:
        raise ProvenanceError("qualifying clone fitting requires frozen VM --execution-manifest")
    spec = json.loads(Path(path).read_text())
    scan(spec)
    if (
        spec.get("kind") != "clone_execution"
        or spec.get("digest") != digest({k: v for k, v in spec.items() if k != "digest"})
        or {k: v for k, v in spec.get("execution", {}).items() if k != "vm"}
        != {"tier": "vm", "purpose": "qualification", "arms": {"clone": list(range(10))}}
    ):
        raise ProvenanceError(
            "guarded clone cohort requires frozen ten-candidate VM qualification design"
        )
    job_path, _ = artifact(spec.get("queue_job"), Path(path).parent)
    job = json.loads(job_path.read_text())
    require_vm_authority(spec["execution"])
    runs = spec.get("runs", [])
    if len(runs) != 10 or sorted(row.get("seed", -1) for row in runs) != list(range(10)):
        raise ProvenanceError("clone execution plan lacks all ten identities")
    rows = [row for row in runs if row["seed"] == args.seed]
    phases = {phase["name"]: phase for phase in job.get("phases", [])}
    if (
        os.environ.get("JOB") != job.get("name")
        or not os.environ.get("CODE")
        or not os.environ.get("RESULTS")
        or len(rows) != 1
        or rows[0].get("argv", [])[3:] != sys.argv[1:]
        or phases.get(rows[0].get("phase"), {}).get("run") != shlex.join(rows[0]["argv"])
    ):
        raise ProvenanceError("clone fitting is not the exact authorized VM queue phase")
    return {
        "nonqualifying": False,
        "selection_allowed": True,
        "execution_manifest_sha256": checkpoint_hash(path),
    }


def enforce_qualification_execution(args, module):
    """Publishing clone-selection/promotion evidence is a qualification phase, not a local selection probe."""
    import os
    import shlex
    import sys

    path = getattr(args, "execution_manifest", None)
    if path is None:
        raise ProvenanceError("qualification selection requires frozen VM --execution-manifest")
    spec = json.loads(Path(path).read_text())
    scan(spec)
    if (
        spec.get("kind") != "clone_execution"
        or spec.get("digest") != digest({k: v for k, v in spec.items() if k != "digest"})
        or {k: v for k, v in spec.get("execution", {}).items() if k != "vm"}
        != {"tier": "vm", "purpose": "qualification", "arms": {"clone": list(range(10))}}
    ):
        raise ProvenanceError(
            "clone qualification requires the reviewed frozen ten-candidate VM design"
        )
    job_path, _ = artifact(spec.get("queue_job"), Path(path).parent)
    job = json.loads(job_path.read_text())
    require_vm_authority(spec["execution"])
    phases = [
        p
        for p in job.get("phases", [])
        if (
            lambda argv: len(argv) > 3 and argv[1:3] == ["-m", module] and argv[3:] == sys.argv[1:]
        )(shlex.split(p["run"]))
    ]
    if (
        len(phases) != 1
        or os.environ.get("JOB") != job.get("name")
        or not os.environ.get("CODE")
        or not os.environ.get("RESULTS")
    ):
        raise ProvenanceError("qualification selection is not its exact authorized VM queue phase")
    return spec


def require_vm_authority(execution):
    """Read only the current machine's fixed GCP metadata identity; environment prose is not VM authority."""
    import urllib.request

    expected = execution.get("vm")
    if not isinstance(expected, dict) or set(expected) != {"project_id", "instance_id", "zone"}:
        raise ProvenanceError("VM execution requires pre-frozen project/instance/zone identity")
    actual = {}
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for key, endpoint in (
            ("project_id", "project/project-id"),
            ("instance_id", "instance/id"),
            ("zone", "instance/zone"),
        ):
            request = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/" + endpoint,
                headers={"Metadata-Flavor": "Google"},
            )
            with opener.open(request, timeout=2) as response:
                if response.headers.get("Metadata-Flavor") != "Google":
                    raise ValueError("untrusted metadata endpoint")
                actual[key] = response.read(4096).decode().strip().split("/")[-1]
    except (OSError, ValueError) as error:
        raise ProvenanceError("current host cannot prove frozen VM execution authority") from error
    if actual != expected:
        raise ProvenanceError("current VM differs from frozen execution authority")


def search_cohort(path: Path, bank, config, held: dict) -> tuple[dict, list[tuple]]:
    """Resolve an immutable exact cohort; no newest-entry selection or error-driven top-up."""
    import hashlib

    obj = json.loads(path.read_text())
    scan(obj)
    if obj.get("schema") != 1 or obj.get("digest") != digest(
        {k: v for k, v in obj.items() if k != "digest"}
    ):
        raise ProvenanceError("invalid frozen search cohort")
    role, rows = obj.get("role"), obj.get("worlds", [])
    if role not in ("training-default", "training-dirichlet", "calibration") or not rows:
        raise ProvenanceError("search cohort role/worlds missing")
    entries = {int(e["id"]): e for e in bank.entries()} if bank else {}
    extras = set(seeds(obj.get("extra_training_roots", []), "registered extra TRAIN roots"))
    other_roles = set(s for name, group in ROLES.items() if name != "teacher_train" for s in group)
    if extras & other_roles:
        raise ProvenanceError("extra TRAIN roots conflict with a fixed seed role")
    train_roots = set(ROLES["teacher_train"]) | extras
    out, identities = [], set()
    openings = daughters = 0
    from .calibration import held_out

    for row in rows:
        seed = seeds([row["seed"]], "search cohort")[0]
        mask, warm, entry_id = (
            int(row["withheld"]),
            int(row["warmup"]),
            int(row.get("bank_entry", -1)),
        )
        snapshot = row.get("snapshot_sha256", "")
        if warm < 0 or mask < 0:
            raise ProvenanceError("invalid cohort fixture")
        if entry_id >= 0:
            entry = entries.get(entry_id)
            if (
                entry is None
                or entry["seed"] != seed
                or entry["mask"] != mask
                or warm != 0
                or hashlib.sha256(bank.snapshot(entry)).hexdigest() != snapshot
            ):
                raise ProvenanceError("frozen search snapshot missing/mutated/mismatched")
            daughters += 1
        else:
            if mask or warm or snapshot:
                raise ProvenanceError("registered search openings must be fresh default worlds")
            openings += 1
        identity = (seed, mask, warm, snapshot)
        if identity in identities:
            raise ProvenanceError("duplicate frozen search trajectory")
        identities.add(identity)
        if role == "calibration":
            if seed not in ROLES["calibration"] or not held_out(held, seed, snapshot):
                raise ProvenanceError("calibration cohort family not held out")
            if snapshot and snapshot not in held.get("calibration_snapshots", []):
                raise ProvenanceError("calibration snapshot not frozen in exclusion manifest")
        elif held_out(held, seed, snapshot) or seed not in train_roots:
            raise ProvenanceError("training cohort family outside declared TRAIN/exclusions")
        out.append((seed, mask, warm, entry_id))
    calibration = role == "calibration"
    if (openings != 100 or daughters > 100) if calibration else (openings != 60 or daughters != 60):
        raise ProvenanceError("frozen search cohort count differs from preregistration")
    if (
        config.points != 4
        or config.villages != 2
        or config.horizon != 6
        or config.tail != (12 if calibration else 4)
        or config.turns != 25
        or config.first_turn != 1
        or not config.daughter_search
        or config.calibration_tail != 12
        or config.bootstrap
        or config.calibration_share != (1 if calibration else 0)
        or config.calibration_min_per_kind != (1 if calibration else 0)
    ):
        raise ProvenanceError("search flags differ from frozen cohort protocol")
    if any(not getattr(config, key) for key in ("native", "regions", "mixes", "fission", "trade")):
        raise ProvenanceError("registered search requires all current alternatives")
    personality = "dirichlet:1" if role == "training-dirichlet" else "default"
    if (
        obj.get("max_alternatives") != 19
        or obj.get("personality") != personality
        or config.personality != personality
    ):
        raise ProvenanceError("search alternatives/personality not frozen")
    return obj, out


def inventory(spec: dict, *, base: Path | None = None) -> dict:
    """Inventory explicitly enumerated prior-use artifacts; unknown evidence never proves absence of exposure."""
    base = base or Path.cwd()
    if spec.get("schema") != 1 or not spec.get("artifacts") or not spec.get("scope_roots"):
        raise ProvenanceError("prior-use inventory needs explicit artifact and scope-root coverage")
    roles = {k: set(v) for k, v in ROLES.items()}
    extras = set(seeds(spec.get("extra_training_roots", [])))
    if extras & set(s for k, v in roles.items() if k != "teacher_train" for s in v):
        raise ProvenanceError("extra TRAIN roots conflict with a fixed seed role")
    roles["teacher_train"].update(extras)
    paths, families, records = set(), {}, []
    for ref in spec["artifacts"]:
        path, sha = artifact(ref, base)
        if path in paths:
            raise ProvenanceError("duplicate inventory artifact")
        paths.add(path)
        role = ref.get("role")
        if role not in roles and role != "prior_use":
            raise ProvenanceError("unknown inventory role")
        kind = ref.get("kind")
        if kind == "checkpoint":
            import torch

            value = torch.load(path, map_location="cpu", weights_only=False)
            if not value.get("family_lineage"):
                raise ProvenanceError("checkpoint lacks prior-use lineage evidence")
        elif kind in ("json", "report", "manifest", "receipt"):
            value = json.loads(path.read_text())
        elif kind == "bank_index":
            import hashlib
            import zlib

            value = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for entry in value:
                raw = zlib.decompress((path.parent / entry["file"]).read_bytes())
                if hashlib.sha256(raw).hexdigest() != entry.get("snapshot_sha256"):
                    raise ProvenanceError(
                        "bank inventory lacks exact decompressed snapshot evidence"
                    )
                paths.add((path.parent / entry["file"]).resolve())
        elif kind == "world_list":
            value = {
                "seeds": [
                    int(line.split(",")[0]) for line in path.read_text().split() if line.strip()
                ]
            }
        else:
            raise ProvenanceError("unsupported inventory artifact; cannot infer families")
        scan(value, str(path))
        found = set()

        def visit(node):
            if isinstance(node, dict):
                for key, item in node.items():
                    if key == "seed" and isinstance(item, int):
                        found.add(item)
                    elif (key == "seeds" or key.endswith("_seeds")) and isinstance(item, list):
                        found.update(x for x in item if isinstance(x, int))
                    visit(item)
            elif isinstance(node, list):
                for item in node:
                    visit(item)

        visit(value)
        if not found:
            raise ProvenanceError(f"inventory artifact has no attributable families: {path}")
        if role != "prior_use" and not found <= roles[role]:
            raise ProvenanceError("artifact family differs from its registered role")
        families.setdefault(role, set()).update(found)
        records.append({"path": str(path), "sha256": sha, "role": role, "seeds": sorted(found)})
    for root in spec["scope_roots"]:
        directory = Path(root)
        directory = directory if directory.is_absolute() else base / directory
        if not directory.is_dir():
            raise ProvenanceError("inventory scope root missing")
        # An explicit ignore list is hash-bound to the inventory specification; no silent omission of possible evidence.
        if spec.get("ignored_files"):
            raise ProvenanceError("prior-use evidence cannot be silently ignored")
        unlisted = [p for p in directory.rglob("*") if p.is_file() and p.resolve() not in paths]
        if unlisted:
            raise ProvenanceError(f"unlisted prior-use evidence in scope: {unlisted[0]}")
    cal = roles["calibration"]
    if any(cal & values for role, values in families.items() if role != "calibration"):
        raise ProvenanceError("calibration roots have prior non-calibration exposure")
    result = {
        "schema": 1,
        "status": "pass",
        "spec": spec,
        "base": str(base.resolve()),
        "spec_digest": digest(spec),
        "records": records,
        "families": {k: sorted(v) for k, v in families.items()},
    }
    result["digest"] = digest(result)
    return result


def verify_inventory(path: Path) -> dict:
    saved = json.loads(Path(path).read_text())
    if saved.get("digest") != digest({k: v for k, v in saved.items() if k != "digest"}):
        raise ProvenanceError("inventory receipt digest mismatch")
    rebuilt = inventory(saved.get("spec", {}), base=Path(saved.get("base", ".")))
    if rebuilt != saved:
        raise ProvenanceError("inventory receipt does not match current evidence contents")
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(json.loads(args.spec.read_text()), base=args.spec.parent)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
