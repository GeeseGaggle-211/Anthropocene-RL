"""Content-bound final-budget and best-selection evidence; filenames do not establish checkpoint forms."""  # noqa: E501 - docstring/prose line kept verbatim

import json
import math
from pathlib import Path

import torch

from .protocol import artifact
from .provenance import ProvenanceError, checkpoint_hash, digest


def build(directory, frozen):
    directory = Path(directory)
    refs = {
        k: {"path": str((directory / p).resolve()), "sha256": checkpoint_hash(directory / p)}
        for k, p in (
            ("trajectory", "trajectory.jsonl"),
            ("best", "best.pt"),
            ("latest", "latest.pt"),
        )
    }
    body = {
        "schema": 1,
        "kind": "run_completion",
        "arm": frozen["options"]["arm"],
        "seed": frozen["options"]["seed"],
        "run_manifest_digest": digest(frozen),
        "effective_iterations": frozen["effective_iterations"],
        **refs,
    }
    body["digest"] = digest(body)
    validate_body(body, directory)
    return body


def validate_body(body, base):
    if (
        body.get("schema") != 1
        or body.get("kind") != "run_completion"
        or body.get("digest") != digest({k: v for k, v in body.items() if k != "digest"})
        or body.get("effective_iterations") != {"critic_only": 5, "actor": 50, "total": 55}
    ):
        raise ProvenanceError("completion receipt missing final 5+50 budget/identity")
    path, _ = artifact(body["trajectory"], base)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if "iteration" in row]
    if [r["iteration"] for r in rows] != list(range(55)) or any(
        r.get("critic_warmup") is not (i < 5) for i, r in enumerate(rows)
    ):
        raise ProvenanceError(
            "completion trajectory is early/incomplete or has wrong effective actor/critic budget"
        )
    eligible = [
        (i, r["eval"]["selection_score"])
        for i, r in enumerate(rows)
        if r.get("eval", {}).get("eligible") is True
    ]
    if not eligible or any(not math.isfinite(score) for _, score in eligible):
        raise ProvenanceError("completion lacks eligible finite checkpoint-selection evidence")
    selected = min(eligible, key=lambda x: (-x[1], x[0]))[0]
    for form, iteration in (("best", selected), ("latest", 54)):
        path, _ = artifact(body[form], base)
        state = torch.load(path, map_location="cpu", weights_only=False)
        run = state.get("run_manifest", {})
        if (
            digest(run) != body["run_manifest_digest"]
            or state.get("iteration") != iteration
            or run.get("options", {}).get("arm") != body["arm"]
            or run.get("options", {}).get("seed") != body["seed"]
            or run.get("effective_iterations") != body["effective_iterations"]
        ):
            raise ProvenanceError(
                "completion selected/final checkpoint does not match immutable run/budget"
            )
        worker = run.get("worker_contract")
        if worker is None:
            raise ProvenanceError("completion lacks frozen pre-execution producer evidence")
        expected_producer = {
            "simulator": worker["simulator"],
            "bridge_sha256": worker["bridge_sha256"],
            "flags": worker["flags"],
            "raw_materials": False,
        }
        for index, _score in eligible:
            evidence = rows[index]["eval"]
            if (
                evidence.get("worker_contract") != worker
                or evidence.get("producer") != expected_producer
            ):
                raise ProvenanceError(
                    "completion selection metrics lack exact worker producer evidence"
                )
            sources = list(evidence.get("regimes", {}).values())
            if run.get("options", {}).get("personality") != "default":
                from .ppo import SELECTION_POINTS

                points = evidence.get("spread", {}).get("points", {})
                if set(points) != set(SELECTION_POINTS):
                    raise ProvenanceError(
                        "completion lacks complete personality selection producers"
                    )
                sources += list(points.values())
            if not sources or any(
                p.get("worker_contract") != worker or p.get("producer") != expected_producer
                for p in sources
            ):
                raise ProvenanceError(
                    "completion selection regime/point producer evidence missing or different"
                )
    return body


def validate(ref, checkpoint, form, base=None):
    path, _ = artifact(ref, base)
    body = validate_body(json.loads(path.read_text()), path.parent)
    if form not in ("best", "latest") or checkpoint_hash(checkpoint) != body[form]["sha256"]:
        raise ProvenanceError("checkpoint is not the receipt's selected best/final latest")
    return body
