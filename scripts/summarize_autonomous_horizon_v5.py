"""Compact versioned evidence; preserve complete raw runs and checkpoints in place."""

import argparse
import json
from pathlib import Path

from preflight_autonomous_v4_residual import sha


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    preflight_path = Path("runs/autonomous-residual-preflight-v4-fixed.json")
    canary_path = Path("runs/autonomous-critic-horizon-gate-v5/summary.json")
    diagnostics_path = Path("runs/autonomous-horizon-diagnostics-v5/summary.json")
    resume_path = Path("runs/autonomous-horizon-resume-v5.json")
    panel_path = Path("runs/autonomous-horizon-welfare-v5/summary.json")
    heuristic_path = Path("runs/autonomous-heuristic-welfare-v1/summary.json")
    preflight, canary, diagnostics, resume, panel, heuristic = map(
        read,
        (preflight_path, canary_path, diagnostics_path, resume_path, panel_path, heuristic_path),
    )
    pins = read("configs/training/autonomous-horizon-postflight-v5.json")
    for pin in [*canary["config"]["inputs"].values(), *pins["inputs"].values()]:
        assert sha(pin["path"]) == pin["sha256"]
    raw_paths = [
        preflight_path,
        canary_path,
        diagnostics_path,
        resume_path,
        panel_path,
        heuristic_path,
    ]
    result = dict(
        version="autonomous-horizon-evidence-v5",
        script_sha256=sha(__file__),
        raw_evidence={str(p): sha(p) for p in raw_paths},
        technical_pass=(
            preflight["complete"]
            and preflight["passed"]
            and canary["complete"]
            and diagnostics["complete"]
            and diagnostics["passed"]
            and resume["complete"]
            and panel["complete"]
        ),
        economic_outcome_is_acceptance_threshold=False,
        independent_review_required=True,
    )
    result["preflight"] = {
        k: v for k, v in preflight.items() if k not in ("transfer", "adaptation")
    }
    result["preflight"]["transfer"] = preflight["transfer"]
    result["preflight"]["adaptation"] = [
        {k: v for k, v in row.items() if k != "mse_trace"} for row in preflight["adaptation"]
    ]
    result["canary"] = {k: v for k, v in canary.items() if k != "cells"}
    result["canary"]["cells"] = []
    for cell in canary["cells"]:
        compact = {k: v for k, v in cell.items() if k not in ("updates", "episodes")}
        compact["ppo_updates"] = len(cell["updates"])
        compact["minibatches"] = sum(r["minibatches"] for r in cell["updates"])
        compact["actor_minibatches"] = sum(r["actor_updates"] for r in cell["updates"])
        keys = (
            "actor_step_l2",
            "critic_step_l2",
            "actor_relative_step",
            "critic_relative_step",
            "actor_first_layer_grad",
            "critic_first_layer_grad",
            "return_mean",
            "return_std",
            "critic_bias",
            "explained_variance",
            "critic_mse_over_return_variance",
            "maximum_kl",
        )
        compact["update_ranges"] = {
            k: dict(
                minimum=min(r[k] for r in cell["updates"]),
                maximum=max(r[k] for r in cell["updates"]),
            )
            for k in keys
        }
        compact["episodes"] = len(cell["episodes"])
        compact["training_focal_events"] = {
            k: sum(r["events"][k] for r in cell["episodes"])
            for k in ("movement", "quotes", "focal_trades", "focal_fissions", "focal_migrants")
        }
        compact["training_family_episodes"] = {
            f: sum(r["family"] == f for r in cell["episodes"])
            for f in sorted({r["family"] for r in cell["episodes"]})
        }
        result["canary"]["cells"].append(compact)
    result["retained_diagnostics"] = dict(diagnostics, cases=[])
    for item in diagnostics["checkpoints"]:
        path = Path(item["output"])
        raw_paths.append(path)
        for task, row in read(path)["tasks"].items():
            result["retained_diagnostics"]["cases"].append(
                dict(
                    checkpoint=item["checkpoint"],
                    task=task,
                    **{k: v for k, v in row.items() if k != "updates"},
                )
            )
    result["exact_native_resume"] = resume
    result["welfare_panel"] = panel
    result["heuristic"] = {k: v for k, v in heuristic.items() if k != "groups"}
    result["heuristic"]["groups"] = [
        {k: v for k, v in g.items() if k != "episodes"} for g in heuristic["groups"]
    ]
    result["raw_evidence"] = {str(p): sha(p) for p in raw_paths}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                technical_pass=result["technical_pass"],
                cells=len(canary["cells"]),
                retained_cases=len(result["retained_diagnostics"]["cases"]),
            )
        )
    )


if __name__ == "__main__":
    main()
