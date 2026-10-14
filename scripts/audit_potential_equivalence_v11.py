"""Cached arithmetic only: bounded potential equivalence on actual V10 PPO rows."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from run_autonomous_learning_v10 import sha

from anthropocene_rl.autonomous_v6.reward_objectives import Objective
from anthropocene_rl.training.returns import generalized_advantage_estimate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    objective = Objective("potential")
    result = dict(
        version="bounded-potential-actual-rows-v11",
        model_calls=0,
        native_resolves=0,
        optimizer_steps=0,
        rows=0,
        objective=objective.wire(),
        cells=[],
    )
    for folder in ["autonomous-learning-first1024-v10", "autonomous-learning-final4096-v10"]:
        summary = Path("runs") / folder / "summary.json"
        stage = json.loads(summary.read_text())
        for cell in stage["cells"]:
            trace_path = summary.parent / f"seed{cell['seed']}-training-traces.json"
            traces = [r for e in json.loads(trace_path.read_text()) for r in e["trace"]]
            offset = 0
            for audit in cell["rollout_audits"]:
                if sha(audit["path"]) != audit["sha256"]:
                    raise ValueError("Changed actual rollout audit")
                data = torch.load(audit["path"], map_location="cpu", weights_only=False)
                rows = data["rows"]
                receipts = traces[offset : offset + len(rows)]
                offset += len(rows)
                shaped = []
                for row, receipt in zip(rows, receipts, strict=True):
                    if (
                        row["elapsed"] != receipt["elapsed"]
                        or row["terminal"] != receipt["terminal"]
                    ):
                        raise ValueError("Trace/actual-row alignment differs")
                    if not np.array_equal(
                        row["action"]["workforce"][0].numpy(), receipt["requested_workforce"]
                    ):
                        raise ValueError("Trace/actual worker request differs")
                    gamma = np.exp(-row["elapsed"] / 400)
                    now = objective.potential(receipt["population_start"])
                    following = objective.potential(receipt["population_end"], row["terminal"])
                    shaped.append(
                        dict(
                            row,
                            reward=row["reward"] + gamma * following - now,
                            value=row["value"] - now,
                            next_value=row["next_value"] - following,
                        )
                    )

                def gae(batch, lambda_ref=data["config"]["lambda_ref"]):
                    return generalized_advantage_estimate(
                        np.array([r["reward"] for r in batch]),
                        np.array([r["value"] for r in batch]),
                        np.array([r["next_value"] for r in batch]),
                        np.array([r["elapsed"] for r in batch]),
                        np.array([r["terminal"] for r in batch]),
                        np.array([r["truncated"] for r in batch]),
                        horizon_years=400,
                        lambda_ref=lambda_ref,
                    )

                plain_adv, plain_returns = gae(rows)
                shaped_adv, shaped_returns = gae(shaped)
                cached_adv = np.array([r["advantage"] for r in rows])
                phi = np.array([objective.potential(r["population_start"]) for r in receipts])
                errors = dict(
                    cached_gae=float(np.max(np.abs(plain_adv - cached_adv))),
                    potential_advantage=float(np.max(np.abs(plain_adv - shaped_adv))),
                    potential_return=float(np.max(np.abs(plain_returns - phi - shaped_returns))),
                )
                if max(errors.values()) > 1e-10:
                    raise ValueError("Potential/cached GAE equivalence fails")
                result["cells"].append(
                    dict(
                        stage=folder,
                        seed=cell["seed"],
                        rows=len(rows),
                        audit_sha256=audit["sha256"],
                        errors=errors,
                    )
                )
                result["rows"] += len(rows)
            if offset != len(traces):
                raise ValueError("Incomplete actual-row coverage")
    result["status"] = "PASS"
    result["maximum_errors"] = {
        key: max(c["errors"][key] for c in result["cells"]) for key in result["cells"][0]["errors"]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["status", "rows", "maximum_errors"]}))


if __name__ == "__main__":
    main()
