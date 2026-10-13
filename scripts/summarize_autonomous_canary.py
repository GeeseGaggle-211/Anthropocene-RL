"""Compact measured health evidence from an immutable canary directory."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = {}
    for path in sorted(args.run.glob("seed-*-health.json")):
        seed = path.name.split("-")[1]
        frames = json.loads(path.read_text())
        updates = json.loads((args.run / f"seed-{seed}-updates.json").read_text())
        fields = [
            "actor_step_l2",
            "critic_step_l2",
            "actor_first_layer_grad",
            "critic_first_layer_grad",
            "critic_residual_mse",
            "maximum_kl",
        ]
        report[seed] = dict(
            boundaries=[
                dict(
                    transitions=f["additional_transitions"],
                    passed=f["recovery"]["passed"],
                    welfare=f["evaluation"]["welfare"],
                    actor_saturation=[r["saturated_fraction"] for r in f["health"]["actor_layers"]],
                    critic_saturation=[
                        r["saturated_fraction"] for r in f["health"]["critic_layers"]
                    ],
                    actor_derivative=[r["mean_derivative"] for r in f["health"]["actor_layers"]],
                    critic_derivative=[r["mean_derivative"] for r in f["health"]["critic_layers"]],
                    active_clamp_fraction=f["health"]["active_clamp_fraction"],
                    critic_probe_mse=f["recovery"]["critic_residual_mse"],
                    workforce_probe_separation_gain=(
                        f["recovery"]["separation_after"] - f["recovery"]["separation_before"]
                    ),
                )
                for f in frames
            ],
            updates=len(updates),
            optimizer={
                k: dict(
                    minimum=min(r[k] for r in updates),
                    median=float(np.median([r[k] for r in updates])),
                    maximum=max(r[k] for r in updates),
                )
                for k in fields
            },
            actor_minibatch_execution_fraction=(
                sum(r["actor_updates"] for r in updates) / sum(r["minibatches"] for r in updates)
            ),
            final_heads=frames[-1]["health"]["categorical"],
        )
    args.output.write_text(
        json.dumps(
            dict(
                seeds=report,
                artifacts={
                    p.name: dict(
                        bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()
                    )
                    for p in sorted(args.run.iterdir())
                    if p.is_file()
                },
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
