"""Compact derived report; raw ablation artifacts remain immutable."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path("runs/autonomous-layernorm-ablation-v3")
    output = Path("evidence/autonomous-layernorm-ablation-v3")
    output.mkdir(exist_ok=True)
    results = {}
    for path in sorted(root.glob("*.json")):
        r = json.loads(path.read_text())
        if "tasks" in r:
            r = {
                **r,
                "tasks": {
                    name: {
                        **{k: v for k, v in task.items() if k != "updates"},
                        "final_enable_mechanism": task.get("updates", [{}])[-1].get(
                            "enable_mechanism"
                        ),
                        "active_draws": sum(s["active_draws"] for s in task.get("updates", [])),
                        "active_positive_label": sum(
                            s["active_positive_label"] for s in task.get("updates", [])
                        ),
                        "active_negative_label": sum(
                            s["active_negative_label"] for s in task.get("updates", [])
                        ),
                    }
                    for name, task in r["tasks"].items()
                },
            }
        if path.name == "native-entry.json":
            r = {
                **r,
                "results": {
                    k: {n: v for n, v in x.items() if n != "episodes"}
                    for k, x in r["results"].items()
                },
            }
        results[path.name] = dict(
            raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), result=r
        )
    total = sum(t["draws"] for x in results.values() for t in x["result"].get("tasks", {}).values())
    (output / "summary.json").write_text(
        json.dumps(
            dict(
                results=results,
                ablation_draws=total,
                historical_draws=86016,
                total_discarded_draws=86016 + total,
            ),
            indent=2,
        )
        + "\n"
    )
    print("Discarded diagnostic draws:", 86016 + total)


if __name__ == "__main__":
    main()
