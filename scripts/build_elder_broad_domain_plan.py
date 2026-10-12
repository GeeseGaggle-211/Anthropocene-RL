"""Write the deterministic 50-decision broad-domain cohort plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anthropocene_rl.datasets.broad_domain_plan import save_elder_broad_domain_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/elder-broad-domain-50d-v1.yaml"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("evidence/elder-broad-domain-50d-v1/plan-summary.json"),
    )
    args = parser.parse_args()
    summary = save_elder_broad_domain_plan(
        config_path=args.config,
        summary_path=args.summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
