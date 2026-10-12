#!/usr/bin/env python3
"""Write the fresh, evidence-weighted 50-decision scale-out curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anthropocene_rl.datasets.broad_domain_plan import (
    save_elder_scaleout_curriculum_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/elder-broad-domain-50d-scaleout-v1.yaml"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("evidence/elder-broad-domain-50d-scaleout-v1/plan-summary.json"),
    )
    args = parser.parse_args()
    summary = save_elder_scaleout_curriculum_plan(
        config_path=args.config,
        summary_path=args.summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
