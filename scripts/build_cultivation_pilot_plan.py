#!/usr/bin/env python3
"""Materialize the deterministic cultivation-only pilot generation plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anthropocene_rl.datasets.pilot_plan import save_cultivation_pilot_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/cultivation-pilot-v1.yaml"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("evidence/cultivation-pilot-plan/plan-summary.json"),
    )
    parser.add_argument(
        "--variant",
        choices=("v2", "normalization-v3"),
        default="v2",
    )
    args = parser.parse_args()
    summary = save_cultivation_pilot_plan(
        config_path=args.config,
        summary_path=args.summary,
        variant=args.variant,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
