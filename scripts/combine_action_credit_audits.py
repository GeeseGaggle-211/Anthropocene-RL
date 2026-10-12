from __future__ import annotations

import argparse
import json
from pathlib import Path

from anthropocene_rl.training.action_credit_audit import combine_action_credit_audits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine isolated per-seed PPO action-credit audits"
    )
    parser.add_argument(
        "--input-directory",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument("--evidence-directory", type=Path, required=True)
    parser.add_argument("--expected-seeds", default="1701,1901,2101")
    args = parser.parse_args()
    result = combine_action_credit_audits(
        input_directories=tuple(args.input_directory),
        evidence_directory=args.evidence_directory,
        expected_seeds=tuple(int(value) for value in args.expected_seeds.split(",")),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "evidence": str(args.evidence_directory / "summary.json"),
                "evidence_sha256": result["evidence_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
