"""Write a compact, Git-suitable summary of normalization diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _feature_summary(feature: dict[str, Any]) -> dict[str, Any]:
    slices = feature["slices"]
    return {
        "index": feature["index"],
        "name": feature["name"],
        "kind": feature["kind"],
        "fitted_c_i": feature["fitted_c_i"],
        "scale_fit_mode": feature["scale_fit_mode"],
        "partition_saturation": {
            name: slices[name]["saturation_rate"]
            for name in ("train-fit", "development", "heldout")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    compact = {
        "status": payload["status"],
        "bank_id": payload["bank_id"],
        "normalization_version": payload["normalization_version"],
        "partition_manifest_sha256": payload["partition_manifest_sha256"],
        "fit_partition": payload["fit_partition"],
        "fit_observation_count": payload["fit_observation_count"],
        "fit_source_mix": payload["fit_source_mix"],
        "heldout_or_development_used_for_fit": payload[
            "heldout_or_development_used_for_fit"
        ],
        "saturation_gate_pass": payload["saturation_gate_pass"],
        "saturation_gate_maximum": payload["saturation_gate_maximum"],
        "saturation_max_by_slice": payload["saturation_max_by_slice"],
        "sparse_causal_support_pass": payload["sparse_causal_support_pass"],
        "sparse_causal_support": payload["sparse_causal_support"],
        "features": [_feature_summary(feature) for feature in payload["features"]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(compact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
