from __future__ import annotations

import argparse
import json
from pathlib import Path

from anthropocene_rl.training.headroom_audit import run_constant_grid_headroom_audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the predeclared TRAIN+DEVELOPMENT constant-grid headroom audit"
    )
    parser.add_argument("--evidence-directory", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "datasets/manifests/elder-v2-cultivation-pilot-normalization-v3-scope-safe.json"
        ),
    )
    parser.add_argument("--blob-root", type=Path, default=Path("datasets/blobs"))
    parser.add_argument(
        "--normalizer",
        type=Path,
        default=Path(
            "evidence/cultivation-pilot-normalization-v3-scope-safe/generation/"
            "final-train-normalization.json"
        ),
    )
    parser.add_argument(
        "--snapshot-store-config",
        type=Path,
        default=Path("configs/storage/google-drive-v1.yaml"),
    )
    parser.add_argument(
        "--dataset-manifest",
        default=(
            "gs://anthropocene-rl-datasets-508912/datasets/"
            "elder-v2-cultivation-pilot-normalization-v3-scope-safe/"
            "2026-10-12-manifest-1cfabbc6/manifest.json"
        ),
    )
    parser.add_argument(
        "--dataset-store-config",
        type=Path,
        default=Path("configs/storage/gcs-v1.yaml"),
    )
    parser.add_argument(
        "--bridge-dll",
        type=Path,
        default=Path("integration/bin/Release/net8.0/Anthropocene.RLBridge.dll"),
    )
    parser.add_argument("--dotnet", default="dotnet")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--selected-policy-evidence",
        action="append",
        type=Path,
        default=[],
    )
    args = parser.parse_args()

    def progress(payload: dict[str, object]) -> None:
        print(json.dumps({"event": "progress", **payload}, sort_keys=True), flush=True)

    result = run_constant_grid_headroom_audit(
        evidence_directory=args.evidence_directory,
        bridge_dll=args.bridge_dll,
        dotnet=args.dotnet,
        manifest_path=args.manifest,
        blob_root=args.blob_root,
        normalizer_path=args.normalizer,
        snapshot_store_config=args.snapshot_store_config,
        dataset_manifest=args.dataset_manifest,
        dataset_store_config=args.dataset_store_config,
        repository_root=args.repository_root,
        selected_policy_evidence=tuple(args.selected_policy_evidence),
        progress=progress,
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
    return 2 if result["status"] == "STOP_INTEGRITY_FAILURE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
