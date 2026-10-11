from __future__ import annotations

import argparse
import json
from pathlib import Path

from anthropocene_rl.datasets.object_store import DatasetStoreConfig, build_object_store
from anthropocene_rl.datasets.run_artifacts import publish_run_artifacts, stage_run_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish one immutable RL run bundle")
    parser.add_argument("--config", type=Path, default=Path("configs/storage/gcs-v1.yaml"))
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--include", action="append", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--captured-at", required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--starting-learner-sha", required=True)
    parser.add_argument("--execution-learner-sha", required=True)
    parser.add_argument("--native-simulator-sha", required=True)
    parser.add_argument("--dataset-manifest-uri", required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    args = parser.parse_args()

    config = DatasetStoreConfig.load(args.config)
    manifest_path = stage_run_artifacts(
        repository_root=args.repository_root,
        includes=tuple(args.include),
        staging_root=config.staging_root,
        campaign_id=args.campaign_id,
        run_id=args.run_id,
        captured_at=args.captured_at,
        source_branch=args.source_branch,
        starting_learner_sha=args.starting_learner_sha,
        execution_learner_sha=args.execution_learner_sha,
        native_simulator_sha=args.native_simulator_sha,
        dataset_manifest_uri=args.dataset_manifest_uri,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
    )
    result = publish_run_artifacts(
        manifest_path,
        config=config,
        store=build_object_store(config),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
