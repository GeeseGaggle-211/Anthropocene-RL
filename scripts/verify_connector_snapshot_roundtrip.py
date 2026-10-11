#!/usr/bin/env python3
"""Exercise prune and transparent loader restore from a verified connector download."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from anthropocene_rl.datasets.blobs import sha256_file
from anthropocene_rl.datasets.models import SnapshotBankManifest, SnapshotRecord
from anthropocene_rl.datasets.storage import (
    RemoteObject,
    SnapshotStoreConfig,
    SnapshotStoreManager,
    md5_file,
    remote_key_for,
)
from anthropocene_rl.training.overfit import load_stage0_selection


class ConnectorReceiptStore:
    backend = "google-drive"

    def __init__(self, record: SnapshotRecord, source: Path, file_id: str) -> None:
        digest = record.content_sha256
        self.source = source
        self.remote = RemoteObject(
            file_id=file_id,
            key=remote_key_for(digest),
            size=source.stat().st_size,
            md5_checksum=md5_file(source),
            version=record.remote_storage.version if record.remote_storage else None,
            modified_time=(
                record.remote_storage.uploaded_at if record.remote_storage else None
            ),
            content_sha256=digest,
        )

    def find(self, key: str) -> RemoteObject | None:
        return self.remote if key == self.remote.key else None

    def get(self, file_id: str) -> RemoteObject:
        if file_id != self.remote.file_id:
            raise FileNotFoundError(file_id)
        return self.remote

    def upload(
        self,
        source: Path,
        *,
        key: str,
        content_sha256: str,
        content_bytes: int,
    ) -> RemoteObject:
        raise AssertionError("The connector receipt store is read-only")

    def download(self, file_id: str, destination: Path) -> RemoteObject:
        self.get(file_id)
        shutil.copyfile(self.source, destination)
        return self.remote


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--connector-download", type=Path, required=True)
    parser.add_argument("--drive-file-id", required=True)
    parser.add_argument(
        "--snapshot-id",
        default="f01-21001-f01-young-low-q-d0-f8067a545ec2",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/manifests/elder-v2-stage0-v1.json"),
    )
    parser.add_argument("--blob-root", type=Path, default=Path("datasets/blobs"))
    parser.add_argument(
        "--normalizer",
        type=Path,
        default=Path(
            "evidence/stage0-snapshot-bank/provisional-train-normalization.json"
        ),
    )
    args = parser.parse_args()

    manifest = SnapshotBankManifest.load(args.manifest)
    matches = [record for record in manifest.records if record.snapshot_id == args.snapshot_id]
    if len(matches) != 1:
        raise ValueError("Expected exactly one selected snapshot")
    record = matches[0]
    if record.remote_storage is None:
        raise ValueError("Selected record has no committed remote identity")
    if record.remote_storage.file_id != args.drive_file_id:
        raise ValueError("Drive file ID differs from the committed remote identity")
    source_digest = sha256_file(args.connector_download)
    if source_digest != (record.content_sha256, record.content_bytes):
        raise ValueError("Connector download does not match the snapshot manifest")
    remote = ConnectorReceiptStore(record, args.connector_download, args.drive_file_id)
    config = SnapshotStoreConfig.load("configs/storage/google-drive-v1.yaml")
    manager = SnapshotStoreManager(
        manifest_path=args.manifest,
        blob_root=args.blob_root,
        config=config,
        remote=remote,
    )
    local_path = manager.cache.path_for(record)
    if not local_path.is_file():
        raise FileNotFoundError("Local snapshot must exist before the eviction proof")
    total = sum(
        path.stat().st_size
        for path in args.blob_root.glob("sha256/*/*.snapshot")
        if path.is_file()
    )
    prune = manager.prune(max_local_bytes=total - record.content_bytes)
    if local_path.exists() or record.snapshot_id not in prune["evicted_snapshot_ids"]:
        raise RuntimeError("The verified snapshot was not evicted")
    selection = load_stage0_selection(
        manifest_path=args.manifest,
        blob_root=args.blob_root,
        normalizer_path=args.normalizer,
        snapshot_id=record.snapshot_id,
        remote_store=remote,
    )
    restored_digest = sha256_file(selection.snapshot_path)
    if restored_digest != (record.content_sha256, record.content_bytes):
        raise RuntimeError("Transparent loader restore failed integrity verification")
    result = {
        "status": "PASS",
        "transport": "authenticated-google-drive-connector-download-receipt",
        "drive_file_id": args.drive_file_id,
        "remote_key": remote.remote.key,
        "snapshot_id": record.snapshot_id,
        "content_sha256": record.content_sha256,
        "content_bytes": record.content_bytes,
        "connector_download_md5": remote.remote.md5_checksum,
        "prune": prune,
        "loader_restored": True,
        "restored_path": selection.snapshot_path.as_posix(),
        "manifest_identity_sha256": manifest.sha256(),
        "limitation": (
            "The authenticated app connector supplied the Drive download because its OAuth "
            "grant is not exposed as local CLI Application Default Credentials."
        ),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
