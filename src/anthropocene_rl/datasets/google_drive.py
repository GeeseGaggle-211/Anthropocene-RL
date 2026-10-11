"""Google Drive implementation of the immutable snapshot remote-store contract."""

from __future__ import annotations

import json
import os
from importlib import import_module
from pathlib import Path
from typing import Any

from anthropocene_rl.datasets.storage import RemoteObject, remote_key_for

_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
_FOLDER_MIME = "application/vnd.google-apps.folder"
_API_RETRIES = 5
_FILE_FIELDS = (
    "id,name,size,md5Checksum,sha256Checksum,version,modifiedTime,parents,appProperties"
)


class GoogleDriveRemoteStore:
    """Store immutable SaveCodec files beneath one explicitly configured folder."""

    backend = "google-drive"

    def __init__(
        self,
        *,
        root_folder_id: str,
        credentials_path: str | Path | None = None,
        service: Any | None = None,
    ) -> None:
        if not root_folder_id:
            raise ValueError("Google Drive root folder ID is required")
        self.root_folder_id = root_folder_id
        self.credentials_path = Path(credentials_path) if credentials_path else None
        self._service_override = service
        self._service_instance: Any | None = None
        self._folder_cache: dict[tuple[str, str], str] = {}

    def find(self, key: str) -> RemoteObject | None:
        segments = _validate_key(key).split("/")
        parent = self.root_folder_id
        for folder_name in segments[:-1]:
            folder = self._find_child(parent, folder_name, folders_only=True)
            if folder is None:
                return None
            parent = str(folder["id"])
        file_metadata = self._find_child(parent, segments[-1], folders_only=False)
        return (
            None
            if file_metadata is None
            else self._to_remote_object(file_metadata, expected_key=key)
        )

    def get(self, file_id: str) -> RemoteObject:
        metadata = (
            self._service()
            .files()
            .get(fileId=file_id, fields=_FILE_FIELDS, supportsAllDrives=True)
            .execute(num_retries=_API_RETRIES)
        )
        key = str((metadata.get("appProperties") or {}).get("anthropocene_remote_key", ""))
        return self._to_remote_object(metadata, expected_key=key)

    def upload(
        self,
        source: Path,
        *,
        key: str,
        content_sha256: str,
        content_bytes: int,
    ) -> RemoteObject:
        MediaFileUpload = import_module("googleapiclient.http").MediaFileUpload

        validated_key = _validate_key(key)
        if validated_key != remote_key_for(content_sha256):
            raise ValueError("Remote key does not match the content SHA-256")
        if source.stat().st_size != content_bytes:
            raise ValueError("Upload source size changed before Drive upload")
        existing = self.find(validated_key)
        if existing is not None:
            return existing
        segments = validated_key.split("/")
        parent = self.root_folder_id
        for folder_name in segments[:-1]:
            parent = self._ensure_folder(parent, folder_name)
        body = {
            "name": segments[-1],
            "parents": [parent],
            "appProperties": {
                "anthropocene_remote_key": validated_key,
                "anthropocene_content_sha256": content_sha256,
                "anthropocene_content_bytes": str(content_bytes),
            },
        }
        media = MediaFileUpload(
            str(source),
            mimetype="application/octet-stream",
            resumable=True,
        )
        metadata = (
            self._service()
            .files()
            .create(
                body=body,
                media_body=media,
                fields=_FILE_FIELDS,
                supportsAllDrives=True,
            )
            .execute(num_retries=_API_RETRIES)
        )
        # A resumable Drive upload can return the metadata-request version from
        # ``files.create`` and then advance the object version when the media is
        # finalized.  Re-read the completed object so the manifest records the
        # settled version rather than an immediately stale one.
        return self.get(str(metadata["id"]))

    def download(self, file_id: str, destination: Path) -> RemoteObject:
        MediaIoBaseDownload = import_module("googleapiclient.http").MediaIoBaseDownload

        request = self._service().files().get_media(
            fileId=file_id,
            supportsAllDrives=True,
        )
        with destination.open("wb") as handle:
            downloader = MediaIoBaseDownload(handle, request)
            done = False
            while not done:
                _, done = downloader.next_chunk(num_retries=_API_RETRIES)
        return self.get(file_id)

    def _service(self) -> Any:
        if self._service_override is not None:
            return self._service_override
        if self._service_instance is not None:
            return self._service_instance
        try:
            import google.auth  # type: ignore[import-untyped]
            from google.auth.transport.requests import Request  # type: ignore[import-untyped]
            from google.oauth2.credentials import (  # type: ignore[import-untyped]
                Credentials as UserCredentials,
            )
            build = import_module("googleapiclient.discovery").build
        except ImportError as error:
            raise RuntimeError(
                "Install the 'google-drive' extra before using the Google Drive "
                "snapshot backend"
            ) from error

        token = os.environ.get("ANTHROPOCENE_GOOGLE_DRIVE_ACCESS_TOKEN")
        if token:
            credentials = UserCredentials(token=token, scopes=[_DRIVE_SCOPE])
        elif self.credentials_path is not None:
            payload = json.loads(self.credentials_path.read_text(encoding="utf-8"))
            if payload.get("type") == "authorized_user":
                credentials = UserCredentials.from_authorized_user_file(
                    str(self.credentials_path),
                    scopes=[_DRIVE_SCOPE],
                )
            else:
                credentials, _ = google.auth.load_credentials_from_file(
                    str(self.credentials_path),
                    scopes=[_DRIVE_SCOPE],
                )
        else:
            try:
                credentials, _ = google.auth.default(scopes=[_DRIVE_SCOPE])
            except Exception as error:
                raise RuntimeError(
                    "No local Google credentials are available. Set "
                    "ANTHROPOCENE_GOOGLE_DRIVE_CREDENTIALS to an authorized-user or "
                    "service-account JSON file, or configure Application Default Credentials."
                ) from error
        if getattr(credentials, "expired", False):
            credentials.refresh(Request())
        self._service_instance = build(
            "drive",
            "v3",
            credentials=credentials,
            cache_discovery=False,
        )
        return self._service_instance

    def _find_child(
        self,
        parent: str,
        name: str,
        *,
        folders_only: bool,
    ) -> dict[str, Any] | None:
        escaped_name = name.replace("'", "\\'")
        mime_clause = (
            f"mimeType = '{_FOLDER_MIME}'"
            if folders_only
            else f"mimeType != '{_FOLDER_MIME}'"
        )
        query = (
            f"name = '{escaped_name}' and '{parent}' in parents and "
            f"{mime_clause} and trashed = false"
        )
        response = (
            self._service()
            .files()
            .list(
                q=query,
                fields=f"files({_FILE_FIELDS})",
                pageSize=2,
                spaces="drive",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute(num_retries=_API_RETRIES)
        )
        files = response.get("files", [])
        if len(files) > 1:
            raise RuntimeError(f"Drive contains duplicate objects named {name!r}")
        return files[0] if files else None

    def _ensure_folder(self, parent: str, name: str) -> str:
        cache_key = (parent, name)
        cached = self._folder_cache.get(cache_key)
        if cached is not None:
            return cached
        existing = self._find_child(parent, name, folders_only=True)
        if existing is None:
            existing = (
                self._service()
                .files()
                .create(
                    body={"name": name, "mimeType": _FOLDER_MIME, "parents": [parent]},
                    fields="id,name",
                    supportsAllDrives=True,
                )
                .execute(num_retries=_API_RETRIES)
            )
        folder_id = str(existing["id"])
        self._folder_cache[cache_key] = folder_id
        return folder_id

    @staticmethod
    def _to_remote_object(
        metadata: dict[str, Any],
        *,
        expected_key: str,
    ) -> RemoteObject:
        properties = metadata.get("appProperties") or {}
        name = str(metadata.get("name", ""))
        filename_digest = (
            name.removesuffix(".snapshot") if name.endswith(".snapshot") else ""
        )
        derived_key = remote_key_for(filename_digest) if len(filename_digest) == 64 else ""
        key = str(
            properties.get("anthropocene_remote_key") or expected_key or derived_key
        )
        content_sha256 = properties.get("anthropocene_content_sha256") or (
            filename_digest if key == derived_key else None
        )
        return RemoteObject(
            file_id=str(metadata["id"]),
            key=key,
            size=int(metadata.get("size", 0)),
            md5_checksum=metadata.get("md5Checksum"),
            sha256_checksum=metadata.get("sha256Checksum"),
            version=str(metadata["version"]) if metadata.get("version") else None,
            modified_time=metadata.get("modifiedTime"),
            content_sha256=content_sha256,
        )


def _validate_key(key: str) -> str:
    parts = key.split("/")
    if len(parts) != 3 or parts[0] != "sha256":
        raise ValueError("Google Drive snapshot key must use sha256/<prefix>/<digest>.snapshot")
    filename = parts[2]
    if not filename.endswith(".snapshot"):
        raise ValueError("Google Drive snapshot object must end in .snapshot")
    digest = filename.removesuffix(".snapshot")
    expected = remote_key_for(digest)
    if key != expected:
        raise ValueError("Google Drive snapshot key has an invalid digest prefix")
    return key
