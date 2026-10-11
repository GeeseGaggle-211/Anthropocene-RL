"""Content-addressed storage for authoritative SaveCodec payloads."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BlobReference:
    sha256: str
    size: int
    relative_path: str
    absolute_path: Path


def sha256_file(path: str | Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


class ContentAddressedBlobStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, sha256: str) -> Path:
        if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
            raise ValueError("Expected a lowercase SHA-256 digest")
        return self.root / "sha256" / sha256[:2] / f"{sha256}.snapshot"

    def put_file(self, source: str | Path) -> BlobReference:
        source_path = Path(source)
        digest, size = sha256_file(source_path)
        target = self.path_for(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing_digest, existing_size = sha256_file(target)
            if existing_digest != digest or existing_size != size:
                raise ValueError("Existing content-addressed blob failed integrity validation")
        else:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                with source_path.open("rb") as source_handle:
                    shutil.copyfileobj(source_handle, temporary)
            try:
                copied_digest, copied_size = sha256_file(temporary_path)
                if copied_digest != digest or copied_size != size:
                    raise ValueError("Snapshot changed while entering content-addressed storage")
                os.replace(temporary_path, target)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
        relative = target.relative_to(self.root).as_posix()
        return BlobReference(digest, size, relative, target)

    def verify(self, reference: BlobReference) -> bool:
        if not reference.absolute_path.is_file():
            return False
        digest, size = sha256_file(reference.absolute_path)
        return digest == reference.sha256 and size == reference.size
