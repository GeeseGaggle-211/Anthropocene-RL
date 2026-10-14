"""Verify a selected managed GCP identity and the immutable dataset manifest.

No keys are created or copied. Credential files are SDK selectors; never read or
print their contents. There is no fallback to an unrelated/default identity.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse


def identity_selectors(manifest, alias):
    if manifest.get("version") != 1:
        raise ValueError("Unsupported outbound identity manifest version")
    entries = [c for c in manifest.get("connections", []) if c.get("alias") == alias]
    if len(entries) != 1 or entries[0].get("provider_kind") != "gcp":
        raise ValueError("Selected GCP identity alias is absent or ambiguous")
    c = entries[0]
    fields = {
        "CLOUDSDK_CONFIG": "config_dir",
        "CLOUDSDK_ACTIVE_CONFIG_NAME": "configuration_name",
        "GOOGLE_APPLICATION_CREDENTIALS": "credentials_file",
    }
    if any(not isinstance(c.get(k), str) or not c[k] for k in fields.values()):
        raise ValueError("Selected managed identity lacks required SDK selectors")
    return {env: c[key] for env, key in fields.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--identity-alias", required=True)
    p.add_argument("--project", required=True)
    p.add_argument(
        "--catalog-audit",
        type=Path,
        default=Path("evidence/autonomous-continuation-v4/gcs-catalog-audit.json"),
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = os.environ.get("OIC_MANIFEST_PATH")
    if not manifest_path:
        raise RuntimeError(
            "Attach a GCP outbound identity in the runtime settings; "
            "OIC_MANIFEST_PATH is not configured."
        )
    selectors = identity_selectors(json.loads(Path(manifest_path).read_text()), args.identity_alias)
    os.environ.update(selectors)
    # Call only after cloud environment status reports this alias ready with current observations.
    from google.cloud import storage

    audit = json.loads(args.catalog_audit.read_text())
    expected = audit["gcs_published_manifest"]
    uri = urlparse(expected["uri"])
    if uri.scheme != "gs" or uri.netloc != "anthropocene-rl-datasets-508912":
        raise ValueError("Unexpected dataset bucket")
    client = storage.Client(project=args.project)
    blob = client.bucket(uri.netloc).blob(
        uri.path.lstrip("/"), generation=int(expected["gcs_generation"])
    )
    blob.reload(timeout=30)
    if str(blob.generation) != expected["gcs_generation"] or blob.size != expected["bytes"]:
        raise ValueError("Pinned manifest generation/size mismatch")
    data = blob.download_as_bytes(timeout=30, if_generation_match=int(expected["gcs_generation"]))
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected["anthropocene_sha256_metadata"]:
        raise ValueError("Pinned manifest content digest mismatch")
    result = dict(
        version="gcs-runtime-identity-verification-v1",
        alias=args.identity_alias,
        project=args.project,
        bucket=uri.netloc,
        manifest_uri=expected["uri"],
        generation=str(blob.generation),
        size=blob.size,
        sha256=digest,
        role_required="roles/storage.objectViewer",
        writes=0,
        result="PASS",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
