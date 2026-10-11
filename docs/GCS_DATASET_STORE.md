# GCS dataset store

## Architecture

The simulation still emits authoritative, individually verifiable `SaveCodec`
snapshots. The training repository groups complete trajectory lineages into
deterministic immutable tar shards, stages them locally, uploads finalized
shards with bounded concurrency, and publishes `manifest.json` only after every
shard verifies remotely.

```text
native generation -> local content store -> 64 MiB tar shards
                  -> bounded resumable uploads -> immutable manifest
                  -> on-demand shard cache -> Elder Reset/Step workers
```

The `.snapshot` members already use gzip. Measurements on 1,014 current
snapshots found 192,625/213,673/232,917/262,989 byte min/median/P95/max sizes
and 217,716,578 bytes total. A second gzip pass grew the bytes by about 4%, so
shards are uncompressed tar containers of already-compressed members. The 64
MiB default closes roughly every 300 states, allows generation and upload to
overlap, preserves trajectory boundaries where possible, and yields useful
parallelism for the present roughly 208 MiB bank. Increase the setting toward
256 MiB for much larger campaigns after remeasurement.

Each run uses:

```text
gs://<bucket>/datasets/<dataset-id>/<run-id>/
  manifest.json
gs://<bucket>/objects/sha256/<prefix>/<sha256>.tar
gs://<bucket>/runs/<campaign-id>/<run-id>/
  manifest.json
gs://<bucket>/run-objects/sha256/<prefix>/<sha256>.tar
```

Shards live in a global content-addressed pool keyed by SHA-256, so identical
bytes are reused across runs. Run IDs and object names are immutable. Uploads
use a zero-generation
precondition, CRC32C transport checking, application SHA-256 metadata, bounded
workers, exponential retry, and a local `upload-state.json` recovery journal.
A pre-existing object is accepted only when its size and SHA-256 match; a
different object at the same name is a hard collision. Local staged files are
never deleted by publication.

## Cloud configuration

- project: `anthropocene-508912` (Google Cloud free trial active)
- bucket: `anthropocene-rl-datasets-508912` (created and integration-tested)
- region: `asia-northeast1` (Tokyo)
- storage class: `STANDARD`
- flat object namespace; no multi-region replication or Autoclass
- uniform bucket-level access and public-access prevention are enforced
- the default seven-day soft-delete policy is active

Tokyo is selected because current Cloud TPU v6e capacity is available in
`asia-northeast1-b`. Co-locating future TPU workers and the bucket avoids
cross-region training reads. Standard storage has no minimum retention period
or retrieval charge, which fits actively iterated datasets and caches.

At the current roughly $0.02/GiB-month single-region Standard rate, the measured
217.7 MB snapshot bank is under half a cent per month before free allowances.
Class A writes are roughly $0.005 per 1,000 operations and Class B reads roughly
$0.0004 per 1,000. Upload ingress is free. Reads to compute in the same region
are normally free; internet downloads and cross-region training can incur data
transfer charges. The free trial does not automatically convert to paid billing;
confirm the live SKU and currency before manually upgrading the account.

## Authentication and bucket setup

Install the Google Cloud CLI and the optional project client, then authenticate
interactively. Never commit credential JSON.

```bash
uv sync --extra gcs --extra dev
gcloud auth login
gcloud auth application-default login
gcloud config set project anthropocene-508912
gcloud services enable storage.googleapis.com
gcloud storage buckets create gs://anthropocene-rl-datasets-508912 \
  --project=anthropocene-508912 \
  --location=asia-northeast1 \
  --default-storage-class=STANDARD \
  --uniform-bucket-level-access
gcloud storage buckets update gs://anthropocene-rl-datasets-508912 \
  --public-access-prevention
```

The live bucket was reviewed before creation. Workload Identity or a narrowly
scoped service account is preferable for unattended remote workers; local
development uses user Application Default Credentials. Never copy a personal
ADC file onto a worker image.

## Pack, publish, resume, list, and read

Pack a current snapshot bank without any cloud dependency:

```bash
uv run anthropocene-rl dataset pack \
  --snapshot-manifest datasets/manifests/elder-v2-stage0-v1.json \
  --blob-root datasets/blobs \
  --dataset-id elder-v2-stage0 \
  --run-id 2026-10-11-stage0 \
  --anthropocene-branch feature/elder-rl-scaffold \
  --training-commit "$(git rev-parse HEAD)"
```

Publish, or rerun the same command after an interruption. Verified shards are
deduplicated from the journal and remote metadata; incomplete shards retry.

```bash
uv run anthropocene-rl dataset publish \
  --manifest datasets/staging/elder-v2-stage0/2026-10-11-stage0/manifest.json \
  --config configs/storage/gcs-v1.yaml

uv run anthropocene-rl dataset list \
  --dataset-id elder-v2-stage0 \
  --config configs/storage/gcs-v1.yaml
```

Training code can construct `DatasetReader` from a local manifest or a
`gs://.../manifest.json` URI. It downloads a shard on first access, verifies the
manifest SHA-256, and caches the shard by immutable checksum. Repeated epochs
are cache hits. The CLI exposes the same path:

```bash
uv run anthropocene-rl dataset fetch-record \
  gs://anthropocene-rl-datasets-508912/datasets/elder-v2-stage0/2026-10-11-stage0/manifest.json \
  <snapshot-id>

uv run anthropocene-rl dataset verify \
  gs://anthropocene-rl-datasets-508912/datasets/elder-v2-stage0/2026-10-11-stage0/manifest.json
```

For local-only development, replace the config with
`configs/storage/local-v1.yaml`; packing, publication semantics, discovery,
cache validation, and readers remain the same without GCS credentials.

Training-run evidence and checkpoints are published as one deterministic tar
bundle plus a manifest created last. The command accepts explicit repository
paths so secrets and unrelated workspace files cannot enter the bundle:

```bash
uv run python scripts/publish_run_artifacts.py \
  --campaign-id <campaign-id> \
  --run-id <immutable-run-id> \
  --captured-at <fixed-utc-time> \
  --source-branch <branch> \
  --starting-learner-sha <sha> \
  --execution-learner-sha <sha> \
  --native-simulator-sha <sha> \
  --dataset-manifest-uri <gs-uri> \
  --dataset-manifest-sha256 <sha256> \
  --include evidence/<campaign-id> \
  --include checkpoints/<campaign-id>
```

## Measured local and GCS benchmark

The checked-in stage-0 bank (28 records, 6,035,379 payload bytes) packed into
one 6,072,320-byte shard in 0.196 seconds of library time. Local publication
completed in 0.720 seconds; an immediate restart uploaded zero objects and
deduplicated the existing shard. Full readback verified all 28 records with one
cache miss and 27 cache hits. The 6,035,379 compressed payload bytes expand to
17,608,427 bytes (34.28% of the decompressed size). Sharding intentionally does
not recompress these project-native gzip members.

The real GCS run `gcs-smoke-20261011` packed in 0.213 seconds, published one
6,072,320-byte shard plus a 14,851-byte manifest in 7.807 seconds wall time,
and wrote 6,087,171 total bytes. This is 0.744 MiB/s effective end-to-end from
the current connection, including process/API/manifest overhead. A restart took
2.386 seconds, uploaded zero shards, and deduplicated the existing shard.

Cold-cache verification downloaded the shard and verified all 28 records in
24.619 seconds (0.235 MiB/s effective, one miss then 27 intra-shard hits). The
warm-cache repeat took 2.929 seconds with 28 hits and zero misses. The configured
uploader has four workers, but this one-shard fixture exercised one active
upload; larger banks overlap up to four finalized shard uploads with continued
generation. The machine-readable results are retained at
`evidence/gcs-dataset-store/benchmark.json`.

The cultivation pilot published 300 snapshots in one 64,706,560-byte shard and
verified all records after cold recovery. Its complete 129-file training bundle
is 224,614,400 bytes. Provider byte count, generation, MD5, CRC32C, custom
SHA-256 metadata, and post-download SHA-256 all agree; details are recorded in
`evidence/cultivation-ppo-pilot-v1/gcs-storage-audit.json`.

## Versioning and recovery

`anthropocene-dataset-v1` is rejected explicitly when unknown rather than being
silently coerced. It records simulator/training Git commits, source snapshot
manifest identity, configuration hashes, scenario/world/seed metadata,
simulation/trajectory/state/sample counts, serialization and compression,
every shard size/checksum/URI, and staged versus complete publication status.

The authoritative remote manifest is created last. A run without that manifest
is incomplete and must not be consumed. On restart, publish reconciles every
staged shard and never overwrites differing bytes. A corrupt cached shard is
discarded and fetched again; corrupt remote bytes fail checksum validation.
The legacy Google Drive snapshot store remains readable for historical
manifests, but new snapshot banks and run outputs use GCS.

## Troubleshooting

- `DefaultCredentialsError`: run `gcloud auth application-default login`.
- `403` on upload: confirm project billing, Storage API, and bucket IAM.
- immutable object collision: choose a new run ID or investigate the differing
  staged bytes; never delete/overwrite the published object to force progress.
- failed upload: retain staging and rerun `dataset publish`; inspect
  `upload-state.json` for the last verified object and error.
- cache checksum failure: rerun the read; the reader removes the corrupt local
  shard before download.
