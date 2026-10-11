# Cultivation readiness and snapshot-storage evidence

Recorded 2026-10-11 on branch `feature/cultivation-pilot-readiness`.

Storage implementation and clean-checkout rerun commit:
`ebeb54fbae2730934dd976d0749e426ac5d7c8a7`.

## Frozen identities and readiness result

- Preserved readiness learner commit: `33a88006ba9f55f80c47a69b91331009973a75ac`
- Frozen native simulator commit: `5559961164d6f9da8e6d20c1dc563ecec3eb04b7`
- Stage-0 manifest identity: `1a6c524bc7e7c443133b56324bd3867682eac66749a3b81476353c93f134e084`
- Readiness aggregate deterministic SHA-256: `2f8daa9e203d51f823d74230aa1c4ba4fb0e281c0ebca02c46fcf146fd825929`
- Readiness aggregate file SHA-256: `5501d5ca6a91dc428a9e694ba10d63773fb239ac277149bbeb623c4d723660ee`

The preserved F01/F03/F13 by seed 1701/1901/2101 matrix is a genuine 9/9
PASS. `cultivation-readiness-verification.json` independently re-hashes all
nine diagnostics and checkpoints, checks their embedded learner/native,
manifest, snapshot, and bridge identities, and confirms every per-run and
aggregate gate. Actor motion replicates in at least two of three seeds for each
snapshot. The evidence recommends retaining the separate feed-forward actor
and critic; it does not justify a GRU, Transformer, or privileged critic.

The complete matrix was then rerun from a clean checkout at implementation
commit `ebeb54fbae2730934dd976d0749e426ac5d7c8a7`. It passed 9/9 again in
217.75 seconds with aggregate deterministic SHA-256
`2bc2b73866f4fc110bd33550559f61fa1029beebecf05b07b312dfec4e5e4d73`
and aggregate file SHA-256
`5014420bccbbb56bb79e212fac5cb5f453197bd152cdcbd0fcac64f5df4c10e9`.
`cultivation-readiness-post-storage-verification.json` independently verifies
that rerun and its nine new checkpoints. The aggregate and all nine small
per-run diagnostics are committed under `post-storage-readiness/per-run/`; the nine
large checkpoint files remain preserved locally under
`checkpoints/cultivation-readiness-ebeb54f/` and gitignored as required.

## Google Drive round trip

- Folder: `Anthropocene / RL Snapshot Banks / Elder v2 / Stage0 and Pilot Banks`
- Folder ID: `1BtKChDAhEpJOWdeEiJyfnowuuN7y_GSN`
- Object key: `sha256/5e/5ef2fc720414671a7d5b9fe66ccb7537b03c4e37264bf38f54083c6d4639f86f.snapshot`
- Drive file ID: `1DTwsgDCI3R8qva7jpTMRLRqA44BtCNZo`
- Bytes: `209704`
- SHA-256: `5ef2fc720414671a7d5b9fe66ccb7537b03c4e37264bf38f54083c6d4639f86f`
- Drive/local MD5: `c048bc1f040de9e17591e24ed3504721`

The authenticated Drive connector uploaded the object and downloaded a fresh
copy. The download matched byte count, SHA-256, and MD5. The cache proof then
evicted the local object, reclaimed 209704 bytes, and restored it through the
normal training loader without another episode-level transfer. See
`storage-roundtrip.json`.

The connector OAuth grant is not available to local processes. The implemented
CLI therefore still needs Application Default Credentials, an authorized-user
credential file, or a service account that has been granted access to the
folder. At this evidence point, 1 of 28 Stage-0 blobs has committed remote
metadata; the remaining 27 must be uploaded with `snapshot-store sync` before
the local bank is fully evictable. This is an operational credential/sync gate,
not an architecture blocker.

## Disk footprint

`disk-footprint.json` records 6,035,379 bytes (5.756 MiB) of Stage-0 blobs,
154,348,898 bytes (147.199 MiB) of checkpoints, 1,806,597 bytes (1.723 MiB) of
artifacts, the learner C# build caches, and the frozen simulator checkout/cache.
After full Drive sync, all 6,035,379 Stage-0 blob bytes are safely reclaimable.
The 5 GiB LRU ceiling bounds future local snapshot storage; checkpoints remain
outside this blob-store policy. At the current Stage-0 mean blob size, the
unlaunched 240-snapshot plan projects to roughly 49.335 MiB of blob bytes.

## Commands and observed status

```text
ruff check .
# PASS

pytest -q --basetemp=<short-workspace-temp>
# 51 passed

anthropocene-rl check-manifest
# PASS; frozen native/schema contract

anthropocene-rl ppo-smoke
# PASS; finite update and active actor gradients

anthropocene-rl audit-snapshot-bank --manifest datasets/manifests/elder-v2-stage0-v1.json --blob-root datasets/blobs
# PASS; 28 records; 11 TRAIN / 10 DEVELOPMENT / 7 HELDOUT

anthropocene-rl check-simulator --root <frozen-simulator> --dotnet <unity-dotnet>
anthropocene-rl adapter-smoke --dotnet <unity-dotnet>
# PASS; observation=2, action=1, environment=2, critic-sidecar=1, width=512

python scripts/audit_cultivation_readiness.py --output evidence/cultivation-storage-readiness/cultivation-readiness-verification.json
# PASS; 9/9 diagnostics and checkpoints independently verified

anthropocene-rl cultivation-readiness --dotnet <unity-dotnet> --run-directory artifacts/cultivation-readiness-ebeb54f-runs --checkpoint-directory checkpoints/cultivation-readiness-ebeb54f --diagnostics artifacts/cultivation-readiness-ebeb54f.json
# PASS; clean implementation commit; 9/9 in 217.75 seconds

python scripts/audit_cultivation_readiness.py --aggregate artifacts/cultivation-readiness-ebeb54f.json --run-directory artifacts/cultivation-readiness-ebeb54f-runs --checkpoint-directory checkpoints/cultivation-readiness-ebeb54f --learner-sha ebeb54fbae2730934dd976d0749e426ac5d7c8a7 --output evidence/cultivation-storage-readiness/cultivation-readiness-post-storage-verification.json
# PASS; new aggregate and all nine checkpoints independently re-hashed

python scripts/build_cultivation_pilot_plan.py
# PLAN_READY_NOT_GENERATED; 240 requested snapshots / 200 worlds

anthropocene-rl snapshot-store status --offline
# PARTIAL; 28/28 valid locally, 1/28 with remote metadata, 0 unrecoverable
```

The full 200-500-state cultivation campaign was deliberately not launched.
