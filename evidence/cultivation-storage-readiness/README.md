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

Local authorized-user Application Default Credentials were subsequently
configured for the private test client. The learner CLI uploaded the remaining
27 Stage-0 blobs, recorded all 28 remote identities atomically, and then
live-verified all 28 by Drive file ID, content-addressed key, byte count,
SHA-256 app property, Drive MD5, and Drive's server-computed SHA-256. The first
live audit exposed that Drive's resumable upload returns metadata version 1
before media finalization advances the object to version 2. The backend now
re-reads the completed object, and
`sync` may refresh only this provider version when verified local bytes, SHA,
size, MD5, key, and file ID are unchanged. Remote-only audit and prune remain
strict about unexpected version changes. The regression is covered by the
snapshot-store tests.

The local CLI round-trip then evicted exactly
`f01-21002-f01-young-viable-d1-87f7b59c91fe` (211883 bytes) only after its
remote identity re-verified, and fetched it back through the normal cache path.
The restored blob matched SHA-256
`d3a85149cb6db298a08e5a6ddc56182faf224ccdb1d2a423a333e9c7765bfc94`.
See `storage-live-sync.json`. There is no remaining credential or Stage-0 sync
gate.

The current OAuth app is External/Testing. Google limits non-basic testing-mode
refresh tokens to seven days, so this authorized-user credential is suitable
for the immediate pilot but is not yet a durable unattended credential. Before
a campaign that may span that boundary, either publish the OAuth app (accepting
Google's production and verification requirements) or use a service account
with access only to the snapshot folder. Credential JSON remains outside the
repository. The full-Drive scope is currently needed because the CLI must reach
an existing user-created folder without a Google Picker flow.

## Disk footprint

`disk-footprint.json` records 6,035,379 bytes (5.756 MiB) of Stage-0 blobs,
154,348,898 bytes (147.199 MiB) of checkpoints, 1,806,597 bytes (1.723 MiB) of
artifacts, the learner C# build caches, and the frozen simulator checkout/cache.
All 6,035,379 Stage-0 blob bytes are now remotely verified and safely
reclaimable.
The 5 GiB LRU ceiling bounds future local snapshot storage; checkpoints remain
outside this blob-store policy. At the current Stage-0 mean blob size, the
unlaunched 240-snapshot plan projects to roughly 49.335 MiB of blob bytes.

## Commands and observed status

```text
ruff check .
# PASS

pytest -q --basetemp=<short-workspace-temp>
# 53 passed

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
# PASS locally; 28/28 valid, 28/28 with remote metadata, 0 unrecoverable

anthropocene-rl snapshot-store sync
# PASS; uploaded 27, deduplicated 1, then reconciled 28/28 settled versions

anthropocene-rl snapshot-store status
# PASS; 28/28 live remote verification, 0 invalid, 0 unrecoverable

anthropocene-rl snapshot-store prune --max-local-gb 0.005423646420240402
# PASS; evicted exactly one verified blob and reclaimed 211883 bytes

anthropocene-rl snapshot-store fetch f01-21002-f01-young-viable-d1-87f7b59c91fe
# PASS; fetched=true; SHA-256 and size verified; cache restored
```

The full 200-500-state cultivation campaign was deliberately not launched.

## Training trajectory interpretation

The 25-point deterministic action paths (initial policy plus 24 PPO updates)
do **not** show a steady increase in cultivation intensity. None of the nine
runs is monotonic in either active cultivation control. Retained-field
operation ends above its initial value in 6/9 runs, with a modest mean endpoint
change of +0.0448, while development budget ends above its initial value in
only 4/9 runs and averages -0.0534. Individual update-to-update increases occur
only 49.5% and 44.9% of the time, respectively.

The fixed-policy before/after comparison is more conservative: mean requested
development budget decreases in all 9/9 runs (mean change -0.1384), and mean
retained-field operation decreases in 7/9 (mean change -0.0787). F03 is the
clearest learned effect: all three seeds improve discounted scaled return by
1.5312 while reducing mean retained-field operation by 0.2473 and development
budget by 0.0770. F01 and F13 returns remain flat rather than regress; F13 shows
the only family-level increase in retained-field operation (+0.0558), driven by
its amenity-constrained adaptation. Thus the readiness evidence supports
stable, context-sensitive policy movement and selective de-intensification,
not a universal or steadily increasing cultivation trajectory. See
`training-trajectory-summary.json`.
