# Model7/native6 opportunity evidence

`canary-summary.json`, `canary-table.json`, `postflight-summary.json`,
`verification.json` and `ledgers.json` are the compact outcome/accounting records.
`artifact-index.json` hashes the preserved local run files, including all failed
attempts and retained/intermediate checkpoints. Large snapshots and checkpoints
remain in the original ignored datasets/runs directories; they are not replaced
by these summaries. `snapshot-manifest.json` is a byte-preserving copy of the
new bank manifest. Original v5 bank/quarantine/HELDOUT flags remain unchanged.

The final v9.7 status is **NO-GO_REVERSAL**, with a separately passing two-Elder
native plug/replay. Earlier reversal `passed` flags measure improvement only;
they must not be interpreted as a completed signed switch. Economic improvement
is not demonstrated. The independent Astra/high review is authoritative for
scope, not a blanket scale-out approval.

Prospective JSON plans in `configs/` pin each actual source/data/DLL/checkpoint
envelope. Archived launch sources under `source-at-generation`,
`source-at-admission2` and `source-at-canary` preserve subsequently changed bytes.
The original deadline-cleanup race and v9.6 reporting typo are archived; the
final current source fixes only teardown/reporting after the completed canary.
Old-version contracts, manifests, experiments and checkpoints remain available.

The new `rollout_audit.py` saver is tested preparation for a future bounded
experiment; it was not retroactively applied to v9 traces. Old frozen-actor
targets cannot establish current post-PPO on-policy critic readiness.
