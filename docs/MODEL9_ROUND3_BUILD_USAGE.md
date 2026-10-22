# Round-3 learner implementation and validation

The normative design is `MODEL9_ROUND3_BUILD_ARCHITECTURE.md`. This note describes the implemented interfaces;
passing local tests does not constitute the round-3 experiment or calibration gate.

## Head loading and resume

All actor-loading tools accept `--region-head hierarchical|flat` and the compatibility alias `--flat-region-head`.
The shared loader resolves an explicit flag, then the saved mode, then hierarchical for old weights. An old optimizer
resume is flat, excludes dormant group parameters, and retains its parameter ordering. A head or cost-bootstrap
change on resume fails: start a new named run from weights instead. Same-round PPO resume compares a frozen manifest
of resolved flags, input hashes, learner identity and the live simulator identity.

The bridge project embeds its simulator checkout's full Git commit and clean status at build time. Build it with an
explicit `AnthropoceneSimRoot`; a failed Git query produces an untrusted identity. A production PPO run probes the
same configured bridge once before collection, without resetting a world. It requires clean full learner/simulator
commits and freezes the actual schema build hash. `ANTHROPOCENE_SIM_COMMIT`, if supplied by a launcher, must match
the embedded commit; it cannot replace the live identity. Each rollout bridge must match the frozen identity.

## Source and calibration manifests

Generated labels contain a full `source_record` and `source_run`; branch metadata includes those identities, the bank
entry/snapshot hash, a full trajectory identity and a digest of the stored NPY payloads. Branch world IDs are rebased
on loading. Duplicate trajectories and changed payloads fail rather than merging sequences.

Training defaults to strict provenance. To deliberately train on multiple compatible, attributable source runs, supply
`--source-manifest sources.json` with the exact allowed IDs:

```json
{"source_runs": ["<full source_run SHA-256>", "<another full source_run SHA-256>"]}
```

This does not waive unknown/dirty sources or incompatible simulator/contract/utility identities. `flag` is for analysis;
`allow` is for tests/migrations. Production training cannot use either policy for labels or branch shards.

PPO and offline critic training compare the union of loaded label/branch sources with the live schema-only bridge
probe before evaluation or optimization. Exact full commit/build equality is the default. An independently approved
flag-off semantic comparator may be pre-registered in the same frozen source manifest as `simulator_compatibility`:
each entry contains `source` and `live` objects with full `commit`, full `buildHash` and `clean: true`, a nonempty
`reason`, the comparator report's full `evidence_digest` SHA-256, and `flags` containing the exact `sim_rules` and
`sim_statics` strings. A mapping only covers its named source-to-live pair and frozen flags. The code does not create
or approve comparator evidence. The current dirty simulator is not production-eligible, with or without a mapping.

Search, PPO, stage-1 training and `branch_bias` accept `--calibration-manifest`. Its format is:

```json
{
  "calibration_seeds": [10000, 10001],
  "calibration_snapshots": [],
  "training_seeds": [1000, 1001],
  "training_snapshots": [],
  "selection_seeds": [2160, 2161],
  "selection_snapshots": [],
  "waivers": []
}
```

This abbreviated shape illustrates exclusion fields only, not a qualification-ready cohort. Production includes all
100 roots 10000-10099, the frozen snapshot list, `protocol_digest`, an absolute pre-frozen `audit_registry`, and empty
waivers. The complete seed-role contract is in the reviewed architecture; examples do not authorize smaller cohorts.

Snapshot entries are full content SHA-256 identities. Families must be disjoint, and the manifest digest is frozen in
generated source metadata. Training/selection check `source_seed` and `source_snapshot`, not rebased sequence IDs.
The `training_*` and `selection_*` role lists are declared exclusion evidence, not exhaustive positive allowlists;
actual role-family closures are persisted and every calibration-family intersection is rejected. Exact intentional
source-mix control is `source_manifest.source_runs`, including when daughter-family sets grow dynamically.
Held-out labels and branch shards cannot train. `branch_bias` verifies that its source families and manifest digest
match before allowing a pass. Waivers name exact nonmandatory cells, such as `trade/trade_utility` or
`trade/population/0-29`; native/region/mix welfare cells cannot be waived.

Saved checkpoints carry `family_lineage`: actual training/selection seed and snapshot families plus recursive,
content-hash-bound initialization/teacher proofs. `branch_bias` requires the evaluated checkpoint's own matching
exclusion manifest, clean frozen run identities/configuration, and the complete inherited lineage. A legacy
initialization without this evidence stays explicitly unverified; clean held-out shards do not make it calibration-safe.

Calibration requires the same complete eight-turn windows for bias and coverage, including a recorded living endpoint
or an earlier terminal lineage. The numerical near-zero rule, independent subset gates and sample minimums are those
in the reviewed architecture. All living endpoint settlements are included, even daughters founded on the final turn.

## Decision diagnostics

Schema-v1 `decisionDiagnostics` is an authoritative sidecar with trajectory, decision-turn and camp identity. Initial
reset has none. Fork/restore retains the preceding main-path sidecar; the next branch action may acquire a new
trajectory. The client checks identity and every value/mask layout. PPO and evaluation report per-component change
rates, magnitudes, mask transitions and unavailable/inactive counts. Branch shards also serialize the raw sidecars
separately. These fields never enter actor tensors or rewards. Historical schema-0 bridges remain readable for
compatibility tests and report no authoritative decision diagnostics.

The separate physical sidecar is top-level `diagnosticSchemaVersion`, `trajectoryId` and `rawMaterialDiagnostics`.
The dictionary key is `campId:numericMaterialId`; each record carries `schemaVersion`, `campId`, `decisionTurn`,
`material`, `requestId`, `requestedUnits`, `acceptedUnits` and `realisedUnits`. These are physical quantities from the
simulator's typed requests, grants and tagged extraction receipts, not actor-head values. Python exposes
`raw_material_diagnostics`, `diagnostic_schema_version` and `diagnostic_trajectory_id`, validates identities and finite
nonnegative units, and checks exact sidecar alignment on fork/restore. Reset has no preceding receipts. The schema
advertises envelope `diagnosticSchemaVersion: 2` and `rawMaterialDiagnosticsSchema: 2`, while contained generic
decision-head records retain schema 1. Envelope 1 with raw materials enabled fails closed; envelope 1/raw off is
legacy compatibility only. Older bridges without diagnostics provide an absent physical sidecar.
Neither physical sidecars nor decision sidecars enter observation tensors or rewards.

Raw gathering is latent only. Normal native fallback commits all residual workers, so normal Prepare must authorize
zero; synthetic positive settlement tests validate the kernel only. Useful positive full-turn gathering awaits a
separately reviewed workforce-idleness contract. Round 3 remains raw OFF.

## Local regression command

When using another worktree's shared virtualenv, set `PYTHONPATH` to **this** checkout's `src` so its editable install
cannot select the other checkout. Run from this checkout with a dedicated temporary directory:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -B -m pytest -p no:cacheprovider --basetemp=work/pytest-round3 `
  tests/test_autonomous_v7.py tests/test_clone_stability.py `
  tests/test_autonomous_v7_round3_contracts.py tests/test_autonomous_v7_round3_build.py `
  tests/test_autonomous_v7_round3_search.py tests/test_autonomous_v7_round3.py `
  tests/test_autonomous_v7_search.py tests/test_autonomous_v7_ppo.py `
  tests/test_movement_audit_inspect.py tests/test_autonomous_v7_bridge.py `
  tests/test_autonomous_v7_round3_protocol.py
```

The `work` parent must exist. The contract tests compare flat initialization, logits, sampled actions, likelihoods,
Adam updates and RNG state with the pre-hierarchy Git baseline for both pooling and token actors. Bridge tests require
a freshly built local binary. No cloud execution is needed for this validation.

## Strict qualification interfaces

These interfaces do not authorize generation, training or submission. Production gates intentionally refuse the
current dirty source trees. Local training probes are diagnostic only and limited to three seeds; readiness requires
the reviewed VM cohort of R3def/R3per/R3lo with paired optimizer seeds 170-174. Smaller VM designs need review.

- `python -m anthropocene_rl.autonomous_v7.protocol --spec inventory-spec.json --out inventory.json` inventories
  actual hashed prior-use files under every explicit `scope_roots` directory. The schema-1 specification enumerates
  `artifacts` (`path`, `sha256`, `kind`, `role`) and optional `extra_training_roots`. Unknown, unlisted, mutated,
  unattributed or confirmation-family inputs fail. Consumers re-read the evidence, not just the receipt digest.
- `python -m anthropocene_rl.autonomous_v7.teacher_data --seeds <explicit-list> --withheld 0 --warmup 0
  --players <frozen-count> --turns <frozen-count> --prior-use-inventory inventory.json --out <new-directory>` uses the
  exact production bridge. `dataset.json` binds sorted full trajectory IDs to published metadata/payload hashes;
  `meta.json` is published last. The production `train` loader rejects old t1 shards; no metadata migration exists.
- `python -m anthropocene_rl.autonomous_v7.clone_evaluate --checkpoint candidate.pt
  --calibration-manifest held.json --out <new-report.json>` evaluates one sealed candidate on all clone-selection
  roots. `promotion --inputs candidates.json --calibration-manifest held.json --out promoted.pt` requires ten input
  entries, each with hash-bound `checkpoint` and `report` references (`path`, `sha256`). All losing closures remain.
  Promotion removes optimizer/RNG continuation and preserves model/normalizer semantics exactly.
- Every production clone supplies `train --clone-protocol clone-plan.json --preregistration preregistration.json
  --protocol-approval approval.json ...`. The schema-1 clone plan has `kind: guarded_clone_cohort`,
  `projection_version: 1`, `protocol_digest` and `cohort_digest`; all ten candidates must share this exact artifact
  and complete non-seed/non-output configuration and teacher-input cohort. Recursive proofs reopen/re-hash actual
  ancestor checkpoints and teacher dataset/payload artifacts; copying an embedded proof or self-hashing a fabricated
  dataset manifest is insufficient. Those referenced artifacts must remain available to qualification.
- Preregistration uses `protocol_digest_version: 1`. Its canonical protocol projection excludes only `digest` and the
  explicitly late-bound `clone_protocol_sha256`; every other field, including newly introduced fields, affects the
  protocol digest. Freeze decision rules first, then the exclusion manifest, teacher inputs and clone plan; bind the
  exact full clone-plan SHA into final preregistration bytes. An external immutable schema-1 approval receipt has
  `kind: protocol_approval`, `reviewer: Astra`, `decision: APPROVE`, the projected `protocol_digest` and the full-byte
  `preregistration_sha256`. Approval is not embedded in the hashed preregistration. Changing late binding requires a
  new matching approval even though the projected protocol digest stays stable. Each clone validates both the full
  plan and external approval before fitting; promotion and qualification repeat those checks.
- `curriculum.freeze_bank(bank, image_path)` publishes a new schema-1 initial image containing `entries`, compressed
  `files` references and canonical `digest`; entries include decompressed `snapshot_sha256`. Each PPO arm supplies
  `--curriculum r3-balanced-v1 --bank-image image.json --daughter-bank <its-own-output>/bank`. Reusing another arm's
  bank fails. Resume stores both cursors, exact bank index/content identities and RNG continuation.
- `search_teacher --worlds-manifest cohort.json ...` requires a digest-bound schema-1 cohort: `role` is
  `training-default`, `training-dirichlet` or `calibration`; `personality` is `default` or `dirichlet:1`;
  `max_alternatives` is 19; each `worlds` row has `seed`, `withheld`, `warmup`, `bank_entry` (-1 for an opening),
  and `snapshot_sha256` (empty for an opening). Training cohorts contain 60 fresh roots plus 60 exact daughter
  snapshots; calibration contains all 100 calibration roots and at most 100 frozen calibration snapshots. Exact
  source families and snapshots are checked before and after workers run. No dynamic newest-entry selection,
  additional fixtures, output overwrite or adaptive top-up is allowed. Specify the reviewed points/villages/horizon,
  tail, personality, bootstrap and calibration coverage flags explicitly.
- Search cohort flags also require `turns: 25`, `first_turn: 1`, `daughter_search: true` and `calibration_tail: 12`.
  Calibration generation publishes `calibration-inputs.json` binding the complete cohort, audited promoted SHA,
  every branch metadata hash and every payload digest. A subset or another producer cannot satisfy the audit.
- `branch_bias --checkpoint promoted.pt --branch-shards <held-shards> --calibration-manifest held.json
  --calibration-inputs calibration-inputs.json --provenance reject --out <new-audit.json>` claims the central frozen
  `audit_registry` before evaluating. The exclusive protocol/candidate lock and complete cohort/payload claim are
  independent of output paths. Changing output paths or input inventories cannot repeat the one-shot audit. A failure
  is not permission to remove the claim or tune/refit. Qualification requires all four critic heads, frozen
  `waivers: []`, and recomputes numerical gates/coverage from the actual payloads without native generation or fitting.
- Registered PPO completion publishes hash-bound `completion.json` before its JSON `done` marker. It binds all 55
  iterations (five critic-only then fifty actor), the actual selected best and final latest checkpoint hashes, eligible
  selection scores and deterministic earliest-iteration tie rule. `personality_sweep --completion completion.json
  --checkpoint-form best|latest ...` validates that receipt; a renamed intermediate checkpoint is not final evidence.
  Actual rollout worker bridge/simulator/flags identities must match across workers and the checkpoint. Reports
  require exactly the canonical seven personality vectors, 25 turns, withheld=0/warmup=0 and per-world focal evidence.
- `formal_report --inputs report-inputs.json --out report.json` consumes hash-bound `arms[arm][best|latest][all|focal]`
  sweep lists. Each list contains five uniquely matched seeds; focal per-world outcomes and denominators are required.
  It computes personality/default-world gates only, not missing global-regime or reference-collapse evidence.
- `qualification --manifest submission.json` is read-only preflight, not a queue submitter. A frozen schema-1 manifest
  has canonical `digest`, `protocol_digest`, `execution` (`tier`, `purpose`, `arms`), full `learner_commit`, `simulator`,
  `simulator_repository`, `bridge_sha256`, `flags`, `clone_protocol_sha256`, and hash-bound `queue_job`,
  `calibration_manifest`, `prior_use_inventory`, `initialization`, `comparators`, `gates` and 15 `runs`
  (`arm`, `seed`, `phase`, actual fully expanded `argv`, `resolved_config` reference, `bank_image_sha256`,
  `search_cohort_digest`). Required gates
  are protocol/source/clone/search/calibration/movement-integration/submission. Each schema-1 receipt has matching
  frozen identities, `status: pass`, its `kind`, nonempty hash-verified `outputs` and typed `evidence`; a generic pass
  shell or bare done marker fails. Source/clone/search/calibration/movement-integration receipts additionally bind the
  actual qualification queue `job`, runner `status`, `phase` and structured `done` artifact. The done receipt names
  the exact queue-job SHA, frozen identities and complete output inventory, and runner status must show exit zero.
  Both repository commits must be clean and reachable from fetched remote refs. Queue execution still needs separate
  authority. Immutable per-run resolved configurations include every checked treatment plus the complete CLI/default
  expansion; a preflight result does not substitute for independent architecture review.

Typed `evidence` maps are: protocol `{preregistration, review}`; source `{inventory, datasets}`; clone
`{promoted, candidates: [{checkpoint, report}]}`; search `{training-default, training-dirichlet}`, each containing
`{cohort, summary, branch_indexes, labels}` and `bank_index` when snapshots are present; calibration that same cohort
map plus `{inputs, report, checkpoint}`; movement-integration `{rollouts, forks, judgment, integration_reports}`;
submission `{queue_job}`. Every named artifact is a `{path, sha256}` reference and appears in the receipt output
inventory. Validators consume actual contents: all ten clone/report artifacts, complete search inventories,
calibration gates, paired grouped/native/stay movement and MOVE/STAY evidence, and historical paired metrics.

Each final queue phase is direct fully expanded `python -m anthropocene_rl.autonomous_v7.ppo ...`, not an opaque
shell loop. Every registered run includes `--arm`, `--execution-manifest`, `--resume`, all frozen inputs and its own
output/private bank. The actual PPO entrypoint invokes full preflight in the matching queue environment before RNG
initialization or optimization. Preflight parses the actual argv through the PPO parser/resolver and compares the
complete options, resolved Hyper/Stage and effective 5+50 budget against each immutable resolved configuration.
It enforces paired arms/seeds, welfare-only base weights, commitments OFF, roots 2060-2099, identical initial bank
payloads, the same Dirichlet search cohort for R3per/R3lo, and equality of all non-treatment defaults.

All protocol-bearing fitting entrypoints enforce execution authority before RNG initialization. Frozen VM designs
also name `execution.vm = {project_id, instance_id, zone}`; the actual host must match fixed GCP metadata identity,
not an environment claim. Guarded-clone fitting, clone selection and promotion require `--execution-manifest` naming
the exact ten-candidate `clone_execution` queue plan and actual phase argv. Local probes explicitly use
`--nonqualifying --local-execution-design local.json`, with a canonical-digest schema-1 execution design naming
`tier: local`, `purpose: validation|stability`, and `arms: {diagnostic: [at most three seeds]}`. They cannot carry
protocol/qualification inputs, select `best.pt`, emit formal completion/readiness evidence, or become verified
lineage; completion is `diagnostic.done`, not the qualifying `done` marker.

Qualification emits all fifteen full frozen run manifests and their digests. Formal input includes a hash-bound
`qualification` reference to that result, which itself binds the original immutable submission artifact. Formal
reporting calls the same `validate_qualified_submission` semantic validator as initial preflight: it reopens and
rechecks all non-training gates, including actual teacher/ancestor payloads, all-ten promotion, calibration arithmetic
and central one-shot claim, approved protocol/configuration/bank/search bindings and queue evidence. It does not
repeat fitting or expensive evaluations. An immutable, self-consistent `status: pass` shell is not evidence of truth.
Each checkpoint/completion must belong to the exact qualified 3x5 cohort, never merely match its own manifest.

Simulator compatibility mappings require an actual hash-bound comparator receipt (`evidence` path/SHA-256), not only
an `evidence_digest`. Its kind is `simulator_semantic_comparison`; it names both full identities and exact flags,
nonempty verified output references, and per-root/turn equal source/live fingerprint comparisons. Each comparison
also names actual `source_trace` and `live_trace` output references: schema-3 `simulator_semantic_trace` objects with
the exact identity/flags/seed/turn count, hash-bound `bridge` binary reference, ordinary `reset_snapshot`, canonical
`initial_state`, and contiguous action/background/successor transitions under
`model9-step-actions-and-native-unsubmitted-v1`. The verifier starts a pinned producer, restores the ordinary reset
snapshot, checks its canonical initial payload, replays every exact action/background input and verifies every
successor. Copying equal valid frames without proving their transitions fails. Each frame binds
its complete canonical gzip SaveCodec `snapshot` artifact, native `fingerprint`, `normalized_payload_sha256`, `header_sha256`
and `binary_tail_sha256`; all binaries/snapshots/traces are listed in the comparator output inventory. The validator
uses that exact bridge's read-only `fingerprint_snapshot` export contract
`savecodec-canonical-writepayload-sha256-v1`: the exporter constructs SaveCodec's canonical ordered copy, proves its
decompressed WritePayload SHA-256 equals the current live native fingerprint, and proves export did not change it.
Trace field `fingerprint_contract` names that exact version. Gzip artifact SHA and uncompressed/native fingerprint
remain separate identities. `validate_fingerprint_snapshot` requires a byte-exact typed header round trip, verifies
complete save/root/turn structure and recomputes
`payloadFingerprint` directly from complete decompressed bytes; its lossy restored-world `fingerprint` is not used
as the original live-frame identity. Canonical fingerprint headers have `Generation=null`; because ordinary
SaveCodec.Load requires nonphysical generator metadata, the verifier supplies a fresh GenerationReport only in a
temporary validation copy. It never changes the compared header/tail artifact or uses the restored hash as identity.
The exact trace normalization is
`{"version":1,"kind":"savecodec-complete-identity","excluded_fields":[]}`: gzip decompression only, no world field
removed or rewritten. Source/live complete uncompressed saves must be byte-identical, including the entire binary
tail. Arbitrary `world` dictionaries/partial physical projections fail. Legacy/current non-identity normalization v2
requires separate architecture review; v1 does not grant that waiver. Fresh Round-4 teacher lineage uses the exact
current build, without a cross-build waiver. Exact identity
remains the default. Calibration manifest role lists are declared exclusion evidence, not exhaustive positive
allowlists; the exact input mix is `source_manifest.source_runs`, while actual role closures are persisted.
Before either replay, paired traces must also have identical reset descriptors, root/seed/turn/player/movement
inputs, complete ordinary reset-save semantics, canonical initial payload and exact per-turn action/background
inputs. Independent replays that reach equal outputs under compensating different interventions do not qualify.

Both Python and native reset boundaries deny confirmation seeds before opening generation. Captured snapshots are
loaded only for root validation first, without environment reset/observation; their root must be unsealed and match
the declared seed. An immutable worker contract freezes actual loaded bridge SHA/MVID, full simulator identity,
head/envelope/raw schema versions, actual native rule/static strings, the complete native Boolean feature flags,
and the native full configuration SHA-256 with exact player count. No numeric configuration/player-count allowances
are registered: bank snapshots and fresh openings must match the frozen full configuration. The separately frozen
reviewed capability mask is excluded by SaveCodec's configuration hash, never silently inferred from it.
Metadata `players` means `world.Config.Players`, the initial reset configuration; `livePlayerCount` separately reports
the dynamic `World.PlayerCount`, which may grow after fission and is not a configuration waiver.
Before any search/evaluation/PPO/daughter Reset, the read-only `reset_metadata` operation verifies the pinned world
root/turn/snapshot hash, full configuration/player identity and raw-OFF feature flags without generation, Reset,
observation, warm-up or action. World descriptors bind movement; bound Reset rejects unregistered size overrides.
The Python bridge then rechecks the bound producer before Reset/Step/fork/restore and rejects changed Reset
arguments. Parent aggregators revalidate worker/world proofs before metrics, optimization or best selection;
selection and formal per-world evidence retain those proofs. Training search cohorts must share exact canonical
world/snapshot/extra-root declarations; their role/personality/digest alone differ. Registered search rejects flat
inference before simulation, and source producers bind the complete canonical effective summary config, resolving
calibration-manifest paths to their verified content rather than comparing a pathname to an embedded dictionary.
