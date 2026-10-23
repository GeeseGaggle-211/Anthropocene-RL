# Model9 Gate 1 production evidence collection and freeze protocol

Status: **DRAFT -- production evidence `HOLD`**. This document is a design and
freeze procedure. It is not a substitute for the schema-1, digest-sealed JSON
protocol, review, manifest, report, verification or packet artifacts required
by `MODEL9_DIAGNOSTIC_CLOSURE_ARCHITECTURE.md`. The terminated-VM bundle is now
present locally and its 396/396 source/local SHA-256 inventory compares with zero
mismatches. A corrected read-only evidence package records the recovered
lineage, but a later raw-state check corrected one material label: recovered
`stage1-token-s6.pt` is the **unguarded T3 initialization clone**, not the
distinct T2m guarded clone named by the historical account. A separately
hash-verified T2m retrieval contains nine guarded actor-only BC phases (seeds
0,1,2,3,4,6,7,8,9), including their original normalizers and BC configurations;
the logs stop during critic fitting and no completed critic checkpoint was
recovered. Seed 6 is included as an independent actor-only evaluation node
because it matches the historical T3 initialization seed, not because of an
outcome. It is not a T3 ancestor or a completed clone. That is not proof of
global historical nonexistence, so completed guarded-clone critic provenance
remains explicitly unresolved. No
checkpoint evaluation, simulator reset, same-machine benchmark or production
result has been collected. The exact fresh cohort and executable proposal
remain `HOLD` until their sealed artifacts receive the required review.

The Gate 1 diagnostic evaluator capability implementation at commit
`2a6565e46c43b425b1f9cc6d220995818d42061d` has Astra High implementation
approval recorded as acceptance6. That approval does not approve this draft,
any late-bound value below, or any production evidence. This draft and the
implementation-identity source-list change that binds it require a new Astra
High review before they may govern collection.

This protocol does not authorize a VM start, additional retrieval, training, checkpoint mutation,
arm selection, tuning, checkpoint promotion, model change, confirmation-seed
use, or Gates 2-4. Retrieval, when separately authorized, is read-only.

## 1. Governing rule and state machine

Every production input is either immutable and content-addressed or is an
explicit unresolved item that blocks collection. Strings such as `TBD`, empty
hashes, convenient local replacements, caller assertions and hashes populated
before bytes are present are invalid evidence.

The evidence program advances only through these states:

1. **Draft (`HOLD`).** This document names the required slots, constraints and
   unresolved decisions. It authorizes no retrieval or execution.
2. **Retrieval candidate (`HOLD`).** After separate authorization, copy the
   historical bytes read-only. Hash each byte stream only after it is locally
   present. Inspect raw metadata without upgrade, model construction or
   inferred defaults. Record missing artifacts as missing; do not reconstruct
   or substitute them.
3. **Freeze candidate (`HOLD`).** First resolve only the inputs required for a
   limited baseline-calibration review: exact fixtures, machine/runtime policy,
   warm-ups, repetitions, final pair order, measurement methods and relative
   budget rules. Seal that benchmark subprotocol and obtain a limited Astra
   `APPROVE` for **baseline-only calibration**. Calibration contains no
   diagnostic treatment run and cannot supply a final acceptance pair. Run the
   approved baseline-only calibration, hash its raw receipts and, if useful,
   derive conservative absolute operational guardrails. Then populate those
   hashes and guardrails, populate all other exact hashes and metadata in the
   lineage inventory, evaluation/claim protocol, equivalence protocols and
   verification protocol, resolve every remaining item in section 13, and seal
   the complete proposal digests for state-4 review. No treatment or evaluation
   may run in this state. The limited review authorizes neither diagnostic
   evaluation nor final benchmark acceptance.
4. **Independent review (`HOLD`).** Astra High reviews the complete candidate,
   including all retrieved hashes, cohort, worlds, snapshots, semantics,
   equivalence contracts, commands, coverage scope and performance limits.
   Each schema-1 JSON proposal is bound by its own content-addressed independent
   review artifact. The review scope must explicitly name artifacts,
   equivalence, evaluation cohort, semantics and verification.
5. **Frozen for collection.** This state exists only after all applicable Astra
   decisions are `APPROVE`, proposal digests match byte-for-byte and no
   unresolved or missing required item remains. From this transition onward,
   every listed byte, ID, ordering, command, threshold and semantic declaration
   is immutable. A change returns the program to state 3 and requires a new
   review before collection resumes.
6. **Evidence collection (`HOLD`).** Run only the frozen evaluations and
   verification. Final performance evidence consists of new baseline and
   treatment measurements collected as the frozen paired/interleaved sequence;
   state-3 calibration receipts are not reused as pair members. Store raw
   receipts and all transitive inputs in the packet CAS. A clean execution does
   not by itself pass Gate 1.
7. **Post-collection review.** Astra reviews the reconstructed packet and raw
   evidence. Gate 1 remains `HOLD` unless that review explicitly approves it.

Retrieval hashes and calibration-baseline-derived operational guardrails are
deliberately late-bound because they cannot be honestly known before the bytes
and same-machine baseline-only measurements exist. They become immutable at
state 5, before any treatment or evaluation result is observed.

## 2. Exact lineage inventory

The independently reviewed `lineage_inventory` JSON must prescribe the exact
IDs, roles, cardinalities, hashes, source evidence, run identities, statuses,
applicable panes and edges. The manifest must equal that inventory; self-chosen
roles or a favorable subset cannot establish completeness.

| User-facing slot | Required inventory |
| --- | --- |
| `prior-reference` | Exactly one recovered historical prior/reference node: the contract-8 `stage1-seed2.pt`, used as an evaluation reference and not asserted to be the BC parameter or data parent. |
| `conversion` | `not-applicable` only because the recovered producer script constructs a fresh token actor and performs learned BC; there is no historical parameter-conversion edge. Never synthesize a conversion. |
| `learned-bc-phase` | Exactly one recovered actor-only 12-epoch learned-BC artifact. For the fresh diagnostic composite it uses the recovered unguarded T3 initialization checkpoint's hash-bound normalizer/metadata after literal actor identity is proved. Recovered teacher arrays have a hash-verified 7,920-member inventory, but their exact linkage to the original BC fit and the original BC-time fitted-normalizer identity are not established; no original-BC-normalizer claim is permitted. |
| `t3-initialization-clone` | Exactly one recovered unguarded clone node, `stage1-token-s6.pt`, used as the recovered T3 initialization support and current diagnostic composite carrier. Its normalizer has no guard/dead fields. |
| `guarded-clone` | T2m seed 6 supplies a present, independently hash-bound guarded actor-only BC node with its original normalizer and BC configuration. It is selected only by seed identity, is not a T3 ancestor and has no recovered completed critic checkpoint. The completed guarded-clone critic slot therefore remains `incomplete-unresolved`; absence in the recovered archive is not proof of global nonexistence. |
| `prospective-normalizer-intervention` | Exactly one prospective composition: the bitwise recovered unguarded T3-initialization actor and all its affine normalizer moments, plus only the T2m seed-6 `guard=10` and exact dead masks. All common affine fields must be literally identical before composition. This node is motivated by the pre-existing cloning-stability diagnosis, uses no fresh outcome selection, and is never a historical artifact or ancestry claim. |
| `trained-t3` | Exactly 15 distinct historical trained nodes: five for each of `T3per`, `T3def` and `T3lo`. Every arm/run remains a distinct branch and aggregation identity. |

Historical T3 optimizer identities are disjoint and **unpaired**:

- `T3per`: optimizer seeds 70, 71, 72, 73, 74;
- `T3def`: optimizer seeds 75, 76, 77, 78, 79; and
- `T3lo`: optimizer seeds 80, 81, 82, 83, 84.

Every T3 node must declare `pairing_status=unpaired`, no `pairing_group`, its
exact `arm_id`, optimizer seed and run ID. The prior and clone use explicit
not-applicable arm/optimizer-seed semantics. No report may relabel historical
T3 as matched, and no paired-world bootstrap may be described as optimizer
uncertainty.

The reviewed edge inventory must preserve the shared historical base sequence
and each T3 branch without duplicating the base edge. Edge chronology must
follow DAG ancestry. Incomparable branches receive path-specific results; they
do not acquire a fabricated global first edge. Required present boundaries may
not be omitted from the comparison order. Every present required node and
comparable edge must have all prescribed panes, worlds and metrics.

Each node must bind at least: checkpoint and raw state-dict bytes; exact raw
schema; actor, critic and normalizer contracts; producing learner; worker and
producer evidence; run evidence; source artifacts; historical status; loader
implementation; default-resolution and migration records; simulator and bridge
identities; and run/arm/seed/pairing identity. A required missing artifact keeps
the gate on `HOLD`. An optional missing or not-applicable item remains visible
but is not silently promoted to present.

### 2.1 Read-only retrieval slots

Retrieval is an inventory operation, not permission to evaluate. Every copied
file retains its original path, size, modification metadata where available and
post-copy SHA-256. A directory listing or run-name convention is not evidence of
a file. Remote directory roots remain late-bound until a contemporaneous
manifest, completion record or other reviewed producer evidence identifies
them; the protocol must not guess a root from an arm name.

The primary evaluator artifact is the fixed-budget `latest.pt` for each T3 run.
`best.pt` is separately preserved selection evidence and must never replace
`latest.pt` in the primary endpoint. The retrieval candidate has these exact
T3 run slots:

| Canonical node ID | Historical run label | Arm identity | Optimizer seed | Required primary | Supporting selection artifact |
| --- | --- | --- | ---: | --- | --- |
| `t3per-s70` | `T3per-s70` | `T3per` | 70 | contemporaneous run root + `latest.pt` | same root + `best.pt` |
| `t3per-s71` | `T3per-s71` | `T3per` | 71 | contemporaneous run root + `latest.pt` | same root + `best.pt` |
| `t3per-s72` | `T3per-s72` | `T3per` | 72 | contemporaneous run root + `latest.pt` | same root + `best.pt` |
| `t3per-s73` | `T3per-s73` | `T3per` | 73 | contemporaneous run root + `latest.pt` | same root + `best.pt` |
| `t3per-s74` | `T3per-s74` | `T3per` | 74 | contemporaneous run root + `latest.pt` | same root + `best.pt` |
| `t3def-s75` | `T3def-s75` | `T3def` | 75 | contemporaneous run root + `latest.pt` | same root + `best.pt` |
| `t3def-s76` | `T3def-s76` | `T3def` | 76 | contemporaneous run root + `latest.pt` | same root + `best.pt` |
| `t3def-s77` | `T3def-s77` | `T3def` | 77 | contemporaneous run root + `latest.pt` | same root + `best.pt` |
| `t3def-s78` | `T3def-s78` | `T3def` | 78 | contemporaneous run root + `latest.pt` | same root + `best.pt` |
| `t3def-s79` | `T3def-s79` | `T3def` | 79 | contemporaneous run root + `latest.pt` | same root + `best.pt` |
| `t3lo-s80` | `T3lo-s80` | `T3lo` | 80 | contemporaneous run root + `latest.pt` | same root + `best.pt` |
| `t3lo-s81` | `T3lo-s81` | `T3lo` | 81 | contemporaneous run root + `latest.pt` | same root + `best.pt` |
| `t3lo-s82` | `T3lo-s82` | `T3lo` | 82 | contemporaneous run root + `latest.pt` | same root + `best.pt` |
| `t3lo-s83` | `T3lo-s83` | `T3lo` | 83 | contemporaneous run root + `latest.pt` | same root + `best.pt` |
| `t3lo-s84` | `T3lo-s84` | `T3lo` | 84 | contemporaneous run root + `latest.pt` | same root + `best.pt` |

Only the lowercase first column is used as the manifest's canonical `node.id`.
The historical run label and `arm_id` are separate, case-sensitive provenance
fields and must equal the contemporaneous producer records.

The prior/reference is recovered in the verified bundle as
`producer/inputs/stage1-seed2.pt`. It was read-only reverified as 4,497,255 bytes, SHA-256
`c2f424289029e57ecbb0371a99ce373d6324e4a9aff232f348bdcacc14ff7e99`, raw
contract 8. Those bytes also match immutable GCS object
`gs://anthropocene-rl-datasets-508912/model9/checkpoints/contract8-pilot/checkpoints10/stage1-seed2.pt`,
generation `1792308001259477`, created `2026-10-18T07:20:01Z`, with matching
4,497,255-byte size and MD5 `xreqIFyu1BFNM24l8lB8Zg==`. The local bytes are now
bound by the 396-file retrieval inventory; the remote metadata remains
acquisition provenance rather than a fresh download receipt. A weights-only preflight inspection on 23
October 2026 confirmed contract 8, core width 207, present actor/critic/
normalizer mappings, actor-key digest
`8a058d7cd13308b85ba5dcdf880db47daf81a93c1bef1b50fba374ea7323ea2d` and
normalizer-key digest
`a39f21fc45621b0604195194066ce6b14ab6486284ea93292faa6fbd3f6acdca`.
It had no embedded worker-contract digest and was not constructed or evaluated.
The unguarded T3 initialization clone is recovered as `stage1-token-s6.pt`; its
raw normalizer contains mean/std fields and no guard/dead fields. The actor-only
learned BC checkpoint is recovered as `t2/stage1-token-s6.bc.pt`. No completed
separate T2m guarded critic clone was recovered. The separate T2m retrieval does
contain nine guarded actor-only BC checkpoints with their original normalizers;
seed 6 is an executable independent branch, while completed-clone critic
provenance remains an unresolved user-facing slot. A prospective guard-only
composition is also explicit rather than masquerading as recovered history.
All 15 T3 branches have recovered `latest.pt`,
`best.pt` and resume artifacts in their verified run directories. `latest.pt`
is the evaluated fixed-budget endpoint; `best.pt` remains selection evidence
only.

A read-only search of the currently inventoried GCS dataset found no object
matching `T3per`, `T3def`, `T3lo` or `stage1-token-s6` (including the searched
case variants). This prior-session report means only that no named matches were
reported in that searched scope. It proves neither archive completeness nor
historical nonexistence and is no longer the source of the recovered endpoints.

The recovered bundle originated from the previously inventoried disk
`projects/anthropocene-508912/zones/us-central1-c/disks/anthropocene-trainer-60`,
disk ID `744315408768270797`, 100 GiB `pd-balanced`, status `READY`, attached to
the terminated `anthropocene-trainer-60` instance and created
`2026-10-19T09:02:43.341-07:00`. No disk snapshot, attachment change or mutation
was performed here. Those cloud metadata remain discovery provenance. The
current proposal uses only the local hash-verified retrieval and does not access
that disk.

An archived prior producer-log candidate is named
`gs://anthropocene-rl-datasets-508912/model9/local-archive/model9-data/logs10/train-2.txt`
(generation `1792495448019703`, size 8,736, MD5
`RZWk4BrJbMZC11DkmD+U8Q==`, created `2026-10-20T11:24:08Z`), with empty
`train-2.err` (generation `1792495447816742`, size 0, MD5
`1B2M2Y8AsgTpgAmY7PhCfg==`). Its reported 600 training worlds, 60 development
worlds, 12 behavioural-cloning epochs, four critic rounds and final critic-dev
fit statistics can support only the historical optimization-target-fit estimand
after retrieval and hashing. It lacks an independently identified continuation
return and cannot by itself establish Gate 3 critic course-correction.

For the prior, clone and every T3 run, retrieval records the following distinct
slots. A single file may satisfy multiple slots only when its raw schema proves
the relevant byte sections are embedded, and each section remains named in the
inventory.

| Retrieval slot | Required contents | Missingness consequence |
| --- | --- | --- |
| `evaluation-checkpoint` | Exact actor and normalizer state, plus critic state when the historical contract declares one, used for the primary endpoint (`stage1-seed2.pt`, the unguarded T3 initialization clone, the T2m seed-6 guarded actor-only BC phase, or T3 `latest.pt`), with raw schema and contract. | Required. Missing or ineligible expected bytes block Gate 1. A content-backed historically absent critic is explicit `not-applicable`; an expected-but-lost critic is `missing`. The prospective guard-only composition references its two immutable sources and is not a checkpoint. |
| `selection-checkpoint` | T3 `best.pt`, selection metadata and the selection criterion/seed-family binding. | Supporting unless the reviewed graph evaluates or claims the selected endpoint. Absence forbids `best.pt`/selection conclusions but does not substitute it for or invalidate a complete `latest.pt` endpoint. |
| `resume-state` | Optimizer state, scheduler/scaler state, update counter, Python/NumPy/Torch/simulator RNG state and dataloader/order state, whether embedded or sidecar. | Not required for a fresh Gate 1 endpoint rollout. Absence blocks exact optimizer-state and update-sequence reconstruction from the checkpoint alone. Complete surviving contemporaneous transition/update traces may still support a separately reviewed, estimand-specific historical diagnosis; they do not recreate missing optimizer bytes. |
| `run-manifest` | Completion manifest, artifact inventory, hashes, arm/run/seed identity, parent/init checkpoint and done status. | Required for Gate 1 unless equivalent contemporaneous producer evidence independently establishes every field. |
| `producer-artifact-inventory` | A complete content-addressed listing of every file named by the contemporaneous producer/run record, including failed, partial and unexpected files; each entry retains source path, local path, size, modification metadata when available and post-retrieval SHA-256. | Required for the retrieval candidate. A favorable subset cannot establish endpoint completeness. |
| `args-config` | Original command/arguments, resolved configuration, environment flags, personality/arm settings, horizon, worker count and seed roles. | Required to establish historical semantics and producer compatibility; missing fields block the affected Gate 1 node/pane. |
| `producer-build-provenance` | Source commit/tree status, package lock/runtime, simulator and bridge build/hash, worker contract, host/job identity and launch receipt. | Required. Missing or incompatible producer semantics block imported or historical Gate 1 evidence unless a reviewed comparator proves compatibility. |
| `normalizer` | Exact core/set moments, guarded fields/dead masks and schema; embedded state is indexed separately by key and byte hash. | Required for evaluation. No synthesis, coercion or default is allowed. |
| `critic` | Exact critic tensors/schema and initialization/continuation parent evidence when the historical contract declares a critic. | Required for Gate 1 critic identity/provenance when expected. Proven historical absence is `not-applicable`; expected-but-lost bytes are `missing` and block the affected claim. Missing optimizer state blocks exact reconstruction, while complete contemporaneous target/advantage/update traces may support only the historical estimands they directly bind. |
| `adapter-default-migration` | Contract-8 adapter implementation/source/protocol, historical source identity, default-resolution record and any actual schema-migration receipt. | Required for every applicable legacy/reconstructed path. Absence blocks Gate 1; current `upgrade()` is not a fallback. |
| `training-logs` | Stdout/stderr, structured progress/update rows, checkpoint-write events, failures/restarts and resource logs. | Preserve when present. If they are the only producer/run/parent evidence, absence blocks that claim. Otherwise absence is explicit supporting missingness: it blocks exact log-dependent reconstruction, but complete independently sealed transition/update traces may still support separately reviewed estimand-specific diagnostics. |
| `evaluation-raw` | Original per-world evaluation rows, world/cohort manifests, snapshots, action/trajectory sidecars and command/output receipts. | Required only for imported historical outcome claims or when referenced by the reviewed graph. Absence does not become a fresh result; it blocks that historical claim. |
| `sweep-raw` | Original personality/focal/all-village sweep inputs, rows, selection reports and comparison outputs. | Required for any historical sweep, personality-separation or selection claim. Otherwise it remains explicit supporting evidence and cannot be summarized as if recovered. |

Retrieval also inventories every file in the producer's declared artifact list,
including unexpected and failed/partial files. It does not cherry-pick only the
expected names. Duplicate filenames from different run roots retain distinct
path and run identities. Any conflict among checkpoint metadata, manifest,
arguments, logs or evaluation outputs is a Gate 1 `HOLD` pending review, not an
opportunity to prefer the convenient source.

### 2.2 Executable retrieval preflight

`anthropocene_rl.autonomous_v7.gate1_preflight` implements the local, read-only
schema-1 retrieval candidate check:

```text
python -m anthropocene_rl.autonomous_v7.initialization_lineage preflight --inventory gate1-retrieval-inventory.json --out gate1-preflight-report.json
```

The historical-endpoint preflight inventory has exactly 17 records:
`prior-reference`, an explicit `guarded-clone` slot and the 15 lowercase T3
node IDs in section 2.1. That earlier current-state inventory keeps the completed
guarded-clone slot `incomplete-unresolved`; the recovered unguarded
`stage1-token-s6.pt` may not fill it. The production freeze builder additionally
binds the later T2m retrieval and records 20 executable nodes: the prior,
learned BC phase, unguarded T3 initialization, independent T2m seed-6 guarded
actor-only BC phase, prospective guard-only composition and 15 T3 endpoints.
The completed guarded critic clone remains visibly unresolved. It also has an
explicit conversion record. A conversion may currently be `missing` or
content-backed `not-applicable`; a present conversion intentionally fails until
its actual reviewed node/edge schema is known. Every endpoint carries the exact
ordered slot registry above plus `transition-update-trace`. File records bind
local bytes, SHA-256, size and mandatory observed local modification metadata.
The original path and every ancestor are inspected before resolution, so a
symlink, junction or other reparse alias is not normalized away. Missing source
modification metadata requires a specific unavailability reason in both the
reference and acquisition receipt. For `pre-existing-local`, source modification
time must instead equal the observed `st_mtime_ns` exactly and an unavailability
reason is forbidden. Unrecovered nodes use
a null producer run ID rather than an invented identity and remain `HOLD` until
producer-bound evidence is available. Sensitive embedded slots have exact source
keys and semantic schemas, and typed values use length-delimited recursive
hashing so nesting and container types cannot collide. They separately bind
the acquisition source URI/path, immutable generation, source size/time/ETag,
read-only method and a receipt hash for copied or downloaded bytes. The producer
inventory must contain and rehash exactly one schema-1 acquisition receipt that
binds the same source metadata plus the exact node, slot, artifact role, local
path, destination hash/size, method, acquisition time and tool identity. An
identical local byte stream cannot substitute for the downloaded slot reference.
The producer-artifact inventory must contain every complete file reference bound
by another present slot; cross-slot aliases, duplicate, substituted or omitted
files fail closed. When that inventory was remotely acquired, its own artifact
record must carry a separately validated slot-specific acquisition receipt. An
otherwise valid receipt attached to another copy or slot cannot authorize the
producer inventory.

The preflight inspects checkpoints weights-only without upgrade or defaults,
validates the economic trace's manifest/completion hashes/Merkle root and
transition/update schemas. A trace must bind the exact checkpoint, endpoint
actor, normalizer, run-manifest, producer-provenance and raw transition/update
hashes in both its own manifest and the retrieval slot. Structural byte/identity
consistency is reported separately from historical provenance; it cannot make a
historical diagnostic answerable while authoritative completeness/provenance is
unreviewed. Every update input batch must consist of transitions produced by an
evidenced behavior-policy actor/version no later than that update's before-policy
version. Legitimate PPO rollout reuse by later updates is allowed; transitions
from a future or the update-after policy are rejected. The final completed
event in raw updates-file order—not a maximum-version surrogate—must bind its
after-actor to the endpoint. Starts must precede their completions, version
deltas are limited to zero or one, and adjacent before/after versions and hash
maps must match exactly. No transition produced by the final endpoint policy is
required. Matching run, case-sensitive arm and optimizer-seed strings alone are
insufficient. Its report separates:

- endpoint artifact completeness;
- critic-state identity or content-backed historical absence;
- structural inventory of expected resume sections, without an exact
  continuation claim;
- original-training transition and optimization-target critic-estimand
  answerability;
- independently identified continuation-return critic accuracy, which remains
  blocked without separately frozen return evidence; and
- exact optimizer mutation/refit answerability from paired update boundaries
  with matching batch rows/digest, policy versions and actor/critic/optimizer
  pre/post hashes.

Missing resume bytes block exact endpoint continuation but do not erase a
complete contemporaneous transition/update estimand. A structurally plausible
resume dictionary is only `resume_sections_inventoried`: exact continuation
remains blocked until a separately reviewed historical restoration contract
binds supported optimizer names/parameter-group IDs, RNG algorithms and state
schemas, data-order, scheduler/scaler and simulator-state identities.
Conversely, a trace does not reconstruct missing optimizer state, and a resume
inventory does not reconstruct the ordered optimizer states, batches and RNG
chain that produced that endpoint. Expected-but-lost critic bytes block endpoint
completeness and the protocol-freeze readiness flag. A caller's `complete=true`
or hashed `not-applicable` explanation does not prove completeness or historical
nonexistence; those authorities remain unverified and blocking until separately
reviewed provenance is bound. The report always sets
`evaluation_authorized=false`, `gate1_readiness_claimed=false` and
`production_gate1=HOLD`. `PASS-PREFLIGHT-ONLY` is reserved for a future report
with separately reviewed completeness and nonexistence authorities and is never
an evaluator or Gate 1 pass; self-authored schema-1 fixtures remain `HOLD` and
`ready_for_protocol_freeze=false`. Even a complete preflight only becomes a
candidate input to state-4 independent review; strict
loader/adapter/producer-comparator/evaluation eligibility remains blocked until
the reviewed production protocols and manifest bind it. This executable layer
is a new post-acceptance6 implementation delta and requires Astra review before
use in state 2 or later.

### 2.3 Portable proposal and VM materialization

`anthropocene_rl.autonomous_v7.gate1_freeze_candidate` is the executable builder
for the fresh common-producer proposal. The local stage performs only byte
inspection, seed-use scanning, proposal construction and validation. It binds
the verified retrieval, pinned contract-8 sources, nonempty singleton/batched
corpus, actor-only BC composite, corrected 20-node DAG (four recovered
initialization surfaces, one prospective guard-only composition and 15 T3
endpoints), cohort, RNG schedules,
verification profiles, executing-source hashes, current bridge dependency set
and exact post-review command inventory. It does not construct a bridge, reset a
world, evaluate a checkpoint or collect a treatment benchmark.

From the RL worktree, the current Windows staging command is:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
& 'C:\Users\suyas\Documents\ChatGPT\worktrees\elder-model9-native-rl\.venv\Scripts\python.exe' `
  -m anthropocene_rl.autonomous_v7.gate1_freeze_candidate build `
  --rl-root 'C:\Users\suyas\Documents\ChatGPT\worktrees\r3-build-rl' `
  --retrieval 'C:\Users\suyas\Documents\ChatGPT\worktrees\model9-data\gate1-vm-retrieval' `
  --t2m-retrieval 'C:\Users\suyas\Documents\ChatGPT\worktrees\model9-data\gate1-t2m-retrieval' `
  --evidence-package 'C:\Users\suyas\Documents\ChatGPT\worktrees\model9-data\evidence\model9-gate1-vm-retrieval-analysis-v2' `
  --evidence-review 'C:\Users\suyas\Documents\Codex\2026-10-22\files-pasted-by-the-user-you\work\gate1-recovered-evidence-v2-root-review.json' `
  --expert-expansion-evidence 'C:\Users\suyas\Documents\ChatGPT\worktrees\model9-data\evidence\model9-gate1-expert-expansion-equivalence' `
  --output 'C:\Users\suyas\Documents\ChatGPT\worktrees\r3-build-rl\evidence\model9-gate1-freeze-candidate-final2-20261023' `
  --seed-start 50000
```

That result is deliberately `local-portable-staging` and cannot execute a
checkpoint. Transfer the reviewed source bundle, retrieval and portable
candidate to the VM without changing their bytes. On the VM, run
`materialize-vm` with the actual absolute VM source/retrieval/output roots. The
command refuses a nonempty output directory and rebinds Python/.NET runtime,
package versions, executing source bytes, bridge DLL/deps/runtimeconfig,
retrieval paths and every generated command. The content-addressed T2m archive
must be uploaded beside the T2m retrieval directory, matching the local layout;
materialization rejects a missing or changed archive:

```text
python -m anthropocene_rl.autonomous_v7.gate1_freeze_candidate materialize-vm \
  --portable-candidate ABSOLUTE_PORTABLE_ROOT/freeze-candidate.json \
  --rl-root ABSOLUTE_VM_RL_ROOT \
  --retrieval ABSOLUTE_VM_RETRIEVAL_ROOT \
  --t2m-retrieval ABSOLUTE_VM_T2M_RETRIEVAL_ROOT \
  --output ABSOLUTE_EMPTY_VM_CANDIDATE_ROOT
```

The four absolute VM roots are execution-location inputs, not values that may
be guessed on Windows. The materialized candidate remains
`HOLD-awaiting-astra-high`, `evaluation_authorized=false`; review must bind its
exact digest. Only after that review may the exact no-placeholder commands in
its sealed `execution-commands.json` be run, in order: candidate validation,
contract-8 comparator, guarded-BC pinned-source comparator, current-worker
capture and every present node/pane evaluation. The actor-only guarded node and
the prospective guard-only composition are executable only in their explicit
fresh-common-producer scopes; neither satisfies the unresolved completed-critic
clone slot.

Before resource calibration the command artifact permits one node/pane process.
It proposes, but does not authorize, up to eight concurrent node/pane processes
after paired incremental-RSS and throughput receipts are accepted on the exact
VM runtime. Each process writes one world immediately as its own sealed,
content-addressed artifact and releases it before the next world; it never
retains 64 world results. The reviewed canonical capture ceiling remains
256 MiB per world, so the conservative disk ceiling is 16 GiB per 64-world
node/pane and 640 GiB for all 40 node/pane outputs. Python object/RSS overhead
is not inferred from that canonical-byte ceiling. Running all 40 commands
concurrently is not authorized by this protocol, and the proposed eight-way
ceiling becomes immutable only in a reviewed post-calibration execution record.

## 3. Seed-family exclusions and evaluation cohort

The following known Round-3 roles are excluded from Gate 1 evaluation worlds:

- teacher training: 1000-1599;
- development: 2000-2059;
- PPO selection: 2060-2099;
- teacher development: 2100-2159;
- clone selection: 2160-2179;
- calibration: 10000-10099; and
- confirmation: 3000-3059, which remains sealed.

The executable freeze candidate proposes evaluation-only seeds `50000-50015`.
Its seed audit raw-byte scans every Git tracked/untracked nonignored RL file,
every byte in the verified 396-file VM retrieval and every corrected retrieval-
analysis file, while separately listing the three files that merely declare
this proposal. It also rejects every known train, development, selection,
calibration and confirmation family. No candidate checkpoint output informed
the choice. This is a proposal, not authority: the family remains prohibited
until Astra approves the exact audit, world expansion and candidate digest.

The proposal expands each seed over exact withheld-technology masks `0`
(`default`), `10` (`farming`), `6` (`fishing`) and `12` (`herding`), producing
64 stable world IDs. The same worlds apply to every endpoint. The candidate
binds exact seeded resets with no snapshot, reset turn 9, warm-up 0, eight players
and a 25-turn horizon. Action masks are authoritative observation fields rather
than caller-chosen constants: every raw mask is retained and validated against
the exact decoder receipt. Any mask substitution or missing receipt rejects.

Every applicable endpoint of a boundary uses the exact same frozen world
records. A world record binds seed, snapshot descriptor and hash (or explicit
no-snapshot), reset turn, players, warm-up length, horizon, withheld-family
status, cohort digest, environment, background, personality, action mode and
policy RNG schedule. At least two worlds are necessary for a paired interval,
but the final count is not inferred from that minimum; it is independently
reviewed and frozen.

## 4. Snapshots, warm-up and horizon

Snapshot bytes are immutable inputs, not paths by convention. Each snapshot
record binds its SHA-256, producer, schema, reset trajectory identity, reset
turn and expected opening boundary. The evaluator verifies those fields after
reset. Warm-up is an exact nonnegative count; the opening turn must equal the
reset turn plus warm-up under the frozen transition semantics. The retained
reset-to-opening chain proves continuity and prevents a duplicate restored
transition.

The executable candidate proposes a 25-turn horizon for every evaluation world,
matching the recovered historical evaluation horizon. Early termination is
allowed only through the reviewed terminal schema and a raw zero-population/no-
live-successor witness. Truncation reason, executed-step count and terminal
boundary are evidence, not summaries. The horizon remains unapproved until the
complete candidate receives Astra review.

Snapshot selection may not be changed to avoid a fission, extinction,
migration, compiler intervention or service result. If the reviewed cohort
requires transition envelopes absent from available snapshots, that is a
missing capability/evidence `HOLD`, not permission to choose easier worlds.

## 5. Panes and execution semantics

The evaluation/claim protocol freezes every complete pre-result node, edge,
pane and world record. Historical interpretation and fresh diagnostic execution
remain semantically separate:

- `historical-semantics` is required for every prescribed historical endpoint
  and boundary. It uses each node's bound historical flags, actor/decoder,
  normalizer and loader. Missing historical coverage blocks the historical-
  replay/provenance interpretation.
- `fresh-common-producer-diagnostic` evaluates every recovered actor under one
  exact current simulator/worker/runtime, identical frozen worlds and flags.
  A reviewed external producer binding is allowed because the recovered
  checkpoints do not embed historical worker contracts. It preserves
  `historical_producer_status=unresolved` and cannot establish historical
  replay, historical producer equivalence, original BC-normalizer behavior,
  selection, readiness or a historical provenance pass.
- `standardized-flags-intervention` remains an available additive kind but is
  not proposed for this collection.

These are separately classified scopes. Complete required evidence for all 20
nodes in the fresh-current pane can receive a scoped diagnostic `PASS` and
support current-diagnostic work in Gates 2-4 while the historical replay/
provenance claim remains `HOLD`. This is not a waiver of any current-pane node,
world, metric, comparator or verification requirement, and it is never a
training-readiness, selection or promotion decision.

The only currently supported background declaration is exactly
`{"kind":"none","actor_scope":"all-live-villages"}`: the checkpoint actor
controls all live villages. The global personality vector, observation
contract, action heads, learned movement and region-movement semantics,
environment flags, raw-material state and worker/producer contracts are frozen
per pane. Personality maps and unsupported background policies are rejected.

The executable candidate proposes two separate `fresh-common-producer-diagnostic` panes:
deterministic and seeded-stochastic. It proposes no standardized intervention,
because one is not needed to answer the recovered historical-boundary question
and must not substitute for historical semantics. The seeded schedule couples
only the common active heads (`mix`, `strategic`, `service`, `clothing`,
`shelter`); movement, fission, commitments and trade remain delegated/inactive
for these recovered stages. These are reviewed proposal values, not defaults by
omission, and remain prohibited before Astra approval.

Each T3 initialization-to-endpoint boundary also contains a reconstructed
parameter-expansion substep: pinned PPO constructs `Actor(experts=True)` from
the `experts=False` initializer before updates. That is an architecture
expansion, not a decoder flag. The final adapter layer of the expert module is
expected to be zero at entry, so behavior preservation is an evidence-bearing
pinned-source/weight claim, not an assumption. The portable candidate binds the
recovered-source audit across all optimizer seeds 70-84 and fixed batch-1/24
teacher-state/mask cases; copied tensors are bit-exact, only `experts.*` is
missing, final expert adapters are exactly zero, and original/expanded forward
outputs and decoded modes match on that corpus. The 20-node rollout cohort does
not add a synthetic expansion node. Closed-loop endpoint interpretation still
treats the boundary as expansion plus PPO training—never as a pure-gradient or
pure-PPO causal effect.

## 6. Deterministic and seeded-stochastic schedules

Deterministic execution binds the exact decoder implementation, runtime,
dtype/device, active/delegated heads, masks, forced region selection and output
contract. Each action retains a forward decoder receipt from the exact raw
float32 head values to the emitted bridge action.

For every seeded-stochastic node/world/step, the reviewed protocol additionally
freezes:

- RNG implementation, algorithm/version, device and state serialization;
- seed and initial-state digest;
- exact active-head draw order and draw count;
- masks and per-head distribution/decoder semantics;
- post-state digest derived from the actual action-bearing rows; and
- whether source/destination schedules are coupled.

Coupling is permitted only when worker contracts, active heads and draw
semantics are compatible. Nodes with different active heads receive distinct
reviewed schedules or are explicitly incompatible; a pane-wide generic draw
order is insufficient. Execution uses a local generator and proves that global
Python, NumPy and Torch RNG states are bitwise unchanged on success and failure.

## 7. Equivalence claims

Each reviewed edge explicitly states whether an equivalence claim is
applicable. Absence or substitution under an old lineage/evaluation review is
invalid. A claim freezes before results:

- kind: exact tensor preservation, mapped-distribution numerical equivalence,
  deterministic behavioral equivalence or stochastic behavioral equivalence;
- exact source/destination checkpoint and inference implementation identities;
- executor/comparator implementations and runtime/dtype/device;
- canonical corpus cases and input hashes;
- mapped output scopes, actor/normalizer/decoder/head semantics;
- exact per-case/per-scope output contracts: dtype, shape, ordered component
  labels, boolean mask schema/shape and decoded-action schema;
- appended-output disposition;
- absolute and relative tolerances; and
- for stochastic claims, the exact RNG schedule and coverage.

The corpus, cases, output contracts and tolerances are unresolved until the
historical artifacts and applicable edges are known. Astra freezes them at
state 5. Exact tensor preservation uses literal contiguous tensor bytes after
dtype/shape checks. Numerical/behavioral claims retain complete source and
destination raw outputs and execution receipts and recompute errors, masks and
actions. Summary booleans or max-error assertions are not evidence.

## 8. Mandatory diagnostic evidence

Every applicable node/world report carries all mandatory metrics. A partial
report blocks classification even when an earlier edge already changed:

- welfare;
- authoritative population-years;
- ending residents;
- authoritative QoL-years;
- realised trade utility;
- extinctions and settlements;
- the complete registered service-outcome object;
- compiler interventions; and
- canonical trajectory digest.

All headline values are recomputed from a complete typed raw trajectory. The
trajectory contains the reset record, warm-up chain, opening policy observation
and fingerprint, contiguous transition IDs, source/action/successor evidence,
rewards/outcomes, terminal/truncation receipt, economic rows, durations and
final boundaries. Imported and live reports use the same validator.

The exact 41-field service registry preserves cultivation and herding food,
eleven named manufacturing goods, Stone/Clay/Wood material receipts, settled
trade, realised utility and imported/exported quantities for eleven named trade
goods. Each field carries its registered native unit and stable identity. Zero
is observed zero. Null requires a nonempty availability reason and keeps
production evidence on `HOLD`. Compiler interventions retain raw camp/turn/
domain/reason rows and deterministic named totals; services are never collapsed
to a mixed-unit scalar.

Population-years and QoL-years use authoritative transition duration and turn
boundaries. The canonical trajectory digest includes the frozen semantic
state/action/reward/terminal/economic surface. Semantically unordered entity
collections are identity-sorted; temporal and action-component order remain
sensitive.

Gate 1 must also preserve critic identity and provenance: critic state hash and
schema, initialization/continuation edge, producer/run evidence and whether a
critic is absent. Independent critic calibration, target construction and
historical update-causality claims belong to Gate 3 and cannot be invented here.
If Astra requires an additional Gate 1 critic-forward diagnostic, its exact
corpus and metric are an unresolved design amendment and block state 5.

## 9. Loader and producer eligibility

Raw metadata is inspected before construction. Current-contract actors and
normalizers use exact strict raw-key, shape, dtype and device coverage. The
pinned contract-8 path requires its reviewed adapter protocol, source bytes,
input mapping, actor/normalizer schema, default-resolution record, migration
record, decoder and output corpus. No path may invoke current `upgrade()`, load
hooks that synthesize absent tensors or unreviewed defaults.

Historical worker-to-pane compatibility uses the same semantic comparator for
live and imported evidence. Comparator acceptance requires a reviewed protocol,
implementation identity, corpus, raw reset/configuration/action/observation/
reward/state trajectories and execution receipts. A matching self-asserted
digest or booleans are insufficient. Producer simulator, bridge SHA/flags and
raw-material state must equal the frozen worker contract.

## 10. Verification freeze

The verification protocol freezes exact executable/runtime identities,
commands, test IDs, expected zero exits, coverage denominator/scope/exclusions/
tool identity, diagnostic-off parity profiles, benchmark fixtures, warm-ups,
repetitions, ordering and resource method. `python -c pass` and command
substitution are invalid.

The required test-command set is:

```text
python -m pytest -q tests/test_initialization_lineage.py
python -m pytest -q tests/test_gate1_preflight.py
python -m pytest -q tests/test_gate1_diagnostics.py tests/test_gate1_legacy_contract8.py
python -m pytest -q tests/test_autonomous_v7_worker_contracts.py tests/test_autonomous_v7_ppo.py
python -m pytest -q tests/test_autonomous_v7_bridge.py tests/test_economic_trace.py tests/test_economic_diagnostics.py
python -m pytest -q tests/test_autonomous_v7_round3_contracts.py tests/test_autonomous_v7_round3_protocol.py
python -m py_compile src/anthropocene_rl/autonomous_v7/initialization_lineage.py src/anthropocene_rl/autonomous_v7/evaluate.py src/anthropocene_rl/autonomous_v7/gate1_diagnostics.py src/anthropocene_rl/autonomous_v7/gate1_preflight.py src/anthropocene_rl/autonomous_v7/legacy_contract8.py
python -m ruff check --select F,E9 src/anthropocene_rl/autonomous_v7/initialization_lineage.py src/anthropocene_rl/autonomous_v7/evaluate.py src/anthropocene_rl/autonomous_v7/gate1_diagnostics.py src/anthropocene_rl/autonomous_v7/gate1_preflight.py src/anthropocene_rl/autonomous_v7/legacy_contract8.py tests/test_initialization_lineage.py tests/test_gate1_diagnostics.py tests/test_gate1_preflight.py tests/test_gate1_legacy_contract8.py
git diff --check
```

Before state 5, the candidate replaces `python` and `git` with frozen executable
identities and records platform/runtime/library versions. If a listed test path
does not exist at freeze time, Astra must approve the exact replacement; it may
not be silently dropped. The exact coverage minimum is not supplied by the
approved architecture and is therefore unresolved. Astra must freeze the
minimum percentage, executable-line denominator, source-file scope, exclusions
and coverage tool/version before evidence collection. Raw test output, coverage
data and coverage report are independently hashed; a scalar percentage is not
enough.

## 11. Relative performance and memory protocol

The reviewed budgets are relative, not arbitrary absolute numbers:

| Profile | Frozen acceptance rule |
| --- | --- |
| Diagnostic-off parity/current fixture | Bitwise identical observations, actions, rewards and trajectory fingerprints; median throughput ratio `diagnostic-off / baseline >= 0.98`. |
| Diagnostic-on/current Model9 fixture | Median step-time ratio `diagnostic-on / baseline <= 1.15`. |
| Diagnostic-on/mature-population fixture | Median step-time ratio `diagnostic-on / baseline <= 1.20`. |
| Eight-worker end-to-end RL | Median transitions/sec ratio `diagnostic-on / baseline >= 0.80`. |
| Worker memory | `peak worker RSS with diagnostics - paired baseline peak worker RSS <= 256 MiB`. |

Each profile uses the same machine identity, power mode, operating-system and
runtime/library versions, worker count, fixture/snapshot bytes, seeds, horizon,
process topology and work units for baseline and diagnostic samples. Baseline
and treatment acceptance measurements are newly collected as a pair in each
repetition and follow a preregistered alternating/interleaved order so drift
cannot be assigned after results. Warm-up count, measured repetition count and
order seed are unresolved and must be frozen by Astra before collection.

Raw receipts retain, for every pair: repetition ID; order; start/end monotonic
times; elapsed time; completed work; worker IDs; baseline and treatment peak
worker RSS; exit status; machine/runtime identity; and fixture hash. The
validator recomputes each per-pair throughput or step-time ratio and the median
of those per-pair ratios. It also recomputes every paired worker-RSS delta and
its maximum. The relative ratios in the table above and maximum incremental RSS
of 256 MiB are the governing acceptance criteria. Total process working set is
not a substitute for incremental peak worker RSS, and optimizer identities are
never pooled.

### 11.1 Baseline-only calibration is not acceptance evidence

Before state 5, the limited reviewed calibration subprotocol may collect only
baseline measurements. The freeze candidate records the calibration raw-report
hash, median baseline value and optional derived absolute operational guardrail:

- throughput minimum = baseline median throughput multiplied by 0.98 or 0.80;
- step-time maximum = baseline median step time multiplied by 1.15 or 1.20; and
- memory operational ceiling = calibration baseline peak worker RSS plus
  256 MiB.

The validator recomputes each guardrail from the hashed calibration baseline
and reviewed relative rule. These absolute values are pre-run abort/diagnostic
guardrails only. A calibration receipt is never an acceptance baseline, is
never paired with a later treatment run, and cannot make a failed relative
criterion pass. Passing an absolute guardrail is not acceptance evidence. If a
guardrail aborts final collection, the required pair is incomplete and the gate
remains `HOLD`.

No treatment measurement may exist before these values and the final pairing
protocol become immutable at state 5. A machine/fixture/runtime change
invalidates calibration and requires a new baseline-only calibration and Astra
freeze review. A treatment result from an invalidated attempt remains in the
audit inventory and must not be used to choose a replacement threshold.

### 11.2 Planned production verification schema extension

The schema-1 verification protocol must replace the current generic absolute
benchmark shape with a reviewed relative-budget row containing, at minimum:

- benchmark ID, scenario and content-addressed fixture/snapshot references;
- exact machine/runtime policy, workers, work unit, warm-ups, repetitions and
  alternating/interleaved pair order;
- metric kind (`throughput-ratio` or `step-time-ratio`), direction, relative
  limit and aggregation exactly `median-of-per-pair-ratios`;
- memory method exactly `paired-incremental-peak-worker-rss`, unit `MiB`,
  aggregation exactly `maximum-of-per-pair-deltas` and limit 256; and
- a distinct optional calibration object containing only its baseline raw
  report, recomputed median and derived operational guardrail.

Final evidence contains the exact reviewed number of unique pair receipts. Each
pair has separate content-addressed baseline and treatment raw reports plus its
preregistered order, elapsed time, work count and per-worker peak RSS. The
validator rejects a calibration ref in either acceptance side, reuse of one run
in multiple pairs, missing/extra repetitions, machine/fixture mismatch,
scalar-only summaries, changed ordering or any post-result threshold mutation.
It derives ratios and RSS deltas directly from raw pair members and applies only
the governing criteria above.

Production collection remains `HOLD` until this schema extension and exact
profile values have Astra approval. Relying on the current generic absolute
`throughput.minimum` / `memory.maximum` fixture schema is invalid.

## 12. Packet and reconstruction requirements

The packet root inventory is exact. Its immutable CAS includes the canonical
manifest; lineage, evaluation/claim, equivalence, comparator and verification
protocols and independent reviews; every checkpoint/source/snapshot/corpus/
schema/output-contract hash; adapter and implementation source bytes;
evaluation reports and raw trajectories; test/coverage/parity/benchmark/memory
reports; implementation identity; and all nested dependencies.

Reconstruction validates every CAS filename, regular-file status and byte hash
before interpretation, then resolves all references from the packet CAS only.
It does not fall back to readable external originals. Validation,
classification, aggregation and decision are recomputed from preserved raw
inputs, not from a serialized derived packet body. Missing, extra, substituted,
corrupt or externally resolved blobs fail closed.

## 13. Unresolved decisions that block collection

All of the following require explicit Astra review and must be resolved before
state 5:

1. Astra approval of the proposed evaluation-only `50000-50015` family, 64
   exact seed/mask world IDs and seed-scan scope;
2. Astra approval of the proposed no-snapshot reset, turn 9, warm-up 0 and
   25-turn horizon;
3. Astra approval of the proposed deterministic plus compatible seeded-
   stochastic historical panes and empty standardized-intervention registry;
4. review of the populated 396-file retrieval hashes/raw schemas and corrected
   DAG, plus explicit blocking treatment of the unrecovered completed T2m
   guarded critic checkpoint and still-unresolved historical producer identities;
5. equivalence-applicable edges, corpus cases, output contracts and tolerances;
6. whether Gate 1 needs a critic-forward diagnostic beyond critic provenance;
7. exact verification executable identities and any missing test-path
   replacements;
8. exact coverage minimum, denominator, scope, exclusions and tool identity;
9. performance warm-ups, repetitions, pair order and frozen fixture hashes;
10. baseline-only calibration raw hashes, optional derived operational
    guardrails and the reviewed final acceptance-pair schema; and
11. independent reviews of every populated schema-1 proposal and this new
    implementation-identity source-list delta.

Any unresolved item, missing required artifact, unavailable mandatory service,
unsupported loader, incomplete trajectory, incompatible producer, absent raw
verification receipt or failed content hash leaves the production decision at
`HOLD`. Nothing in this procedure implies Round 4 readiness.
