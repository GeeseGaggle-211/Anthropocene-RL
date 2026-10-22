# Diagnostic implementation status

This implements a prospective diagnostic path for the existing model9 learner.
It does not implement section 8 of the approved observability architecture and
does not change the existing objectives, personality, observations or rewards.

## Interfaces

The bridge advertises diagnostic envelope 3 and economic schema 1. Ordinary
steps retain envelope 2. Reset has envelope 0. Opt in with
`Model9Bridge(economic_diagnostics=True)` or PPO's `--economic-trace DIRECTORY`.
The directory must be new. The existing execution-authority checks still apply.

`economicDiagnostics` is keyed by camp. Each record binds trajectory, turn,
camp/society and transition IDs and contains population, cultivation, herding,
manufacturing and trade domains, stable goods/material rows, herd IDs, field
capital and typed transfers. Availability and unit paths use
`domains.<domain>.<field>` and indexed paths for row collections. Restored
observations retain the preceding transition ID and are labelled explicitly.
New branch steps receive new trajectory IDs and camp-specific parent IDs.

Capture switches belong to the environment's transient world and resolution
metrics. Model9 Reset/VerifyReplay never changes a process-global capture flag.
Sidecars are absent from authoritative saves, tensors, teacher inputs and reward
functions. Capture indexes turn receipts once by camp and trade transaction;
bounded stable-good loops do not scan the map or historical records.

The learner writes contemporaneous transitions before optimizer mutation and
start/complete records around minibatch and critic-refit mutations. Update rows
bind ordered transition IDs and their digest, rollout identity, policy versions,
checkpoint hashes, PPO ratios/clipping/KL, losses, pre-clipping gradient norms,
clipping scales and parameter-update norms. The writer streams synchronously,
limits captured rollout data to 256 MiB and does not drop records. Reset
snapshots are content-addressed in `snapshots/`. Completion binds stream counts,
SHA-256 values, snapshot hashes and the ordered transition-then-update Merkle
root. Failed traces do not receive `done.json`.

The deterministic aggregator validates the trace manifest, transition stream,
update stream, snapshots and completion hashes before writing its report. In
addition to the flattened decision, domain, calibration, action-alignment,
matched-fork, paired-arm and trade-latency tables, it writes three optimizer
views: the raw validated update records, paired mutation records, and a
per-batch summary joining critic targets/residuals to gradient clipping,
parameter-update norms and before/after hashes. Mandatory independent slices
cover optimizer/world seed, capability regime, personality, population, QoL,
turn and intervention status. Values used only as finite encodings for
unbounded planner returns are excluded from statistics.

## Boundaries and limitations

Population reports retain linked-population semantics. Resident and party
snapshots are separate before economy/demography, after economy and after
reintegration; dispatch/reintegration and close-phase migration/fission have
typed witnesses. Integer carry remains explicitly unavailable as a separately
recorded event. Both committed-linked and gross-event residuals are exposed.

Field opening/mean/ending capital is captured from the corrected authoritative
horizon allocation. Construction-versus-decay components are unavailable.
Herd reconciliation is scoped explicitly to managed-report herd IDs, beginning
after resource allocation; new ownership/acquisition groups are outside that
scope. Fission herd witnesses carry parent and child herd IDs. Nutrition uses
starting-animal-weighted denominators. Unbounded planner returns use a finite
zero encoding plus a positive/negative-unbounded bound-state flag; the zero is
not an observed return and must not enter return statistics.

Opening/closing goods and raw-material stocks and actual recipe consumption
are retained. Housing's inventory row is explicitly legacy carried housing and
excludes fixed housing. Complete food/material/goods/field conservation remains
unavailable until all service wear, consumption, discard and capital/ownership
witnesses are instrumented. These residuals are null, not asserted zero.
Phase-A raw extraction cannot be compared to Phase-B material authorization;
such mixed rows have null requested/granted totals with a reason and separately
named Phase-B amounts.

Compiler request phases retain native demand before action capping, capped
requests, grants and request bundle conversion data. Compiler-control units
are distinct from physical-good units. Current receipts are not yet attributed
to individual requests with opening-stock/intermediate netting, so final-service
fulfilment and cross-unit control-to-physical stages remain explicitly unavailable.
Production-limit attempts, full recipe configuration, human food displaced by
herds and fixed-housing/field transfer closure remain implementation gaps.

Settled TradeRecord quantities already include delivery fractions; multiplying
them again would double-discount. Actual imports/exports and utilities are
retained. Per-camp handshake phases, escrow/refund/storage-rejection amounts and
original requisition/agreements are not present in final TradeRecord and remain
unavailable in this implementation.

The trace contains joint action log probability and distribution tensors.
Per-head log probabilities/entropy and a fixed audit minibatch are not computed
by the existing APIs and are explicitly unavailable. Independent continuations,
preregistered matched forks, frozen population/QoL quintiles and latency evidence
have not been collected. Historical T3 optimizer attribution remains unavailable
without its original contemporaneous traces. Reconstructed data must not be
labelled historical training evidence.

## Validation status

Focused tests cover capture-off/on fingerprint, actor/reward/teacher leakage,
schema/order, herd IDs, linked-population reconciliation, replay tampering,
fork copy/continuation, scalar/worker parity, parser rejection, zero denominators,
writer failure/completion integrity and actual pre-update targets/mutations.
The focused real-bridge replay and the broader autonomous-v7 regression set
pass from the clean simulator commit. The local three-seed prospective cohort
is intentionally deferred until this learner branch is committed and clean,
because the provenance guard rejects a dirty learner before optimization.
The approved throughput/RSS benchmark gates have not been measured; this build
does not establish VM collection readiness. No VM was started and no training
comparison, arm selection or confirmation-seed access was performed.
