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

Each transition also retains the already validated authoritative action
diagnostic masks and a separate `actorPolicyActiveMask` computed from the exact
stage and commitment decision. Action alignment requires both masks: delegated
native movement, disabled heads and capital controls re-emitted during a
commitment hold are excluded from actor evidence. Exclusion-only policy rows are
retained so native delegation is visible rather than silently absent. The report
derives a head-specific intervention numerator from comparable
requested-to-accepted or accepted-to-realised changes. Continuous actions use
Pearson correlation; route, movement, fission, commitment and trade-gate choices
are categorical and are reported as category-mean differences. Constraint reasons
are attached only where a direct economic-funnel mapping exists. Conservation
coverage is emitted separately from non-zero residuals. Every standalone summary
CSV carries the execution-authority cohort classification, arm-selection and
readiness permissions, plus an explicitly descriptive sample-size flag, so an
extracted local table cannot appear experimentally sufficient.

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
pass from the clean simulator commit.

A non-qualifying local prospective cohort was first run from learner commit
`61180a4`, repeated after the active-head report repair at `7a2915c`, and
regenerated from `d6308b4` after adding the actor policy/stage activity mask.
Every pass used simulator commit `e5fb626` and the same identities: exactly
optimizer seeds 90, 91 and 92, one diagnostic arm, world seed 1000 and ten
turns. The final cohort produced 344 decision rows, 25 update rows and 11 joined
mutations. All 344 observe-only native-movement rows are now reported as
policy-inactive, with zero missing policy masks; the 11 commitment-hold rows
exclude their re-emitted capital controls. There were zero non-zero values among
1,032 available conservation checks, while 1,376 food/material/goods/field
residual values were explicitly unavailable. This cohort validates stability,
integrity and report generation only: it has one arm and one world seed,
initializes new vector-critic heads from a legacy scalar checkpoint, and cannot
select an arm, tune a treatment or establish readiness. Its critic calibration
is therefore an instrumentation exercise, not evidence about the historical T3
optimizer.

Astra High independently approved the final policy-mask and standalone-cohort
reporting semantics with no blocking defect. It retained a HOLD on critic-head
removal, personality-attention redesign and Round 4 because independent-return,
matched-fork, trade-latency, conservation-completion and performance evidence is
still missing.

The approved throughput/RSS benchmark gates have not been measured; this build
does not establish VM collection readiness. No VM was started and no training
comparison, arm selection or confirmation-seed access was performed.
