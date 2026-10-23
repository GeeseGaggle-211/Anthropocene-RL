# Model 9 Gate 2 implementation status

**Status (2026-10-23): local implementation candidate; production Gate 2 remains HOLD.**

This delta implements an opt-in reporting surface and deterministic parser for
Gate 2 causal accounting. It does not authorize VM use, evaluation, training,
arm selection, checkpoint promotion, or readiness. It has not received Astra
implementation review and contains no production evidence.

## Implemented diagnostic surface

When economic diagnostics are enabled, the native bridge emits schema 2 rows
that preserve stable camp, society, trajectory, transition, request,
transaction, leg, recipe, good, material, field, and herd identities. The
surface includes:

- the actor distribution/sample and the native-demand, actor-capped request,
  compiler grant, executed input/work claim, committed receipt, inventory
  change, final service, population, and QoL phases for active controls;
- exact native-unit craft-good and physical-material rows in stable registry
  order;
- mutation-site receipts for production, recipe inputs, raw-material
  extraction, authoritative final clothing service, farm/herd/fishing tool
  service, and wear/depreciation, plus explicit provisional service and export
  claims and storage-blocked potential;
- zero-inclusive opening and ending stock boundary receipts for every stable
  craft good and material, so absence of a mutation receipt is distinguishable
  from an omitted inventory identity;
- population, food, field, herd, Stone, Clay, Wood, and every craft-good
  conservation equations with their complete named terms; and
- eligibility/gating, partner search, requisition, rejection/counter/handshake,
  stock versus made-to-order supply, escrow, production, bilateral delivery,
  storage acceptance/block, refund/return, and immediate-utility records for
  trade.

The Python reader rejects unknown units or lifecycle states, incomplete stable
registries, unjoined or conflicting identities, forged conservation residuals,
physical-event/stock disagreement, and trade-part/conservation disagreement.
The same strict parser is used for live and imported schema 2 rows.

## Delayed trade bookkeeping and causal boundary

Accepted imported parts receive bounded trajectory-scoped lot identities.
Offline aggregation uses one documented, deterministic imported-first FIFO
allocation to relate those fungible lots to later recipe use, service, or wear.
It writes separate lot, allocation, and downstream population/QoL-window
tables. This is explicitly bookkeeping, not a claim about physical ownership or
causality. Partial delivery, storage block, refund, mixed local/imported stock,
later consumption, and unconsumed end-of-horizon lots remain visible without
double counting.

The native simulator does not persist provenance ownership in authoritative
inventory. Therefore this implementation binds lot lineage to the captured
evaluation trajectory only. It does not infer lineage across an unbound save or
unrecorded interval.

Immediate utility is observed. Later service, population, and QoL are
observational windows. A delayed *causal* utility/effect requires a
transaction-bound matched intervention with the same opening state and random
stream. No such evidence is synthesized. Every transaction whose delayed causal
effect is unavailable forces the Gate 2 accounting classification to `HOLD`;
a captured cohort with no bilateral transaction also remains `HOLD`. The
sidecar validator can classify a transaction-free row as
`PASS-ACCOUNTING-ONLY`, but the aggregate Gate 2 surface cannot pass until a
reviewed transaction-bound causal-evidence schema and matching evidence close
the terminal link. No current row is interpreted as a training/readiness claim.

## Opt-in and scaling properties

Diagnostics-off execution does not allocate or retain these rows and does not
change policy inputs, actions, rewards, simulator state, save state, or RNG.
Replay compares the complete diagnostic tree when capture is enabled.

Native capture indexes each authoritative receipt list once, then processes
active camps, stable goods/materials, active fields/herds, physical events, and
trade legs. Its intended bound is
`O(R + H + T + V*(G+M))` time and output-proportional memory, where `R` is
authoritative receipts, `H` active fields/herds, `T` trade records/legs, `V`
active camps, and `G+M` the fixed good/material registries. It performs no map
or historical-trajectory scan. Offline lot aggregation sorts retained rows and
events, then consumes each lot allocation once; its bound is
`O(E log E + L + A)` time and `O(L + A)` retained output for events `E`, lots
`L`, and allocations `A`.

No reviewed production throughput or incremental-RSS receipt exists yet. The
relative performance protocol and evidence requirements from the governing
architecture remain a production HOLD.

## Remaining HOLDs

- Astra review of this Gate 2 implementation is pending.
- Production cohort, snapshot, checkpoint, and trace evidence are absent.
- Transaction-bound matched-intervention evidence for delayed causal trade
  effects is absent. Observed downstream windows cannot replace it.
- The historical artifact retrieval and Gate 1 evidence dependencies remain
  unresolved as documented in the Gate 1 status and current-state package.
- Reviewed diagnostic-on/off throughput and incremental-memory acceptance
  receipts are absent.

These limitations are deliberate fail-closed boundaries, not unavailable
values that may be treated as zero.
