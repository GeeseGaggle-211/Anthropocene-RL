# Autonomous Elder institutional-history repair contract v6

The user authorizes proceeding with the correctness repair identified by both
Astra/high reviews. Work remains on feature/elder-autonomous-economy in the two
repositories. Preserve all baselines, worktrees, snapshots, raw evidence and
checkpoints. No main merge, GPU ecology or expensive PPO campaign is included.

## Versioned authority and persistence

Economic action3 is unchanged. Introduce observation5/environment5/replay5 and
learner model/checkpoint/inference5. The first761 learner features retain the
current context/mask/stock layout; append300 explicitly named history features,
giving1061 inputs. The obsolete supervisory receipt slots40:56 are zero in
observation5; actual autonomous actions are never encoded as ElderDirective.

At reset, initialize persistent Camp.ElderMemory only for controlled living
settlements that lack initialized memory. Preserve genuine existing memory,
including its physical timestamps and EMA maturity. A nil legacy snapshot starts
history at the current boundary; never synthesize earlier observations by
scanning World.History. Existing exact piecewise-constant EMA updates at
20/60/180 years remain authoritative. A changed boundary signal affects the
following interval's trace; first-turn means may legitimately still coincide.

Keep one native autonomous receipt per opted-in village, with actual requested
action3, canonical compiled staffing, opening physical state and authoritative
closed-boundary outputs/outcomes. No unbounded trajectory log. The receipt is
published through the shared native resolver after immigration/fission/contact
updates. It must use only that village's own requests and realized events, with
explicit decision presence, physical action age and interval duration.
Compiled staffing is an intended allocation, not a claim of productive labor.
Production and extraction receipts distinguish unsuccessful allocation from output.

Persist receipts in a versioned extension block of ANTHROPOCENE-12. Runtime
receipt storage is excluded from the existing binary-XML world header. Worlds
without autonomous receipts keep byte-identical ANTHROPOCENE-11 serialization;
legacy8/9/10/11 remain readable, and old-mode fingerprints must match the frozen
baseline. New12 snapshots require environment5 for autonomous control. Memory
and extension bytes participate in new state fingerprints. Unknown/corrupt
extension versions and impossible future/negative receipt times are rejected.

A daughter starts its own history when independently opted into control, with no
inherited parent action receipt or EMA bank. Parent reward remains strictly
parent-local; no descendant/lineage welfare is added. Save/load/resume must
preserve each institution independently.

## Observable history layout

The appended vector is ordered by the following groups. Quotes retain the
original decision-boundary partner ordinal; it is explicitly not an address into
today's reordered contact rows. No server-only identity tokens or remote stocks
are exposed. Quote quantities and actual settlements are observed physical units,
not heuristic utility. Scalar numeric values use fixed sign-log compression;
bounded shares/one-hot/presence remain raw.

| Group | Width |
| --- | ---: |
| decision presence, decision age years, decision elapsed years |3|
| requested workforce shares |13|
| requested recipe/activity shares, role-major |195|
| requested field/herd/breeding controls |3|
| requested movement ordinal, one-hot |8|
| quote presence |1|
| quoted original partner ordinal, one-hot |8|
| quoted give/want goods, one-hot |22|
| requested give/want quantities |2|
| compiled profession staffing, resident counts |13|
| actual production outputs, CraftGood order |11|
| actual Stone/Clay/Wood extraction |3|
| prior/end population, QoL, food/water fulfillment |5|
| actual movement and focal settled-trade count |2|
| actual quote give/want quantities settled |2|
| immigrants, emigrants, daughter count |3|
| opening/end domestic animals |2|
| opening/end owned cultivation capital |2|
| institution presence and physical age |2|

With no completed decision, receipt features are zero, while institution presence
and age describe the initialized memory. Neither legacy development-budget traces
nor stock-quote utility fields certify autonomous investment/exchange: zero the
obsolete development EMA context slots114:117 in observation5 and expose actual
owned capital/production/stock exchange evidence in the new receipt instead.
Other temporal state/contact/event channels retain their declared native semantics.

## Learner and mechanical transfer

Preserve the existing256x256 actor with actor LayerNorm, current action heads,
masked Dirichlet/Beta/categorical likelihoods, H100/H400 and both critic variants.
History inputs enter the first actor and trainable critic layers through separate
zero-initialized linear weights. Evaluate the original761-input linear expression
unchanged, then add the history contribution. Both frozen residual value anchors
retain zero history weights. This preserves entry action probabilities and values
for an identical base input, while permitting history gradients in live networks.
It does not promise identical behavior when the repaired native base EMA values
differ from the old defective input.

Explicit, digest-checked v4-to-v5 forks preserve every inherited parameter,
populated named Adam state, hyperparameters, RNG and episode/transition counters;
new history parameters begin with empty optimizer state. Frozen anchors are
serialized/excluded from Adam. Do not silently relabel or overwrite old files.
Checkpoint loaders enforce model/schema/feature-layout provenance. Gym, native
transport, collector, inference and PPO use the new version consistently.

## Scaling and tests

Authoritative memory is one fixed EMA bank and receipt per opted-in village.
Projection scales with controlled villages times the fixed300-feature layout,
plus their own local footprint. Existing EMA capture retains global herd and
event-list lookups; this repair measures their cost without claiming a new local
index for that older component. Group native production/material/trade/social
receipts once per boundary. No history-depth, full-world tile or hidden-state scan
is allowed in observation construction. Existing native memory capture costs must
be measured when activated for multiple villages.

Required checks: reset/persistent maturity; distinct observed histories with equal
current state; correct causal EMA order and physical-time decay; actual workforce,
recipe/control/movement/quote receipts; success/failure/partial settlement amounts;
compiled allocation versus physical output; no mutation by invalid actions or
caller edits; knowledge isolation; parent/daughter separation and local reward;
save/load/continued replay with exact receipt/observation/reward evidence; common
WEGO and scalar/parallel equivalence; old native/learner regression.

Learner checks cover strict masks/schema, finite log features, zero-column entry
equivalence, nonzero live history gradients, immutable frozen anchors, populated
Adam/RNG/counter transfer, exact checkpoint/resume/inference and real PPO update.
A local synthetic history contrast is a mechanical test, not proof of economic
memory sufficiency or GRU eligibility.

Run focused native suites and existing frozen old-mode fingerprints. For the
separate real-bank adapter proof reserve at most128 new validation trajectory
steps plus128 replay resolves,600 seconds, four CPU workers/one Torch thread.
Use TRAIN fixtures only for neural conversion/probes; DEV may be reset/decoded
for schema compatibility without fitting or policy selection. Record actual use,
including failures/retries. No retained native PPO campaign is launched here.
Benchmark full native steps including observations/economic execution at
1/8/80 villages against current heuristic and autonomous4 modes, using one
discarded warmup plus three measured independent resets per mode/size (36
resolves total); report allocations and separately time observation construction.
Native unit-suite resolves are reported separately, not disguised as training.

Independent Astra/high must review the concrete implementation and completed
evidence. A correctness PASS can authorize only a separately bounded memory
ablation/critic-calibration curriculum comparison. Economic scale-out and GRU
comparison remain blocked until their actual held-out gates pass.

## Additional native scenario regeneration

The retained GCS bank has499 records (348 TRAIN,75 DEV,76 HELDOUT), rather than
just the pilot's12 TRAIN states. The user authorizes native regeneration while
arranging cloud access. Create a separate local bank from24 fresh worlds,
partitioned16 TRAIN/4 DEV/4 HELDOUT, with survival, trade, migration, fission,
husbandry and fishing families. Capture actual states at decisions0/2/8/25:
at most576 snapshots and3,600 generation+3,600 replay resolves,600 seconds,
CPU only. Preserve whole-world/lineage/counterfactual partitions. Log failures
and unavailable/extinct endpoints rather than rescuing them. No optimizer,
neural-model sampling or retained PPO training is included in this generation.

Source actions are explicitly scripted native heuristic projections through
action3, with legal stock quotes and movement. Trade/migration/fission remain
enabled; native ecology, workforce, tools, recipes and capital stay authoritative.
The source controller may choose good actions; the compiler cannot repair poor
legal actions. Cold states initialize only current local signals; later states
persist actual mature history and receipts. Cultivation capital in receipts means
owned capital in the village's current local operation footprint.

Husbandry/fishing faces stay in evidence quarantine pending their own neural
action-face validation. Core states are prospective training data pending a
compatibility/mask/history/partition audit. HELDOUT is reserved from neural
fitting and policy selection. Preserve existing GCS/local banks and checkpoints.
Snapshot count alone does not establish economic competence or GRU eligibility.
