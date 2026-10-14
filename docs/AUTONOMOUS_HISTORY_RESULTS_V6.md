# Autonomous Elder institutional-history repair and regenerated curriculum v6

The independent **gpt-6-astra/high** review passes correctness and audited core
curriculum readiness. Economic scale-out and GRU comparison remain **NO-GO**.
This gate repairs the missing information identified in the prior causal
assessment; it does not claim that the repaired MLP has learned better economics.
No retained native PPO training, GPU ecology or main merge occurred.

## Interface and model

The shared [implementation contract](AUTONOMOUS_HISTORY_REPAIR_CONTRACT_V6.md)
preserves economic action3 and introduces observation/environment/replay5,
Save12 and learner model/checkpoint/inference5. The old five-head contract and
the autonomous4 build, banks and checkpoints remain available.

The native resolver commits one bounded receipt per opted-in village at the
common WEGO close: requested workforce, recipe shares, capital/herd/breeding,
movement and quote; compiled staffing; actual production, extraction, settlement,
population, QoL, migration and fission. Compiled staffing is planned allocation,
not proof that workers had materials, tools or useful output. Failed and partial
production/trade therefore remain distinguishable from the requested decision.

Persistent 20/60/180-year EMA banks now survive reset/save/resume. Cold snapshots
start honestly at the current boundary; no earlier action/history is fabricated.
The old boundary signal integrates before the new signal is installed. Obsolete
five-head receipts and generic development-budget EMA slots are zeroed in the
new observation. Daughter institutions start separately; their later welfare is
excluded from the parent's reward.

The 256×256 actor retains its existing LayerNorm and distributions. The new MLP
has **1,061 inputs: the existing 761 plus 300 named history features**. Separate
zero-initialized history weights preserve the original first-layer expression,
inherited parameters, entry probabilities/values, Adam state, hyperparameters,
RNG and counters. Live actor/critic history weights receive gradients; frozen
residual critic anchors retain zero history weights and remain outside Adam.
Both critic variants and **H100/H400** remain supported. No GRU is introduced.

Welfare remains exclusively local:

```text
annual welfare rate = opening resident population × realized QoL
interval welfare = annual welfare rate × elapsed physical years
episode welfare = sum of interval welfare
```

Population, QoL and elapsed years are reported separately. Fission's immediate
effect on the parent remains physical; subsequent daughter welfare is not added.

## Trade input and bounded policy choices

The MLP receives own stock amounts, legal good masks and known counterparty
rows. It receives no counterparties' private stocks or submitted orders. Stock
and other unbounded numeric features use fixed `log1p`/sign-log compression;
there is no fitted normalizer or assumption of a fixed maximum village wealth.
The new history adds the previous give/want goods, requested quantities,
original partner ordinal, and amounts that actually settled. That ordinal is a
past row label, not an address into today's reordered contacts.

The existing learner selects enable/partner/give/want plus a lot of **5%, 20% or
50% of own opening stock** and a want/give ratio of **0.5, 1 or 2**. These choices
produce native stock quotes within `[1e-6, 1e6]` units per leg. PPO evaluates the
conditional categorical choices, including their masks. Native escrow and
matching enforce settlement; a legal bad price is not rescued by utility logic.
The native interface can accept direct bounded quantities; the learner's three
lot and three price categories remain a coarse action-space limitation.

## Verification

- New native history suite: **77 assertions**, 9 validation plus 9 replay
  resolves, including partial/incompatible quotes, terminal receipts, knowledge
  isolation, common close, cold/mature history, corrupted saves and daughters.
- Existing autonomous suite: **471 assertions**; legacy five-head suite:
  **143 assertions**. All eight frozen fingerprints match both a fresh build
  of validated `5559961164d6f9da8e6d20c1dc563ecec3eb04b7` and retained evidence;
  both old-mode replay checks pass.
- Full learner run: **184 passed, 5 skipped**. Subsequent audited-curriculum and progression-admission
  guard cases were added; the final focused history suite passes all **14** tests
  (net 189 passing cases). Ruff passes on changed Python files.
- Real TRAIN bank proof: **100 validation + 100 replay** resolves in 12.272s,
  within the separately frozen 128+128 budget. All eight retained 256-wide
  parents preserve entry behavior; one discarded eight-row native PPO update
  changes live history weights and preserves frozen anchors. Canonical one-
  versus four-worker collection, checkpoint/inference/resume and native replay
  pass. This is mechanical verification, not retained training.
- A real partial trade requested 743.8336 units per leg and settled 5.785829.
  The native receipt stores the settled quantities once, without applying the
  delivery fraction twice.

An initial fission test used an unexplored destination and was corrected. A
later headless build failed because a struct was modified through `foreach`;
attempt3 inadvertently used the previous DLL and is prior evidence only.
Attempt4 is the rebuilt 77-assertion result. An initial old-mode probe selected
a nomad camp and stopped after four heuristic steps per build; those attempts
remain charged and preserved separately from the complete frozen comparison.

## Regenerated local bank

GCS contains a retained 499-record catalog (348 TRAIN/75 DEV/76 HELDOUT); twelve
TRAIN pilot fixtures were never the limit of available curriculum capacity.
With runtime cloud identity attachment still absent, the user authorized local
native regeneration. The existing GCS and local banks remain untouched.

The new bank contains **511 Save12 snapshots from 24 fresh worlds**: 16 TRAIN,
4 DEV and 4 reserved HELDOUT worlds. Six source families cover survival, trade,
movement/migration, fission, husbandry and fishing. Snapshots occur at decisions
0/2/8/25, with genuinely persistent history at later boundaries.

| Partition | Snapshots | Audited core states | Quarantined faces |
| --- | ---: | ---: | ---: |
| TRAIN |335|196|139|
| DEV |92|42|50|
| HELDOUT, reserved |84|43|41|

Eligible TRAIN age coverage is 64/59/43/30 states at decisions0/2/8/25. Eligibility
uses actual current masks, not origin labels: native progression can expose a
herder/fisher face in a cultivation-origin state. The 230 quarantined states
are preserved, not silently admitted to neural training. HELDOUT underwent
schema/metadata audit only: no policy, fitting or model selection used it.

The source controller is explicitly scripted native DemandControl projected
through action3, with own-stock quotes and legal movement. Its choices are not
neural training results. The 128 completed trajectories each run 25 decisions
and pass parallel replay. They contain 389 focal settled trades, 22 moves,
4,212 migrant counts and 156 daughter events, across all source partitions.

Generation used **3,200 generation + 3,200 replay resolves**, cumulatively
**167.565s**, within 3,600+3,600/600s caps. First and second attempts failed on
physical site/herd-capacity strata. Resumes preserve their manifests, blobs,
counts and time; no completed trajectories are regenerated and no infeasible
stratum is clamped. There are **65 unique unavailable endpoints**, so
511+65=576 planned endpoints. The raw ledger has69 rows because a failed fishing
stratum was logged twice across resumes; this duplicate is retained and
deduplicated in the audit. The future resume code skips already unavailable IDs.

All511 hashes, Save12 envelopes, reset fingerprints, configuration hashes,
masks, chronology, history presence, maturities and finite encodings pass the
native audit. World/lineage/counterfactual/content partition leakage is absent.
This audit used511 reset requests, **zero world steps, model calls or updates**.
The audited loader connects these entries to the v5 collector and rejects
changed manifests, quarantined states and unapproved HELDOUT use. Core-only
collection also stops before actor/value input if native progression reveals
an unvalidated herder/fisher face. This preserves native progression; only a
separately declared explicit boolean override can admit a validated broader gate.

Raw snapshots remain locally in `datasets/autonomous-history-fresh-v5/`,
excluded from Git. Hash-pinned manifests, audit, generation envelopes and the
independent review are retained in `evidence/autonomous-history-v6/`.

## Performance

Complete native steps include observation construction, background planning,
economic execution, receipt close and fingerprints. One discarded warmup and
three measured resets per mode/size consumed36 resolves total.

| Villages | Heuristic median ms | Autonomous4 ms | History5 ms |
| --- | ---: | ---: | ---: |
|1|3.523|2.906|3.002|
|8|18.220|16.860|21.344|
|80|207.738|90.161|59.481|

At80 villages, History5 ranged50.330–128.866ms and allocated53.18MB per step;
heuristic allocated54.99MB and autonomous4 allocated49.65MB. Separately timed
History5 observation construction had31.209ms median. Sequential order, JIT/GC,
few samples and different mode outcomes make the apparent speedup inconclusive.
This establishes measured current-scale cost, **not** mature-empire or scale-out
performance clearance. Existing EMA capture still has global herd/event scans.

## Next safe training gate

Proceed next with a separately frozen CPU comparison on the audited core bank,
not economic scale-out. Keep movement, stock trade, migration and fission active.
Declare budgets, whole-world splits, reset ages and sampler seeds before launch;
keep H100/H400 and eight/25-decision resets as distinct experimental axes.

Before a 25-decision collection, validate any newly acquired action faces in a
bounded mechanics gate; the core guard must otherwise stop at an unlock.
First freeze the actor and calibrate the critic on actual TRAIN returns,
including late and mature states; validate bias/error by world, family and depth
before interpreting H400 actor learning. Then compare the same MLP with permitted
history versus a declared history ablation at equal native budgets, with paired
seed schedules and separately recorded model/optimizer provenance. Subsequent
promotion needs at least three declared neural replicates and a genuinely
untouched evaluation gate; common-parent forks are not independent initializations.

Measure fixed-state distributions, actor/critic saturation, gradients, KL stops,
history sensitivity and response latency after revealing evidence. Report
population, QoL, annual welfare and integrated welfare separately. A poor welfare
trajectory is acceptable evidence of poor choices if meaningful adaptation is
still present; numerical saturation, inert weights, excessive stability or
oscillation must not be mistaken for an economic failure alone.

Before training broader domains, validate those actual action faces and version
the remaining authority gaps: direct cultivation-hex selection, herd growth
retention beyond opening herd, and executable autonomous quotes from ordinary
non-controlled counterparties. Daughter villages currently use separate native
heuristic Elders. The GRU gate requires an actual failure of the repaired MLP on
paired histories/evidence; low aggregate welfare or this mechanical test does
not establish it. No economic scale-out GO is issued here.

## Runtime GCS status

The user's GCP principal is
`codex-gcs-reader@anthropocene-508912.iam.gserviceaccount.com`, with bucket-scoped
object viewer on `anthropocene-rl-datasets-508912`. The fresh managed-runtime
status at revision65 has current observations but **empty outbound identities**;
the required storage/authentication endpoints are absent from the enforced
allowlist. Bucket verification was therefore not attempted. No credential values
were printed, no service-account key created and no interactive login started.
The read-only identity/immutable-manifest verifier is ready for a later attached
runtime identity. Cloud-side IAM provisioning alone does not attach it here.
