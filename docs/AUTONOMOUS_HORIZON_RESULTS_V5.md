# Autonomous Elder model4: critic plasticity and H100/H400

Scale-out is **NO-GO** because the autonomous observation path does not supply
the persisted institutional history promised by the memory-equipped MLP design.
Two independent Astra/high agents confirmed this defect. The native experiment,
inference and retained-PPO results below are complete. Independent review passes
the frozen mechanical experiment, but rejects further training promotion and
economic scale-out. Economic welfare is not a mechanical acceptance threshold.

## Implemented model and preserved contracts

Model/checkpoint/inference4 keeps the current761-feature,256×256 actor MLP,
actor LayerNorm, masked workforce/recipe Dirichlet allocations, Beta capital
controls, movement and conditional stock-quote distributions. The old five-head
contract and all previous datasets, worktrees, experiments and checkpoints remain
intact. Native action3, observation/environment/replay4 and ANTHROPOCENE-11 saves
are unchanged. Native source SHA256 remains
`8a1319571e1204a8330306277b96529de23ec3932efff11c61d1964efe996f52`.

Both candidates use critic LR .0001 instead of the old .0003. One retains the old
critic architecture. The other supplies a normalized trainable correction:

`V(s) = frozen_original(s) + (trainable_LN(s) - frozen_initial_LN(s))`.

The subtraction happens first, making entry values exactly equal to the parent.
Actor/head parameters, original named Adam moments, RNG and episode cursors are
preserved. Both reference networks are serialized, immutable and excluded from
the optimizer. Collection, bootstrapping, PPO, checkpoint and inference paths
use the composite value. This is not ordinary LayerNorm insertion, which failed
the prior conversion gate; the frozen original representation stays frozen.

The TRAIN-only preflight uses128 construction rows and64 whole-world-disjoint
validation rows, with no fitting or evaluation-data normalization. All12 transfers
preserve values exactly. All112 retained critic cases (two variants×56 common
original value-plus-label targets) meet final AND maximum-last-eight MSE<.9.
Small-step worst final/last-eight errors are .1534/.2608; residual-LN .4407/.4417.
The small-step traces exactly reproduce the prior .0001 remedy experiment.
This establishes transfer and retained adaptation, not optimal economic weighting.

## Physical horizon and bounded native experiment

H100 and H400 are both implemented. Each native turn remains20 years. The return
uses interval integration `rate × H × (1-exp(-dt/H))` and discount `exp(-dt/H)`.
The corresponding20-year discounts are .81873 and .95123. Lambda stays .95 per
20 years, reward scale120 and actor LR .0003. H400 deliberately changes the
target of a warm start trained at H100; it is not a previously trained H400 critic.
Episode reset duration remains eight decisions for this comparison.

Eight cells (two critic variants×two horizons×seeds29/43) each completed4,096
additional native transitions and64 real PPO updates. The total32,768 took
1,610.272 seconds with four native CPU workers and one Torch thread. All24
entry/2,048/4,096 boundary fresh recovery and retained-Adam critic checks passed.
Every rollout update moved actor and critic parameters; no optimizer stall,
numeric/replay or resource stop fired. Frozen anchors remain exact at every
saved boundary, independently checked by Astra/high.

The old98,304-transition additional-training ledger stays intact. Cumulative
additional native training is now131,072; the original6,144 pilot transitions
remain a separate historical entry. The new campaign additionally used6,144
development steps and6,144 replay resolves, totaling45,056 native resolves.
No GPU training, ecology work or main merge was performed.

## Completed mechanical verification

The final retained-Adam diagnostics pass66/66 actor/composite tasks across12
selected checkpoints:135,168 sampled diagnostic actions and2,880.167 seconds.
Every task runs64 actual retained PPO updates of32 samples. Worst final and
maximum-last-eight composite MSE are .227588/.295443 against a .9 limit; minimum
actor separation gain is .095209 against .01. Exposure and final separation
criteria also pass. This is local adaptation evidence, not global neural-capacity
or economic-competence proof.

Coverage is four core tasks on all eight final candidates, all13 covered tasks
at each critic variant's worst-confidence checkpoint, and four core tasks at each
variant's worst actor-saturation checkpoint, with duplicates removed. Selected
extrema include intermediate checkpoints. All13 tasks were not run on every
final candidate; unrepresented husbandry/advanced faces remain unverified.

All12 selected checkpoints pass exact native continuation with one versus four
active workers, requested-action likelihoods, replay, the next PPO model/Adam/RNG
state and model4 inference. This charges384 discarded validation+384 replay steps
to the full existing shared cap and takes50.250 seconds. Astra/high independently
audited all24 saved boundary restores and all66 retained traces without adding
native transitions beyond the shared envelope.

Learner checks total174 distinct passing tests: the172-test regression run
(six opt-in native skips) plus the two subsequently added food-slot and sequential
welfare-accumulation regressions. Changed implementation/driver files pass Ruff.
These include likelihood/masking, populated-Adam conversion, frozen anchors,
H400/truncation, budget/selection and old-mode regressions. Native assertions
listed below are explicitly carried forward for the unchanged simulator source.

Eight-decision development scores below use all16 fixed snapshots and two
samplers per candidate, normalized per initial population and averaged equally
across four curriculum families. These are training entry→final comparisons,
not welfare change within one simulated episode.

| Critic | H | Seed | Entry score | Final score |
| --- | ---: | ---: | ---: | ---: |
| Small step |100|29|171.644|151.426|
| Small step |100|43|135.528|138.911|
| Small step |400|29|171.644|138.284|
| Small step |400|43|135.528|135.959|
| Residual LN |100|29|171.644|167.798|
| Residual LN |100|43|135.528|144.203|
| Residual LN |400|29|171.644|154.008|
| Residual LN |400|43|135.528|109.992|

Welfare can decline while the networks remain responsive. That distinction is
the point of the adaptation gate; favorable welfare cannot excuse a stuck model.

## Population, QoL and welfare

Each interval contributes `prior controlled-village population × clipped
end-interval QoL × elapsed years`. Daughters' subsequent welfare is excluded.
They currently have separate native heuristic controllers; automatic learned
daughter assignment remains incomplete. Immigration/fission can change parent
population without equaling deaths. Shared MLP weights must still use separate
village observations, rewards and trajectories.

QoL is food fulfillment×water fulfillment×the average of expectation-adjusted
clothing/housing fulfillment×social cohesion. Raw welfare sums the intervals.
Displayed score divides each episode by its own initial population, then gives
families equal weight. PPO instead trains on raw reward/120 with physical discount
and GAE. Dividing mean raw welfare by mean initial population is incorrect.

The25-decision panel uses the original eight development snapshots and two
samplers:16 episodes per candidate,500 simulated years, mean initial population
246.25. Weighted QoL below is person-year weighted, so mean raw welfare decomposes
exactly as time-mean population×weighted QoL×500. These eight-snapshot scores are
distinct from the full16-snapshot development scores above.

| Policy | Mean population over time | Final population | Weighted QoL | Raw welfare | Displayed score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native heuristic |371.88|427.38|.8967|166,726.38|1,121.25|
| Small step H100/29 |166.74|133.25|.9096|75,835.62|486.25|
| Small step H100/43 |168.52|164.75|.8657|72,940.25|434.53|
| Small step H400/29 |122.55|91.38|.8426|51,630.82|333.27|
| Small step H400/43 |164.49|134.94|.8701|71,559.87|431.07|
| Residual LN H100/29 |197.21|131.81|.8907|87,829.29|627.43|
| Residual LN H100/43 |205.29|196.06|.8695|89,248.67|546.90|
| Residual LN H400/29 |162.42|132.00|.8780|71,303.11|509.17|
| Residual LN H400/43 |128.92|85.88|.8192|52,806.52|273.32|

Native has one deterministic episode per snapshot; learned policies have two
samplers per snapshot. Native trading interfaces differ; the public-market-peer
arm controls peer behavior but cannot equalize the quote interface. Native's
matched public-peer score is1,144.41 on this eight-snapshot25-turn panel.

H40025-turn welfare is lower than matched H100 in all four critic/seed comparisons.
Two seeds and4,096 transitions per cell do not establish an optimal H. Longer reset
duration training has not yet been compared. Every candidate has survival-family
episodes ending with1–3 residents, and all16 panel episodes per candidate settle
zero focal trades. Training did settle1 focal trade for small-step H400/29 and2
for residual-LN H400/43, each across512 training episodes. Background trades are
not learned focal trade competence. Positive cumulative welfare does not imply
improving welfare per year, population or QoL.

## Heuristic headroom and measured costs

The separate matched read-only audit ran native DemandControl versus both
original retained v3 parents on all16 development snapshots at eight and25 turns.
Native scores were210.883 and926.065, versus171.644/621.583 for seed29 and
135.528/371.938 for seed43. Native advantage is22.9–55.6% at eight turns and
49.0–149.0% at25. The matched public-market-peer arm also passes the prospective
20% aggregate threshold. Every native family mean beats its comparator, but some
family gains are below20%. This is descriptive economic headroom, not an isolated
causal allocation experiment. It cost2,656 trajectory+2,656 replay resolves,
87.037 seconds and zero PPO updates.

Residual-LN composite forward cost on the fixed probe is at most3.568× the parent
(prospective bound4.5×), versus1.056× for small-step. Model parameter bytes are
4,456,476 versus2,359,316, including both frozen references. Actual complete
collection-plus-PPO cost is40.50–41.20ms/transition for residual-LN and38.39–41.68ms
for small-step; measured peak total process RSS stays900.9–931.6MiB across cells.
These run costs include inference/economy/PPO but exclude scheduled diagnostics;
the1,610-second campaign wall includes those diagnostics and evaluations.

The unchanged native simulator's earlier80-camp complete-world benchmark remains
81.775ms heuristic versus62.959ms autonomous, including observation construction
and economic execution. It was not remeasured here and is separate from PPO
throughput. Native69 check groups/471 autonomous assertions,143 legacy assertions
and eight baseline fingerprints remain carried-forward evidence for this exact
unchanged source digest; the learner regression suite separately covers the old
five-head model. New native replay/parallel checks exercise model4 directly.

At final H400, small-step critic tanh saturation is74.3%/96.5% for seed29 and
78.5%/83.9% for seed43. It still passes the current retained critic probes. Residual
adaptive saturation is .0020%/0% and0%/0%, with LN affine scale maxima1.0571/1.0576.
Its frozen teacher remains highly saturated. The new trainable path supplies
plasticity; normalization does not normalize returns or guarantee future stability.

H400 still substantially underpredicts its native targets. Mean preupdate critic
MSE/return-variance is1.364–2.066 across H400 cells versus .223–.311 at H100. H400
mean value bias is approximately -36.4 to -51.1 scaled value units. These use actual
return variance, with only a1e-12 numerical floor and an explicit near-constant
flag. No near-constant batch occurred. Synthetic retained adaptation does not
certify native value accuracy; all economic decline cannot be causally assigned
to actor weights alone. The observed H400 target shift is an additional learning
problem, alongside poor decisions, despite plastic critic parameters.

## Trade input and direct-control limits

The MLP receives own stocks, recent imports/exports and up to eight known-contact
rows: distance, knowledge age, affinity, bilateral flows per resident-year,
migration, kinship and visibility. Remote hidden stocks are excluded. Own stocks
use log1p; signed context uses sign(x)×log1p(abs(x)). These fixed transforms compress
large magnitudes without evaluation fitting or discarding differences through
hard clipping. Realized trade feeds back through stocks, flows, needs and welfare.

The initial quote policy emits bounded conditional choices: enable, known partner,
stocked offered good, different requested good,5/20/50% offered stock fraction,
and .5/1/2 exchange ratio, adjusted to native1e-6..1e6 quantity bounds. Conditional
PPO likelihoods score requested latent choices; inactive branches contribute zero,
and joint entropy weights downstream choices by reach probability. Native escrow,
matching and settlement remain authoritative without a heuristic utility veto.
This does not yet demonstrate learned reciprocal exchange.

Current controls are labor allocation, recipes, farmer expansion effort, breeding
coefficient and retained-herd fraction. Native logic still selects cultivation
hexes. Herd retention compiles to a target no larger than the opening herd, and
postbreeding excess is slaughtered, restricting retention of herd growth. Those
are direct-control limitations, distinct from absent husbandry/advanced-face
training coverage. See `AUTONOMOUS_DIRECT_CONTROL_SCOPE_V4.md`; this gate cannot
certify explicit hex authority, husbandry or all-domain scale-out.

## Missing history and GRU eligibility

All192 fixed TRAIN probe rows have previous-decision slots40–55 and EMA maturity
slots117–119 exactly zero. Every signal's20/60/180-year channels84–116 are identical.
All28 bank snapshots contain four nil ElderMemory records. ScenarioFactory clears
memory; autonomous Reset does not initialize it; observations therefore rebuild
a transient current-state view. Compile supplies no Elder decision receipt, so
the autonomous early return also bypasses legacy previous-decision recording.

This is an input-contract defect, not proof that correctly persisted memory is
insufficient. Its causal contribution to welfare requires an ablation. Current
local responsiveness tests cannot repair it or certify the intended history-aware
MLP. Add actual autonomous requests/outcomes with validity and physical ages;
do not force workforce/recipe/quote actions into obsolete five-head fields.
Version observation/replay semantics and test initialization, EMA maturity,
distinct histories, chronological decay, previous receipts and save/replay.

The Drive architecture's§41.6 GRU comparison gate is **not reached**: no paired
history failure has shown a decision-relevant distinction present in ordered
permitted history but absent from a corrected current-state/EMA/action-outcome
representation. The≥three-seed held-out frontier-regime sandbox and EMA ablation
also remain unperformed on this autonomous interface. Repair permitted history
and test the MLP before considering a time-aware GRU. No GRU was implemented.

The separate [Astra causal assessment](AUTONOMOUS_TRAINING_ASSESSMENT_ASTRA_V5.md)
maps every relevant architecture clause and ranks the competing causes. It is
also published in the user's private [Google Doc](https://docs.google.com/document/d/1tgk3YC4R-10axzjRA9tQuRURjsnsbqD9m5OZMOV9W04/edit),
with a verified final mechanical-results addendum. The original architecture
document was read without modification.

Raw village welfare also loses to native in every one of32 model×family matched
25-turn comparisons, by34.8–81.3%; this is not just displayed-score normalization.
The training bank has only12 snapshots from three worlds, with eight-turn resets.
These coverage limits, H400 target mismatch and sparse/nonreciprocal trade are
competing explanations. No observation-only audit can apportion losses among them.

## Preserved failures and next gate

The first conversion preflight failed before any updates due to a path-type bug.
Its source/config/report remain preserved; a separately pinned narrow fix passed.
The first long panel replayed25 turns successfully, then compared Python3.12's
compensated sum against the collector's sequential+= accumulator. The first
episode differs by1.82e-14 in normalized welfare while its fingerprint matches.
The corrected separately pinned runner preserves exact sequential arithmetic
and saves replayed raw episodes before assertions. All128×25 new episodes pass
every available original8-turn summary and final fingerprint; full new replay is
exact. Original intermediate8-turn action/population/QoL traces were not saved.
The failed25+25 are charged: inclusive panel use is3,225 trajectory+3,225 replay,
with3,200+3,200 in the successful retry and zero PPO updates. No threshold changed.

The historical25-turn reporter also mislabeled cloth stock index0 as food.
New measurements use native food index3. The original evidence stays preserved
with `AUTONOMOUS_HORIZON_EVIDENCE_ERRATA.md`; that old field supports no food-reserve
claim. Welfare, QoL, population, food fulfillment and replay were unaffected.

The next training comparison must retain movement, trade, migration and fission,
local village rewards and whole-world partition isolation. Compare eight-versus-
25-decision training with equal native budgets, keeping critic architecture fixed
and reporting population/QoL/welfare rate and capital retention by family. Keep
H100/H400 as explicit matched arms; H400's current economic disadvantage must not
be hidden or treated as proof that a longer horizon cannot learn.
Before increasing H400 actor training, retain actual TRAIN reward/value/return
traces and test bounded critic-only adaptation under a frozen actor, with disjoint
TRAIN validation and preserved actor/Adam/RNG evidence. This addresses the measured
target mismatch without fitting development data or confusing it with saturation.
The new32,768 native-training allowance is exhausted. These follow-ups need their
own frozen transition/wall-time budgets and a new native-training allowance;
they are recommendations, not extra work hidden inside the completed envelope.

Before broader economic scale-out, version policy-selected legal cultivation
hexes and breeding versus postgrowth slaughter/retention, add husbandry snapshots,
assign learned daughters separate observations/rewards/trajectories, and validate
reciprocal learned peers. GCS metadata candidates remain unmaterialized: live
Storage credentials were unavailable, and no candidate entered this experiment.
Future reuse must verify immutable generations/digests, native load/replay, rebuild
current observations/masks and preserve world/lineage partition groups.

Compact hash-anchored [evidence](../evidence/autonomous-horizon-v5/README.md) and
the [independent final verdict](../evidence/autonomous-horizon-v5/independent-final-review.md)
are retained in the repository. Verdict: **PASS frozen mechanics; NO-GO further
training promotion or scale-out; GRU comparison gate not reached**. The first
corrective gate is versioned institutional history and autonomous action/outcome
receipts, before any new training allowance or recurrent comparison. No
unrestricted GPU/all-domain promotion or main merge is included.
