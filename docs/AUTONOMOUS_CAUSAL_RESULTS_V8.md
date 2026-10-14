# Autonomous Elder direct-control and causal gate v8

Independent Astra/high review: **GO for simulator correctness, learned-Elder
plugging and bounded trainability. NO-GO for broad economic scale-out,
institutional-competence claims or a GRU/capacity conclusion.** Poor economic
choices are acceptable canary outcomes. The outstanding gates concern actual
opportunities and update coverage, not a requirement to equal heuristic welfare.

## Implemented repair and model

The native breeding coefficient already respected [0,1]. Action3's opening-herd
retention target immediately slaughtered births above that opening target.
Action4 instead retains the chosen fraction of the actual herd after nutrition
and reproduction. Retention1 keeps every survivor;0 slaughters every survivor.
The native stable proportional allocation across multiple herds remains.
Nutrition, starvation, labour, ecology and phase order remain authoritative.
The action4 compiler clears the old persistent target; turn-local authority is
nonserialized and cleared before committed boundaries and daughter creation.
Action3, old saves and the five-head contract retain their original semantics.

Native action4 / observation6 / environment6 / replay6 / Save13 use a versioned
post-growth receipt. Old Save12 receipts remain distinguishable: the appended
history coordinate is0 absent,3 legacy action,4 new action. Model6 features have
width1062, including301 receipt coordinates; the prior1061 coordinates remain.
Gym, multi-Elder inference and checkpoints validate version, names, widths,
save envelope and control semantics. The shared contracts and their prospective
amendments preserve the actual chronology of changes and failed attempts.

The MLP retains the previous256×256 architecture, existing actor LayerNorm and
residual LayerNorm live critic. Actor LayerNorm was already present; adding it
again would not resolve output clamps or scarce reciprocal quotes. Conversion
preserves inherited actor tensors, populated Adam state, RNG, counters and frozen
critic anchors. New history columns start at zero. Physical values require new
calibration because native authority changes; tensor equality is not value parity.

The workforce and conditional activity heads remain masked Dirichlets. Three
capital/livestock controls remain latent Betas, mapped once to physical controls
by clip((x−.05)/.90,0,1), with explicit threshold branches giving exact endpoints.
PPO stores and scores the latent sample. It does not substitute the executed
endpoint into a Beta likelihood: physical space has endpoint atoms. A
deterministic request is T(E[X]), which need not equal stochastic E[T(X)].

Existing surplus production, resource constraints, integer staffing, native
recipes, movement, escrow trade, fission and migration remain intrinsic. This
repair does not introduce direct cultivated-hex choices or new fission heads.

## Physical proof and preservation

The frozen64 paired-head scenarios used512 validation and512 replay resolves;
all128 arm replays passed. Workforce changed actual production in8/8 pairs;
activity allocations7/8 (empty materials legitimately blocked one), cultivation
8/8, old breeding4/8, retention8/8, movement8/8 and quotes6/8. Aquatic harvest
changed meaningfully in7/8 fishing pairs; an already saturated case changed only
at floating-point scale. These are within-pair effects, not comparisons between
unmatched average-policy and endpoint worlds.

Compatible scripted quotes settled10:20; incompatible and nonparticipating
quotes settled zero; a stock-limited quote settled3:6. Gross legs conserve goods.
All resource-empty scenarios retained native consequences. Removing fishing
tools does not imply a hard fishing prohibition; actual native tool effects
must be distinguished from invented vetoes. The existing crafting checks retain
the meaningful missing-material/tool/workforce coverage.

The repaired-herd proof passed63 assertions. From100 animals, breeding0/.5/1
produced0/22/49 births with retention1 preserving all births. Multiple-herd
retention0/.5/1, starvation, terminal receipts, mixed legacy/new receipts,
Save13 round trips, resumed/parallel replay and daughter isolation passed.
The first proof attempt failed a starvation fixture that left grace enabled;
its17+16 resolves remain charged. The successful attempt used19+19. No failed
attempt, checkpoint or source snapshot was overwritten.

Old-mode verification passed471 autonomous assertions,143 old five-head
assertions and77 history assertions. Eight old-mode fingerprints exactly match
the validated5559961 baseline. The final learner test count is in
`evidence/autonomous-causal-v8/verification.json`; intermediate suites passed
202 tests with5 skips before the last schema-guard test was added.

## Critic targets and short training matrix

Frozen actors collected600 native decisions and600 replay decisions at25 turns
per episode. The cohort comprises eight TRAIN fitting worlds and four distinct
TRAIN critic-validation worlds; validation worlds never entered PPO. Only two
DEV worlds were evaluated; HELDOUT snapshots were never used by a neural model.
The existing511-snapshot bank and all prior checkpoints are unchanged.

Targets are discounted native n-step returns plus a frozen, censored tail.
At500 years the reset tail weight is.006738 for H100 and.286505 for H400. Zero
and doubled-tail sensitivities are recorded. These are not infinite-horizon
ground truth. H100 and H400 start from different historical trained actors, so
between-H results do not isolate the effect of the discount horizon.

H100 full/receipt-none critics passed after32 epochs with validation MSE2817/2799
against a TRAIN-derived constant's6530. H400 initially failed the bias gate
after256 epochs; the live network was responsive, rather than numerically frozen.
One prospectively declared TRAIN-only analytic intercept update changed only
the live critic's final bias, with no actor, Adam, RNG, anchor or counter changes.
The same held-apart validation then passed: MSE44159/44662 versus constant61760,
normalized bias−.339/−.338. The failed calibration artifacts remain intact.

The first matrix attempt failed a driver conversion before returning any retained
rows: GAE received a list where it required an array. Fourteen attempted native
calls and the completed50+50 baseline remain charged. The corrected driver has
an end-to-end native-stub→real-GAE→real-PPO test. The prospective resume amendment
reuses only the identical completed baseline; historical pins remain archived.

The completed matrix is24 cells: H100/H400 × full/receipt-none ×8/25-turn
collection × paired seeds11/23/61. Each cell retains400 transitions, totaling
9600 and48 rollout-update calls. These are common-parent continuations with
different sampling seeds, not three independent initializations. Each cell's
full and receipt-none parent actor is initially identical; ablation removes
only receipts/version, preserving native EMA and institutional memory.

All48 updates changed actor and live critic parameters. Actor step L2 was
.1179–.2519; critic.4180–.8822. No nonfinite actions, illegal-mask execution or
replay failure occurred. All24 disposable two-direction workforce reversal
probes passed (768 nonretained optimizer steps). Their fresh Adam and two selected
roles do not prove reversal for every head or retained optimizer state.

On64 fixed cold probes, maximum actor/live-critic Tanh saturation was.78%/1.33%,
with no units saturated across every probe. The inherited frozen reference critic
can be strongly saturated; it is not the trainable critic. Workforce/activity
outputs sometimes hit the softplus input clamp; the three control heads did not
on those probes. On198 mature fitting-world states from frozen pre-PPO traces,
no coordinate was clamped in every state where it was available. Local clamps
remain a limitation. Full-mode actor receipt sensitivity was weak but nonzero;
critic sensitivity was clearer. This does not establish memory's economic value.

The concentration floor/cap also limits deterministic workforce means: with13
available roles a single role's mean cannot exceed approximately.676 (11 roles:
.714). Stochastic Dirichlet support is much wider, but useful near-exclusive
staffing can be rare. This is an output-representation limitation, not an
environmental rescue or proof of global collapse. A prospective distribution
comparison needs a new version and likelihood/retained-optimizer tests; it was
not silently introduced into the frozen causal matrix.

All rollouts triggered the KL guard. Actor updates used only163/1344 scheduled
minibatches (2–5 of28 per rollout), with measured maximum KL.0574–.2642. This is
a concrete optimization limitation. Successful parameter changes cannot be
reinterpreted as unrestricted effective training.

## Population, QoL and welfare

For each interval, raw welfare is **focal opening population × realized
end-interval QoL × elapsed years**. Daughter villages' subsequent welfare is
excluded; they run independent native heuristic Elders. Annual welfare divides
the integrated sum by measured years. Mean population and mean QoL are reported
separately; multiplying their means need not equal mean welfare.

Each final DEV panel has two25-turn/500-year episodes. The table averages across
the three continuation seeds;500-year welfare is per episode, not the sum of two.

| H | Receipts | Collection turns | Mean population | Mean QoL | Annual welfare | Mean500-year welfare |
|---|---|---:|---:|---:|---:|---:|
|100|full|8|178.61|.779|159.63|79,814|
|100|full|25|179.89|.819|159.61|79,804|
|100|none|8|166.30|.730|145.18|72,590|
|100|none|25|182.39|.842|162.38|81,190|
|400|full|8|196.83|.725|168.14|84,070|
|400|full|25|195.67|.845|174.20|87,101|
|400|none|8|183.89|.832|164.98|82,492|
|400|none|25|223.87|.880|205.36|102,682|
|native heuristic|—|25|349.28|.956|333.29|166,643|

Individual learned panels span69.58–257.03 annual welfare. Twenty-seven of48
DEV trajectories declined from their first completed interval to the last;
21 improved. The pooled first/last annual welfare was209.78→172.83. No DEV
trajectory terminated. This is not a monotonic collapse shared by every run.
No uniform history/episode-length winner is established by this short matrix.

The native heuristic comparator reaches substantially higher welfare from the
same two starting snapshots, with final population mean382.5. Its annual welfare
is about30% above the best single learned panel and62% above the strongest
three-seed group. Every village runs native heuristics there; learner panels
instead use controlled focal authority and a fixed scripted peer. These are
descriptive attainable outcomes, not a causal architecture or reward comparison.

## Trade as MLP input and output

The MLP receives own stock quantities through fixed log1p compression, legal
good/counterparty masks and ordered known-contact rows (distance, confidence,
trust and permitted past trade signals). Versioned history provides the prior
requested quote and its actual settled count/legs alongside production and
ending stocks. It receives no counterparty inventory, private utility or future
quote. Physical units are compressed; observations are not arbitrarily fitted
to a new normalizer for each evaluation.

Outputs factor trade into enabled/partner/give-good/want-good/lot/price categories.
Lots request5/20/50% of own opening stock; exchange ratios are.5/1/2. Native
positive quantities are bounded by1e−6..1e6, and real own stock still limits
escrow. Thus an unbounded-looking stock economy becomes a finite set of
stock-relative decisions with authoritative native settlement, rather than
an MLP asked to emit arbitrary raw quantities. PPO scores all conditional choices;
it does not replace them with settled quantities or apply a heuristic utility veto.

The price/lot grid is coarse, there is one quote slot, and current counterparty
offers are not visible. Training produced1367 quotes/zero settlements; final DEV
134/zero. The fixed peer only offers food for hand tools at1:1 to its
first known contact, which can be another village. Those sparse reciprocal
conditions and other uncontrolled native counterparties confound trade learning.
Positive scripted mechanics do not prove neural trade competence. The next
curriculum must measure eligible reciprocal opportunities separately from quotes.

Training nevertheless exercised4548 moves,4070 migrant people and1292 daughter
events; final DEV355 moves,819 migrants and86 daughters. Movement, migration
and fission were not removed to stabilize the network. Husbandry controls were
eligible in2379 training decisions: retention1 appeared133 times, breeding1
once. Cultivation expansion1 was0/9600, often reflecting low learned requests.
The two DEV states had no eligible herd controls; learned herd-management
competence remains unmeasured despite the repaired physical authority.

## Native performance and plug test

One warm reset and three measured independent resets per case, complete native
world step including economic execution, observations and fingerprinting:

| Villages | Heuristic median/range ms | Native6 median/range ms | Observation median, heuristic/native6 ms | Allocated MB, heuristic/native6 |
|---:|---:|---:|---:|---:|
|1|3.913 (3.681–4.244)|3.065 (2.802–3.366)|1.018/.897|.737/.732|
|80|261.590 (246.321–265.937)|76.532 (64.343–92.969)|112.283/30.381|54.993/53.604|

Reset/setup, serialization transport and MLP inference are outside the timer.
The modes choose different actions/outcomes; observation timing is also measured
separately. This small cold-world benchmark does not establish a production
speedup, mature-empire scaling or full training throughput.

Two learned Elders passed four common-boundary decisions repeated identically,
with separate own-observation inference instances/RNG streams. The repeated run
saved/reloaded state and both sampler states after two decisions. Actions,
observations, fingerprints, receipts and per-village rewards matched exactly;
all8 validation and8 replay resolves passed, and checkpoint bytes were unchanged.

The complete bounded campaign, including failed attempts, analytic/read-only
audits, performance and plug verification, charged **14991 native resolves and
569.407 active elapsed wall seconds** against15080/1800 limits. Mechanical validation
used124/128 and replay43/128. Regression/build work is separately logged.
No broad PPO scale-out, main merge or additional GPU ecology work occurred.

## Next safe training gate

Use the staged plan `configs/training/autonomous-scaleout-readiness-v9-plan.json`.
First freeze a curriculum with confirmed reciprocal two-Elder trade opportunities
and DEV husbandry/fishing states. Select by world-level partitions, validate source
hashes and chronological receipts, and never reveal foreign private stocks or
the other policy's current action. Failed/incompatible quotes must remain possible.
Keep movement/migration/fission active and preserve the village-local reward.

Then compare smaller actor steps with the current KL guard on an equal native
budget, measuring accepted actor minibatches, each head's response, output-clamp
persistence and reversibility with populated optimizer state. Use25-turn episodes
for long payoffs and report H400 tail uncertainty. Require independent initializations
before claiming robustness; sample seeds on one old parent are insufficient.

Bad welfare from responsive choices need not fail that gate. Nonfinite likelihoods,
systematic illegal actions, dead gradients, persistent head clamps, negligible
accepted updates or absent reciprocal opportunities do. A GRU comparison requires
evidence that permitted mature history is insufficient after observation/action
and optimization issues are addressed. Current welfare alone has not reached it.

The511 local snapshots can support the next cohort now. Runtime revision75 has
current observations but **no attached outbound identity** and lacks the required
GCS domains; read-only bucket verification remains blocked. GCP IAM provisioning
alone does not attach the identity to this runtime. No key or interactive login
was created. Integrate the offered GCS bank later by manifest/hash/partition audit.
