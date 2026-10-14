# Autonomous Elder learning and subsistence needs: V10–V13

The new opt-in native profile provides Gatherers with self-made gear at a .80
baseline, with manufactured Cloth/AnimalFabric and real housing able to improve
it. A food-rich24-person scripted camp grows to29 in100 years at QoL≈.79;
the identical legacy camp remains24 with QoL0. Both food-scarce cases fall to14.
Astra/high gives the native mechanics and integration GO. Training scale-out
remains NO-GO until critics and a bounded PPO canary are checked under the new
physics. Old checkpoints are preserved as behavioral probes.

## What was learned before changing the game

V10 first retained the original focal reward and continued all three existing
low-LR actors by4096 additional transitions each at H400/25 decisions. No actor
reset, imitation restart or new architecture was used. All three completed the
fixed64+64 signed preference reversal;1097/1184 scheduled actor minibatches were
active. Population and original welfare improved, but pooled endpoint QoL fell
and extinctions rose9→10. Current-policy critic accuracy remained incomplete.

V11 introduced model/checkpoint8 with explicit learner objectives, exact model7
parameter/populated-Adam import and unchanged1062 actor inputs. Three matched
retained arms used original, midpoint population, and P0-normalized alpha1.5
training welfare. Each actor/arm retained1024 transitions, five update calls and
128 scheduled minibatches. All1152 actor minibatches were active, all eight action
heads had positive gradients, and all nine fixed signed switches passed. No
persistent Tanh saturation, concentration clamp failure or sustained optimizer
oscillation was found. The existing actor LayerNorm and residual critic LayerNorm
remain; current evidence does not justify GRU.

On the reused36-episode DEV panel, alpha mean original integrated welfare was
3725.518 versus3619.807 for continued original (+2.92%). Midpoint was3343.995.
The independently generated two-world panel then gave:

| Old-physics policy | Mean endpoint population | Mean endpoint QoL, zero if extinct | Mean original integrated welfare | Extinctions/36 |
| --- | ---: | ---: | ---: | ---: |
| Continued original |1.666667|.188254|4146.630|3|
| Alpha1.5 |2.416667|.246566|5023.235|3|

This is a point estimate over two worlds, not broad superiority. Seed101 QoL
declined and seed307 gained an extinction. Alpha critics101/211 failed the strict
EV>=.60, MSE/actual-return-variance<=.40, absolute raw bias<=2 gate. A single
P0=340 trade-family context dominated their errors, including a101 trajectory
with no settled trades. Its opening targets213.63/141.45 exceeded every target
in the later nine-episode TRAIN repair (max73.82/71.51). That fixed64-epoch repair
lowered TRAIN MSE to.218/.062, but validation MSE/variance remained.547/.488;
both failed zero/double-tail sensitivity too. This supports coverage/value
generalization investigation, not additional epochs on those same443 rows,
a saturation diagnosis or a proved trade-credit/GRU failure.

The four steering ideas remain represented: midpoint and alpha1.5 were compared;
valid potential shaping was tested, including12,288 actual cached rollout rows
with maximum GAE discrepancy6.04e-14. With V_shaped=V_base-Phi it is an equivalent
credit calculation, so duplicate native training was omitted. Optional
Crowding-only transferred-headcount credit has an opt-in tested basis but was
not trained. It never credits daughter future welfare, distress, emigration or
starvation. Native evaluation always remains unchanged focal P_open*Q_realized*dt.

## Native change V13

[The shared contract](AUTONOMOUS_SUBSISTENCE_NEEDS_V13.md) defines the actual job
cohort and the partial gear ceiling. For N residents and G Gatherers, automatic
coverage contributes .80G person-equivalents; physical goods fill N-.80G.
The self-made part cannot gain an early expectation-ramp bonus above.80.
Other jobs retain the existing expectation rule. Real shelter determines social
cohesion, so an unsheltered100-person camp remains growth-limited; at the normal
24-person opening the food/water-satisfied value is.790536, above Q*=.745556.
No arbitrary population floor or removal of food shortages was introduced.

Cloth alone can lift that small camp above.85, and AnimalFabric plus real housing
above.95, with faster growth. A general Craftsperson produces20 cloth units per
20-year turn at default efficiency.25; an unlocked Weaver's nominal capacity is80.
Actual production requires workforce, food feedstock, tools and storage. The
native entry point returns labor used (80 reference intervals here), while the
authoritative production receipt/stock records20 output units.

Hunters already lie outside the resident clothing/housing cohort; their food
requirements remain. Implicit and explicitly assigned Gatherers share the gear
baseline. Fission/immigration forecasts use projected jobs, including daughters'
Farmer-to-Gatherer conversion and immigrants' default Gatherer roles, without
turning automatic coverage into transferable cloth. Physical clothing coverage
is separately retained. Needs change the next turn's committed-QoL growth ceiling;
actual recorded deaths continue to come from starvation. Manufacturing, escrow,
settlement, ecology, phase order, demographics and knowledge restrictions retain
their mechanics.

Enable NeolithicConfig.SubsistenceNeedsCovered explicitly, or use the validated
learner profile-derivation helper. Default-false historical worlds stay unchanged. New desktop games enable the
profile after tuning is loaded; this small Unity startup hunk was reviewed but
no Unity build/manual UI run was performed here.
Enabled worlds use Save14, including the mode in fingerprint/configuration hash.
Action4 and observation/environment/replay6 layouts and the old five-head
contract remain intact. Native6 advertises additional Save14/profile capability;
an older bridge cannot silently accept the helper's request. The original banks
and all checkpoint files remain untouched. The prepared V12 old-physics
collection plan was never executed and is suspended; its pins are preserved.

## Verification and performance

Full learner regression:239 passed,5 opt-in native integration skips,101.40s.
Focused new native proof:83 assertions,26 resolves+24 replay; actual enabled
save/resume is exact. Existing native suites passed471 autonomous,77 history,
63 growth (birth coefficients0/.5/1 still produce0/22/49),143 five-head and3622
other adjacent assertions. All eight original-mode fingerprints match validated
commit5559961164d6f9da8e6d20c1dc563ecec3eb04b7 exactly. Enabled migration/fission
job projections were checked; enabled full migration/fission events and actual
hunter reintegration are still further integration coverage, not claimed here.

Two independent model8 Elders executed eight common WEGO boundaries plus eight
replay resolves under Save14. On-disk sampler/P0 sidecars survived reconstruction;
actions, rewards, staffing, owner observations, histories and fingerprints were
identical. Focal reward identities and retained checkpoint hashes were preserved.
This proves execution integration, not new-profile critic or economic readiness.

Complete native-step benchmark, one warm and three measured resets per cell:

| Villages | Profile | Heuristic median ms | Autonomous6 uniform median ms |
| --- | --- | ---: | ---: |
|1|Legacy|4.408|2.841|
|1|Subsistence .80|3.543|3.551|
|80|Legacy|252.336|136.984|
|80|Subsistence .80|96.674|76.385|

Planning/economic execution, owner observations and fingerprinting are included;
setup/reset and neural inference are excluded. Eighty-village allocation is
56.081/53.245MB legacy heuristic/autonomous and56.290/53.501MB new profile.
Different legal needs change heuristic search work, and ordered runs/VM timing
add noise: these numbers are not an isolated algorithmic speedup claim. Mature
empire and complete native-plus-neural training throughput remain unmeasured.
The preserved earlier heuristic25-turn comparator achieved mean endpoint
P349.28/Q.956/annual welfare333.29, far above earlier learned results; its
authority/peers/cohort differ and it is not a new-profile matched comparison.

## Trade in the MLP

The actor sees compressed own stocks, structurally legal goods/professions,
known contact slots and owner-only prior quote/settlement history. It sees no
foreign private inventory or current simultaneous rival quote. Six categorical
quote outputs choose enablement, known partner, give/want goods, lot5/20/50% of
own stock and exchange ratio.5/1/2. Native quantities remain bounded1e-6..1e6.
Conditional masks and latent joint likelihoods are scored by PPO. Deterministic
matching and authoritative escrow settle compatible stock-only quotes; heuristic
utility cannot veto them. This is an intentionally bounded market language,
not unrestricted price discovery. Fine quantities/prices, multiple offers and
current public-offer inputs remain incomplete.

## Accounting and next safe gate

V10 closed51913 charged resolves /1695.869524 cumulative active seconds. V11
closed75835 /2476.987313, including failed calibration/repair attempts. V13's
16-resolve model probe adds2.242510s, giving75851 /2479.229823. Native regression,
build, profile activation and benchmark work are separately recorded. Two new
proof-harness failures involved no world resolves (labor return mistaken for
output); the first benchmark failure involved one world resolve before missing
history initialization was fixed. All logs are retained. No expensive PPO,
additional GPU ecology or main merge was started.

The next safe gate is specified in [the new-profile readiness plan](AUTONOMOUS_SUBSISTENCE_TRAINING_GATE_V14.md).
It requires actual enabled migration/fission/reintegration coverage, current
three-critic calibration at original focal welfare/H400/25, followed by a bounded
three-seed canary with durable actual rollouts and independent review. Only then
expand into fixed tranches with stopping criteria. Direct cultivated-hex choices,
automatic learned daughter assignment, broad matched economic competence and
GRU necessity remain incomplete/unproved.

Evidence and independent reports: [artifact index](../evidence/autonomous-learning-v10-v13/artifact-index.json),
[Astra V13 review](../evidence/autonomous-learning-v10-v13/independent-v13.md).
Final Git SHAs are reported separately to avoid self-referential manifests.
GCS runtime revision102 has no attached outbound identity; its HTTP policy state
is unknown and GCS read access is unverified. No key/login/secret read was used.
