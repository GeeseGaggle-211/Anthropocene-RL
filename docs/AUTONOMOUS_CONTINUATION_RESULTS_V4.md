# Autonomous Elder continuation v4: outcomes and next gate

The capped native continuation finished. Further training promotion is **NO-GO**:
selected retained-Adam diagnostics exposed critic update instability. Economic
score is not an acceptance threshold. Actors in the failed composite tests still reversed their
preferences; the critic learned initially, then oscillated. Smaller critic steps
address that observed failure, but do not establish resistance to saturation over
longer training. Critic LayerNorm was assessed separately with common original
value targets and preserved optimizer state; direct insertion disrupted values
and did not meet the fixed-budget recovery criteria.

## Completed native work and preserved contracts

Seeds 29 and 43 each completed 24,576 additional native transitions, 49,152 total,
with four CPU native workers and one Torch thread. This exhausts the cumulative
98,304-transition additional-training envelope. Native continuation took
2,087.157 seconds; cumulative native campaign time is 3,993.923 seconds. All
14 entry/scheduled fresh recovery checks passed, and both seeds made 384 actual
PPO updates with finite, nonzero actor and critic parameter movement. These
fresh checks are distinct from the stricter retained-optimizer final gate.

All six technically selected checkpoints pass exact one-versus-four-worker
native rollouts, requested actions, likelihoods, next PPO update, populated Adam,
RNG and inference equivalence. A verifier initially rejected JSON-stringified
integer provenance keys before collecting any rollout. Its failure is preserved;
the reader now restores only the explicitly declared integer-key ledger and
keeps all other provenance comparisons strict. No checkpoint was edited.

The retained production model is still model/checkpoint v3: 761 input features,
independent 256×256 actor/critic MLPs, actor LayerNorm, legal categorical mixture
and reverse-KL regularization, masked workforce/activity Dirichlet shares and
Beta controls. Actor/critic native learning rates remain .0003. Critic LayerNorm
and smaller critic learning rates have not been applied to retained policies.
Native action 3, observation/environment/replay 4 and ANTHROPOCENE-11 saves remain
unchanged. The old five-head contract remains intact.

Retained adaptation coverage comprises 13 task groups: workforce, seven
categorical choices, four recipe faces and field expansion. It does not certify
every workforce contrast, rare livestock control or advanced recipe face.
Learned reciprocal trade and newly created daughter controllers remain separate
integration gaps; no unrestricted all-Elder readiness is claimed.

The native source digest remains
`8a1319571e1204a8330306277b96529de23ec3932efff11c61d1964efe996f52`.
Previously verified native coverage remains applicable: 69 check groups,
471 autonomous assertions, 143 legacy assertions, and eight legacy fingerprints
against validated base `5559961164d6f9da8e6d20c1dc563ecec3eb04b7`.
The complete-world benchmark, including observations and economic execution,
remains 81.775 ms heuristic versus 62.959 ms autonomous median at 80 camps.
This is separate from rollout-plus-PPO throughput and was not remeasured during
this source-unchanged continuation.

Complete rollout-plus-PPO cost averaged 39.39/39.17 ms per transition for seeds
29/43, with peak measured process RSS 944,017,408/938,496,000 bytes. No resource
growth stop fired. The learner regression suite passed 163 tests with six native
opt-in skips; selected-checkpoint native equivalence checks ran separately and
passed all six checkpoints. The three new critic-conversion/evidence tests also
passed independent review. Native training settled 10/3 focal trades for seeds
29/43, versus 1,914/1,745 total world trades, so background activity must not be
mistaken for learned focal trade competence.

## Welfare, population and quality of life

Each native interval contributes
`prior focal population × clamp(end-of-interval QoL, 0, 1) × elapsed years`.
Raw episode welfare is the sum of those contributions. The displayed evaluation
score divides each episode by its initial focal population, averages within each
curriculum family, then gives the four families equal weight. Training instead
uses raw reward divided by 120 with a continuous 100-year discount horizon and
GAE. The displayed score is undiscounted and is not identical to the PPO target.

QoL is food-needs fulfillment × water-needs fulfillment × the average of
expectation-adjusted clothing and housing fulfillment × social cohesion.
Expectations increase with stationary settlement age. Population is the
controlled parent settlement; daughters' subsequent welfare is excluded.
Fission and migration can therefore reduce the measured parent population
without representing deaths or a decline in the whole lineage.
The user explicitly confirmed this local reward scope: daughters must have
separate Elders and must not contribute reward to their parent. Shared MLP
weights may serve those Elders, but observations, local rewards and trajectories
must remain separate. Current daughter controllers are native heuristics;
automatic assignment of newly created daughters to learned controllers is not
implemented in this curriculum.

All rows below use the full 32 fixed development episodes per boundary, eight
decisions/160 years, mean initial population 246.25. QoL is weighted by prior
population × elapsed years, so raw welfare decomposes exactly as time-mean
population × weighted QoL × 160. The normalized score must be aggregated per
episode; dividing mean raw welfare by mean initial population is incorrect.

| Seed/boundary | Time-mean population | Final population | Weighted QoL | Mean raw welfare | Displayed score |
|---|---:|---:|---:|---:|---:|
| 29 entry | 198.68 | 171.41 | .7180 | 22,824.81 | 141.60 |
| 29 final | 206.41 | 203.81 | .8381 | 27,679.76 | 171.64 |
| 43 entry | 208.78 | 161.41 | .6411 | 21,417.56 | 121.93 |
| 43 final | 186.04 | 153.50 | .7953 | 23,673.75 | 135.53 |

Development evaluation scores at training boundaries were nonmonotonic:

- Seed 29: 141.60, 146.96, 148.23, 147.89, 153.84, 142.10, 171.64.
- Seed 43: 121.93, 148.05, 149.43, 131.49, 133.45, 127.59, 135.53.

Thus entry-to-final evaluation improved by 21.2% and 11.2%. That does not mean
every settlement improved within an episode. Cumulative nonnegative welfare
always accumulates; population, QoL and welfare per year can deteriorate.

## Twenty-five-turn evaluation

A preregistered inference-only check ran 32 episodes across both final models,
eight fixed development snapshots and two sampler seeds. It used 800 native
steps in 34.375 seconds, with zero training updates. All 32 controlled Elders
survived 25 decisions/500 years. Original eight-decision population/QoL/reward
prefixes and the eight-decision world fingerprint matched; full new 25-decision
action replay was exact. Original intermediate action/fingerprint traces were
not retained, so their comparison is not claimed.

The matched survival-family subset demonstrates why aggregate performance is
insufficient. These rows each average four episodes, starting at population 100:

| Seed | Population at decision 8 → 25 | QoL at decision 8 → 25 | Welfare/year at decision 8 → 25 |
|---|---:|---:|---:|
| 29 | 52 → 26 | .328 → .806 | 30.12 → 23.13 |
| 43 | 49 → 12.75 | .484 → .436 | 26.73 → 6.14 |

One seed-43 episode ended with one resident. Fission/emigration also occurred,
so the parent-population losses are not a death count. Migration-family growth
can dominate normalized aggregate welfare. All 32 episodes settled zero focal
trades despite submitted quotes; background trades are reported separately.
Mechanically valid quote heads do not yet establish learned exchange competence.

Twenty-five-turn episodes are a justified curriculum comparison because they
expose post-reset consequences. They do not by themselves establish how slowly
cultivation pays off. With the present 100-year discount horizon, a benefit 500
years away has weight exp(-5), approximately .0067 before GAE decay. Episode
length, discount horizon and estimator trace length are separate controls.
Correct truncation bootstrapping already exists. Measure actual cultivation
payoff delays before choosing a longer discount horizon; compare fixed-prefix
or welfare-per-year metrics, not raw totals over unequal durations.

Recorded field tiles, food stocks and food-needs fulfillment add context but are
not direct harvest measurements. Native current/future field-support fields
both alias managed recovery coverage; the latter is not a forecast. No new
legal coordinate bits appeared beyond the fixed probe's union, which does not
certify coverage of all mask combinations or advanced action faces.

## Critic instability and LayerNorm

The seed-29/20,480 recipe-4 and recipe-9 tests share exactly the same critic
observations, labels and error trajectory. Both actors pass. Critic MSE falls
to .16875, then oscillates and ends at 1.39740, failing the frozen .9 limit.
A discarded critic-only replay reproduced every original error value exactly.

| Critic learning rate | Final MSE | Maximum MSE in last eight updates |
|---|---:|---:|
| .0003 | 1.39740 | 1.56398 |
| .0001 | .02416 | .02507 |
| .00003 | .00641 | .00868 |

This is evidence of step instability and retained learnability on that target,
not evidence against later representation collapse. Selected original critic
tanh saturation is roughly 55–78% across layers. At final seed 29, 8.96% of
active continuous outputs also touch the existing raw-output clamp, even though
actor hidden tanh saturation is low. Retained per-head responsiveness matters
more than one favorable activation statistic.

An independently reviewed read-only critic-LayerNorm audit ran all six selected
checkpoints on the frozen TRAIN-only probe. Identity-affine LayerNorm before
each critic tanh reduced saturation to essentially zero, but shifted value
predictions by RMS 20.9–32.8 units and changed approximately 7–18% of pairwise
value orderings. Actor/head parameters, original Adam tensors and checkpoint
bytes remained exact. No candidate policy was saved. This supports evaluating
normalization, while exposing the bootstrap disruption an upgrade must address.
Mean tanh derivatives rose to roughly .51–.55 from .075–.20, but centered hidden
activation effective rank decreased on this probe. Neither lower saturation nor
higher local derivatives alone proves a more useful representation. Future
native comparisons must also track LayerNorm affine-scale growth, value-target
scale and retained adaptation; LayerNorm does not normalize reward targets or
guarantee that learned affine parameters cannot saturate tanh again.

The subsequent comparison is defined in `AUTONOMOUS_CRITIC_REMEDIES_V1.md`:
all unique targets from all six selected checkpoints, common ORIGINAL value
targets, retained named moments, and prospective final/last-eight stability
criteria. Smaller-step success alone cannot establish native long-term stability.

The original diagnostics are now complete: 78/78 actor adaptation tests pass,
while 73/78 composite tests pass. Five failures are critic errors on two unique
targets: seed 29/20,480 recipes 4 and 9 (MSE 1.39740), and seed 43/8,192 workforce
and recipes 4 and 9 (MSE 1.83489). Both final checkpoints pass all 13 tasks. The
frozen selection of intermediate saturation/confidence extremes was therefore
necessary; inspecting final checkpoints alone would have missed these failures.
All 159,744 reserved diagnostic draws were used, with no retained checkpoint
modified. These are discarded synthetic PPO draws, separate from native samples.

The matched remedy comparison is complete. All 56 unique original critic traces
were reproduced exactly, covering the 78 original tasks. Four variants used the
same original targets and retained named moments. It took 135.871 seconds and
57,344 critic minibatches, with zero native steps, actor updates or action draws.

| Variant | Targets meeting final and last-eight MSE < .9 | Worst final MSE | Worst last-eight MSE |
|---|---:|---:|---:|
| Original LR .0003 | 51/56 | 1.83489 | 1.83489 |
| LR .0001 | 56/56 | .15339 | .26076 |
| LR .00003 | 56/56 | .08808 | .09658 |
| Critic LayerNorm, LR .0001 | 3/56 | 78.79275 | 98.61215 |

The added last-eight criterion belongs to this new comparison; it does not
rewrite the original 73/78 composite gate. LR .0001 is the primary mechanical
candidate because it is the smaller intervention and has a large margin. No
claim of optimal learning rate or longitudinal stability follows. LayerNorm
entry errors against original targets were 194.95–1,251.84 MSE; direct insertion
therefore incurs substantial calibration cost. This result does not show that
LayerNorm cannot help longer training, a properly calibrated conversion or a
model initialized with normalization from the beginning. It rules out treating
this particular insertion as a transparent, validated upgrade.

## Trade as an MLP input and output

The 761-feature input includes own commodity stocks, recent imports/exports,
local economic context and known-contact rows. Up to eight contacts expose
distance, knowledge age, affinity, recent bilateral flows per resident-year,
migration, kinship and visibility. No hidden remote inventory is exposed.
Own stocks use log1p; signed context values use sign(x) × log1p(abs(x)). These
fixed transforms compress large magnitudes without fitting evaluation data or
hard-clipping every large trade to the same value. A historical trade-utility
channel exists, but autonomous policy quotes carry no planner utility and that
channel is zero for them; feedback comes from realized flows, stocks, needs and
welfare, not a heuristic utility judgment.

Actions are bounded conditionally: quote enable, known partner, offered stocked
good, a different requested good, an offered fraction of 5/20/50% of own opening
stock after quantity-cap adjustment, and a requested/offered ratio of .5/1/2. The native API also bounds each
quantity to 1e-6..1e6, escrows actual opening stocks and settles deterministically
without utility vetoes. Thus the initial MLP does not emit arbitrary unbounded
prices or quantities. Correct conditional likelihoods score requested actions;
inactive branches contribute zero log-likelihood, while prospective joint
entropy weights downstream choices by their reach probability.

## Optional GCS curriculum reuse

The retained broad-domain manifest contains 499 records. Metadata identifies
234 cultivation-only TRAIN candidates in 161 complete world/lineage/counterfactual
groups across nine families. There is no detected content/seed overlap with the
current bank and no partition-group collision. These are metadata candidates,
not verified native-compatible snapshots. The historical action/observation
contract and normalizer cannot be reused with the new model.

This environment has neither a configured GCP identity nor permitted Storage
endpoint access, so no live bucket read or blob download was performed. The
catalog records immutable manifest/shard generations and digests for later
materialization. Future reuse must verify those pins, load unchanged native saves,
rebuild current knowledge-isolated observations/masks, preserve group splits and
HELDOUT isolation, and pass native replay/conservation and action-coverage gates.
No GCS record was mixed into the completed continuation.

## Next safe training gate

Preserve the failed continuation verdict. Before a native LayerNorm comparison,
define a value-preserving conversion/calibration gate using TRAIN data, preserve
the actor and all existing evidence, and validate values on disjoint TRAIN
groups without fitting DEV/HELDOUT. Do not hide conversion damage by rebasing
diagnostic targets. The smaller-step candidate and any calibrated normalization
candidate still need a new explicitly bounded native canary: retain
movement, trade, migration and fission, and hold the eight-turn bank and reward
horizon fixed while comparing smaller critic steps against critic normalization.
Track value disruption, critic error/conditioning, actor likelihoods, actual
optimizer movement and final/worst-checkpoint retained adaptation. Do not infer
long-term readiness from a short synthetic recovery alone.

Only after that gate should a matched 8-versus-25-turn curriculum comparison
change reset duration, followed by a separately justified discount comparison.
Keep native transition budgets equal and report per-family population, QoL,
welfare rate, capital retention, food and focal trade outcomes. GCS expansion,
learned reciprocal counterparties and daughter-policy coverage are subsequent
gates. Broad/GPU scale-out, main merges and additional ecology work remain
outside this completed envelope.

Keep reward exclusively village-local, as explicitly confirmed by the user.
Learned daughter integration must create separate observation/reward/trajectory
streams for each new Elder; never add a daughter's welfare to its parent's
reward. No lineage-based reward change is proposed.

Evidence is under `evidence/autonomous-continuation-v4/`; raw runs and immutable
checkpoints remain under `runs/` and are not replaced by compact summaries.

Independent final review verifies all 78 original cases, all 56 matched target
traces and input pins, welfare/trade/GCS qualifications, and independently
reproduces exact selected-checkpoint native replay/parallel/PPO/Adam/RNG/inference
checks on all six checkpoints. Its verdict is **NO-GO for further training
promotion**. LR .0001 is a mechanical candidate only. The next preparatory gate
is value-preserving, TRAIN-only critic-LayerNorm calibration with disjoint TRAIN
validation; any subsequent native comparison requires a new sample budget.
See `evidence/autonomous-continuation-v4/independent-final-review.md`.
