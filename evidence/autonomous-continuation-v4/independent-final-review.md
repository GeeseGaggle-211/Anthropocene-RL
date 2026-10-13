# Independent continuation-v4 final review

Reviewer: `/root/independent_review`.

**Verdict: NO-GO for further training promotion or broad scale-out.** The frozen native continuation completed its authorized work, but the selected-checkpoint retained-optimizer gate failed. The full 98,304-transition additional-native-training envelope is exhausted. Neither unused wall time nor successful diagnostic replicas creates a new training allowance.

## Verified implementation and native evidence

The continuation kept the native simulator, model-v3 policy, reward, training bank and optimizer configuration fixed. I checked actual source/config hashes against recorded provenance, cumulative accounting, exact checkpoint restore and deterministic selection. Both seeds completed 24,576 additional transitions: 49,152 in this continuation and 98,304 cumulatively. Native continuation wall time was 2,087.157 seconds; cumulative campaign time was 3,993.923 seconds. All 14 entry/scheduled fresh recovery checks passed. Both seeds made 384 actual PPO updates with finite, nonzero actor and critic parameter displacement.

I independently reran `scripts/verify_autonomous_v4_selected_resume.py` on all six selected checkpoints after reviewing its narrowly typed provenance reconstruction. One-versus-four-worker native actions, likelihoods, replay fingerprints, next PPO update, model/Adam/shuffle state and inference equivalence passed; originals remained unchanged. `/tmp/independent-autonomous-v4-selected-resume.json` exactly matches the retained selected-resume evidence. The initial verifier rejection of stringified integer ledger keys is preserved and was not a model or replay failure.

Independent focused tests during this review: continuation budget/selection and completion checks **3 passed**; typed provenance **1 passed**; critic-conversion/deduplication/partial-evidence checks **3 passed**. The primary full suite reports 163 passed with six native opt-in skips. Existing native physical/legacy checks and complete-world benchmarks were carried forward under the unchanged native source, not falsely presented as new measurements.

## Frozen gate and causal attribution

All six selected checkpoint reports completed: **78/78 actor adaptation criteria passed; 73/78 composite criteria passed**. Five failures represent two distinct critic targets:

- Seed 29, boundary 20,480: recipe tasks 4 and 9, critic MSE **1.39740479**.
- Seed 43, boundary 8,192: workforce and recipe tasks 4 and 9, critic MSE **1.83488667**.

Both final checkpoints passed their 13 composite tasks. The preregistered intermediate checkpoint selection therefore found failures that final-only inspection would miss. The full 159,744 discarded policy-diagnostic draws were used. Actor success and nonzero gradients do not waive the critic failures; observed critic learning followed by oscillation is evidence of update instability rather than an irreversibly dead network.

I independently checked the completed critic comparison's source/input/checkpoint/report pins, deduplication multiplicities, thresholds and all 56 original 64-point traces against the original task reports. The comparison represents all 78 tasks and uses common original targets, retained named Adam moments and identical shuffle/clipping. It completed 57,344 critic minibatches in 135.871 seconds with zero native steps, actor updates or action draws.

| Discarded variant | Targets passing final and last-eight MSE < .9 | Worst final MSE | Worst last-eight MSE |
|---|---:|---:|---:|
| Original critic LR .0003 | 51/56 | 1.834887 | 1.834887 |
| Critic LR .0001 | 56/56 | .153391 | .260756 |
| Critic LR .00003 | 56/56 | .088082 | .096585 |
| Direct critic LayerNorm insertion, LR .0001 | 3/56 | 78.792747 | 98.612152 |

The last-eight requirement is a prospective criterion for this separate comparison; it does not rewrite the original 73/78 gate. **LR .0001 is a mechanically validated primary candidate**, not a production-ready or optimal setting. It is the smaller change with a substantial stability margin. There is no proof that it prevents representation saturation under longer native training.

Direct LayerNorm insertion reduced hidden saturation but produced original-target entry MSE 194.95–1,251.84. Its failure here measures conversion and retained-state calibration cost. It does not establish that normalization cannot help longer training, calibrated conversion or initialization from scratch. Lower saturation and larger local derivatives alone are insufficient promotion evidence.

## Economic measurements and scope

I checked the raw welfare digest, table values and both raw and normalized decompositions. Native reward is intentionally **exclusive to the local village**: prior focal population × clipped end-interval QoL × elapsed years. The displayed score normalizes each episode by its own initial population and then weights families equally; it is distinct from the discounted PPO target. Development entry-to-final scores improved **141.60 → 171.64** and **121.93 → 135.53** despite nonmonotonic intermediate outcomes.

The 25-decision inference check was separately bounded: 800 native steps, 32 episodes, zero training updates, exact saved eight-decision scalar prefixes and final-prefix fingerprints, and full new replay. All 32 focal Elders survived, but the matched survival-family populations fell **52 → 26** and **49 → 12.75** between decisions 8 and 25. Those are focal population changes, not death counts; fission and migration also occurred. Larger cumulative welfare over 500 rather than 160 years is not itself improvement. No final development or 25-decision focal settlements demonstrate learned trade competence. Native training recorded 10/3 focal trades, which must remain separate from background world trades.

Automatic learned control of newly created daughter settlements is still missing. Each daughter needs its own Elder observation, local reward and trajectory stream; shared policy weights are compatible with that requirement. **No daughter welfare should be added to the parent's reward, and no lineage-reward redesign is recommended.** Current daughters use native heuristic controllers.

GCS inspection remains metadata-only. The 234 prospective TRAIN records in 161 complete groups have not been loaded as current native states. Schema, mask, physics, replay and partition compatibility still require a separate materialization gate. The original normalization/action traces are not reusable learner inputs.

## Concrete next gate

Preserve the original failure and immutable checkpoints. For the normalization candidate, first preregister a bounded TRAIN-only value-preserving conversion/calibration protocol, including initialization, allowed parameter changes, retained/new optimizer-state handling, calibration budget and disjoint TRAIN validation groups. Check original-value preservation and common-original-target adaptation; do not hide conversion disruption by rebasing targets or fit DEV/HELDOUT data.

The smaller-step candidate and any successfully calibrated normalization candidate then require a **new explicitly bounded native canary** with actor, village-local reward, eight-decision bank and discount horizon fixed. Track value disruption, critic conditioning, actual optimizer updates and retained adaptation at final and preregistered extreme checkpoints. This review approves neither that unimplemented canary nor additional native samples.

Only subsequently compare eight versus 25 decisions or change the physical discount horizon, using equal transition budgets and per-family trajectories. Twenty-five decisions span 500 years in these fixtures, but actual cultivation payoff delay must be measured before inferring the appropriate discount. Rare action faces, learned reciprocal peers and automatic learned daughter lifecycle remain separate integration/coverage gates. No main merge, GPU ecology or unrestricted training is approved.
