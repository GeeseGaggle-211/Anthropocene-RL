# Autonomous Elder collapse investigation and v3 gate

Here model v3 uses native action3/observation4/environment4/replay4. It is distinct from the historical native-contract-v3 vertical slice.

The historical v2 native canary remains a failed gate. It consumed40,960 additional native transitions and stopped all three seeds on the disposable critic endpoint test. Longer and lower-step-size trajectories showed optimizer overshoot, not a dead critic. Retained-Adam action tests separately exposed a real inability to adapt several categorical choices. The fixes target that mechanism; low economic reward is not treated as collapse.

## What changed

Model v3 retains the 761-feature MLP interface, separate 256×256 actor/critic trunks, masked workforce/recipe Dirichlet shares and explicit Beta controls. It adds actor-only LayerNorm before each tanh, exact 95% softmax/5% legal-uniform categorical distributions, exact conditional entropy and a .05 prospective reverse KL(Ulegal||raw-softmax) penalty. The critic has no LayerNorm; its actual learning rate is .0003. All existing weights and populated Adam moments survive conversion by parameter name and shape; only new LayerNorm affine parameters start fresh. New actions are collected under the converted policy. Separate model/checkpoint/inference versions protect old experiments and the five-head contract.

LayerNorm is not a sufficient fix for overconfident outputs. At coefficient 0.01, seed 17 quote-enable adaptation gains were .002018 without LayerNorm and .001618 with it; both failed. With .05, both recovered. The exact-state-baseline diagnostic also showed that some earlier control failures were noisy credit assignment in the diagnostic, rather than a dead control head. This baseline is used only in discarded tests, not native training.

| Seed17 task | v2 without LayerNorm, .05 | v3 actor LayerNorm, .05 |
|---|---:|---:|
| Movement signed response gain | .788338 | .762246 |
| Quote enable gain | .779869 | .701093 |
| Field-expansion control gain | .646197 | .666940 |

All three paired tests pass for both candidates. The comparison does not establish overall superiority of LayerNorm. V3 is the selected candidate because it keeps hidden derivatives active and passes all 13 covered action-group adaptation tests; its possible information-loss/behavior-shift cost is measured rather than ignored. Entry actor saturation fell from54%/83% to0%/0% at seed 17. Full action-distribution KL from v2 to v3 is .680/.565/.497 nats at seeds 17/29/43. Native entry evaluation survived192/192 paired episodes with exact replay, while v3 welfare was lower than v2. That economic decline is acceptable evidence, not a reason to conceal or reject the conversion.

## How trade enters the MLP

Trade is represented in both observations and conditional actions. The fixed context contains up to eight known contacts with distance, knowledge age, affinity, recent bilateral trade utility/flow per resident-year, migration, kinship and visibility. Commodity rows provide own stocks, recent imports/exports and production/trade availability. The learner concatenates own 11-good stocks, legal-good and counterparty masks, plus the other economy/movement masks into761 features. Counts/flows use fixed signed-log compression; own stocks use log1p. No hidden remote inventory or evaluation-fitted normalizer is exposed. Contact slots and quote slots use the same native ordering.

The initial quote policy makes bounded decisions in sequence: enable; known counterparty; offered stocked good; different requested good;5/20/50% of own opening stock; requested/offered quantity ratio .5/1/2. Thus it need not emit an unbounded price or quantity. The native quote API additionally enforces quantities1e-6..1e6 and authoritative opening-stock escrow/settlement. Legal masks, conditional sampling and PPO likelihoods use the same distribution. A sampled disabled downstream trade branch contributes zero log-likelihood and consumes no downstream RNG. Exact joint entropy still includes prospective downstream entropy weighted by quote-enable probability; structurally impossible branches contribute zero to it. The5% mixture preserves legal exploration, and the reverse-KL term retains a useful raw-logit gradient when a softmax head becomes overconfident. There is no heuristic utility veto or policy rescue.

This gate verifies bounded quote mechanics and adaptable choice heads. Profitable learned exchange, richer price/quantity resolution, learned reciprocal counterparties and unrestricted economic competence remain separate work.

## Validation and limits

The learner suite passed 155 tests with 6 native opt-in skips. The focused native/model suite passed 7, including v2/v3 probability scoring, explicit conversion, actual populated-Adam save/resume, inference equivalence and exact 1-versus-4 collection/update equality. Source/manifest checks preserve the native source SHA256 `8a1319571e1204a8330306277b96529de23ec3932efff11c61d1964efe996f52` and the previously validated 69 native check groups,471 autonomous checks,143 legacy checks and eight old-mode fingerprints. The simulator source did not change during this collapse investigation.

Native complete-boundary timing, including observation construction and economic execution, remains the measured81.775 ms heuristic versus62.959 ms autonomous median at 80 camps (seven samples each). These are native timings, not a claim about end-to-end PPO throughput. New canary timing and resource results are recorded in its summary.

The cultivation curriculum retains survival, trade, migration and fission families with movement/trade/migration/fission enabled throughout. The current 13-task coverage comprises workforce, seven categorical choices, four active recipe faces and one field-expansion control. Rare livestock/advanced-era faces are not covered, and fission daughters still use heuristic controllers. Short-horizon survival and mechanically responsive heads do not establish economic competence or long-horizon stability.

## Native v3 canary outcome

Both seeds completed 4,096 additional transitions each, 8,192 total in 373.58 seconds. Seed 29 development welfare changed 140.02→141.60; seed 43 changed 136.19→121.93. Final 64/64 development episodes survived and replayed exactly. Development welfare is cumulative native QoL × population × elapsed time per initial focal resident, averaged equally across the four families. The welfare decline is retained as an economic outcome, with no rescue or retrospective acceptance threshold. All covered fresh recovery checks passed at both final boundaries. Actor tanh saturation ended at .21%/0% and .059%/0%; critic fresh-probe MSE was .268/.214. Actual actor and critic updates remained nonzero. Seed 29 produced three focal training trades; final development policies settled zero, so learned trade competence is not claimed.

Mean complete rollout plus PPO-update time was 39.87/39.58 ms per training transition, with peak measured main/native-process RSS 891/896 MiB. Compared with the old canary's roughly 33 ms, this is a modest throughput cost under changed policies/world trajectories, not a controlled LayerNorm-only timing estimate. No memory/latency-growth or optimizer-stall pause fired. The earlier native complete-world heuristic comparison remains separate.

Actual final checkpoint verification passes exact 1-versus-4 native rollouts, next PPO update, populated Adam/RNG state and v3 inference likelihood equivalence. Immutable checkpoints:

- `runs/autonomous-collapse-canary-v3/seed-29-step-4096.pt`: SHA256 `3f50dfd885f8d07d53783a39461cdb4dda7ea1914928cf4d7ad4dcb23291c847`.
- `runs/autonomous-collapse-canary-v3/seed-43-step-4096.pt`: SHA256 `a073f667869c8111d988865a2e8a1159d165aec11ef1b9a8cffa0c5e5cbf0516`.

Final retained-Adam tests pass all 13 covered action groups on both checkpoints (26/26). Minimum signed response gain is .047291; all final signed separations exceed .01 and critic target MSE remains below .9. Every task meets endogenous branch-exposure thresholds. The entire discarded diagnostic campaign used 192,512 draws, below its 200,000 cap; none of those artificial-reward updates entered the retained models. Independent final review gives **GO for model-v3 plug-in readiness and capped CPU continuation within cultivation/parent-Elder scope**. It independently reproduced both final checkpoint replay/parallel/PPO/Adam/inference checks. The next permitted gate is at most 24,576 more native transitions each for seeds 29/43 (49,152 total), four native CPU workers, one Torch thread and 12,493.234 seconds of remaining cumulative native time. Freeze its configuration/hashes before launch and apply the exact worst-checkpoint selection/diagnostic cap in the scale-out plan. Broader or GPU training is not approved by this gate.

## Plug-in boundary

Use `anthropocene_rl.policies.autonomous_inference_v3.AutonomousElderPolicyV3` with an explicitly pinned checkpoint and the expected provenance recorded in the canary evidence. `act_many({camp_id: native_wire_observation, ...})` returns native action dictionaries for a shared `AutonomousTransport.step_world` WEGO boundary. It sorts camp IDs and evaluates each knowledge-isolated observation; actor LayerNorm operates within each observation, never across settlements. Validate the checkpoint SHA256 from `evidence/autonomous-collapse-canary-v3/checkpoints.json` before loading. The actual final-checkpoint replay test verifies loader likelihood equivalence. Training continuation must restore `AutonomousTrainerV3` with the exact checkpoint configuration/provenance and preserved optimizer/RNG state.
