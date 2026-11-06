# r5 critic: decision traces on fit8-b01 (E5 and F1)

Tracer report, 28 Oct 2026, written about 11:30 IST. Read-only. No code was changed, nothing was pushed and no VM was touched.

## Scope and provenance

| Item | Value |
|---|---|
| Code | Anthropocene-RL `821660b`, the clean checkout `worktrees/r5-multi-dataset-rl` (a byte-identical copy was used for running) |
| Simulator source, read only | `worktrees/r5-sim` `8bd480b` |
| Data | the local fit8-b01 (8 worlds 90100–90107, 33,450 rows, 0 quarantined, 0 tail-unavailable) and its `actor-normalizer.pt` |
| Checkpoints | `pilot/ckpt/e5/{checkpoint-000.pt, critic.pt}` (sha256 `220dafa7…`, `fc974dd6…`, both verified) and `ckpt/shared/priv-normalizer-b01.pt` |
| F1 | `pilot/ckpt/f1/{checkpoint-000.pt, critic.pt}` (sha256 `0e09da5c…`, `f5d00977…`, both verified), uploaded 06:01Z. Separate QoL encoder, no QoL loss weight, otherwise the same as E5 (lr 1e-4, 4 epochs, 524 steps, same family schedule, trained on b01, so **in-sample** too). |
| E5 run | E5 trained on b01 itself, so every E5 number here is **in-sample**. It is the `critic.pt` summary, not my reconstruction: shared encoder, logistic QoL head with scale 2 (range 0 to 2W = 820.17 QoL-years), TD(λ = 0.8), QoL loss weight 100, lr 1e-4, 4 epochs, 524 steps (131 per epoch), batch 256, family weights `default,under-clothed=2` with cosine decay. |

**Local loader workaround.** Both E5 checkpoints fail `LaborTaskCritic.check_support` (`labor_model.py:1670`) on CPU. The saved `bins` buffer differs from the CPU `torch.linspace(-0.5, 2.5, 255)` in 79 of 255 entries, by at most 2.4e-7, which suggests a CUDA linspace. My scripts patch the check to `allclose(atol = 1e-6)`; the repository is untouched.

- **Observed:** a CPU `evaluate`/`load` of a GPU-written checkpoint will raise.

**Units:**
- QoL-years. W = 410.083 (`labor_model.py:204`). The head decodes 2W·sigmoid(z), and the BCE target is p* = G / 2W (`labor_model.py:1635-1640`).
- Population-years, with reference scale s = 85,133.98 (`contracts.py:37`). The two-hot target is symlog(G / s), clamped to [-0.5, 2.5] over 255 bins (`labor_model.py:179`, `:1724-1745`).
- Discount γ = exp(-20/400) = 0.95123 per 20-year turn.

**Where the scripts and outputs are.** They are in the session scratchpad under `trace/work/`: `trace.py`, `step.py`, `globalstats.py`, `actorcmp.py`, `worldid.py`, `V_e5init.pt`, `V_e5.pt` (decoded predictions for all 33,450 rows), `trace_e5.json`, `trace_last.json`, `step_e5.json` and `globalstats.json`.

## Selected decisions

| Case | Row | World / camp / turn | Window position | Why chosen |
|---|---|---|---|---|
| A, ordinary | 6775 | 90101 / 26 / 35 | 25 of 50 | large-camp only, label valid, a stationary 1,000-resident camp |
| B, delayed investment | 13475 | 90103 / 29 / 18 | 8 of 50 | first Farmer staffing of the trajectory (0 → 128 farmers), tagged `first-use-farming` |
| C, boundary | 17948, then 17984 and 20938 | 90104 / 24 / 9 → 11 → 58 | 0 of 49 | the **only** `transition_terminal` row in all of b01. The camp then reappears |
| D, label unavailable | 21145 | 90105 / 33 / 15 | 6 of 50 | `label_specialty_valid = False` (one of 5 such rows in b01) |

Last rows used for the tail part: 9189 (A), 17835 (B), 20938 (C) and 23679 (D).

Facts about the data that shaped the selection (all observed):

- **No rejected or retried attempts exist.** `initial_accepted == total_decisions` and `quarantine == []` in all 8 b01 shards. The same is true in every published b02–b07 shard (`qmeta.txt`). The retry edge has never fired in r5 teacher data.
- **No camp ends inside the window.** Every trajectory runs to the world's last turn (58 or 59). The single terminal flag is a camp that **comes back** (case C).
- **One camp per world never has a trade label.** It is the player-0 camp: camps 25, 24 and 20 in 90100, 90104 and 90105, with 49–50 rows each. That is why `full_action_label_valid = total - 50`, and it matches "player 0 never trades".
- **The heuristic teacher falls into a 2-turn limit cycle** in most mature camps. Clothing per capita alternates ~0.93 ↔ 0.00 and residents oscillate (B: 849 ↔ 749). The QoL reward alternates 8.2 ↔ 15–17.7 QoL-years per turn.

---

## A. Ordinary mid-window decision: row 6775 (90101, camp 26, turn 35)

**1. Knowledge.**

Privileged raw values, critic only:

| Field | Value |
|---|---|
| residents | 1000 (linked 1000) |
| food | 88,713 (plant only) |
| cloth/cap | 1.00 |
| housing/cap | 1.00 |
| Farmer / Gatherer workers | 90 / 866 |
| meals per turn | 80,000 |
| footprint-food-over-meals | 1.10 |
| site-food-over-meals | 2.20 |
| episode-age-turns | 25 |
| era-elapsed-years | 680 |
| acquired-cultivation | 1 |
| daughters | 1 |

- The camp holds exactly 1,000 residents with 20,000 population-years per turn from turn 32 to 59, while food rises by about 6,600 per turn. That looks like a binding size limit, not a growth equilibrium (**hypothesised**).
- Actor inputs clipped at ±10: 7 values in `labor_resource_features`, 0 in core. Privileged inputs clipped: 0.

| Critic | QoL prediction | Population prediction |
|---|---|---|
| init (E5 checkpoint-000) | 434.5 QoL-years (z = 0.119) | 246,182 |
| E5 | 285.8 (z = -0.626) | 383,658 |
| F1 | init 469.7 → trained 288.2 | init 246,182 → trained 384,377 |

**2. Sampling and probability.** Nothing was sampled. The shadow heuristic step produced the action (`shadow_teacher_labels`, `labor_bc.py:915`).

- The actor is trained on `target_specialty = counts / P` (`labor_bc.py:642`): Gatherer 0.871, Farmer 0.090, Weaver 0.013, Stonecutter 0.012, Toolmaker 0.004, Builder 0.007, Carpenter 0.002, Potter 0.001. It uses a masked soft-label cross entropy weighted by `row_weight × label_specialty_valid` (`labor_bc.py:1989-2001`).
- The executed payload is the label's `drive_sample` (`labor_bc.py:722`). Here counts sum to P = 1000, so drive equals label.
- **No behaviour log-probability exists.** `_retained_action` hard-codes `behavior_labor_logp = 0.0` and `behavior_nonlabor_logp = 0.0` (`labor_bc.py:836`, `:851`), and `record_teacher_transition` stores no logp column at all (`labor_bc.py:1461-1548`).
- **The PPO path is different.** `_execute_single_step_detailed` samples from the actor and stores `labor_log_prob` / `nonlabor_log_prob` (`labor_training.py:330-334`, saved at `:436-446`, retry logp at `:378`). The ratio uses them (`labor_ppo.py:554-555`).
- The frozen actor in this checkpoint (iteration 0) puts at most 0.11 on any specialty. Its L1 distance to the teacher label is 1.55 out of 2, and its CE is 2.32 against a label entropy of 0.52. The critic is therefore learning V of the teacher, not of this actor.

**3. Native execution.** `validate_labor` (pure) accepted the initial payload (`labor_ppo.py:259`), so no retry factory ran (`:276`). `step_labor` executed it (`:296`) with feasibility cost 0. The only clamping is the simplex canonicalisation of the wire payload (`labor_ppo.py` `RetainedLaborAction.validate`). Executed equals label.

**4. Reward and successor.**

| Quantity | Value |
|---|---|
| QoL reward | 14.597 QoL-years (lineage QoL 0.730 × 20 y) |
| Population reward | 20,000 |
| Feasibility / guard | 0 / 0 |
| Elapsed | 20 y, γ = 0.9512 |
| Terminal | no |
| Successor | row 6870 (turn 36) |
| Trajectory tail (last row 9189, turn 59, cutoff 60) | method 2 (independent randomized), QoL 288.55 ± 60.46 SE, population 395,358 ± 82,832 |
| World schedule factor | 0.964 (`labor_pretrain_eval.py:742`) |

The last row's TD and MC target is 14.60 + 0.9512 × 288.55 = 289.08. Its tail weight is 0.95.

**5. Target and loss.**

| Target | Value (QoL-years) |
|---|---|
| MC (`labor_bc.py:1761`) | 296.22, of which the tail carries 0.279 (coefficient 0.287) |
| TD(0.8) with init V, the step-0 refresh at `labor_bc.py:3371` | 415.69 |
| TD(0.8) with E5's V | 287.88 |
| n-step k = 5 (`labor_pretrain_eval.py:911`), init V | 418.61 |
| n-step k = 5, E5's V | 288.29 |

- TD recursion with E5's V (`labor_advantage.py:1751`): 14.597 + 0.9512 × (0.2 × V(s′) = 285.30 + 0.8 × G′ = 287.79) = 287.88.
- In the TD(0.8) target the tail coefficient is only 0.00135, against 0.287 in MC. At mid-window the λ-target sees the tail almost only through bootstrapped V.
- Population: the MC target is 405,864, symlog 1.752, two-hot bins 190 and 191 with weights 0.313 and 0.687. CE is 5.017 at init and 2.992 after E5.
- QoL BCE against the init TD target (p* = 0.507) is 0.6941, with entropy floor 0.6931, so the excess is only 0.0011.
  - **The init loss is already at its floor for this row because the init TD target is the init V.**
  - Against MC (p* = 0.361): BCE 0.7115, floor 0.6541.
  - After E5, against its own TD (p* = 0.351): BCE 0.64808, floor 0.64806, excess 1.4e-5.
- Weights:
  - World-balanced row weight 1.027.
  - Family multiplier 1, so the sampling probability is 1.56e-5 at step 0 and 3.07e-5 at the end; about 0.5 draws per epoch early on.
  - Labels valid for specialty, movement, fission and trade; activity roles valid only where staffed.

**6. One update.** One Adam step from init: lr 1e-4, clip 1.0, the batch is this row plus 255 rows drawn with the step-0 family-weighted probabilities (seed 20261028 + row), QoL weight 100 as in E5, init TD targets.

- Pre-clip gradient norm 43.5.
- This row's QoL prediction went 434.5 → **249.0** (z 0.119 → -0.830), so its BCE rose 0.6941 → 0.7826.
- Population went 246,182 → 249,904, with CE 5.017 → 4.990.
- The batch-mean QoL BCE rose 0.6776 → 0.7523.
- **Observed: the first step overshoots.** Adam's first update is about lr·sign(g) on every parameter (clipping does not change it), so the bounded head's logit shifts by about -0.95 for all rows together. The step went straight past the target of 415.7.
- **Control, the same step with QoL weight 1:**
  - pre-clip gradient norm 23.3;
  - QoL 434.5 → 399.7 (z -0.170), so the row BCE fell 0.69411 → 0.69382;
  - batch QoL BCE 0.6776 → 0.6772;
  - population CE 5.684 → 5.581.
- **Observed:** with weight 100, QoL decides the sign of every shared parameter's first step.
- **F1, from its own init with no weight:**
  - pre-clip gradient norm 23.3;
  - QoL 469.7 → 323.5 (z 0.293 → -0.429), so the row BCE rose 0.6943 → 0.7299;
  - batch QoL BCE 0.6686 → 0.7089;
  - population CE 5.017 → 4.964.
- So the separate encoder overshoots too: all of its parameters serve QoL alone. See item 9.
- After training, E5 predicts 285.8 against MC 296.2, TD 287.9 and n-step 288.3: within 0.025W of all three.
- F1 predicts 288.2 against its own TD 291.2 and n-step 291.9.

---

## B. Delayed investment: row 13475 (90103, camp 29, turn 18, first Farmer staffing)

**1. Knowledge.**

| Privileged field | Value |
|---|---|
| residents | 366 |
| food | 5,176 |
| cloth/cap | 1.00 |
| housing/cap | 0.71 |
| Farmer workers before the decision | 0 |
| FarmTools | 0 |
| footprint-potential-food-over-meals | 6.82 |
| sustainable-herd-over-animals | 51.7 |
| last-growth | 0.65 |
| episode-age | 8 |
| era-elapsed-years | 340 |
| cultivation and husbandry acquired | 1 |

- Clipped inputs: 6 values in `labor_resource_features`; nothing in core or privileged.

| Critic | QoL prediction | Population prediction |
|---|---|---|
| init | 293.8 | 249,137 |
| E5 | 273.4 | 246,648 |
| F1 | init 276.0 → trained 266.3 | init 249,137 → trained 257,051 |

**2. Sampling and probability.** Teacher label: Gatherer 0.552, Farmer **0.350** (128 workers), Builder 0.038, Weaver 0.019, Craftsperson 0.014, Stonecutter 0.014, Herder 0.008 and Toolmaker 0.005. Counts sum to P = 366, so drive equals label. No logp is stored (as in A). The actor puts 0.106 on Farmer.

**3. Native execution.** Accepted at attempt 0 and executed as labelled.

The investment itself then oscillates. Farmers by turn:

| Turn | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|---|
| Farmers | 128 | 14 | 17 | 164 | 305 | 41 | 254 |

From turn 22 the camp settles into the 2-cycle: farmers swing 290 ↔ 60 and cloth/cap 0.93 ↔ 0.

**4. Reward and successor.**

| Quantity | Value |
|---|---|
| QoL reward | 10.42 |
| Population reward | 7,320 |
| Feasibility / guard | 0 / 0 |
| Successor | row 13525 |
| Later QoL rewards (turn 19 onward) | 10.12, 11.21, 10.88, 9.31, then alternating 16.9–17.0 / 9.1 … to 8.20 / 14.87 at turn 58–59 |
| Population-years per turn | 7,320 → 16,980 at turn 59 (residents 366 → 849) |
| Last row 17835 | turn 59, cutoff 60, QoL tail 321.71 ± 56.25, population tail 432,348 ± 76,374 |
| World schedule factor | **1.247** (the highest in b01) |

**5. Target and loss.**

| Target | Value (QoL-years) |
|---|---|
| MC | 257.98, tail weight 0.153 |
| TD(0.8), init V | 265.81 |
| TD(0.8), E5 V | 266.18 = 10.42 + 0.9512 × (0.2 × 273.74 + 0.8 × 267.65) |
| n-step k = 5, init V | 243.03 |
| n-step k = 5, E5 V | 274.30 |

- Tail coefficient under TD(0.8): 1.3e-5.
- The population MC target is 251,385: symlog 1.374, bins 158 and 159 with weights 0.298 and 0.702. CE 6.273 at init, 3.453 after E5.
- QoL BCE against the init TD target (p* = 0.324): 0.6325, floor 0.6299.
- **Family multiplier 10** (`first-use-farming`, `labor_scenarios.py:917`): sampling probability 1.37e-4 at step 0 (about 4.6 draws per epoch), decaying to 2.66e-5. World-balanced weight 0.891.

**Credit for the investment.**

- The QoL payoff of this decision is not visible in the rewards. Per-turn QoL after turn 18 is no higher than before (10–13). The camp's later pattern is the teacher's 2-cycle.
- The population payoff is large but spread over 40 turns: 7,320 → about 16,000 population-years per turn.
- Under TD(0.8) the weight on a reward k decisions ahead is (λγ)^k = 0.761^k: 0.25 at k = 5 and 0.065 at k = 10. Credit for a turn-30 harvest reaches row 13475 only through V(s′) at turn 19–23. So V must already encode "128 farmers staffed, fields started" in its state features (**hypothesised**; this cannot be checked without a counterfactual).
- Population-years still use MC (`labor_bc.py:2954`), so their credit is not truncated.

**6. One update.** Pre-clip gradient norm 44.0. QoL went 293.8 → **155.3** (z -0.583 → -1.454), so the row BCE rose 0.6325 → 0.6812. Population went 249,137 → 250,532, with CE 6.273 → 6.188.

After training, E5 predicts 273.4 against MC 258.0, TD 266.2 and n-step 274.3. It sits between its own bootstrap and MC, about 0.04W above MC.

F1 predicts 266.3 (TD 270.0, n-step 273.0). Its F1 one-step drop is 276.0 → 171.1.

---

## C. Boundary: row 17948 (90104, camp 24, turn 9), the only terminal, which then revives

**1. Knowledge at turn 9.**

| Field | Value |
|---|---|
| residents | **1** |
| linked population | **5** (4 people away in hunter groups) |
| food | 0 |
| cloth/cap and housing/cap | 0 / 0 |
| site-food-over-meals | 171 |
| footprint-potential | 1446 |
| episode-age | 0 |
| era-elapsed-years | 160 |
| cultivation acquired | 1 |

- Clipped inputs: core[96, 97, 98] (native names `v2-96..98`), 2 values in `labor_resource_features` and 3 privileged core values (the site and footprint ratios rail at +10).
- This is the player-0 camp, so it has no trade label.

| Critic | QoL prediction | Population prediction |
|---|---|---|
| init | 391.5 | 241,862 |
| E5 | 207.0 | 121,154 |
| F1 | init 402.3 → trained 192.6 | init 241,862 → trained 164,759 |

**2. Sampling and probability.** Label: Gatherer 1.0 (1 worker), specialty label valid, trade gates invalid. No logp is stored.

**3. Native execution.** Accepted and executed. After resolution the camp has `Population <= 0`, so the bridge sets `Terminated = true` (`ElderModel9.cs:1075`: `camp == null || camp.Population <= 0`). The QoL and population rewards are 0 and 0, and the guard reward is 0.

The camp **reappears at turn 11** (it is absent at turn 10). It has 1 resident, privileged `episode-age-turns = 2` (it counts on from turn 9), and stays live to turn 58, growing to 350–413 residents.

- **Observed:** the same camp id carries a terminal row followed by 48 live rows.
- **Hypothesised mechanism:** "terminated" means "zero residents", while linked hunter groups survive and return.

**4. Reward and successor.**

- Row 17948: terminal, so no successor inside its return. Its stored next row in the trajectory group is 17984 (turn 11).
- Row 17984 (the revival): QoL reward 0, population reward 20, guard 1. The QoL reward stays 0 for turns 11–13 although the camp is alive. Across b01, 252 live rows (0.75%) have QoL reward 0, all post-Neolithic.
- Last row 20938 (turn 58, cutoff 59): QoL tail 318.36 ± 62.34, population tail 194,551 ± 38,891, world factor **1.144**. Last-row target = 17.67 + 0.9512 × 318.36 = 320.50; tail weight 0.945.

**5. Target and loss.**

At row 17948:
- `discounted_returns` resets at the terminal (`labor_bc.py:1822`), so MC = 0, TD(0.8) = 0 (`labor_advantage.py:1812`, `g = reward`) and n-step = 0 (closure `NSTEP_TERMINAL`).
- p* = 0, so the entropy floor is 0 and **the whole BCE is excess**: 0.649 at init and 0.291 after E5. This is the largest per-row QoL excess among the traced rows.
- The population target 0 sits at symlog 0, bins 42 and 43 (0.667 / 0.333). CE 5.36 at init, 6.48 after E5, so it got *worse*.
- Family multiplier 5 (farming-unlocked, small-camp, under-housed, under-clothed), so the row is drawn about 3.5 times per epoch early on.

At row 17984 (revival, turn 11): MC 206.8, TD with E5 V 224.9, n-step 219.9, E5 prediction 236.9. Its multiplier is 10 (`first-use-farming`).

**6. One update.** Pre-clip gradient norm 46.1. QoL went 391.5 → 254.7 (BCE 0.649 → 0.372); population went 241,862 → 244,297.

- After E5 the prediction is still 207 QoL-years and 121k population-years against a target of 0. The critic cannot fit a zero that its inputs do not distinguish from a 1-resident camp with 4 hunters out. It is right not to.
- F1 behaves the same: 192.6 QoL-years and 164.8k population-years against 0, with population CE 5.36 → 7.42 (worse).
- **Observed: this row teaches "a 1-resident camp is worth 0", which contradicts the 48 rows that follow.**

---

## D. Label unavailable: row 21145 (90105, camp 33, turn 15)

**1. Knowledge.**

| Field | Value |
|---|---|
| residents | 160 |
| food | ~0 |
| cloth/cap | 0.97 |
| housing/cap | 0.16 |
| Farmer / Gatherer workers | 5 / 84 |
| watercraft acquired | 1 (first Fisher staffing this turn) |
| episode-age | 6 |
| era-elapsed-years | 280 |

- Clipped inputs: 2 values in `labor_resource_features`.

| Critic | QoL prediction | Population prediction |
|---|---|---|
| init | 475.9 | 245,906 |
| E5 | 330.0 | 148,240 |
| F1 | init 298.6 → trained 280.5 | init 245,906 → trained 140,053 |

**2. Sampling and probability.**

- `label_specialty_valid = False` and every `label_activity_role_valid` is False. Movement (code 1), fission and trade labels are valid.
- World 90105's reasons are one `craft-work-unknown` and one `fishing-harvest-activity-unavailable`. With 9 Fishers newly staffed, this row is most likely the fishing reason (**hypothesised**; reasons are not stored per row).
- The stored label values are counts / P: Gatherer 0.794, Craftsperson 0.094, Fisher 0.056, Farmer 0.044, Toolmaker and Stonecutter 0.006 each.
- **The actor is not trained on them.** The BC weight is `row_weight × spec_valid` = 0 (`labor_bc.py:1999`), and every activity role weight is 0.
- No logp is stored.

**3. Native execution.**

- The payload is still `drive_sample` (`labor_bc.py:722`). It is not over budget, so the drive counts equal the counts and the executed staffing is the teacher's.
- Accepted at attempt 0, no retry.
- `convert_heuristic_labor_label` marks the label unavailable (`labor_bc.py:721`) but the action is still executed.
- So the first-use-fishing decision is executed and **its reward enters the critic**, while **the actor never sees its label**.

**4. Reward and successor.**

| Quantity | Value |
|---|---|
| QoL reward | 6.90 |
| Population reward | 3,200 |
| Successor | row 21172 |
| Later QoL rewards | 14.76, 7.84, 9.69, 15.95, … then stable at about 18.4 from turn 28 (the cycle stops here) |
| Last row 23679 | turn 58, QoL tail 233.78 ± 66.25, population tail 153,278 ± 42,908 |
| World schedule factor | **0.683** (the lowest in b01) |

The last-row target is 15.67 + 0.9512 × 233.78 = 238.04.

**5. Target and loss.**

| Target | Value (QoL-years) |
|---|---|
| MC | 290.95 |
| TD(0.8), init V | 408.47 |
| TD(0.8), E5 V | 286.06 = 6.90 + 0.9512 × (0.2 × 307.23 + 0.8 × 290.03) |
| n-step k = 5, init V | 392.94 |
| n-step k = 5, E5 V | 281.74 |

- The population MC target is 168,797: symlog 1.093, bins 134 and 135 (0.140 / 0.860). CE 6.06 at init, 3.95 after E5.
- QoL BCE against the init TD target (p* = 0.498): 0.7068, floor 0.6931. Against MC (p* = 0.355): 0.7533, floor 0.6503.
- Family multiplier 5, world weight 1.541. Critic label validity is irrelevant: the critic keeps every non-quarantined, tail-available row (`labor_bc.py:2907`).

**6. One update.** Pre-clip gradient norm 41.5. QoL went 475.9 → 259.9 (BCE 0.7068 → 0.7636); population went 245,906 → 246,246. After training, E5 predicts 330.0 against MC 291.0, TD 286.1 and n-step 281.7: **+0.10W too high**, the largest residual among the four.

F1 predicts 280.5 (TD 266.5, n-step 260.7), so the separate encoder removes most of E5's error here. F1's one-step drop is 298.6 → 176.7.

At the last row, E5 predicts 279.0 and F1 280.3, against a target of 238.0 (+41). At world factor 0.683 the target is low and the critic sits above it.

---

## Whole-dataset context (E5, in-sample, units of W = 410.08)

| E5 vs | Bias | MAE |
|---|---|---|
| MC | +0.013 | 0.054 |
| own TD(0.8) | +0.004 | 0.023 |
| n-step k = 5 | +0.002 | 0.020 |
| MC, rows with tail weight ≤ 0.5 | +0.014 | 0.045 |
| MC, last 10 rows | +0.017 | **0.076** |

- The correlation of E5 predictions with MC is 0.78; at init it is -0.14.
- The init critic is biased +0.12W against MC.
- The population head improves from a constant ~246k at init (correlation -0.08) to a correlation of 0.977 with 9.2% relative MAE.
- On the late 2-cycle, first differences of E5 predictions are uncorrelated with first differences of MC (correlation 0.07). Mean |Δpred| is 2.94 against |ΔMC| 2.73 QoL-years, so the critic does not track the cycle's phase.

## High-level mismatches, ranked

1. **The world-level roulette draw drives the per-world targets, and the critic partly memorises it.**
   - *Observed:* over the 8 worlds, corr(schedule factor, world-mean MC QoL target) = **0.979**, slope 138.5 QoL-years per unit factor; corr(factor, world-mean tail) = 0.995. One stop schedule is shared by every camp of a world, so a factor of 0.68 to 1.25 rescales all its tails together. The SE scope "excludes world uncertainty"; the stop indices are in the meta.
   - *Observed:* E5's world-mean prediction follows the factor with corr 0.79 and slope 43.7 (init: 0.05 and 4.1). The TD(0.8) target follows it with slope 66.6. F1 behaves the same: corr 0.78, slope 37.4 (its init: -0.88, -13.2).
   - In-sample, the critic has therefore learned about a third of a pure sampling artefact as if it were state value.
   - *Observed:* a 1-NN world-identification test (`worldid.py`) on the normalized blocks, excluding the query's own trajectory (2,000 queries; chance is 0.125), shows which inputs identify the world:

     | Block | Accuracy |
     |---|---|
     | privileged **unlocks** (era and technology timing, lifetime practice, era-elapsed-years) | **0.584** |
     | all privileged blocks together | 0.25 |
     | actor core | 0.20 |
     | privileged herds | 0.16 |
     | privileged economy | 0.16 |
     | privileged site fields | 0.15 |
     | privileged core | 0.14 |
     | turn only | 0.14 |

     The unlocks family is effectively a world fingerprint: camps of a world share a technology and era timeline. Site constants are not.
   - *Hypothesised:* the unlocks block is how the critic fits this offset in-sample.
   - Expected consequence: a held-out world gets no matching offset. This matches the b04 nMAE ~0.97 against fit nMAE ~0.5, and the "late stratum worst".
2. **Late targets are tail-dominated, and the tail noise is as large as the residual.**
   - *Observed:* mean MC tail weight 0.40; 31% of rows above 0.5 and 3.5% above 0.9. On the last rows the target is r + 0.951·tail with tail weight 0.93–0.97.
   - The tail SE is 56–66 QoL-years, about 0.14–0.16W, against E5's last-row errors of -6, -61, -21 and +41 QoL-years. The last-10-row MAE (0.076W) is the worst position band.
   - Per row, the critic cannot beat this noise. Only the world-shared part (item 1) is learnable, and it is the wrong thing to learn.
3. **The critic learns V of the teacher, not of the actor, through a 2-cycle.**
   - *Observed:* the critic is trained on V of the heuristic teacher. The teacher's behaviour includes a 2-turn clothing and population limit cycle in most mature camps, with QoL reward 8.2 ↔ 15–18.
   - The checkpoint actor (iteration 0) is near-uniform: L1 to the teacher label 1.33–1.68 out of 2, CE about 2.1–2.3 against label entropy 0.0–1.08. No behaviour logp is stored (`labor_bc.py:836`, `:851`).
   - Any later off-policy use must treat these values as V of the teacher. PPO's own rollouts store logp (`labor_training.py:333-334`, `:436-446`).
   - *Hypothesised:* the cycle is an artefact of the heuristic's clothing and food rule. A critic fitted to it will rate states by cycle phase rather than by structure.
4. **Credit for delayed investment beyond the TD horizon goes only through V.**
   - *Observed:* the TD(0.8) per-decision weight is 0.761; the tail coefficient is about 1e-3 to 1e-5 at mid-window.
   - Case B's farming payoff is a population payoff spread over 40 turns, with essentially none on QoL.
   - Population keeps MC targets; QoL must carry the investment in V(s′) features. *Hypothesised:* with the cycle and world offsets dominating the QoL variance (within-world QoL spread is small; MC std 0.108W), the QoL head has little signal left to attribute to the investment.
5. **Terminal handling is wrong for "zero residents with hunters out".**
   - *Observed:* the only terminal in b01 is a camp that revives two turns later and lives 48 more rows: `Terminated = Population <= 0` (`ElderModel9.cs:1075`).
   - The row trains V = 0 against a near-identical revived state worth about 207–237 QoL-years. The population CE on that row gets worse with training (5.36 → 6.48).
   - *Observed:* there are no genuine extinctions in the window at all, so the critic has no real terminal evidence.
6. **Rejected-attempt rows: none exist.**
   - *Observed:* 0 rejected or quarantined decisions in b01, and in every published b02–b07 shard.
   - The retry edge, `_idle_retry` (`labor_bc.py:855`) and the AttemptGAE retry terms are untested by teacher data. PPO's first retries will be off the critic's support (**hypothesised** risk).
7. **Labels are unavailable on informative rows, but those rows still drive the critic.**
   - *Observed:* 5 specialty-invalid rows in b01, plus the one player-0 camp per world with no trade label (49–50 rows each).
   - The invalid row in D is a first-use-fishing decision: executed and valued, but absent from actor BC.
   - Impact is small by count, but those rows are concentrated on first-use events.
8. **Clipped features.**
   - *Observed:* core column 182 is clipped on 1,337 rows (4%). Core 33 on 50 rows, core 96–98 on 6, 8 and 3 rows, trade column 1 on 2,090 rows, and sites columns 4–6 on 731 rows each. The privileged core site and footprint ratios rail at +10 for tiny camps (case C: site-food-over-meals 171, footprint 1446).
   - `labor_resource_features` has 2–7 clipped values on every traced row.
   - *Hypothesised:* modest. The case most affected is exactly the small-camp boundary, where the inputs saturate and the target is degenerate.
9. **Optimiser start (secondary).**
   - *Observed:* one Adam step from init (lr 1e-4, clip 1.0) overshoots the QoL head in both arms. Clipping does not change Adam's first, sign-like step.

     | Arm | Pre-clip gradient norm | Traced QoL predictions | Δz | Batch QoL BCE |
     |---|---|---|---|---|
     | E5 (QoL weight 100) | 41–46 | -137 to -216 QoL-years | -0.7 to -1.1 | 0.678 → 0.752 |
     | F1 (separate QoL encoder, weight 1) | 22–24 | -105 to -146 | -0.65 to -0.74 | about 0.669 → 0.70 |
     | E5, control with weight 1 | 23 | -35 | | slightly lower |

   - The overshoot comes from every parameter that serves QoL alone taking a ±lr step in the same direction: the separate QoL encoder in F1, or the shared trunk when QoL's weight dominates its gradient sign in E5.
   - The init QoL head starts biased (E5 +0.12W, F1 +0.07W against MC), and the step-0 TD targets are the init V itself (row A's excess over its floor is 0.001–0.003). The early steps therefore spend their effort on a bias the bootstrap partly feeds back.
   - Both arms reach TD consistency (MAE 0.023W) by step 524, so this is a warm-up issue, not the main mismatch. *Hypothesised:* an Adam warm-up or a lower first-step learning rate on the QoL encoder would remove it.

## F1 against E5 (in-sample, units of W)

| Measure | E5 | F1 |
|---|---|---|
| Bias / MAE vs MC | +0.013 / 0.054 | -0.009 / 0.054 |
| Bias / MAE vs own TD(0.8) | +0.004 / 0.023 | -0.003 / 0.023 |
| Bias / MAE vs n-step 5 | +0.002 / 0.020 | -0.004 / 0.021 |
| MAE vs MC, last 10 rows | 0.076 | 0.074 |
| Correlation with MC | 0.775 | 0.768 |
| Prediction std (MC std 0.108) | 0.094 | 0.081 |
| Late 2-cycle corr(Δpred, ΔMC) | 0.07 | -0.01 |
| World factor: corr / slope of world-mean prediction (MC slope 138.5) | 0.79 / 43.7 | 0.78 / 37.4 |
| Population: correlation / relative MAE | 0.977 / 9.2% | 0.981 / 8.6% |
| b04 validation QoL BCE at epoch 4 (`critic.pt` summary) | 0.6594 | 0.6562 |

On the traced rows F1 matches E5 to within about 20 QoL-years, with one exception: on row D it removes most of E5's +0.10W error (F1 280.5 against MC 291.0). The separate encoder does not change any of the high-level mismatches above. In particular, F1 learns the world roulette offset almost as strongly as E5.
