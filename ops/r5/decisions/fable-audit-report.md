# Fable escalation audit: r5 critic (E5 / F1 / F2)

28 Oct 2026, read-only. Code `feature/r5-integration` 821660b (checkout `worktrees/r5-multi-dataset-rl`, HEAD verified). Evidence: pilot `cal-v2.json` for e1/e3/e5 (out-v2) and f1/f2, stability logs, the tracer's decoded b01 predictions (`trace/work/V_*.pt`), and two new local measurements in the session scratchpad `fable\` (`relmetrics.py` on all 8 b01 worlds; `heldout.py` runs the E5 and F1 checkpoints on b04 worlds 90124/90125 and the matched in-sample pair 90100/90101). Units: W = 410.08 QoL-years (head), S = 55.16 (gate scale); 1 W = 7.43 S.

## 1. Is the synthesis right?

Mostly, with three corrections.

**Confirmed.** No saturation anywhere; the clip is active every step; the QoL weight sets the head balance (PopArt-equivalent ~21); float32 is fine. World factor vs world-mean MC target corr 0.979; critics follow it with slope 44/138 (32% of the artefact learned); the unlocks block identifies the world (0.584). Population n-step nMAE 0.08-0.09 passes; the top-bin mass is the softmax floor over the 22-46 empty bins above the observed max (upper_excess 0.008 S on b04, 0.028 on b01; `value_support`), not a control problem.

**Correction 1: the n-step k=5 nMAE (`labor_pretrain_eval.py:911`) certifies almost nothing.** Its target is 78% the critic's own V (gamma^5 = 0.78), so a *constant* predictor at the mean MC return scores 0.145 (b01, all 8 worlds) and 0.173 (b04 pair), against E5 0.157 / 0.166 and F1 0.161 / 0.170 (`fable/constpred.log`, `heldout.json`). An oracle camp-mean scores 0.137. The "0.17 pass" is the reward-sum noise floor plus 0.22 x any level offset; it does not show accuracy, and the small fit/held-out gap (0.014) is the same floor on both sides. Accuracy is visible only in MC error (E5 0.054 W in-sample, 0.091 W on the b04 pair; constant 0.081 W) and in ordering (section 4).

**Correction 2: the world-tail artefact matters for absolute calibration only.** Within a trajectory both rows carry the same tail, so ordering is immune; the n-step gate never sees a tail. On the b04 pair E5's world-mean errors (+0.11 W at f = 0.64, +0.05 W at f = 0.83) match what an exact critic would show under the (1 - 1/f) attribution (`labor_pretrain_eval.py:1281-1287`), i.e. they are the target's error, not the critic's. Per-row MC target noise is large: mean c_i SE_i = 0.54 S over all rows (0.99 S in the last 10 rows, 0.36 S on tail-cut rows; `tailse.log`), plus the world-factor term 0.37 S. The critic's n-step MAE (0.15-0.17 S) is below both.

**Correction 3: the F1/F2 negative bias is a training-dynamics level offset, not an encoder property** (section 2).

**Missed:** the teacher 2-cycle sets the floor of any lag-1 "next state better?" test (section 4); the w100 cost to population is 0.008 nMAE (E5 0.093 vs E3 0.085), i.e. the "imbalance" has no victim.

## 2. Why is the separate-encoder bias negative? Baseline choice

Tail-cut MC bias in S, observed / predicted-by-tail / residual (`cal-v2.json`, `qol_nstep.strata.all.bias_tailcut`):

| arm | fit (mean f 0.934) | cal b04 (mean f 1.022) |
|---|---|---|
| E1 w30 | +0.126 / +0.115 / +0.010 | +0.064 / +0.020 / +0.045 |
| E5 w100 | +0.094 / +0.115 / -0.022 | +0.049 / +0.020 / +0.030 |
| E3 w1 | +0.046 / +0.115 / -0.069 | -0.034 / +0.020 / -0.053 |
| F2 sep, 8 ep | -0.017 / +0.115 / -0.133 | -0.077 / +0.020 / -0.097 |
| F1 sep, 4 ep | -0.087 / +0.115 / -0.202 | -0.152 / +0.020 / -0.172 |

The level is monotone in effective QoL optimisation (E1 > E5 > E3 > F2 > F1), in-sample as much as held-out, and F1 -> F2 halves it with 4 more epochs. Mechanism: the QoL head is a default `nn.Linear(width, 1)` (`labor_model.py:1525`), so the init sits at z ~ 0, i.e. ~W = 410 against a mean target of ~300 (+0.9 S); the step-0 refresh (`labor_bc.py:3371`, `:3161`) builds TD(0.8) targets from that init V; the first Adam steps overshoot downward by -0.7 to -1.1 in z (decision-traces, item 9); then the level relaxes slowly because TD(lambda) targets follow the critic's own level with gain gamma(1 - lambda)/(1 - gamma lambda) = 0.80 (`labor_advantage.py:1751`), and the n-step gate sees only 1 - gamma^5 = 0.22 of a uniform offset. A separate encoder with weight 1 and 524 steps simply recovers least (its shared-trunk cousin E3 is next lowest). This is the brief's candidate (a), under-training of the level, in a specific form; Adam/eps and loss geometry are not involved (saturation audit section 4b).

**Baseline for the scale-out: the shared encoder, weight 30 (E1 config), plus the level fix in section 6.** F1/F2 bring no nMAE gain, lose ordering (section 4: all-pairs 0.61 vs 0.65 held-out, Spearman 0.22 vs 0.43), cost +34% parameters and +30% step time, and carry the offset. E5 (w100) is marginally better than E1 on every metric (0.170 vs 0.183 nMAE) at a 0.008 population cost; keep it as the second arm, not the default, because the dominance monitor flags it.

## 3. Should the tail redesign go ahead now? No.

| option | removes | leaves | cost |
|---|---|---|---|
| Redesign (per-step rewards, independent per-camp stops, recollect) | world-correlated factor (0.37 S/row in MC targets; ~0.14 S per held-out world level); most per-trajectory SE if closed by exact truncation + V bootstrap | the 2-cycle, level drift, teacher V, false terminal; ordering and n-step unchanged (0 effect on relative gates) | 1.5-2 days + all batches |
| Per-world tail normalisation (divide stored tails by f_w from `stop_indices`, `labor_pretrain_eval.py:742`) | the same world factor, exactly for a constant-QoL continuation, approximately otherwise; in-sample MC MAE 0.054 -> 0.043 W (E5, `relmetrics.json` `mae_W_vs_MC_factor_corrected`) and the per-world level errors above | per-trajectory SE (0.54 S) | loader change, no recollection, old data valid |
| Mask / regularise unlocks | the world fingerprint, but also real signal (era and technology timing; farmer_locked nMAE is already 0.39-0.48) | the factor still in the targets, so memorisation moves to other blocks (all-privileged 1-NN 0.25) | retrain only |
| Train on n-step / TD with smaller tail weight | last-row tail noise (0.99 S) | the level becomes unanchored: a uniform offset is a near fixed point (section 2); tail coefficient is already 1e-3 to 1e-5 mid-window | config only |

Under relative-first gates the redesign is not necessary: no relative metric moves. Do the normalisation tweak now and keep the redesign in r6, where recollection is already queued (normaliser widening, simulator leak).

## 4. What would E5 / F2 score on relative gates?

Within-trajectory ordering, sign agreement of (V_j - V_i) with (G_j - G_i), MC targets (world tail cancels within a camp); margin = |G_j - G_i| threshold:

| set | model | all pairs: no margin / >0.02 W / >0.05 W | lag 5: >0.02 W | per-traj Spearman (median) | TD(0) MAE S | n-step k5 |
|---|---|---|---|---|---|---|
| b01, 8 worlds, in-sample | E5 | 0.655 / 0.701 / 0.750 | 0.697 | 0.48 | 0.085 | 0.149 |
| | F1 | 0.630 / 0.678 / 0.727 | 0.699 | 0.38 | 0.080 | 0.153 |
| | init | 0.43 / 0.41 / 0.40 | 0.42 | -0.24 | 0.35 | 0.50 |
| b01 pair (matched) | E5 | 0.694 / 0.760 / 0.824 | 0.733 | 0.55 | 0.087 | 0.150 |
| **b04 pair, held-out** | **E5** | **0.651 / 0.700 / 0.745** | **0.719** | **0.43** | 0.093 | 0.166 |
| | F1 | 0.609 / 0.639 / 0.664 | 0.715 | 0.22 | 0.088 | 0.170 |

Pair-level shares beyond the margins: 67% of pairs exceed 0.02 W, 41% exceed 0.05 W (b04). Lag-1 ("is the next state better?") is 0.57-0.58 for every model, because lag-1 return differences are r_i - 0.049 G_{i+1} ~ +-4 QoL-years, i.e. the 2-cycle phase, which the critics do not track (corr(dV, dMC) 0.07-0.12; rows with strong phase carry n-step MAE 0.19 vs 0.13). Held-out ordering drops 0.04-0.06 from matched in-sample for E5 and 0.05-0.11 for F1. Expected E5 score on a "next-state advantage sign" gate: ~0.70 at a 0.02 W margin and lag >= 5, ~0.75-0.80 at 0.05 W; ~0.57 if judged at lag 1 without a margin. F2 was not downloaded; from F1's gap I expect it between F1 and E5 (unverified). Same-state A ranking cannot be estimated from teacher data (one action per state).

## 5. Teacher cycle and V-of-teacher: PPO threat?

Moderate, and already mitigated by design. The critic is V^rho of the heuristic (no behaviour logp stored, `labor_bc.py:836`, `:851`); PPO rollouts store their own logp (`labor_training.py:330-334`) and refit V on actor data, so the pretrained V is an initialiser and the G5 refit plus DAgger teacher continuations at actor-visited roots (25%) give V^rho on the actor's support. Residual risks: (i) the phase-blind V turns the cycle into +-4 QoL-years (0.07 S) advantage noise, harmless; (ii) the cycle itself (cloth/cap 0.93 <-> 0 every turn, farmers 290 <-> 60) looks like a heuristic clothing-rule artefact that PPO may learn to exploit, a user decision on the teacher, not the critic (hypothesised); (iii) the false terminal `Terminated = camp == null || camp.Population <= 0` (`ElderModel9.cs:1075`) with hunters away trains V = 0 on a revivable state and, in PPO, penalises sending every resident out; one row in b01, a simulator fix for the user; (iv) zero rejected attempts means the feasibility head is a constant-zero head; PPO's first retries will be off-support, so judge the cost critics only after the G5 refit.

## 6. Ranked plan

Tweaks (no user decision needed):
1. Initialise the QoL head bias at logit(mean p*) of the fit set (~ -0.62) and take the step-0 targets from MC rather than init V; expected to remove the overshoot and the level offset (section 2). Pilot: E1 config, b01 -> b04, check tail-cut residual within +-0.05 and n-step bias within +-0.02.
2. Per-world tail normalisation in the loader (section 3); report the mean factor per set.
3. Gate restructure: make within-trajectory ordering at lag >= 5 with a 0.02 W margin (threshold 0.70) and all-pairs ordering primary; keep TD(0) residual MAE (0.09 S) as self-consistency; demote n-step nMAE to diagnostic and always print the constant-predictor reference next to it; keep MC bias as a loose sanity bound after tail attribution.
4. Loosen `check_support` (`labor_model.py:1670`) to `allclose(atol=1e-6)` or rebuild `bins` on load; CPU evaluation of GPU checkpoints raises today.
5. Scale-out baseline: shared encoder, w30, 4 epochs, best-val; E5 (w100) as the second arm. Drop the separate encoder from the campaign default (`--campaign-r5` now defaults to separate, `labor_model.py:345-358`).

Overhauls (to the user): the r5 tail redesign (recommend r6); the teacher clothing cycle and the terminal rule (simulator); a phase-aware or action-conditioned target for lag-1 tests only if the A head cannot cover it.

**Unverified:** F2 ordering (checkpoint not downloaded); held-out numbers are 2 of 8 b04 worlds; the overshoot-plus-relaxation account of the F1 level is inferred from arm ordering and the gain formula, not from a controlled rerun; the n-step reward-noise floor was measured with a constant predictor, not with counterfactual rollouts.
