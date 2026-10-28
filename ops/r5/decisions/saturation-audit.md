# Critic saturation / normalization audit (r5 pilot, E5 and F1)

Read-only measurement. Code: GeeseGaggle-211/Anthropocene-RL feature/r5-integration at 821660b. Scripts and raw JSON are in the session scratchpad under `satAudit\` (`evalset.py`, `dynfit.py`, `res_*.json`, `dyn_*.json`).
Models: `init` = ckpt/e5/checkpoint-000.pt, `e5` = ckpt/e5/critic.pt, `f1` = ckpt/f1/critic.pt. The F1 init checkpoint was not downloaded.
Data: **b01 (training)** = worlds 90100 and 90101 only (9,284 rows). **b04 (held-out)** = worlds 90124 and 90125 only (10,414 rows). Disk and RAM limits forced 2 of 8 worlds per batch, so held-out numbers come from 2 worlds and are noisy. Models were run in `train()` mode (dropout 0) under `no_grad` so the fused fast path does not bypass the hooks.
Calibration targets are the **MC discounted returns closed by the randomized tail** (`discounted_returns`), not the TD-lambda QoL target that E5 and F1 were trained on.

## 0. Framing facts
* QoL target fraction p* = G/(2W) has mean 0.34-0.36 and **std 0.046-0.047** (CV 14%). A constant predictor is already within roughly 0.005 nats of the best BCE. The learnable QoL signal is about 0.002-0.005 nats on a 0.64 floor.
* Population targets in symlog units: b01 median 1.19, 1% / 99% = 0.17 / 1.755. b04 median 1.10, 1% / 99% = 0.23 / 1.72. Std 0.43-0.47. The support is [-0.5, 2.5]. **Targets use 56% of it (0.09 to 1.77); none are clamped.**
* No head saturates (section 3). QoL z lies in [-0.9, 0] for 99.9% of rows.

## 1. Input rail (normalized values, present entries only)
Clip means |x| >= 9.9999 (the +-10 guard).

| family | b01 clip | b04 clip | b01 >5 | b04 >5 | columns ever clipped (b01/b04, width) |
|---|---|---|---|---|---|
| core (436; 100+ dead, zeroed) | 0.011% | 0.004% | 0.042% | 0.049% | 4/4 |
| sites | 0.076% | 0.074% | 0.41% | 0.48% | 3/3 of 14 |
| fishing | 0.011% | 0.003% | 0.13% | 0.003% | 1/1 |
| herds | 0 | 0 | 0.20% | 0.20% | 0 |
| commodities | 0.008% | 0.003% | 0.011% | 0.004% | 2/2 |
| contacts | 0 | 0 | 0.21% | 0.26% | 0 |
| regions | 0 | 0 | 0.085% | 0.067% | 0 |
| trade | 0.18% | 0.071% | 0.24% | 0.13% | 4/4 of 10 |
| activities | 0 | 0 | 0.001% | 0 | 0 |
| labor_activity (78) | 0 | 0 | 1.46% | 1.46% | 0 |
| **labor_resource (32)** | **0.74%** | **0.71%** | 1.78% | 1.74% | 3/3 |
| labor_specialty | 0 | 0 | 0 | 0 | 0 |
| priv core (10) | 0.026% | 0.006% | 0.22% | 0.14% | 3/2 |
| priv fields (19) | 0 | 0 | 0.13% | 0.17% | 0 |
| priv herds (24) | 0.050% | **0.149%** | 0.25% | 0.37% | 3/4 |
| priv unlocks (28) | 0 | 0 | 0.05% | 0.03% | 0 |
| priv economy (64) | 0.011% | 0.005% | 0.27% | 0.19% | 4/3 |

Top 15 features by clip share (b01 / b04):

| feature | clip b01 | clip b04 | >5 b01 / b04 |
|---|---|---|---|
| labor_resource[5] `locallyAccessible` | **17.1%** | 16.7% | 22.5 / 22.7% |
| labor_resource[29] (history column) | 5.9% | 5.7% | 20.6 / 20.8% |
| core[182] | 4.85% | 1.64% | 9.0 / 4.4% |
| trade[1] | 1.54% | 0.64% | 2.1 / 1.2% |
| priv herds[13] unmet-water | 0.40% | 1.12% | same |
| priv herds[15] deficit-severity | 0.00% | 0.98% | 0.89 / 1.17% |
| priv herds[11] nutrition | 0.40% | 0.74% | same |
| priv herds[10] water-nutrition | 0.40% | 0.74% | same |
| labor_resource[4] `opening` | 0.72% | 0.43% | 13.1 / 11.8% |
| sites[4], [5], [6] | 0.36% | 0.35% | 1.07 / 1.23% |
| priv economy[59] partner-stock-AnimalFabric | 0.33% | 0.07% | 0.59 / 0.27% |
| priv economy[13] stock-AnimalFabric | 0.25% | 0.10% | 0.42 / 0.24% |
| priv core[4] footprint-food-over-meals | 0.17% | 0.05% | 0.86 / 0.48% |

Only 5 of 819 features are clipped more than 1% of the time; 17 more than 0.1%. Clipped more often in b04 than b01: only the privileged herd-deficit fields (+0.6 to +1.0 point; rare rows, 3,018 vs 4,202 herd tokens, so a handful of herds). Everything else is equal or lower in b04. **The input clip does not explain a train/held-out gap.**
`labor_resource[4..6]` are not clipped by the fitted normalizer. `labor_bridge.py:1560-1570` applies `sign*log1p` and then a hard clip at 10 (stock of about 22k units or more). Production caps are 10M units (log1p = 16), so quantity information is lost for 17% of resource tokens.

## 2. Hidden activations (hooks; init vs E5 vs F1; b01 vs b04)
* No block approaches a rail. The only |x| near 10 is where clipped privileged inputs enter the global tokens (privileged_project.core output max 9.6, carried into global attention input). Elsewhere max |x| is 2-5.
* SiLU inputs (both trunk SiLUs, shared and F1 qol_encoder): max |x| 2.3-3.4. Share below -4 and above +8 is 0.000%, at init and after training.
* LayerNorm inputs: row variance 0.08-1.7 (trunk) and 0.24-0.60 (global tokens). The smallest row variance seen in training probes was 5.4e-3 (global attn norm1) against eps 1e-5, so eps changes it by under 0.2%. Rows with variance under 1e-3: none.
* LayerNorm gains after 500 steps: 0.99-1.05 (trunk.5 up to 1.05); biases within +-0.02. LN has barely moved from init.
* FiLM output is exactly 0 before and after training (personality constant at 0.5/0.5 in this data), so the conditioning path is inert.
* Attention (E5/F1; init is uniform): local layers normalized entropy 0.90 / 0.86 (init 1.00), global 0.97 (init 0.98). Fraction of queries with max attention above 0.99: 0.0%. Max pre-softmax logit 3.5-4.7 (init 1.6-3.0). Attention is diffuse, not saturated.
* Every activation statistic differs between b01 and b04 only in the second decimal.

## 3. Output heads (b01 = training subset, b04 = held-out subset)

| metric | E5 b01 | E5 b04 | F1 b01 | F1 b04 | init b04 |
|---|---|---|---|---|---|
| QoL z 0.1% / 50% / 99.9% quantile | -3.9 / -0.55 / -0.10 | -0.91 / -0.49 / 0.03 | -3.7 / -0.59 / -0.20 | -0.94 / -0.55 / -0.15 | -1.3 / -0.37 / 0.6 |
| QoL share abs z > 4 / > 6 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| QoL slope of pred on target | 0.84 | 0.58 | 0.70 | 0.43 | -0.33 |
| QoL correlation | 0.84 | 0.62 | 0.82 | 0.58 | -0.21 |
| QoL MAE (p units; x2 for W) | 0.023 | 0.046 | 0.022 | 0.037 | 0.093 |
| QoL bias pred - target (W units) | +0.027 | +0.085 | +0.005 | +0.057 | +0.14 |
| QoL BCE / entropy floor | 0.6496 / 0.6475 | 0.6437 / 0.6363 | 0.6493 / 0.6475 | 0.6414 / 0.6363 | 0.665 / 0.636 |
| Pop logit max (mean / max) | 4.2 / 5.4 | 4.1 / 5.3 | 4.7 / 5.9 | 4.6 / 5.7 | 1.8 / 2.8 |
| Pop row entropy (nats) / effective bins | 3.99 / 56 | 4.04 / 59 | 3.72 / 43 | 3.77 / 45 | 5.4 / 216 |
| Pop mass first-5 / last-5 bins | 0.11% / 0.11% | 0.09% / 0.10% | 0.12% / 0.12% | 0.12% / 0.12% | 1.7% / 2.2% |
| Pop slope (symlog) | 0.90 | 0.93 | 0.90 | 0.92 | 0.00 |
| Pop correlation (symlog) | 0.988 | 0.981 | 0.992 | 0.982 | -0.08 |
| Pop MAE (symlog) | 0.057 | 0.080 | 0.053 | 0.085 | 0.375 |
| Pop bias symlog / years | +0.011 / +3.6% | +0.062 / +12.5% | +0.016 / +4.6% | +0.070 / +13.9% | -0.06 / +31% |
| Pop CE (two-hot floor about 0.5) | 3.40 | 3.78 | 3.09 | 3.72 | 5.73 |

Calibration by target decile (E5, held-out b04). QoL: target 0.256 / 0.311 / 0.334 / 0.356 / 0.392 / 0.419 maps to prediction 0.346 / 0.362 / 0.371 / 0.390 / 0.419 / 0.435. The range is compressed (target spread 0.163, prediction spread 0.089) with a positive offset of about +0.04. Population (symlog), b04: decile 1 0.36 -> 0.50; decile 5 1.03 -> 1.08; decile 10 1.69 -> 1.67. On b01: 0.33 -> 0.48, 1.11 -> 1.12, 1.74 -> 1.68. Both heads regress toward the middle: low targets over-predicted by 0.12-0.15 symlog, the top 5% under-predicted by 0.02-0.06.
Bin resolution: step 3.0/254 = 0.0118 symlog, about 1.2% relative. At the median target (symlog 1.19, about 195k person-years) one bin is about 3.3k person-years (1.7%). The model MAE is 5-7 bins, so resolution does not bind. About 44% of bins (the top 0.73 and bottom 0.59 symlog) carry no targets and hold 0.1% mass at each end.

## 4. Training dynamics

### 4a. Existing pilot stability logs (out-b, d1, d2, e1-e5; clip 1.0)
"mult" is clip_multiplier. The clip is active on every logged step of every run.

| run | lr | QoL weight | global pre-clip norm first / median / max | mult median (min) | task_encoder grad median | pop head grad | QoL head grad | shared-encoder grad from QoL / pop (weighted, median) |
|---|---|---|---|---|---|---|---|---|
| out-b | 3e-4 | 1 | 2.2 / 3.7 / 9.1 | 0.27 (0.11) | 3.6 | 0.45 | 0.035 | 0.08 / 7.3 |
| out-d1 | 3e-4 | 1 | 1.7 / 3.5 / 9.0 | 0.28 (0.11) | 3.3 | 0.43 | 0.044 | 0.11 / 6.6 |
| out-d2 | 1e-4 | 1 | 3.1 / 4.1 / 24.3 | 0.24 (0.04) | 3.5 | 0.44 | 0.038 | 0.12 / 7.0 |
| out-e1 | 1e-4 | 30 | 6.1 / 3.5 / 7.0 | 0.29 (0.14) | 2.3 | 0.44 | 1.07 | 2.8 / 2.3 |
| out-e2 | 1e-4 | 1 | 1.7 / 3.1 / 7.9 | 0.32 (0.13) | 1.9 | 0.45 | 0.33 | 0.95 / 3.4 |
| out-e3 | 1e-4 | none | 1.7 / 3.3 / 8.7 | 0.31 (0.11) | 1.7 | 0.45 | 0.040 | 0.12 / 3.4 |
| out-e4 | 1e-4 | 30 | 4.5 / 4.1 / 9.5 | 0.25 (0.11) | 3.3 | 0.43 | 0.92 | 2.6 / 5.3 |
| **out-e5** | 1e-4 | **100** | **19.4** / 5.8 / 19.4 | **0.17 (0.05)** | 3.9 | 0.44 | 3.6 | **7.4** / 1.7 |

* Clipped gradient norm is exactly 1.0 on every step, so the optimizer sees a normalized gradient. Under Adam the constant factor cancels; what changes is the relative weight of steps and the head mix. At E5's start the shared-encoder gradient is 29 from QoL against 1.7 from population, so the clipped direction is about 99% QoL.
* Population CE is not the large gradient: population-head gradient norm is 0.41-0.59 in every run. Gradients grow only when the QoL weight is 30 or more, because that weight multiplies a tiny-signal BCE.
* The biggest block is task_encoder (median 1.7-3.9; up to 15 at E5 start). param_step_norm for task_encoder is 0.013-0.048 per step.

### 4b. Instrumented CPU replica
Task critic only, E5 init, 2 b01 worlds, batch 256, lr 1e-4, clip 1.0, 200 steps, MC QoL targets, QoL weight 100 and 1. Cost critics and TD-lambda omitted, so absolute numbers differ from the pilot at step 0 (MC targets are biased high against the init head, giving 157 vs the pilot's 19). Recorded every 25 steps.
* **Weight 100:** global pre-clip norm 157, 59, 70, 24, 29, 63, 26, 13, 8.5 (mult 0.006 rising to 0.12). Shared-encoder gradient from QoL (x100): 140 falling to 6.6, against 2.8 to 1.7 from population (QoL share of squared norm 99.9% at the start, 94% at step 100, 94% at the end). Head gradients: QoL head 70 to 4.8; population head 1.56 to 0.90.
* **Weight 1:** global norm 2.2-3.7, mult 0.27-0.47, QoL contributes 0.07-0.6 against population 1.8-3.2 on the encoder.
* Logit-gradient norm for the batch-mean loss: QoL unweighted 0.007 falling to 0.001 (rms residual 0.11 falling to 0.016); population constant 0.05. At weight 100 the QoL logit gradient is 14 times the population one at the start and 2 times at the end.
* Update-to-weight ratio per step: 2e-4 to 6e-4 for transformer and trunk blocks; 1e-3 to 2e-3 for the population head and token embeddings; 3e-5 for LayerNorm parameters. FiLM moves (5e-2 at step 25) but its output stays zero because its input is zero. Total movement over 200 steps is under 10% of weight norm.
* Adam step-cap proxy (share with |m/sqrt(v)| above 0.9): 0-1% in every block, except the population head (28% early at weight 1; 11% late at weight 100). **Updates are not excessive in size.**
* Attention: max logit 1.5-2.9 rising to 2.3-2.9; peaked fraction 0; normalized entropy about 0.97 (global) and 0.9 (local) throughout.
* Float precision: BCE fp32 vs fp64 differ by 1e-9 to 8e-8, against an excess over entropy of 9e-4 to 6e-3 (resolved to 4-5 digits); eps times loss = 8e-8; logit-gradient relative error 1.5e-7 to 7e-7. Two-hot CE fp32 vs fp64 differ by 2e-9 to 4e-7 (gradient relative error 3e-8). The CE excess over its entropy floor is 3.4-5.2 nats, far above float noise. LayerNorm eps is irrelevant (minimum row variance 5e-3). **Float32 does not lose the KL signal.**

## 5. Would PopArt-style per-head normalization change the balance?
* QoL: sigma(p*) = 0.047, so unit-variance normalization scales the QoL error by 1/sigma = **21x** (450x for squared error). E5 weight 100 is already 4.7x past that; weight 30 is 1.4x past it. Measured shared-encoder balance (QoL : population, weighted): weight 100 gives 4.3 : 1 (E5); weight 30 gives 1 : 1 to 0.5 : 1 (E1, E4); weight 1 gives about 1 : 35 (E2, E3). A QoL weight near 21 would balance the heads to within about 1.5x.
* Population: a two-hot categorical on a fixed symlog support is not a regression target; PopArt does not apply (row gradient is softmax minus two-hot, unit scale). The analogue is re-centering the support on the data ([0.09, 1.77] of [-0.5, 2.5]); that changes resolution by at most 1.6x (0.0118 to 0.0074 symlog) and moves the support centre (1.0), which is where an untrained or shrunken prediction sits.
* Neither changes input or hidden saturation, because there is none.

## 6. Verdict (observed vs hypothesis)
Observed:
* Inputs, hidden activations, SiLU, LayerNorm, attention and both output heads are far from any rail. The one real input clip is the bridge's log1p-then-clip-at-10 on `locallyAccessible` resource stock (17% of resource tokens), which is equal in train and held-out.
* QoL fits train (corr 0.84) much better than held-out (0.62); F1 similar (0.82 vs 0.58). Population is good in both (0.99 vs 0.98).
* Bias measured against MC returns is positive, not negative, for both heads (QoL +0.03 to +0.09 W; population +4% to +14%). I could not reproduce the -0.2 QoL or -0.04 to -0.15 population underestimation. It must come from the TD-lambda target, the n-step residual or the tail factor, which I did not rebuild.
* Both heads compress toward the middle (slopes 0.43-0.93).
* The head imbalance is real and set by the QoL loss weight (4.3 : 1 at 100, 1 : 35 at 1).

Hypotheses (not tested):
* QoL overfit is driven by a weak-signal target (std 0.047, learnable KL about 0.003 nats) trained at weight 100 through a shared encoder with 2 to 8 worlds; the weight multiplies noise. F1's separate encoder did not cure it, which supports data/signal rather than shared-trunk interference.
* The population shrinkage is the usual regression-to-the-mean of an expected-value decode from a broad softmax (effective bins 43-59 of about 135 used), pulled toward the support centre.
* Clipping `locallyAccessible` may hide stock information that matters for material decisions; effect unmeasured.

## 7. Ranked fixes
1. (tweak) Set the QoL loss weight to about 20-30, or normalize per head, instead of 100. Removes the 4:1 QoL dominance and the 99%-QoL clipped direction. Cheap, testable on b01.
2. (tweak) Regularize the QoL head: early stop on held-out, weight decay or lower LR on the QoL head/encoder, and more worlds. The remedy for the gap is wider data, not looser gates.
3. (tweak) Predict QoL relative to a baseline (for example the world/age mean of p*) or standardize the QoL logit offset, so the head spends capacity on the 0.047 spread. Medium effort; a centered head is closer to an architecture change.
4. (tweak) Raise the bridge clip on `locallyAccessible` or encode it as log10 or a fraction of cap; run an ablation. Low priority until a resource-sensitive gate fails.
5. (tweak) Shrink and re-center the population support to about [-0.1, 2.0]. Marginal.
6. Not recommended: PopArt on the population head, fp64 loss, an excess-KL loss (identical gradient), a LayerNorm eps or placement change, raising the normalizer clip guard (clip fires on under 0.1% of entries). None addresses a measured fault.
7. (architecture, only if 1-3 fail) A separately scaled dueling/advantage head for QoL that never shares the trunk and uses its own optimizer group.

## Caveats
2 of 8 worlds per set; MC rather than TD targets; instrumented fit is a task-critic replica; cost critics excluded; the Windows run found a 2.4e-7 float difference between the checkpoint `bins` buffer and `linspace` (strict `check_support` rejects it; I skipped it).
