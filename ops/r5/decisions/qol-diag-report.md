# QoL critic diagnosis

Support [-0.5, 2.5] x 255 (bin 0.0118 symlog). Calibration worlds 20; fitting worlds 10.

## 1. Target structure (calibration)

| head | var total | within frac | within sd (bins) | informative pairs | median abs dG (ref) | median abs dG (bins) | < 1 / 2 / 5 bins |
|---|---|---|---|---|---|---|---|
| qol_years | 1.471 | 0.085 | 8.7 | 0.825 | 0.237 | 7.4 | 0.001 / 0.055 / 0.323 |
| population_years | 0.380 | 0.599 | 20.7 | 0.878 | 0.374 | 19.7 | 0.000 / 0.002 / 0.075 |

Informative share and median abs dG (bins) by decisions remaining:

- qol_years: 1: 0.650 / 8.1 (n=2185); 13+: 0.918 / 7.3 (n=10618); 2-3: 0.708 / 7.7 (n=4297); 4-6: 0.817 / 7.4 (n=6179); 7-12: 0.879 / 7.1 (n=10778)
- population_years: 1: 0.614 / 8.5 (n=2185); 13+: 0.954 / 26.8 (n=10618); 2-3: 0.804 / 13.8 (n=4297); 4-6: 0.888 / 19.5 (n=6179); 7-12: 0.932 / 24.6 (n=10778)

By camp size:

- qol_years: 100-300: 0.818 / 8.5 (n=12557); 300-600: 0.794 / 8.2 (n=9427); <100: 0.853 / 7.7 (n=4574); >600: 0.677 / 6.8 (n=7499)
- population_years: 100-300: 0.698 / 10.9 (n=12557); 300-600: 0.732 / 9.4 (n=9427); <100: 0.665 / 15.6 (n=4574); >600: 0.675 / 7.3 (n=7499)

## 2. Critic error decomposition

### calibration

| head | nMAE | bias | R2 | within R2 | rank (CI) | pooled sd ratio | collapse frac | within slope | entropy (eff bins) | upper mass |
|---|---|---|---|---|---|---|---|---|---|---|
| qol_years | 0.450 | 0.038 | 0.757 | -1.202 | 0.580 (0.57-0.59) | 1.413 | 0.000 | 0.397 | 43.7 | 0.0014 |
| population_years | 0.164 | 0.051 | 0.852 | 0.804 | 0.898 (0.89-0.91) | 1.016 | 0.000 | 0.918 | 29.9 | 0.0025 |

- qol_years ranking by abs dG (ref scales): (0.05,0.10]: 0.527 n=189837; (0.1,0.20]: 0.523 n=305937; (0.2,0.50]: 0.585 n=436603; (0.5,1.00]: 0.653 n=200604; (1,2.00]: 0.815 n=43105; (2,inf]: 0.808 n=4493
- qol_years by remaining: 1: nMAE 0.653 bias 0.653 rank 0.538; 13+: nMAE 0.363 bias -0.093 rank 0.663; 2-3: nMAE 0.461 bias 0.359 rank 0.532; 4-6: nMAE 0.503 bias 0.064 rank 0.541; 7-12: nMAE 0.457 bias -0.100 rank 0.584
- qol_years by camp_size: 100-300: nMAE 0.482 bias 0.027 rank 0.576; 300-600: nMAE 0.430 bias 0.055 rank 0.608; <100: nMAE 0.492 bias 0.022 rank 0.597; >600: nMAE 0.394 bias 0.065 rank 0.552
- qol_years by target_magnitude: q1: nMAE 0.544 bias 0.483 rank 0.533; q2: nMAE 0.490 bias 0.079 rank 0.541; q3: nMAE 0.443 bias -0.065 rank 0.559; q4: nMAE 0.388 bias -0.103 rank 0.573; q5: nMAE 0.375 bias -0.202 rank 0.633
- population_years ranking by abs dG (ref scales): (0.05,0.10]: 0.724 n=139841; (0.1,0.20]: 0.815 n=227175; (0.2,0.50]: 0.902 n=398837; (0.5,1.00]: 0.975 n=319613; (1,2.00]: 0.998 n=167460; (2,inf]: 1.000 n=4031
- population_years by remaining: 1: nMAE 0.231 bias 0.231 rank 0.890; 13+: nMAE 0.149 bias 0.006 rank 0.939; 2-3: nMAE 0.167 bias 0.139 rank 0.856; 4-6: nMAE 0.180 bias 0.056 rank 0.864; 7-12: nMAE 0.152 bias 0.019 rank 0.916
- population_years by camp_size: 100-300: nMAE 0.119 bias 0.044 rank 0.887; 300-600: nMAE 0.173 bias 0.072 rank 0.827; <100: nMAE 0.125 bias 0.035 rank 0.872; >600: nMAE 0.257 bias 0.056 rank 0.617
- population_years by target_magnitude: q1: nMAE 0.119 bias 0.106 rank 0.895; q2: nMAE 0.143 bias 0.091 rank 0.751; q3: nMAE 0.170 bias 0.089 rank 0.698; q4: nMAE 0.195 bias 0.045 rank 0.703; q5: nMAE 0.195 bias -0.069 rank 0.831

### fitting

| head | nMAE | bias | R2 | within R2 | rank (CI) | pooled sd ratio | collapse frac | within slope | entropy (eff bins) | upper mass |
|---|---|---|---|---|---|---|---|---|---|---|
| qol_years | 0.345 | 0.093 | 0.843 | -0.412 | 0.628 (0.62-0.64) | 1.244 | 0.000 | 0.568 | 43.4 | 0.0014 |
| population_years | 0.132 | 0.060 | 0.911 | 0.889 | 0.922 (0.92-0.93) | 1.002 | 0.000 | 0.947 | 30.4 | 0.0035 |

- qol_years ranking by abs dG (ref scales): (0.05,0.10]: 0.543 n=88692; (0.1,0.20]: 0.552 n=142097; (0.2,0.50]: 0.647 n=191348; (0.5,1.00]: 0.755 n=75342; (1,2.00]: 0.914 n=13775; (2,inf]: 0.995 n=2764
- qol_years by remaining: 1: nMAE 0.609 bias 0.609 rank 0.559; 13+: nMAE 0.280 bias 0.001 rank 0.712; 2-3: nMAE 0.384 bias 0.319 rank 0.560; 4-6: nMAE 0.368 bias 0.072 rank 0.591; 7-12: nMAE 0.329 bias -0.001 rank 0.640
- qol_years by camp_size: 100-300: nMAE 0.389 bias 0.118 rank 0.652; 300-600: nMAE 0.324 bias 0.084 rank 0.634; <100: nMAE 0.389 bias 0.084 rank 0.671; >600: nMAE 0.279 bias 0.075 rank 0.563
- qol_years by target_magnitude: q1: nMAE 0.465 bias 0.423 rank 0.552; q2: nMAE 0.360 bias 0.081 rank 0.578; q3: nMAE 0.331 bias 0.033 rank 0.606; q4: nMAE 0.293 bias 0.011 rank 0.607; q5: nMAE 0.278 bias -0.086 rank 0.685
- population_years ranking by abs dG (ref scales): (0.05,0.10]: 0.740 n=57343; (0.1,0.20]: 0.838 n=95519; (0.2,0.50]: 0.934 n=173968; (0.5,1.00]: 0.991 n=147682; (1,2.00]: 1.000 n=80573; (2,inf]: 1.000 n=2139
- population_years by remaining: 1: nMAE 0.212 bias 0.212 rank 0.918; 13+: nMAE 0.130 bias 0.033 rank 0.951; 2-3: nMAE 0.135 bias 0.117 rank 0.887; 4-6: nMAE 0.132 bias 0.049 rank 0.896; 7-12: nMAE 0.115 bias 0.041 rank 0.938
- population_years by camp_size: 100-300: nMAE 0.109 bias 0.056 rank 0.899; 300-600: nMAE 0.129 bias 0.072 rank 0.863; <100: nMAE 0.113 bias 0.038 rank 0.892; >600: nMAE 0.177 bias 0.065 rank 0.669
- population_years by target_magnitude: q1: nMAE 0.106 bias 0.099 rank 0.919; q2: nMAE 0.126 bias 0.102 rank 0.752; q3: nMAE 0.136 bias 0.076 rank 0.741; q4: nMAE 0.152 bias 0.041 rank 0.745; q5: nMAE 0.141 bias -0.019 rank 0.852

## 3. Noise floor

Folds by world (5); 4054 input features, 256 trunk features; k-NN k=16 on 32 PCs.

| target | model | nMAE | within R2 | rank | rank CI |
|---|---|---|---|---|---|
| qol_years/full | critic | 0.450 | -1.202 | 0.580 | 0.57-0.59 |
| qol_years/full | ridge_inputs/cal-cv | 0.731 | -1.491 | 0.490 | 0.48-0.50 |
| qol_years/full | knn_inputs/cal-cv | 0.698 | -2.566 | 0.511 | 0.50-0.52 |
| qol_years/full | ridge_inputs/fit->cal | 0.730 | -5.984 | 0.487 | 0.47-0.50 |
| qol_years/full | knn_inputs/fit->cal | 0.708 | -2.626 | 0.502 | 0.49-0.52 |
| qol_years/full | ridge_critic_trunk/cal-cv | 0.466 | -1.056 | 0.590 | 0.58-0.60 |
| qol_years/full | ridge_critic_trunk/fit->cal | 0.473 | -1.240 | 0.568 | 0.56-0.58 |
| qol_years/full | group mean (LOO) | 0.232 | 0 | 0.5 | |
| qol_years/h1 | ridge_inputs/cal-cv | 0.040 | 0.272 | 0.899 | 0.89-0.91 |
| qol_years/h1 | knn_inputs/cal-cv | 0.036 | 0.611 | 0.892 | 0.89-0.90 |
| qol_years/h1 | ridge_inputs/fit->cal | 0.043 | -1.675 | 0.893 | 0.89-0.90 |
| qol_years/h1 | knn_inputs/fit->cal | 0.038 | 0.586 | 0.882 | 0.88-0.89 |
| qol_years/h1 | ridge_critic_trunk/cal-cv | 0.044 | 0.545 | 0.879 | 0.87-0.88 |
| qol_years/h1 | ridge_critic_trunk/fit->cal | 0.044 | 0.543 | 0.878 | 0.87-0.88 |
| qol_years/h1 | group mean (LOO) | 0.073 | 0 | 0.5 | |
| qol_years/h3 | ridge_inputs/cal-cv | 0.134 | -0.500 | 0.750 | 0.74-0.76 |
| qol_years/h3 | knn_inputs/cal-cv | 0.135 | 0.323 | 0.726 | 0.72-0.74 |
| qol_years/h3 | ridge_inputs/fit->cal | 0.142 | -0.476 | 0.719 | 0.71-0.73 |
| qol_years/h3 | knn_inputs/fit->cal | 0.137 | 0.278 | 0.703 | 0.69-0.71 |
| qol_years/h3 | ridge_critic_trunk/cal-cv | 0.129 | 0.293 | 0.743 | 0.74-0.75 |
| qol_years/h3 | ridge_critic_trunk/fit->cal | 0.131 | 0.150 | 0.716 | 0.71-0.72 |
| qol_years/h3 | group mean (LOO) | 0.115 | 0 | 0.5 | |
| qol_years/h5 | ridge_inputs/cal-cv | 0.275 | 0.035 | 0.585 | 0.57-0.60 |
| qol_years/h5 | knn_inputs/cal-cv | 0.260 | -0.190 | 0.619 | 0.61-0.63 |
| qol_years/h5 | ridge_inputs/fit->cal | 0.274 | -0.069 | 0.574 | 0.56-0.59 |
| qol_years/h5 | knn_inputs/fit->cal | 0.263 | -0.295 | 0.601 | 0.59-0.61 |
| qol_years/h5 | ridge_critic_trunk/cal-cv | 0.218 | -0.081 | 0.671 | 0.66-0.68 |
| qol_years/h5 | ridge_critic_trunk/fit->cal | 0.221 | -0.363 | 0.646 | 0.64-0.65 |
| qol_years/h5 | group mean (LOO) | 0.148 | 0 | 0.5 | |
| population_years/full | critic | 0.164 | 0.804 | 0.898 | 0.89-0.91 |
| population_years/full | ridge_inputs/cal-cv | 0.215 | -5.712 | 0.886 | 0.88-0.89 |
| population_years/full | knn_inputs/cal-cv | 0.276 | 0.579 | 0.830 | 0.82-0.84 |
| population_years/full | ridge_inputs/fit->cal | 0.824 | -18347.448 | 0.870 | 0.86-0.88 |
| population_years/full | knn_inputs/fit->cal | 0.283 | 0.539 | 0.812 | 0.80-0.82 |
| population_years/full | ridge_critic_trunk/cal-cv | 0.160 | 0.822 | 0.904 | 0.90-0.91 |
| population_years/full | ridge_critic_trunk/fit->cal | 0.160 | 0.810 | 0.894 | 0.88-0.90 |
| population_years/full | group mean (LOO) | 0.381 | 0 | 0.5 | |
| population_years/h1 | ridge_inputs/cal-cv | 0.005 | 0.985 | 1.000 | 1.00-1.00 |
| population_years/h1 | knn_inputs/cal-cv | 0.011 | 0.932 | 0.995 | 0.99-1.00 |
| population_years/h1 | ridge_inputs/fit->cal | 0.005 | 0.971 | 1.000 | 1.00-1.00 |
| population_years/h1 | knn_inputs/fit->cal | 0.012 | 0.914 | 0.992 | 0.99-0.99 |
| population_years/h1 | ridge_critic_trunk/cal-cv | 0.008 | 0.965 | 0.999 | 1.00-1.00 |
| population_years/h1 | ridge_critic_trunk/fit->cal | 0.008 | 0.965 | 1.000 | 1.00-1.00 |
| population_years/h1 | group mean (LOO) | 0.049 | 0 | 0.5 | |
| population_years/h3 | ridge_inputs/cal-cv | 0.039 | 0.933 | 0.994 | 0.99-0.99 |
| population_years/h3 | knn_inputs/cal-cv | 0.050 | 0.863 | 0.961 | 0.96-0.96 |
| population_years/h3 | ridge_inputs/fit->cal | 0.041 | 0.839 | 0.988 | 0.99-0.99 |
| population_years/h3 | knn_inputs/fit->cal | 0.054 | 0.841 | 0.950 | 0.95-0.95 |
| population_years/h3 | ridge_critic_trunk/cal-cv | 0.037 | 0.924 | 0.985 | 0.98-0.99 |
| population_years/h3 | ridge_critic_trunk/fit->cal | 0.037 | 0.904 | 0.975 | 0.97-0.98 |
| population_years/h3 | group mean (LOO) | 0.131 | 0 | 0.5 | |
| population_years/h5 | ridge_inputs/cal-cv | 0.080 | 0.881 | 0.968 | 0.97-0.97 |
| population_years/h5 | knn_inputs/cal-cv | 0.097 | 0.797 | 0.926 | 0.92-0.93 |
| population_years/h5 | ridge_inputs/fit->cal | 0.085 | 0.719 | 0.956 | 0.95-0.96 |
| population_years/h5 | knn_inputs/fit->cal | 0.101 | 0.768 | 0.911 | 0.91-0.92 |
| population_years/h5 | ridge_critic_trunk/cal-cv | 0.066 | 0.888 | 0.961 | 0.96-0.97 |
| population_years/h5 | ridge_critic_trunk/fit->cal | 0.066 | 0.856 | 0.945 | 0.94-0.95 |
| population_years/h5 | group mean (LOO) | 0.198 | 0 | 0.5 | |

## 4. Search candidate disparity

232 roots from 29 shards (SearchConfig defaults (no dataset.json with a search record found)).

- Successor tensors are not stored, so the V-based (state-value) advantage of each candidate cannot be computed from this export; only native returns are compared.
- Re-evaluation (dithered H12 continuation streams) exists only at the final horizon and only for H12 and the personality winners beyond the selection margin.
- Single-rollout candidate returns come from the deterministic stream 0; the pairwise noise used for them is the root's paired re-evaluation stream SD at the final horizon (an upper-bound proxy for shorter horizons when noise grows with horizon).
- population_years: re-evaluated roots 172 (zero SE 67), share beyond 2 SE at materiality 0/0.01/0.05: 0.797 / 0.349 / 0.047
  - h=1: roots 232, median range 0.0000, median paired sd 0.0003, share beyond 2 sd at 0/0.01/0.05: 0.000 / 0.000 / 0.000
  - h=2: roots 232, median range 0.0027, median paired sd 0.0003, share beyond 2 sd at 0/0.01/0.05: 0.628 / 0.209 / 0.012
  - h=3: roots 232, median range 0.0072, median paired sd 0.0003, share beyond 2 sd at 0/0.01/0.05: 0.791 / 0.424 / 0.047
- qol_years: re-evaluated roots 172 (zero SE 14), share beyond 2 SE at materiality 0/0.01/0.05: 0.698 / 0.645 / 0.459
  - h=1: roots 232, median range 0.0609, median paired sd 0.0204, share beyond 2 sd at 0/0.01/0.05: 0.616 / 0.610 / 0.442
  - h=2: roots 232, median range 0.0367, median paired sd 0.0204, share beyond 2 sd at 0/0.01/0.05: 0.558 / 0.529 / 0.326
  - h=3: roots 232, median range 0.0590, median paired sd 0.0204, share beyond 2 sd at 0/0.01/0.05: 0.674 / 0.640 / 0.453
- utility:(0.1,0.9): re-evaluated roots 172 (zero SE 14), share beyond 2 SE at materiality 0/0.01/0.05: 0.721 / 0.448 / 0.105
  - h=1: roots 232, median range 0.0062, median paired sd 0.0030, share beyond 2 sd at 0/0.01/0.05: 0.564 / 0.279 / 0.000
  - h=2: roots 232, median range 0.0065, median paired sd 0.0030, share beyond 2 sd at 0/0.01/0.05: 0.581 / 0.337 / 0.017
  - h=3: roots 232, median range 0.0113, median paired sd 0.0030, share beyond 2 sd at 0/0.01/0.05: 0.686 / 0.459 / 0.093
- utility:(0.9,0.1): re-evaluated roots 172 (zero SE 14), share beyond 2 SE at materiality 0/0.01/0.05: 0.698 / 0.640 / 0.453
  - h=1: roots 232, median range 0.0556, median paired sd 0.0186, share beyond 2 sd at 0/0.01/0.05: 0.610 / 0.599 / 0.424
  - h=2: roots 232, median range 0.0338, median paired sd 0.0186, share beyond 2 sd at 0/0.01/0.05: 0.552 / 0.512 / 0.320
  - h=3: roots 232, median range 0.0547, median paired sd 0.0186, share beyond 2 sd at 0/0.01/0.05: 0.669 / 0.628 / 0.442
- utility:balanced: re-evaluated roots 172 (zero SE 14), share beyond 2 SE at materiality 0/0.01/0.05: 0.674 / 0.587 / 0.337
  - h=1: roots 232, median range 0.0309, median paired sd 0.0104, share beyond 2 sd at 0/0.01/0.05: 0.599 / 0.552 / 0.291
  - h=2: roots 232, median range 0.0198, median paired sd 0.0104, share beyond 2 sd at 0/0.01/0.05: 0.547 / 0.459 / 0.233
  - h=3: roots 232, median range 0.0324, median paired sd 0.0104, share beyond 2 sd at 0/0.01/0.05: 0.663 / 0.605 / 0.343
- utility:root: re-evaluated roots 172 (zero SE 14), share beyond 2 SE at materiality 0/0.01/0.05: 0.680 / 0.564 / 0.314
  - h=1: roots 232, median range 0.0306, median paired sd 0.0102, share beyond 2 sd at 0/0.01/0.05: 0.564 / 0.488 / 0.244
  - h=2: roots 232, median range 0.0171, median paired sd 0.0102, share beyond 2 sd at 0/0.01/0.05: 0.558 / 0.436 / 0.215
  - h=3: roots 232, median range 0.0315, median paired sd 0.0102, share beyond 2 sd at 0/0.01/0.05: 0.663 / 0.558 / 0.337
- QoL vs population-years gains over H12: Pearson 0.085, Spearman 0.058, median per-root Spearman -0.250, opposite sign 0.482, argmax differs 0.638 (n=696)

## 5. Verdict

- **i_noise_or_resolution: supported**. Supported when every regressor has within-group R2 <= within_r2_noise, or short-horizon targets rank >= short_horizon_gain better than the full window, or rows far from the window end rank >= remaining_gain better than late rows, or ranking passes at large margins but not at small ones. Points to more bins or a shorter-horizon / randomized-tail target. The bin-share is reported but is not decisive alone: the decoded expectation is continuous.
  - evidence: `{"max_within_group_r2_any_model": -1.0555773977245693, "short_horizon_ridge_ranking": {"h1": 0.8994210965827215, "h3": 0.7500980644557306, "h5": 0.5849593253553302}, "full_window_trivial_ranking": 0.5112711644032293, "ranking_many_remaining": 0.623632501454006, "ranking_few_remaining": 0.5371239676438021, "share_informative_pairs_below_bins": 0.05500944875353534, "critic_rank_small_margins": [0.5269994784999763], "critic_rank_large_margins": [0.6531624494028035, 0.8150562579747129, 0.8077008680169152], "resolution_share_flag": false, "flags": {"noise_dominated": true, "short_horizon_better": true, "tail_truncation": true, "fails_only_at_small_margins": true}}`
- **ii_input_information: not supported**. Supported when the critic, the trivial input regressors and the short-horizon regressors all rank <= rank_weak while the population head (same inputs) passes. Points to intervention A (privileged state); it cannot be separated from intrinsic outcome noise without privileged features.
  - evidence: `{"critic": 0.5803652275705395, "best_trivial_inputs": 0.5112711644032293, "short_horizon_best": 0.8994210965827215, "population_head": 0.8983990701352552}`
- **iii_critic_optimisation: not supported**. Supported when a trivial regressor or a linear probe on the critic's own trunk ranks >= rank_gain above the critic head (the information is there but the head does not read it out), or by half that margin when the critic also fails on its own fitting worlds. A fit-cal gap above fit_gap is a generalisation problem, reported separately.
  - evidence: `{"critic": 0.5803652275705395, "critic_on_fitting": 0.6276472808345233, "best_trivial_inputs": 0.5112711644032293, "trunk_probe": 0.589738594367679, "pooled_sd_ratio": 1.412807517270259, "fit_minus_cal": 0.047282053263983714}`
- **iv_action_level_disparity: supported**. Supported when at least search_root_share of roots have a candidate pair whose QoL return differs beyond 2 SE and beyond search_materiality reference scales. This is escalation-gate evidence that real QoL disparity between actions exists. Whether existing advantage signals capture it cannot be measured here (no successor tensors). A high QoL/population gain correlation means a population advantage already carries most of it.
  - evidence: `{"share_exact_reevaluated": 0.45930232558139533, "share_proxy_final_horizon": 0.45348837209302323, "qol_pop_gain_spearman": 0.05805543094834686, "materiality": 0.05, "qol_redundant_with_population": false}`
- generalisation gap flag: False

Recommended: (b) value representation / target: more bins or shorter-horizon / randomized-tail targets; (a) escalation-gate evidence: real action-level QoL disparity exists

Caveats:

- Within-(world, turn) ranking compares camps, not actions; action-level evidence is the search section only.
- Noise and missing input information are not separable from observational data alone; the privileged re-collection (intervention A) is the decisive test of (ii).
- Ridge and k-NN are lower bounds on input information; a nonlinear model could extract more.
- Bootstrap intervals resample worlds; with 20 calibration worlds they are wide.

Thresholds: `{"rank_pass": 0.7, "rank_weak": 0.65, "rank_gain": 0.05, "within_r2_noise": 0.05, "short_horizon_gain": 0.08, "remaining_gain": 0.08, "fit_gap": 0.1, "small_margin_bins": 2.0, "small_margin_share": 0.5, "search_root_share": 0.3, "search_materiality": 0.05, "qol_pop_corr_redundant": 0.8}`

Timings (s): `{"targets": 4.0, "critic_forward": 141.3, "features": 4.2, "noise_floor": 37.8, "search": 0.4, "total": 189.0}`
