# Cached TRAIN-only output regularization repair v9.4

V9.3 passed the strict value gate for101and211 after native coverage expansion,
but307 still overpredicts trade returns. Preserve both NO-GO stages and all
physical target/checkpoint/source hashes. Genuine rare high-return TRA examples
remain; no clipping, winsorizing, new actor prior or readiness relaxation. Root
read-only audit finds <1% hidden saturation and no numerical collapse diagnosis.

One prospective seed307-only ridge attempt uses its frozen initial normalized
critic trunk, whose301 zero-initialized receipt weights remain zero during this
calibration. Disclose this receipt-blind temporary value representation; retained
PPO may subsequently update the full live trunk/receipt weights. Successful101
and211models/optimizer states remain unchanged. No new native physical targets,
no DEV/HELDOUT use or additional fitting epochs. Existing model7/native6/schema
semantics and fixed full-TRAIN value offset/scale remain.

Choose ridge strength from fixed[.01,.1,1,10,100,1000] using leave-one-fitting-world
out across881001..004. Weight the four CV world MSEs equally; tie selects the
smallest strength. Fixed features come exclusively from initial frozen trunks,
not failed fitted trunks containing held-out-world labels. Each fold uses mean
and max(std,1) from fitting folds only and maps values back to raw reward_scale120
units before scoring. No validation observation/target enters selection. Final
fit uses all existing fitting rows and unchanged full-TRAIN buffers.

Solve output-head displacement from its initial weight/bias with unpenalized
intercept under value=offset+scale*(reference+live-initial). Compute in float64,
apply displacement to the float32 live head, restore only live trunk from initial,
and clear only live critic Adam rather than carry nonlinear-fit stale moments.
Actor/heads/anchors/actor Adam/RNG/counters/config and offset/scale are protected;
unit tests verify raw value mapping, fold-local statistics and deterministic ties.
Report all CV candidates, selected strength and exact checkpoint provenance.

After selection, evaluate the unchanged external six-family reused TRAIN gate
once: EV>=.60,MSE/Var(return)<=.40,absrawbias<=2,nonworse original unfit value.
Require all three actors ready. If this attempt fails, preserve failure and do
not launch its downstream PPO. No seed dropping or broad scale-out promotion.

Conditional downstream canary preserves v9.3's six exact448-transition LR cells
(2688retained), three6-state25-turnDEVbaselines and six finals(1350+1350replay).
At prior22417calls,5388maximum→27805cumulative,186reserve under27991ceiling.
Chain root read-only diagnosis active elapsed time, preserve2369.407-second cap,
4nativeworkers/1Torchthread and shared per-loop deadlines. Freeze all source/data
pins and require Astra/high concrete preflight and actual postflight review.
