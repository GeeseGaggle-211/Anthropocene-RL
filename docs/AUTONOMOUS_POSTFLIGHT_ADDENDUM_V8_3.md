# Native postflight v8.3

This is a no-learning continuation of the reviewed causal campaign. Immutable
plan `configs/training/autonomous-native-postflight-v8-plan.json` pins the source,
native tool, native6 bridge, bank, prior ledger and one completed checkpoint.
Select H400/full/25/seed61 in advance; this is a plug check, not winner selection.

After 14,909 attempted resolves and 561.855 cumulative active seconds, reserve
16 performance resolves, 50 heuristic DEV resolves, 8 learned two-Elder resolves
and 8 replay resolves. Projected total is14,991 within15,080. The validation repair
reserve becomes36 repair +14 aborted matrix +16 performance +50 heuristic +8 plug
=124/128; repair replay becomes35+8=43/128. Failed calls remain charged.
Single native process at a time, one Torch thread, no PPO/optimizer work.

Performance uses the existing seed/configuration `BenchmarkWorld(1/80)`, one warm
and three measured independent resets per heuristic/native6 case. The complete
world-step timer includes observation construction and economic execution;
reset/setup and MLP inference are outside it. Report ranges and allocations;
cold worlds with differing outcomes are not a mature-empire scale proof.

The heuristic panel starts at the exact two DEV snapshots used by the matrix,
25 decisions each, with native heuristics for every settlement. Report focal
opening population, end-interval QoL and their integrated product with years.
Its peer/authority protocol differs from the controlled-agent matrix, so results
describe attainable native welfare and are not a matched causal comparison.

The plug test controls both settlements at a common native6 WEGO boundary with
independent model6 inference instances and RNG streams. Each sees its own
observation only. Four decisions are repeated from the same source and RNG;
the repeated run saves/reloads the native state and inference RNG midway.
Check exact requested actions, observations, rewards, receipts and fingerprints,
both replay traces and unchanged checkpoint bytes. Reward remains village-local.

An additional schema guard checks history names and control semantics, beyond
width/version/save checks. This changes no action distribution or native outcome.
The matrix's exact prior decoder and adapter sources are preserved under
`evidence/autonomous-causal-v8/source-at-launch/` with original pin hashes.

Read-only cold/mature head audits are charged as active CPU time with zero
native or optimizer work. Mature rows come only from the earlier frozen-policy
TRAIN fitting-world trajectories. They assess sensitivity and local clamps,
not learned economic value or an independent memory comparison.
