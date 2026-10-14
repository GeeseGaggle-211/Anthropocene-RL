# Reversal-cohort and durable-canary repair v9.6

V9.5 passed all three unchanged critic gates and transfer parity (maximum2.29e-5),
but after448native training decisions and three PPO updates, the reversal harness
rejected its six-row fitting-only probe (existing guard needs16 on one legal face).
The first6DEVbaseline trajectories/replays consumed146+146. Preserve the failed
summary/metrics/source/checkpoints/ledger. Those448decisions/updates executed but
were not durably checkpointed before the diagnostic exception; never refund or
report them as zero work. Cumulative23157resolves/773.918471157activeelapsedseconds.

Build a source-pinned fitting-only probe from the six PPO starts,23additional
fitting starts and12fitting-world trade observations. Deduplicate complete encoded
actor inputs (features/masks/stocks), preserving source IDs/observations. All35
unique inputs come from881001..004; largest legal face has21rows, and exact
existing probe_face selects20balanced rows, nonconstant feature312(std3.76347),
Gatherer/Farmer roles. Verify guard, variance and absence of contradictory labels
before any collection/update. No DEV or validation optimizer probes.

Restart from the saved ready V9.5 parents, preserving canonical critic transfer,
original independently initialized actors, same six TRAIN/DEV starts and RNG reset
schedule, H400/25decisions/LRpair.0003/.00003. Contract six equal cells from448to400
native transitions each (2400retained), never replenish old stage allowances.
Save exact model/optimizer/counters/RNG/checkpoint SHA and cumulative traces after
each completed rollout, before optional health/reversal diagnostics; diagnostics
fail separately without destroying checkpointed progress. Tests reproduce a
post-save probe exception and exact restore. No full scale-out launch.

Reuse seed101's completed baseline only after exact saved ready-parent SHA,
scenario order, snapshot/selection/bridge/converter/source pins,25turn limit,
episode_seed(seed+90000,index) mapping and full146replay-count evidence match.
No cached baseline chooses a policy or updates weights. Other2baselines plus
6final evaluations consume at most1200DEV+1200replay. New maximum2400+1200+1200
=4800→27957cumulative,34reserve underunchanged27991ceiling. Chain elapsed time,
2369.4072460189927cap,4nativeworkers/1Torchthread/shareddeadline checks.

Freeze concrete source/probe/config/pins and obtain Astra/high preflight before
restart. Report failed448 transitions / three updates separately from final
checkpointed2400canary.
Final independent review must assess actual head updates, populated-Adam reversal,
trade opportunities/settlements, welfare decomposition, legal masks/replay and
remaining limitations; numerical success does not imply economic competence.
