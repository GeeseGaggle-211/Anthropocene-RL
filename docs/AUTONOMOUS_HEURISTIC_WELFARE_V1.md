# Native heuristic welfare comparison v1

The user asks whether existing native heuristic Elders achieve substantially
higher welfare than the retained learned Elders. Measure that claim before
asserting it. Use all 16 development snapshots from the frozen autonomous bank,
the same focal village and physical-time local reward, and both eight and
25 decisions (160 and 500 years). Daughter welfare is excluded.

Compare full native AiPlanning/TurnPlanner heuristics with all villages under
native control, and a second native-focal arm retaining the learner curriculum's
public market peer when present. Record each loaded world's EconomicAllocation
rule, fingerprint and native configuration. The shared-peer arm controls peer
behavior, but native trade and autonomous stock-quote interfaces differ; neither
arm isolates learned workforce allocation alone.

Evaluate both final retained v3 models (seeds29/43) on the same 16 snapshots with
sampler seeds9101/9102. Run25 decisions and reconstruct exact eight-decision
prefix metrics; assert original eight-decision scalar traces and final-prefix
fingerprints. Native heuristic policies are deterministic, so repeated sampler
labels are not independent heuristic replications. Verify every new trajectory
by replay and save/load, and require heuristic eight-decision fingerprints and
intervals to match the corresponding25-decision prefix.

Report population, QoL, raw welfare, welfare per initial village resident and
finite-trajectory discounted welfare at H100 and H400, with equal-family weights.
Report per-family paired differences and focal trades separately from background
trades. Native quote attempts are unavailable and must be null, not inferred
from settled trades. For this audit, “substantially higher” means at least20%
higher equal-family normalized welfare than EACH retained model at BOTH lengths;
the full family breakdown remains visible even if that aggregate criterion passes.
Failure to meet this criterion must be reported, not repaired by changing weights,
selecting snapshots or interpreting long cumulative totals as improvements.

Freeze bank, binaries, source/helper hashes, model checkpoint hashes and previous
welfare evidence before execution. Keep the original learner bridge untouched;
build the diagnostic executable into a separate ignored directory using unchanged
simulation sources. Bound recorded transitions to2700 (planned2656) plus at most
the same number for replay, one Torch thread, one active native worker per arm,
600-second wall limit. Zero PPO updates, fitting, GPU use or retained writes.
Preserve raw and failed/partial output under a new run directory.
