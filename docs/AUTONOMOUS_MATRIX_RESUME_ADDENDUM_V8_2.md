# Causal matrix driver repair and budget amendment v8.2

The first matrix attempt failed before any retained rows or PPO update: its
new episode driver passed lists into the existing GAE helper, whose value
operand requires NumPy arrays. Preserve the original source, config, FAILED
summary and raw budget ledger. Fix the boundary to explicit float64 arrays and
boolean terminal/truncation arrays. Test episode collection through the actual
measured PPO update with mocked native responses, including both censoring and
termination; native physical proof remains separate.

This is an explicit stage-budget amendment. Original attempted work remains
14 training collection resolves +50 DEV +50 replay. None was erased or declared
successful mechanical validation. The14 aborted collection attempts consume
the predeclared unused mechanics-overhead reserve: repair36+aborted14=50≤128,
with repair replay35≤128. Retained native PPO allowance remains9600 (24×400).
No model, objective, target, learning rate or readiness criterion changes.

Reuse the one completed H100/seed11 baseline only with identical actor checkpoint,
native binary, action conversion, DEV entries/order, seed mapping and25-decision
limit. It already consumed50+50. Reserve at most1450 additional DEV and1450
replay. Total projected maximum=2409+9600+1450+1450=14909≤15080; cumulative active
CPU deadline remains1800 seconds, including every failed attempt and stage.

Freeze the amended sources/config and prior FAILED result/budget before restart.
Record raw versus retained transitions separately; retain failed-run artifacts.
The eight interleaved reset scenarios/cycle, three continuation seeds, H100/400,
receipt comparison and episode lengths remain unchanged. Astra/high source and
actual-result review remain required; this amendment gives no scale-out clearance.
