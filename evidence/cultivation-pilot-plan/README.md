# Cultivation-only pilot plan

This directory is superseded historical planning evidence. The 240-snapshot
draft was amended to the executed 300-snapshot open-economy campaign; see
`evidence/cultivation-pilot-v1/` and `evidence/cultivation-ppo-pilot-v1/`.

`configs/datasets/cultivation-pilot-v1.yaml` requests 240 snapshots across 200
independent worlds: F01 (40), F02 (40), F03 (40), F13 (48), F17 (24 from eight
three-timestamp memory lineages), and F18 (48 from 24 matched
primitive/plough pairs). Every trajectory is cultivation-only; movement,
herding, and fishing remain structurally unavailable.

The plan uses deterministic constrained-pairwise selection rather than a
Cartesian product. Complete world/lineage identities are assigned to
TRAIN/DEVELOPMENT/HELDOUT before timestamp selection. Requested bands are not
promotion evidence: generation must still pass the cultivation-pilot audit on
measured native strata before the bank is frozen.

After that freeze, fit a new final normalization-v2 from PILOT TRAIN resets and
short legal calibration rollouts only. DEVELOPMENT and HELDOUT remain
transform-only diagnostics; the Stage-0 provisional normalizer is not reused.
