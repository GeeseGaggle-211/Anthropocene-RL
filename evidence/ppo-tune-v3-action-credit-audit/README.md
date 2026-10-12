# TRAIN-only action-credit audit

This directory is the combined evidence boundary for the three isolated
update-168 policy audits. The canonical raw evidence was produced from learner
commit `5cbdf16291fdb500e147e26974f2fe3d82ed34e3` against native simulator
`5559961164d6f9da8e6d20c1dc563ecec3eb04b7`.

- Status: `ACTION_CREDIT_FAILURE`
- Canonical evidence SHA-256:
  `c81dbea2490e1efff0267b5eacde1ef1796489c879d296ea1a8cab3997d9c3be`
- Snapshot manifest SHA-256:
  `1cfabbc6057c063d2bca8d27c4bda861a090d5811e2b02f3fe51adc9fd2a6b83`
- Scope: TRAIN only; DEVELOPMENT and HELDOUT were not read
- Contract result: all native replay, requested-action, and cultivation-only
  scope checks passed

## Replicated findings

| Seed | Development acceptance-responsive | Physically responsive probes | Physical directional agreement | F03 accepted-but-unresponsive | Separate/joint actor displacement | Separate KL |
|---:|---:|---:|---:|---:|---:|---:|
| 1701 | 0.750 | 21 | 0.333 | 0.667 | 1.144 | 0.0140 |
| 1901 | 1.000 | 23 | 0.565 | 0.000 | 1.293 | 0.0214 |
| 2101 | 0.854 | 21 | 0.714 | 0.500 | 1.520 | 0.0684 |

Acceptance responsiveness failed its replicated gate in seeds 1701 and 2101.
Physical directional credit failed in seed 1701. Separate clipping did not
meet its complete replicated escalation gate. This evidence does not justify an
architecture, normalizer, activation, or clipping-policy change.

The multi-megabyte raw JSON files in this directory are durable run artifacts,
not source-controlled evidence. Publish and verify them through the repository's
immutable GCS run-artifact workflow.
