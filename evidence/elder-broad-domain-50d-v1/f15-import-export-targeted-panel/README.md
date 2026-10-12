# Bounded F15 targeted-panel outcome

**Status: NATIVE_SCENARIO_CONSTRUCTION_BLOCKED**

The predeclared 24-pair / 48-record panel produced 7 accepted records and 1 complete causal pair. The success gate required at least two complete pairs.

- Complete pair: f15-import-export-targeted-001
- Rejection reasons: {'InvalidOperationException: No physically compatible open-economy partner site.': 2, 'InvalidOperationException: Requested stratum was not achieved: food-export': 8, 'InvalidOperationException: Requested stratum was not achieved: food-import': 8, 'InvalidOperationException: Requested stratum was not achieved: trade-active': 23}
- Deterministic byte-for-byte reproduction: True
- Native result SHA-256: `522b6373acce9e5a2a8f3b2845de1390a0aaf7379f92563353a73a58183d548d`
- Frozen native SHA: `5559961164d6f9da8e6d20c1dc563ecec3eb04b7`

No crash, replay failure, schema failure, non-determinism, or native defect was observed. The blocker is the reliability of realizing both delivered-trade directions at one declared timestamp under the current frozen scenario constructor.

The result provides no evidence for escalating beyond the feed-forward MLP: policy training did not begin because a valid frozen bank was not available.
