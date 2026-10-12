# Bounded F15 targeted-panel outcome

**Status: PASS**

The predeclared 24-pair / 48-record panel produced 8 accepted records and 4 complete causal pairs. The success gate required at least two complete pairs.

- Complete pair: f15-import-export-event-aligned-002, f15-import-export-event-aligned-007, f15-import-export-event-aligned-017, f15-import-export-event-aligned-021
- Capture-decision distribution: {1: 2, 2: 2}
- Rejection reasons: {'InvalidOperationException: Requested stratum was not achieved: shared F15 delivered-direction event within the fixed capture horizon.': 40}
- Deterministic byte-for-byte reproduction: True
- Native result SHA-256: `84890c689a1d0df81892d38ba2742c7bcfdb1b0c81332b813d13409c8fa18394`
- Frozen native SHA: `5559961164d6f9da8e6d20c1dc563ecec3eb04b7`

No crash, replay failure, schema failure, non-determinism, or native defect was observed. The bounded gate passed using the existing deterministic heuristic counterparties and ordinary native delivered-trade ledger.

The result provides no evidence for escalating beyond the feed-forward MLP.
