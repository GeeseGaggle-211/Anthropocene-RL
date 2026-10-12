# F15 mild-import / surplus-export construction diagnostic

This is a non-promotable scenario-construction diagnostic. The strict run kept all campaign gates; the companion observational run removed only trade acceptance predicates so failed worlds retained their reset and calibration traces.

- Strict candidates: 21 / 48
- Complete causal pairs: 3 / 24
- Complete-pair partitions: {'development': 1, 'train': 2}
- Complete pair IDs: f15-open-economy-import-export-supplement-002, f15-open-economy-pair-007, f15-open-economy-pair-012
- Failure modes: {'contacts-present-but-no-delivery': 27, 'strict-pass': 21}

## Pair outcomes

| Group | Split | Mild import | Surplus export | Complete |
|---|---|---:|---:|---:|
| f15-open-economy-import-export-supplement-001 | train | contacts-present-but-no-delivery | PASS | no |
| f15-open-economy-import-export-supplement-002 | development | PASS | PASS | yes |
| f15-open-economy-import-export-supplement-003 | train | contacts-present-but-no-delivery | PASS | no |
| f15-open-economy-import-export-supplement-004 | train | contacts-present-but-no-delivery | contacts-present-but-no-delivery | no |
| f15-open-economy-import-export-supplement-005 | train | contacts-present-but-no-delivery | contacts-present-but-no-delivery | no |
| f15-open-economy-import-export-supplement-006 | train | PASS | contacts-present-but-no-delivery | no |
| f15-open-economy-import-export-supplement-007 | heldout | PASS | contacts-present-but-no-delivery | no |
| f15-open-economy-import-export-supplement-008 | train | contacts-present-but-no-delivery | PASS | no |
| f15-open-economy-import-export-supplement-009 | development | contacts-present-but-no-delivery | PASS | no |
| f15-open-economy-import-export-supplement-010 | train | PASS | contacts-present-but-no-delivery | no |
| f15-open-economy-import-export-supplement-011 | heldout | contacts-present-but-no-delivery | PASS | no |
| f15-open-economy-import-export-supplement-012 | development | contacts-present-but-no-delivery | contacts-present-but-no-delivery | no |
| f15-open-economy-import-export-supplement-013 | heldout | contacts-present-but-no-delivery | PASS | no |
| f15-open-economy-import-export-supplement-014 | train | PASS | contacts-present-but-no-delivery | no |
| f15-open-economy-pair-002 | train | contacts-present-but-no-delivery | contacts-present-but-no-delivery | no |
| f15-open-economy-pair-007 | train | PASS | PASS | yes |
| f15-open-economy-pair-012 | train | PASS | PASS | yes |
| f15-open-economy-pair-017 | train | contacts-present-but-no-delivery | contacts-present-but-no-delivery | no |
| f15-open-economy-pair-022 | train | contacts-present-but-no-delivery | PASS | no |
| f15-open-economy-pair-027 | development | PASS | contacts-present-but-no-delivery | no |
| f15-open-economy-pair-032 | train | PASS | contacts-present-but-no-delivery | no |
| f15-open-economy-pair-037 | heldout | contacts-present-but-no-delivery | contacts-present-but-no-delivery | no |
| f15-open-economy-pair-042 | heldout | PASS | contacts-present-but-no-delivery | no |
| f15-open-economy-pair-047 | train | PASS | contacts-present-but-no-delivery | no |

## Aggregate comparison

```json
{
  "complete_pairs": {
    "contact_count": 1.0,
    "delivered_trade_total": 551.6134668032329,
    "field_q": 0.2869918012350643,
    "population": 266.5,
    "site_candidates": 253.33333333333334,
    "site_support_quantile": 0.4,
    "site_viability": 180.0,
    "visible_contact_count": 0.0
  },
  "incomplete_pairs": {
    "contact_count": 1.0,
    "delivered_trade_total": 319.11428571428576,
    "field_q": 0.2523810330826417,
    "population": 286.1666666666667,
    "site_candidates": 192.0952380952381,
    "site_support_quantile": 0.5142857142857142,
    "site_viability": 169.29040027352497,
    "visible_contact_count": 0.0
  }
}
```

Full per-sibling physical measurements, contacts, rejection evidence, and six-observation calibration traces are in the JSON report.
