# Preserved horizon evidence erratum

The historical `scripts/evaluate_autonomous_25_turns_v1.py` labeled `Stocks[0]`
as `food_stock_before/after`. Native TradeGood/Trade.Registry order is Cloth0,
AnimalFabric1, HousingMaterial2, Food3. Those historical fields measure cloth,
and must not support food-reserve or cultivation-harvest claims. Original source,
report bytes and checkpoints remain preserved. Population, native QoL, welfare,
food-needs fulfillment, field tiles, replay and fingerprints are unaffected.

The v5 panel uses named `FOOD_STOCK_INDEX=3` and a regression check with distinct
cloth and food values. Its food stocks still include consumption and trade;
their changes are not direct harvest measurements. Managed field coverage is
not a forecast. Coordinate-mask coverage does not certify all mask combinations.

The v5 boundary evaluations retained aggregate summaries and final fingerprints,
not intermediate actions/population/QoL traces. A longer episode's eight-turn
prefix is compared with all available original summary values and its final
world fingerprint. Full new25-turn recorded-action replay is verified. Do not
describe this as an independent comparison of unavailable intermediate traces.
