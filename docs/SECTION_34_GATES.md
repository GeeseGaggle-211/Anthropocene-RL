# Section 34 gates

The skeleton intentionally does not implement PPO or normalization while these
items remain unresolved.

## Yellow approvals

- robust tanh scale calibration from the training partition;
- absolute population and herd-scale representation;
- treatment of capability Experience;
- per-capita/per-horizon commodity flow representation;
- long-range contact distance transform;
- signed ExpectedStayerGain transform.

## Native red fixes required before serious training

- convert/remove global turn-time features in favor of agent-local physical time;
- remove/replace planner-derived Commodity.MarginalValue;
- replace unsafe fission DaughterGain/sentinel with finite exit pressure;
- populate or reserve placeholder contact/fishing features;
- replace/remove dimensionally ambiguous mixed-good RecentTradeValue.

## Adapter decisions still open

- four bounded continuous action distributions;
- masked movement distribution;
- PPO likelihood when the native compiler repairs an action;
- variable-delta-time discount and GAE semantics;
- C#/Python process and batching protocol;
- activation and initialization;
- serialized normalization/checkpoint manifest;
- movement-disabled first smoke versus full hybrid bring-up.
