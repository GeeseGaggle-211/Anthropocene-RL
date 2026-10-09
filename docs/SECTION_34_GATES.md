# Section 34 implementation gates

This file mirrors implementation status from the normative Google Drive design
document. If this file and the design document disagree, the design document
wins and this file must be corrected.

## Approved normalization defaults

The normalization-v1 design is locked at the policy level:

- Stage 1: semantic/reference-relative representation wherever defensible;
- Stage 2: monotonic compression using log-ratio, asinh, log1p, or affine
  mapping according to semantic class;
- Stage 3: tanh as the primary soft-saturation function;
- robust tanh scale calibration from the frozen training partition;
- explicit absolute population scale plus relative herd intensity;
- capability Experience retained only when threshold-relative and nonredundant;
- commodity stock/need/capacity/flows expressed per capita and per physical
  decision horizon rather than divided blindly by LocalNeed;
- long-range contact distance expressed relative to local movement scale;
- ExpectedStayerGain uses signed asinh/tanh;
- binary masks/presence flags remain binary;
- no running inference normalizer.

## Native observation-semantic plumbing required before serious training

The design choices are resolved; implementation is still pending in the native
C# observation layer:

- use agent-local physical delta-t = current decision time - previous decision
  time; convert deficit/contact/trend timing away from turn-count/global-age
  semantics;
- remove planner-derived `Commodity.MarginalValue` from the policy input;
- populate fission exit opportunity with bounded `ExitOpportunityPressure`,
  not `DaughterGain`;
- expose realized signed migration flow/pressure rather than a fabricated
  forward-looking `RelativeAttractiveness` field;
- expose fishing management intensity q directly and a separate normalized
  required-maintenance labor burden; do not expose action-dependent build labor
  as a pre-decision fact;
- replace mixed-good `RecentTradeValue` with realized trade utility in the
  common person-QoL-year numeraire, normalized per capita per physical year;
  preserve at least net utility and, where schema space permits, gross trade
  intensity.

## Adapter decisions still open

- four bounded continuous action distributions;
- masked movement distribution;
- PPO likelihood when the native compiler repairs an action;
- physical-time discount and GAE semantics;
- C#/Python process and batching protocol;
- activation and initialization;
- serialized normalization/checkpoint manifest;
- movement-disabled first smoke versus full hybrid bring-up.

Do not implement PPO around the old observation semantics simply because the
Python skeleton can already represent a 512-float tensor.
