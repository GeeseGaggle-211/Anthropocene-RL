# Cultivation constant-grid headroom audit

This is the completed zero-update Phase 0a audit frozen at learner commit
`b86740dbe7e2a0678a7b9b4ac9c2f2d02e4fe280` against native simulator
`5559961164d6f9da8e6d20c1dc563ecec3eb04b7`.

- Status: `NO_GO_CONSTANT_GRID_ROUTE`
- Evidence SHA-256:
  `92e9ee135b651d7328eed27ce42413a6d434e5bdab84e3fc3d8c39dca75bd5f8`
- Scope: all 210 TRAIN and 46 DEVELOPMENT snapshots
- HELDOUT: unread
- Work: 6,656 episodes; at most 33,280 forward transitions; zero optimizer updates
- Integrity: complete PASS with no scope, replay, request, numerical, or reward-denominator failures
- F15 delivered-trade contrast: `UNEVALUABLE`

## Primary three-decision endpoint

The per-snapshot hindsight best of the fixed 5×5 constant-action grid was
compared with Option A using clustered scenario uncertainty.

| Partition | Snapshots | Independent clusters | Mean headroom | One-sided 95% lower | One-sided 95% upper |
|---|---:|---:|---:|---:|---:|
| TRAIN | 210 | 144 | +0.8913 pp | +0.5080 pp | +1.3166 pp |
| DEVELOPMENT | 46 | 32 | +0.8118 pp | +0.3890 pp | +1.2846 pp |

Both upper confidence bounds are below the predeclared 1.5-point useful-effect
floor, so the constant-grid route is a `NO_GO`. No family met the complete
replicated qualifying gate. F16 was descriptively strongest but had only one
independent DEVELOPMENT cluster.

## Secondary five-decision endpoint

| Partition | Mean headroom | One-sided 95% lower | One-sided 95% upper |
|---|---:|---:|---:|
| TRAIN | +1.6514 pp | +1.2130 pp | +2.1495 pp |
| DEVELOPMENT | +1.5704 pp | +0.7746 pp | +2.3481 pp |

This positive secondary result cannot change the frozen status or authorize a
five-step PPO arm. It is evidence that the broader duration/formulation question
remains open.

The TRAIN-selected global constant action was field operation 0 and Development
0. It was worse than Option A by 1.91 points on TRAIN and 2.40 points on
DEVELOPMENT. Therefore the limited hindsight headroom is state-dependent rather
than a globally superior constant setting.

The hindsight grid exceeded the three selected update-168 policies on
DEVELOPMENT by 5.53, 3.22, and 4.54 points for seeds 1701, 1901, and 2101,
respectively. This does not make the grid deployable; it shows that the policies
missed even the limited state-specific headroom.

The large episode traces and complete summary are durable run artifacts and
must be stored in the immutable GCS run store rather than committed to Git.
