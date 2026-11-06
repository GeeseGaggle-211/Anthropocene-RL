# r5 operations and decision record

This branch (`ops/r5-runbook`) adds review material only. It contains no learner code changes, and nothing here is imported by the package.

## Code under review

| Component | Repository | Branch | Commit | Contents |
|---|---|---|---|---|
| Learner | GeeseGaggle-211/Anthropocene-RL | `feature/r5-integration` | `b6a1290` | All r5 learner code, including the Model9 bridge (`integration/Model9BridgeProgram.cs`) |
| Simulator | GeeseGaggle-211/Anthropocene | `feature/r5-privileged` | `8bd480b` | The simulator the VMs run |

Useful diff ranges in the learner:

- `git diff bf39c7c..b6a1290`: all r5 work, from the A head onward.
- `git diff 51c139a..b6a1290`: 26–27 Oct, covering the CUDA parity redesign, multi-dataset critic fits and collector checkpoint/resume.

## Contents

| Path | What it holds |
|---|---|
| `decisions/G0-manifest-draft.md` | The authoritative decision log. §8–11 cover r5, and dated entries are appended at the end. |
| `decisions/RECALL.md` | The co-architect's running state notes, in chronological order. |
| `decisions/intervention-brief-2026-10-26.md` | The approved brief: privileged inputs and an action-conditioned A head. |
| `decisions/qol-diag-report.md` | The QoL critic diagnosis that motivated amendment r5-b. |
| `decisions/pilot-plan.md` | The partial-data critic pilot run before the scale-out. |
| `decisions/cpu-operator-handoff.md` | The full description of the CPU-VM collection setup as of 27 Oct 03:30Z. |
| `gpu-vm/` | Creation, offline wheelhouse, bootstrap and recreate scripts for the L4 fitting VM. |
| `cpu-vm-snapshots/` | Read-only copies of the live CPU-VM driver scripts, crontab and the guardrail units. |
| `operator-tools/` | Helper scripts the operators used: polling, smoke tests and scenario/food-trade checks. Also the preemption tally. |

## Known open items, for r6 planning

These are recorded in the manifest.

1. **Actor normalizer floor and clip.** Trade columns, imported food among them, saturate: the std floor is 0.1 and values clip at ±10σ, so imported food tops out at about 1.72 food. The approved r6 path widens only those columns, freezes every other column's statistics at r5, warm-starts the r5 critic for a fine-tune, and reruns the gate.
2. **Simulator transient-state leak.** `EconomicLaborPlan.ActorActivityAuthority` and the `Actor*Workers` fields are not cleared at the turn boundary. Measured on one world, the leak changes no rows, labels or observations, only trace scores and the world fingerprint. It is patched in r6.
3. **Critic harness early stops.** Live early-stop guards, a per-epoch validation loss and a seed-0 gate are in progress on `feature/r5-early-stop`.
