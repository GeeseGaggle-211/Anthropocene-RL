# Native v2 process bridge

`Anthropocene.RLBridge.csproj` is a learner-owned stdio adapter. It compiles
the simulation C# files from the exact checkout pinned by
`configs/simulator/elder-v2.yaml` and exposes a JSON-lines protocol:

- `schema`
- `reset`
- `step`
- `verify_replay`
- `close`

No simulation rule is implemented here. Reset/Step and replay verification
delegate directly to `NativeElderEnvironment`; tensor encoding delegates to
`ElderObservationSchema`. Step responses also expose non-authoritative wall
time measured immediately around the native `Step` call for local profiling.

Build the bridge after preparing the simulator checkout:

```bash
uv run python scripts/bootstrap_simulator.py
uv run anthropocene-rl adapter-smoke
```

The reverse game-repository pin remains metadata-only. Once a learner commit is
selected for an experiment, record that exact SHA in the game-side
`Tools/ElderRL/training-repository.json` without introducing a submodule.

## Native snapshot-bank generator

`Anthropocene.SnapshotBank.csproj` is a second learner-owned executable compiled
from the same pinned simulation sources. `SnapshotBankProgram` coordinates
native validation and `SnapshotScenarioFactory` creates bounded scenario
fixtures using `WorldGenerator`, native state types, native capability rules,
native ecology queries, and `SaveCodec`. It does not implement simulation
formulas.

For every accepted snapshot it performs a canonical Save/Load/Fingerprint
roundtrip, Reset, at least one legal Step, replay verification, schema/config
checks, finite-time checks, structural-mask validation, and measured-stratum
validation. Failed trajectory candidates are returned to Python with a reason
instead of being silently repaired. Wall-clock generation timings and
process-memory warnings are canonicalized because `SaveCodec.Fingerprint`
explicitly excludes that non-authoritative telemetry; physical state remains
untouched.

The executable emits temporary native files and calibration observations.
`anthropocene-rl generate-snapshot-bank` moves the SaveCodec files into
gitignored SHA-256 paths, commits only the manifest/reports, and performs a
second identical run to verify deterministic regeneration.
