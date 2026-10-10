# Native v2 process bridge

`Anthropocene.RLBridge.csproj` is a learner-owned stdio adapter. It compiles
the simulation C# files from the exact checkout pinned by
`configs/simulator/elder-v2.yaml` and exposes a JSON-lines protocol:

- `schema`
- `reset`
- `step`
- `close`

No simulation rule is implemented here. Reset/Step delegate directly to
`NativeElderEnvironment`; tensor encoding delegates to
`ElderObservationSchema`.

Build the bridge after preparing the simulator checkout:

```bash
uv run python scripts/bootstrap_simulator.py
uv run anthropocene-rl adapter-smoke
```

The reverse game-repository pin remains metadata-only. Once a learner commit is
selected for an experiment, record that exact SHA in the game-side
`Tools/ElderRL/training-repository.json` without introducing a submodule.
