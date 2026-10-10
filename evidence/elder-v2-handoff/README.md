# Elder v2 native and bridge evidence

This archive records the requested post-review checks against the frozen native
handoff and the selected learner implementation.

- Native repository: `GeeseGaggle-211/Anthropocene`
- Native commit: `5559961164d6f9da8e6d20c1dc563ecec3eb04b7`
- Learner commit under test: `33a655cee76a7bfa3eb8667388c1889e104daac4`
- Archived at: `2026-10-10T16:23:28Z`
- .NET SDK: `8.0.425`
- Python: `3.12.14`

The native worktree was detached at the exact commit and had no tracked or
untracked changes when the commands ran. All commands exited zero:

```text
dotnet Tools/ElderRL/bin/Release/net8.0/Anthropocene.ElderRL.dll schema
dotnet Tools/ElderRL/bin/Release/net8.0/Anthropocene.ElderRL.dll reset-smoke
dotnet Tools/ElderRL/bin/Release/net8.0/Anthropocene.ElderRL.dll step-smoke
dotnet Tools/ElderRL/bin/Release/net8.0/Anthropocene.ElderRL.dll replay-smoke
dotnet Tools/Headless/bin/Release/net8.0/Anthropocene.Headless.dll --elder-rl-only
```

The focused suite reported `PASS: 143 Elder RL assertions`. Structured native
identities and fingerprints are in `native-smoke.json`.

`native-fixture-9301.snapshot` is an actual 7,934-byte `SaveCodec` payload
exported from the exact native Elder tool fixture, not a hand-authored tensor.
The learner-owned JSON-lines bridge consumed those bytes through its normal
base64 JSON request path. One process handled Schema, Reset, two Steps, another
Reset and another Step. The second Step truncated at its configured decision
limit; Reset reproduced the initial and first-step fingerprints exactly. See
`bridge-snapshot-smoke.json`.

The learner commit passed its local CI-equivalent gate and GitHub Actions run
[`38067182809`](https://github.com/GeeseGaggle-211/Anthropocene-RL/actions/runs/38067182809).
See `learner-validation.json`.

This evidence covers the exact native smoke/regression and real persistent
bridge steps requested here. It does not claim completion of the later
deterministic tiny-snapshot PPO overfit, throughput, fitted training-bank
normalizer, checkpoint, or final partition-manifest gates in section 42.9.
