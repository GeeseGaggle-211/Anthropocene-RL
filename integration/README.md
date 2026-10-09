# Game-repository hook

Copy the metadata in `integration/anthropocene-game-hook.json` into
`Tools/ElderRL/training-repository.json` on a child branch of
`feature/elder-rl-scaffold`, add the exact training-repository commit SHA, and
add a short README link.

The game-side manifest owns that reverse pin:

- this repository pins the exact Anthropocene simulator commit and schemas;
- the Anthropocene RL branch pins the exact Anthropocene-RL commit it expects.

The hook is metadata only. There is no Git submodule, and no Python dependency
enters the Unity/C# repository. A PR should target
`feature/elder-rl-scaffold` so the native handoff remains reviewable.
