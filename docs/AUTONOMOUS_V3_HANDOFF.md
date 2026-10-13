# Autonomous Elder economy v3 handoff

Experimental branch: `feature/elder-autonomous-economy`. Native simulator:
`5463fd3c5ca34a65fef5a57239f5a0aa65d1e50b`, built from the requested
`5559961164d6f9da8e6d20c1dc563ecec3eb04b7` base. Learner base:
`fa71f370aa50982552584861e61682d7cd2897e0` (verified main).

The shared [implementation contract](AUTONOMOUS_ELDER_ECONOMY_V1.md) was written
before implementation and is byte-identical in the simulator repository.
`configs/simulator/elder-autonomous-v3.json` pins source, commit, contract,
versions, dimensions and policy scope independently of `elder-v2.yaml`.
Action 2 / observation 3 / environment 3 / replay 3 JSON Schemas are beside it.

## Implemented slice

The native compiler now accepts masked workforce shares over all 13 professions,
uses stable largest remainder over actual residents, and reserves conditional
material/recipe activity time without economic rescue. Native production can
accumulate material and manufactured-good surplus beyond local-use demand.
Physical resources, tools, dependent recipes, storage and phase ordering remain
in the simulator. Field expansion, herd retention and breeding are explicit.

The new environment accepts simultaneous controlled-Elder submissions, provides
owner-isolated observations, and supports bounded reciprocal stock quotes with
native escrow. The native scripted tests demonstrate production -> surplus ->
next-boundary two-settlement trade, including conservation and incompatible
quotes. Replay captures requests, staffing, background orders and full native
fingerprints; persisted replay is verified.

The learner adds:

- `autonomous.env.AutonomousTransport` for multi-Elder WEGO requests and quote
  actions, and `AutonomousElderEnv` for a focal Gymnasium agent with scripted peers;
- separate 753-feature observations and native workforce/activity/control masks;
- masked Dirichlet workforce and recipe distributions and masked Beta controls;
- zero density/entropy for deterministic absent/singleton faces, no inactive RNG
  consumption, and likelihoods on requested shares rather than rounded staffing;
- an actor/critic and bounded PPO mechanics update, leaving the old policy,
  optimizer, datasets, normalizers and checkpoints unchanged;
- conditional bridge compilation: the original pinned source still builds and
  exposes the original five-head protocol.

## Results

Native final focused suite: **462 assertions passed**. Full native fixed suite
passed (69 PASS records); old Elder 143-assertion baseline passed. Four heuristic
and three original five-head transitions have **exact baseline/candidate
fingerprint parity**, archived in `evidence/elder-autonomous-v3/`.

Learner baseline: **131 passed, one skipped**. Final learner suite with native
integration enabled: **136 passed, one skipped** (76.65 seconds). All five new
autonomous tests pass, including a real three-boundary native Gym/PPO update,
replay and reset determinism. Ruff passes. Both bridge variants compile.
The existing skipped test remains skipped; native v3 integration was exercised.
Only CPU mechanical updates were performed, with PyTorch 2.14.1+cpu and
Gymnasium 1.2.2. No expensive PPO campaign, GPU ecology or merge was run.

Independent review found and verified fixes for contact-slot alignment, stale
relocated contacts, Housing Material tool coverage, and finite timber pressure
requests. Its independent randomized probe passed **420 labor/replay checks**;
that probe is now included in the native regression suite. Final review found
no remaining blocker for this mechanics slice.

Complete native-step medians (heuristic/autonomous milliseconds), with economic
execution, per-camp observation and fingerprinting included: 1 camp 1.574/1.161;
2 camps 2.156/1.602; 8 camps 8.337/6.091; 32 camps 29.844/23.012; 80 camps
71.125/58.876. At 80 camps, allocated bytes averaged 54.43/47.05 MB. Seven samples
after two warmups, scalar .NET Release, tiered JIT disabled, seed 9301, 120
residents/camp. The maps are compact synthetic fixtures, not generated mature
worlds; timing outliers and an earlier 8-camp regression preclude broad speedup
claims. Raw data are in `evidence/elder-autonomous-v3/performance.json`.

Detailed native methodology and limitations:
[AutonomousElderValidation.md](https://github.com/GeeseGaggle-211/Anthropocene/blob/5463fd3c5ca34a65fef5a57239f5a0aa65d1e50b/Docs/AutonomousElderValidation.md).

## Reproduce

Use a checkout of the pinned simulator SHA and install this repository's adapter,
training and test dependencies. These commands do not start a training campaign:

```sh
python scripts/check_autonomous_manifest.py --root /absolute/Anthropocene-elder
dotnet build -c Release integration/Anthropocene.RLBridge.csproj \
  -p:AnthropoceneSimRoot=/absolute/Anthropocene-elder
dotnet run -c Release \
  --project /absolute/Anthropocene-elder/Tools/Headless/Anthropocene.Headless.csproj \
  -- --autonomous-elder-fixture /tmp/autonomous-fixture.anthro
AUTONOMOUS_BRIDGE='["dotnet","/absolute/Anthropocene-RL/integration/bin/Release/net8.0/Anthropocene.RLBridge.dll"]' \
AUTONOMOUS_FIXTURE=/tmp/autonomous-fixture.anthro AUTONOMOUS_CAMPS=1,3 \
  python -m pytest tests/test_autonomous_economy.py
python -m pytest
ruff check .
```

The native integration test intentionally skips unless those bridge/fixture
variables are set; use them for the entire pytest run to reproduce the full
136-pass result. The original `check-manifest`, adapter and policy commands
continue to target the old frozen contract. Use the autonomous manifest checker
for v3; old datasets and checkpoints are not v3-compatible.

## Remaining work and safe gate

Learned quote heads are deferred: native and scripted quotes work, while the
initial Gym learned action space fixes no quote. Quotes currently match controlled
Elders only. Movement is stay-only; food harvest retains native demand/reserve
semantics; made-to-order trade is excluded. There is no calibrated v3 normalizer,
campaign runner, persistent optimizer/checkpoint contract, ONNX export or
distributed autonomous learner. Fission daughters remain native background Elders.

Next safe gate is bounded CPU-only scripted/randomized multi-world rollouts,
covering deprivation, storage, resource competition, livestock/fishing and
reciprocal trade. Audit physical/replay ledgers and generated-world full-step
cost, then freeze v3 TRAIN-only normalization and training contracts before a
small learning pilot. This result does not claim policy quality or authorize
expensive training.
