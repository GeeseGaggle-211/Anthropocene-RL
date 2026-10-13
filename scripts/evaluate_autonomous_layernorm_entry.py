"""Pre-learning paired stochastic native evaluation; frozen bank and replay checked."""

import json
import time
from pathlib import Path

import torch
from probe_autonomous_state_baseline import make_trainer

from anthropocene_rl.autonomous_v4.collection import Collector


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = Path("runs/autonomous-layernorm-ablation-v3/native-entry.json")
    if output.exists():
        raise ValueError("Never overwrite evidence")
    bank = Path("datasets/autonomous-ppo-gate-v1")
    entries = [
        e
        for e in json.loads((bank / "manifest.json").read_text())["entries"]
        if e["partition"] == "development"
    ]
    results = {}
    started = time.monotonic()
    with Collector(
        [
            "/workspace/Anthropocene/.tools/dotnet/dotnet",
            str(Path("integration/bin/Release/net8.0/Anthropocene.RLBridge.dll").resolve()),
        ],
        workers=4,
        deadline=started + 300,
    ) as collector:
        for seed, boundary in [(17, 32768), (29, 4096), (43, 4096)]:
            checkpoint = Path(f"runs/autonomous-collapse-canary-v2/seed-{seed}-step-{boundary}.pt")
            for variant in ["v2", "v3"]:
                trainer = make_trainer(checkpoint, variant)
                jobs = [dict(entry=e, sampler_seed=rng) for e in entries for rng in [9101, 9102]]
                _, episodes = collector.collect(
                    trainer.model, bank, jobs, trainer.config, collect=False, verify=True
                )
                key = f"{seed}-{variant}"
                results[key] = dict(
                    episodes=episodes,
                    welfare=sum(e["welfare"] for e in episodes) / len(episodes),
                    survived=sum(e["survived"] for e in episodes),
                    count=len(episodes),
                    focal_trades=sum(e["events"]["focal_trades"] for e in episodes),
                )
                print(key, {k: v for k, v in results[key].items() if k != "episodes"}, flush=True)
                output.write_text(json.dumps(dict(results=results, partial=True), indent=2) + "\n")
    output.write_text(
        json.dumps(
            dict(
                results=results,
                seconds=time.monotonic() - started,
                replay_verified=True,
                training_updates=0,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
