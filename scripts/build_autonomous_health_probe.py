"""Freeze train-only native observations before inspecting canary outcomes."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from anthropocene_rl.autonomous_v4.collection import Collector
from anthropocene_rl.autonomous_v4.training import Trainer


def main():
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior probes")
    torch.set_num_threads(1)
    s = json.loads((args.pilot / "summary.json").read_text())
    t = Trainer(s["protocol"], 17, s["provenance"])
    t.restore(args.pilot / "seed-17-final.pt")
    entries = [
        e
        for e in json.loads((args.bank / "manifest.json").read_text())["entries"]
        if e["partition"] == "train"
    ]
    jobs = [dict(entry=e, sampler_seed=seed) for seed in [731, 732] for e in entries]
    with Collector(
        ["/workspace/Anthropocene/.tools/dotnet/dotnet", str(args.bridge.resolve())],
        workers=4,
        deadline=time.monotonic() + 120,
    ) as collector:
        rows, summaries = collector.collect(t.model, args.bank, jobs, t.config, verify=True)
    obs = {k: torch.cat([r["obs"][k] for r in rows]) for k in rows[0]["obs"]}
    torch.save(obs, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                version="health-probe-v2",
                rows=len(rows),
                sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                episodes=[
                    dict(
                        id=j["entry"]["id"],
                        sampler_seed=j["sampler_seed"],
                        fingerprint=s["fingerprint"],
                    )
                    for j, s in zip(jobs, summaries, strict=True)
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
