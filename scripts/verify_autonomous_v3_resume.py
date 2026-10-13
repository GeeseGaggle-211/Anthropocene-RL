"""Verify discarded continuation copies; never modify retained training checkpoints."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from anthropocene_rl.autonomous_v4.collection import Collector
from anthropocene_rl.autonomous_v4.training import state_hash
from anthropocene_rl.training.autonomous_ppo_v3 import AutonomousTrainerV3 as Trainer


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    campaign = json.loads((args.run / "summary.json").read_text())
    config_path = Path("configs/training/autonomous-ppo-layernorm-v3-recovery.json")
    assert (
        hashlib.sha256(config_path.read_bytes()).hexdigest()
        == campaign["config"]["model_config_sha256"]
    )
    protocol = json.loads(config_path.read_text())
    entries = [
        e
        for e in json.loads((args.bank / "manifest.json").read_text())["entries"]
        if e["partition"] == "train"
    ]
    reports = []
    for result in campaign["seeds"]:
        seed, boundary = result["seed"], result["additional_transitions"]
        checkpoint = args.run / f"seed-{seed}-step-{boundary}.pt"
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        provenance = dict(
            campaign["provenance"],
            training_seed=seed,
            parent_checkpoint_sha256=campaign["config"]["parent_checkpoint_sha256"][str(seed)],
        )
        models = [Trainer(protocol, seed, provenance) for _ in range(2)]
        collected = []
        for trainer, workers in zip(models, [1, 4], strict=True):
            trainer.restore(checkpoint)
            with Collector(
                ["/workspace/Anthropocene/.tools/dotnet/dotnet", str(args.bridge.resolve())],
                workers=workers,
                deadline=time.monotonic() + 120,
            ) as collector:
                rows, summaries = collector.rollout(
                    trainer, args.bank, entries, seed, 16, verify=True
                )
            collected.append((rows, summaries, trainer.update(rows)))
        a, b = collected
        assert [r["fingerprint"] for r in a[1]] == [r["fingerprint"] for r in b[1]]
        assert a[2] == b[2]
        for x, y in zip(a[0], b[0], strict=True):
            assert torch.equal(x["logp"], y["logp"])
            for name in ["workforce", "activities", "controls", "movement", "trade"]:
                assert torch.equal(getattr(x["action"], name), getattr(y["action"], name))
        assert state_hash(models[0].model) == state_hash(models[1].model)
        assert equal(models[0].optimizer.state_dict(), models[1].optimizer.state_dict())
        assert torch.equal(models[0].shuffle.get_state(), models[1].shuffle.get_state())
        from anthropocene_rl.policies.autonomous_inference_v3 import AutonomousElderPolicyV3

        policy = AutonomousElderPolicyV3(checkpoint, expected_provenance=provenance, seed=seed)
        models[0].restore(checkpoint)
        for row in a[0]:
            assert torch.equal(
                policy.model(row["obs"], row["action"])[1],
                models[0].model(row["obs"], row["action"])[1],
            )
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
        reports.append(
            dict(
                seed=seed,
                boundary=boundary,
                checkpoint_sha256=digest,
                rows=len(a[0]),
                native_replay=True,
                exact_workers_1_4=True,
                exact_model_optimizer_rng=True,
                original_unchanged=True,
                inference_matches_checkpoint=True,
            )
        )
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports))


if __name__ == "__main__":
    main()
