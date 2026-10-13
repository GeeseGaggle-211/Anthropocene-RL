"""Prospectively fixed fresh-probe preflight for the bounded v3 native canary."""

import hashlib
import json
from pathlib import Path

import torch

from anthropocene_rl.autonomous_v4.diagnostics import full_recovery_probe, health
from anthropocene_rl.training.autonomous_ppo_v3 import AutonomousTrainerV3


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = Path("runs/autonomous-layernorm-ablation-v3/native-preflight-k05.json")
    if output.exists():
        raise ValueError("Never overwrite evidence")
    probe = Path("runs/autonomous-canary-preflight-v1/probe-v2.pt")
    obs = torch.load(probe, weights_only=True)
    config_path = Path("configs/training/autonomous-ppo-layernorm-v3-recovery.json")
    config = json.loads(config_path.read_text())
    result = {}
    for seed in [29, 43]:
        checkpoint = Path(f"runs/autonomous-collapse-canary-v2/seed-{seed}-step-4096.pt")
        state = torch.load(checkpoint, weights_only=False)
        trainer = AutonomousTrainerV3(config, seed, state["provenance"])
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        trainer.warm_start(checkpoint, expected_sha256=digest)
        recovery = full_recovery_probe(trainer.model, obs, steps=16, lr=0.001, critic_lr=0.0001)
        result[str(seed)] = dict(
            checkpoint_sha256=digest, health=health(trainer.model, obs), recovery=recovery
        )
        print(seed, recovery["passed"], flush=True)
    output.write_text(
        json.dumps(
            dict(
                config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),
                results=result,
                passed=all(r["recovery"]["passed"] for r in result.values()),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
