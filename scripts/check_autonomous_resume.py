"""Exact native rollout and update continuation after populated Adam state."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from anthropocene_rl.autonomous_v4.env import AutonomousTransport
from anthropocene_rl.autonomous_v4.training import Trainer, rollout, state_hash


def tree_hash(value):
    digest = hashlib.sha256()

    def visit(v):
        if isinstance(v, torch.Tensor):
            digest.update(str((v.dtype, tuple(v.shape))).encode())
            digest.update(v.detach().cpu().numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=repr):
                digest.update(repr(k).encode())
                visit(v[k])
        elif isinstance(v, (list, tuple)):
            for item in v:
                visit(item)
        else:
            digest.update(repr(v).encode())

    visit(value)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    summary = json.loads((args.run / "summary.json").read_text())
    config, provenance = summary["protocol"], summary["provenance"]
    entries = [
        e
        for e in json.loads((args.bank / "manifest.json").read_text())["entries"]
        if e["partition"] == "train"
    ]
    results = []
    with AutonomousTransport(
        ["/workspace/Anthropocene/.tools/dotnet/dotnet", str(args.bridge.resolve())]
    ) as transport:
        for seed in config["initialization_seeds"]:
            a, b = Trainer(config, seed, provenance), Trainer(config, seed + 1, provenance)
            a.restore(args.run / f"seed-{seed}-final.pt")
            b.restore(args.run / f"seed-{seed}-final.pt")
            if not a.optimizer.state:
                raise ValueError("Test needs populated Adam moments")
            rows_a, summaries_a = rollout(a, transport, args.bank, entries, verify=True)
            updates_a = a.update(rows_a)
            rows_b, summaries_b = rollout(b, transport, args.bank, entries, verify=True)
            updates_b = b.update(rows_b)
            checks = dict(
                fingerprints=[x["fingerprint"] for x in summaries_a]
                == [x["fingerprint"] for x in summaries_b],
                likelihoods=all(
                    torch.equal(x["logp"], y["logp"]) for x, y in zip(rows_a, rows_b, strict=True)
                ),
                model=state_hash(a.model) == state_hash(b.model),
                optimizer=tree_hash(a.optimizer.state_dict())
                == tree_hash(b.optimizer.state_dict()),
                sampler=torch.equal(a.sampler.get_state(), b.sampler.get_state()),
                shuffle=torch.equal(a.shuffle.get_state(), b.shuffle.get_state()),
                updates=updates_a == updates_b,
            )
            if not all(checks.values()):
                raise ValueError(f"Continuation divergence: {checks}")
            results.append(
                dict(
                    seed=seed,
                    checks=checks,
                    duplicate_validation_transitions=len(rows_a) * 2,
                    optimizer_sha256=tree_hash(a.optimizer.state_dict()),
                    model_sha256=state_hash(a.model),
                )
            )
    args.output.write_text(
        json.dumps(
            dict(status="PASS", seeds=results, retained_training_checkpoints="unchanged"), indent=2
        )
    )
    print("PASS: exact native resume after nonempty Adam state for all three seeds")


if __name__ == "__main__":
    main()
