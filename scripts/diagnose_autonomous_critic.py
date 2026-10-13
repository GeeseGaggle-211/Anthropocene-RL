"""Preserve fixed-step critic recovery trajectories; do not alter a failed gate."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from anthropocene_rl.autonomous_v4.diagnostics import probe_face
from anthropocene_rl.autonomous_v4.training import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite attribution evidence")
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, weights_only=False)
    trainer = Trainer(state["config"], 0, state["provenance"], width=state["width"])
    trainer.restore(args.checkpoint)
    obs, labels, _, _ = probe_face(torch.load(args.probe, weights_only=True))
    result = {}
    for lr in [0.0003, 0.0001, 0.00003]:
        model = copy.deepcopy(trainer.model)
        target = model.value(obs).detach() + labels
        optimizer = torch.optim.Adam(model.critic.parameters(), lr=lr, eps=1e-5)
        sequence = []
        for step in range(64):
            loss = (model.value(obs) - target).square().mean()
            optimizer.zero_grad()
            loss.backward()
            grad = float(
                torch.nn.utils.clip_grad_norm_(
                    model.critic.parameters(), 10, error_if_nonfinite=True
                )
            )
            optimizer.step()
            mse = float((model.value(obs) - target).square().mean().detach())
            sequence.append(dict(step=step + 1, mse=mse, gradient_norm=grad))
        result[str(lr)] = sequence
        print(lr, [(r["step"], r["mse"]) for r in sequence if r["step"] in [1, 16, 32, 64]])
    args.output.write_text(
        json.dumps(
            dict(
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                probe_sha256=hashlib.sha256(args.probe.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                trajectories=result,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
