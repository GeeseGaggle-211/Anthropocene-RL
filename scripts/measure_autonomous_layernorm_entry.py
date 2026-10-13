"""Exact pre-learning distribution shift on the frozen train-only probe."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from probe_autonomous_state_baseline import make_trainer
from torch.distributions import Dirichlet, kl_divergence

from anthropocene_rl.autonomous_v4.diagnostics import health
from anthropocene_rl.policies.autonomous_ppo_v2 import distribution


@torch.no_grad()
def entry(a, b, obs):
    hs = [m.actor(obs["x"]) for m in (a, b)]
    alphas = [
        torch.nn.functional.softplus(m.heads["simplex"](h).clamp(-12, 12)) + 0.5
        for m, h in zip((a, b), hs, strict=True)
    ]
    continuous_kl = torch.zeros(len(hs[0]))
    mean_shift = {}
    faces = [("workforce", slice(0, 13), obs["WorkforceMask"])]
    faces += [
        (f"activity_{i}", slice(13 + i * 15, 28 + i * 15), obs["ActivityMask"][:, i])
        for i in range(13)
    ]
    faces += [
        (
            f"control_{i}",
            slice(208 + i * 2, 210 + i * 2),
            obs["ControlMask"][:, i, None].expand(-1, 2),
        )
        for i in range(3)
    ]
    for name, span, mask in faces:
        aa, bb = [alpha[:, span] for alpha in alphas]
        means = [(x * mask) / (x * mask).sum(-1, keepdim=True).clamp_min(1e-10) for x in (aa, bb)]
        mean_shift[name] = float((means[0] - means[1]).abs().max())
        for face in mask.unique(dim=0):
            if face.sum() < 2:
                continue
            rows = (mask == face).all(-1)
            continuous_kl[rows] += kl_divergence(
                Dirichlet(aa[rows][:, face]), Dirichlet(bb[rows][:, face])
            )

    def cats(name, mask):
        safe = mask.clone()
        safe[~safe.any(-1), 0] = True
        return [
            distribution(m.heads[name](h), safe, m.uniform_mix)[0]
            for m, h in zip((a, b), hs, strict=True)
        ]

    def kl(pair):
        return kl_divergence(*pair)

    give_mask = obs["GoodMask"] & (obs["Stocks"] >= 1e-6)
    eligible = obs["CounterpartyMask"].any(-1) & give_mask.any(-1) & (obs["GoodMask"].sum(-1) > 1)
    enables = cats("enabled", torch.stack([torch.ones_like(eligible), eligible], -1))
    offered = cats("give", give_mask)
    downstream = kl(cats("partner", obs["CounterpartyMask"])) + kl(offered)
    for name in ("lot", "price"):
        downstream += kl(cats(name, torch.ones(len(eligible), 3, dtype=torch.bool)))
    for g in range(give_mask.shape[-1]):
        wanted = obs["GoodMask"].clone()
        wanted[:, g] = False
        downstream += offered[0].probs[:, g] * kl(cats("want", wanted))
    categorical_kl = (
        kl(cats("movement", obs["MovementMask"]))
        + kl(enables)
        + enables[0].probs[:, 1] * downstream
    )
    return dict(
        kl_direction="v2_no_norm_to_v3_norm",
        categorical_kl_mean=float(categorical_kl.mean()),
        continuous_kl_mean=float(continuous_kl.mean()),
        joint_kl_mean=float((continuous_kl + categorical_kl).mean()),
        maximum_coordinate_mean_shift=mean_shift,
        categorical_entropy_mean=[float(m.categorical_terms(obs)[0].mean()) for m in (a, b)],
        health=[health(m, obs) for m in (a, b)],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Never overwrite evidence")
    torch.set_num_threads(1)
    obs = torch.load(args.probe, weights_only=True)
    result = {}
    for path in args.checkpoints:
        a, b = [make_trainer(path, v).model for v in ("v2", "v3")]
        result[str(path)] = dict(
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **entry(a, b, obs)
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: {n: v for n, v in r.items() if n.endswith("kl_mean")} for k, r in result.items()}
        )
    )


if __name__ == "__main__":
    main()
