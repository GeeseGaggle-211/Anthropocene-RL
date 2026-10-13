"""Exact discarded native/PPO continuation and v4 inference across worker counts."""

import argparse
import json
import time
from pathlib import Path

import torch
from preflight_autonomous_v4_residual import sha
from run_autonomous_horizon_diagnostics_v5 import diagnostic_jobs
from verify_autonomous_v4_selected_resume import equal

from anthropocene_rl.autonomous_v4.collection import Collector, episode_seed
from anthropocene_rl.autonomous_v4.training import state_hash
from anthropocene_rl.policies.autonomous_inference_v4 import AutonomousElderPolicyV4
from anthropocene_rl.training.autonomous_ppo_v4 import AutonomousTrainerV4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    campaign = json.loads((args.run / "summary.json").read_text())
    cfg = campaign["config"]
    for name in ("bridge", "bank"):
        pin = cfg["inputs"][name]
        assert sha(pin["path"]) == pin["sha256"]
    bank = Path(cfg["inputs"]["bank"]["path"]).parent
    entries = [
        e
        for e in json.loads((bank / "manifest.json").read_text())["entries"]
        if e["partition"] == "train"
    ]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    deadline = started + 600
    jobs = diagnostic_jobs(campaign)
    report = dict(
        version="autonomous-horizon-exact-resume-v5",
        campaign_sha256=sha(args.run / "summary.json"),
        runner_sha256=sha(__file__),
        complete=False,
        checkpoints=[],
        discarded_validation_steps=0,
        replay_steps=0,
        maximum_discarded_validation_steps=384,
        maximum_replay_steps=384,
        maximum_wall_seconds=600,
        retained_training_updates=0,
    )
    try:
        for job in jobs:
            checkpoint = Path(job["checkpoint"]["path"])
            digest = job["checkpoint"]["sha256"]
            assert sha(checkpoint) == digest
            state = torch.load(checkpoint, weights_only=False)
            models = [
                AutonomousTrainerV4(state["config"], 0, state["provenance"], state["width"])
                for _ in range(2)
            ]
            anchors = {
                k: v
                for k, v in state["model"].items()
                if k.startswith(("reference_critic.", "initial_normalized_critic."))
            }
            collected = []
            for trainer, workers in zip(models, (1, 4), strict=True):
                trainer.restore(checkpoint)
                with Collector(
                    [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]],
                    workers=workers,
                    deadline=deadline,
                    request_timeout=60,
                ) as collector:
                    jobs4 = [
                        dict(
                            entry=entries[(trainer.episodes + i) % len(entries)],
                            sampler_seed=episode_seed(
                                state["provenance"]["training_seed"], trainer.episodes + i
                            ),
                            limit=4,
                        )
                        for i in range(4)
                    ]
                    rows, summaries = collector.collect(
                        trainer.model,
                        bank,
                        jobs4,
                        trainer.config,
                        verify=True,
                    )
                    trainer.episodes += len(jobs4)
                report["discarded_validation_steps"] += len(rows)
                report["replay_steps"] += len(rows)
                collected.append((rows, summaries, trainer.update(rows)))
                assert all(
                    torch.equal(v, trainer.model.state_dict()[k]) for k, v in anchors.items()
                )
            a, b = collected
            assert [r["fingerprint"] for r in a[1]] == [r["fingerprint"] for r in b[1]]
            assert a[2] == b[2]
            for x, y in zip(a[0], b[0], strict=True):
                for field in ("logp",):
                    assert torch.equal(x[field], y[field])
                for field in ("value", "next_value", "return", "advantage"):
                    assert x[field] == y[field]
                for name in ("workforce", "activities", "controls", "movement", "trade"):
                    assert torch.equal(getattr(x["action"], name), getattr(y["action"], name))
            assert state_hash(models[0].model) == state_hash(models[1].model)
            assert equal(models[0].optimizer.state_dict(), models[1].optimizer.state_dict())
            assert torch.equal(models[0].shuffle.get_state(), models[1].shuffle.get_state())
            assert torch.equal(models[0].sampler.get_state(), models[1].sampler.get_state())
            assert models[0].episodes == models[1].episodes
            policy = AutonomousElderPolicyV4(
                checkpoint, expected_provenance=state["provenance"], seed=911
            )
            models[0].restore(checkpoint)
            for row in a[0]:
                assert torch.equal(
                    policy.model(row["obs"], row["action"])[1],
                    models[0].model(row["obs"], row["action"])[1],
                )
                assert torch.equal(
                    policy.model.value(row["obs"]), models[0].model.value(row["obs"])
                )
            assert sha(checkpoint) == digest
            report["checkpoints"].append(
                dict(
                    checkpoint=job["checkpoint"],
                    rows=len(a[0]),
                    native_replay=True,
                    exact_workers_1_4=True,
                    exact_model_optimizer_rng=True,
                    frozen_anchors_exact=True,
                    inference_matches_checkpoint=True,
                    original_unchanged=True,
                )
            )
            assert report["discarded_validation_steps"] <= 384 and report["replay_steps"] <= 384
            print(json.dumps(report["checkpoints"][-1]), flush=True)
        report["complete"] = campaign.get("complete", False) and len(report["checkpoints"]) == len(
            jobs
        )
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
