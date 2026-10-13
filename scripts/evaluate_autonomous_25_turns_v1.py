"""Frozen, inference-only extension of original evaluation trajectories to 25 decisions."""

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch
from report_autonomous_welfare_v4 import WelfareTransport, aggregate, sha

from anthropocene_rl.autonomous_v4.contracts import action_wire
from anthropocene_rl.autonomous_v4.training import episode, state_hash
from anthropocene_rl.training.autonomous_ppo_v3 import AutonomousTrainerV3


class HorizonTransport(WelfareTransport):
    def __init__(self, *args, baseline_masks, **kwargs):
        self.total_steps = 0
        self.baseline_masks = baseline_masks
        super().__init__(*args, **kwargs)

    def reset_world(self, snapshot, controlled, limit):
        response = super().reset_world(snapshot, controlled, limit)
        self.previous = response["observations"][self.focal]
        return response

    def step_world(self, actions):
        if self.total_steps >= 800:
            raise ValueError("Inference step budget exhausted")
        self.total_steps += 1
        before = self.previous
        requested = action_wire(actions[int(self.focal)])
        response = super().step_world(actions)
        result = response["result"]
        after = result["Observations"].get(self.focal)
        self.previous = after
        newly_legal = {
            k: np.argwhere(np.asarray(before[k], dtype=bool) & ~mask).tolist()
            for k, mask in self.baseline_masks.items()
        }
        self.series[-1].update(
            fingerprint=result["Fingerprint"],
            requested_action=requested,
            retained_field_tiles_before=before["Context"][16],
            retained_field_tiles_after=None if after is None else after["Context"][16],
            current_field_food_support=before["Context"][19],
            future_field_food_support=before["Context"][20],
            field_operation=before["Context"][17],
            food_stock_before=before["Stocks"][0],
            food_stock_after=None if after is None else after["Stocks"][0],
            food_needs_met=result["Telemetry"]["FoodMet"][self.focal],
            new_legal_coordinates=newly_legal,
        )
        return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    assert cfg["version"] == "autonomous-25-turn-inference-v1"
    assert cfg["maximum_steps"] == 800 and cfg["maximum_wall_seconds"] == 300
    for path, digest in cfg["learner_sources"].items():
        assert sha(Path(path)) == digest, path
    assert cfg["sampler_seeds"] == [9101, 9102]
    for name, pin in cfg["inputs"].items():
        assert sha(Path(pin["path"])) == pin["sha256"], name
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    baseline = json.loads(Path(cfg["inputs"]["welfare"]["path"]).read_text())
    bank = Path(cfg["inputs"]["bank"]["path"]).parent
    entries = json.loads((bank / "manifest.json").read_text())["entries"]
    chosen = [e for e in entries if e["id"] in cfg["development_ids"]]
    assert len(chosen) == 8 and all(e["partition"] == "development" for e in chosen)
    probe = torch.load(cfg["inputs"]["probe"]["path"], weights_only=True)
    masks = {k: probe[k].any(0).numpy() for k in ["WorkforceMask", "ActivityMask", "ControlMask"]}
    started = time.monotonic()

    def expired(*_):
        raise TimeoutError("25-decision inference deadline exceeded")

    old_handler = signal.signal(signal.SIGALRM, expired)
    signal.alarm(300)
    transport = HorizonTransport(
        [
            "/workspace/Anthropocene/.tools/dotnet/dotnet",
            str(Path(cfg["inputs"]["bridge"]["path"]).resolve()),
        ],
        deadline=started + 300,
        baseline_masks=masks,
    )
    reports = []
    try:
        for seed in [29, 43]:
            checkpoint = Path(cfg["inputs"][str(seed)]["path"])
            state = torch.load(checkpoint, weights_only=False)
            trainer = AutonomousTrainerV3(
                state["config"], seed, state["provenance"], state["width"]
            )
            trainer.restore(checkpoint)
            model_hash = state_hash(trainer.model)
            original = next(
                r for r in baseline["reports"] if r["seed"] == seed and r["boundary"] == 24576
            )
            rows = []
            for entry in chosen:
                for sampler in cfg["sampler_seeds"]:
                    _, row = episode(
                        trainer.model,
                        transport,
                        bank,
                        entry,
                        torch.Generator().manual_seed(sampler),
                        dict(state["config"], episode_limit=25),
                        verify=True,
                    )
                    series = transport.series
                    expected = next(
                        e
                        for e in original["episodes"]
                        if e["id"] == entry["id"] and e["sampler_seed"] == sampler
                    )
                    assert len(series) >= len(expected["intervals"])
                    for actual, prior in zip(series, expected["intervals"], strict=False):
                        assert all(actual[k] == v for k, v in prior.items())
                    assert (
                        series[len(expected["intervals"]) - 1]["fingerprint"]
                        == expected["fingerprint"]
                    )
                    years = sum(s["years"] for s in series)
                    person_years = sum(s["prior_population"] * s["years"] for s in series)
                    raw = sum(s["raw_welfare"] for s in series)
                    row.update(
                        sampler_seed=sampler,
                        intervals=series,
                        raw_welfare=raw,
                        person_years=person_years,
                        time_mean_population=person_years / years,
                        time_mean_qol=sum(s["qol"] * s["years"] for s in series) / years,
                        person_year_weighted_qol=raw / person_years,
                        final_qol=series[-1]["qol"],
                        original_eight_decision_prefix_matches=True,
                    )
                    rows.append(row)
                    (args.output / f"seed-{seed}-episodes.json").write_text(
                        json.dumps(rows, indent=2) + "\n"
                    )
            assert state_hash(trainer.model) == model_hash
            assert sha(checkpoint) == cfg["inputs"][str(seed)]["sha256"]
            reports.append(
                dict(
                    seed=seed,
                    aggregate=aggregate(rows),
                    families={
                        f: aggregate([r for r in rows if r["family"] == f])
                        for f in sorted({r["family"] for r in rows})
                    },
                )
            )
            print(json.dumps(reports[-1]), flush=True)
    finally:
        transport.close()
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    (args.output / "summary.json").write_text(
        json.dumps(
            dict(
                version=cfg["version"],
                complete=True,
                config_sha256=sha(args.config),
                script_sha256=sha(Path(__file__)),
                config=cfg,
                seeds=reports,
                native_steps=transport.total_steps,
                native_training_updates=0,
                wall_seconds=time.monotonic() - started,
                limits=(
                    "8-step prefix checks compare original population/QoL/rewards and final "
                    "world fingerprint; original action/intermediate fingerprint traces were "
                    "not retained. Full new 25-step replay verifies recorded actions."
                ),
                output_measurement=(
                    "Current/future field food-support fields alias ManagedRecoveryCoverage; "
                    "the future field is not a forecast. Stock changes include consumption/trade; "
                    "none is direct harvest. New legal coordinates do not detect "
                    "new mask combinations."
                ),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
