"""Inference-only 25-decision village-local welfare panel for all v5 finalists."""

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch
from benchmark_autonomous_heuristic_welfare_v1 import RecordedWelfareTransport, aggregate, enrich
from preflight_autonomous_v4_residual import sha

from anthropocene_rl.autonomous_v4.training import episode, state_hash
from anthropocene_rl.training.autonomous_ppo_v4 import AutonomousTrainerV4

FOOD_STOCK_INDEX = 3  # Native TradeGood: Cloth=0, AnimalFabric=1, HousingMaterial=2, Food=3.


def food_stock(observation):
    return observation["Stocks"][FOOD_STOCK_INDEX]


def sequential_welfare(frames):
    # Match episode()'s actual accumulation order; Python3.12 sum compensates floats.
    total = 0.0
    for frame in frames:
        total += frame["raw_welfare"]
    return total


class PanelTransport(RecordedWelfareTransport):
    def __init__(self, *args, baseline_masks, **kwargs):
        self.total_steps = 0
        self.baseline_masks = baseline_masks
        super().__init__(*args, **kwargs)

    def reset_world(self, snapshot, controlled, limit):
        result = super().reset_world(snapshot, controlled, limit)
        self.previous = result["observations"][self.focal]
        return result

    def step_world(self, actions):
        if self.total_steps >= 3200:
            raise ValueError("V5 inference step cap exhausted")
        self.total_steps += 1
        before = self.previous
        response = super().step_world(actions)
        after = response["result"]["Observations"].get(self.focal)
        self.previous = after
        self.series[-1].update(
            field_tiles_before=before["Context"][16],
            field_tiles_after=None if after is None else after["Context"][16],
            managed_field_coverage=before["Context"][19],
            food_stock_before=food_stock(before),
            food_stock_after=None if after is None else food_stock(after),
            new_legal_coordinates={
                k: np.argwhere(np.asarray(before[k], dtype=bool) & ~mask).tolist()
                for k, mask in self.baseline_masks.items()
            },
        )
        return response


def prefix(row, limit):
    frames = row["intervals"][:limit]
    last = frames[-1]
    return enrich(
        dict(
            row,
            welfare=sequential_welfare(frames) / row["initial_population"],
            final_population=last["final_population"],
            survived=last["final_population"] > 0,
            food_met=sum(f["food_met"] for f in frames) / limit,
            events={k: sum(f["events"][k] for f in frames) for k in last["events"]},
            fingerprint=last["fingerprint"],
        ),
        frames,
        limit * 20,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    retry = json.loads(args.config.read_text())
    assert retry["maximum_total_panel_steps"] == retry["maximum_total_replay_steps"] == 3225
    for name, pin in retry["inputs"].items():
        assert sha(pin["path"]) == pin["sha256"], name
    failed = json.loads(Path(retry["inputs"]["failed_panel"]["path"]).read_text())
    assert not failed["complete"] and failed["trajectory_steps"] == failed["replay_steps"] == 25
    args.output.mkdir(parents=True, exist_ok=False)
    campaign = json.loads((args.run / "summary.json").read_text())
    cfg = campaign["config"]
    assert (
        cfg["final_inference_maximum_steps"] == cfg["final_inference_maximum_replay_steps"] == 3200
    )
    assert cfg["final_inference_maximum_seconds"] == 600
    assert campaign["complete"] and len(campaign["cells"]) == 8
    for name in ("bank", "bridge", "probe"):
        pin = cfg["inputs"][name]
        assert sha(pin["path"]) == pin["sha256"]
    prior_cfg = Path("configs/training/autonomous-25-turn-inference-v1.json")
    ids = json.loads(prior_cfg.read_text())["development_ids"]
    bank = Path(cfg["inputs"]["bank"]["path"]).parent
    entries = [
        e for e in json.loads((bank / "manifest.json").read_text())["entries"] if e["id"] in ids
    ]
    assert len(entries) == 8 and all(e["partition"] == "development" for e in entries)
    assert len({e["family"] for e in entries}) == 4
    probe = torch.load(cfg["inputs"]["probe"]["path"], weights_only=True)
    masks = {k: probe[k].any(0).numpy() for k in ("WorkforceMask", "ActivityMask", "ControlMask")}
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    result = dict(
        version="autonomous-horizon-welfare-v5-panel-fix1",
        retry_config_sha256=sha(args.config),
        previous_failed_panel_sha256=retry["inputs"]["failed_panel"]["sha256"],
        prior_panel_steps=25,
        prior_replay_steps=25,
        campaign_sha256=sha(args.run / "summary.json"),
        runner_sha256=sha(__file__),
        prior_panel_config_sha256=sha(prior_cfg),
        metric_helpers_sha256=sha("scripts/benchmark_autonomous_heuristic_welfare_v1.py"),
        development_ids=ids,
        sampler_seeds=[9101, 9102],
        training_updates=0,
        complete=False,
        cells=[],
    )

    def expired(*_):
        raise TimeoutError("V5 long inference deadline exceeded")

    old_handler = signal.signal(signal.SIGALRM, expired)
    signal.alarm(600)
    transport = PanelTransport(
        [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]],
        deadline=started + 600,
        baseline_masks=masks,
    )
    try:
        for cell in campaign["cells"]:
            frame = cell["health"][-1]
            checkpoint = Path(frame["checkpoint"]["path"])
            assert sha(checkpoint) == frame["checkpoint"]["sha256"]
            state = torch.load(checkpoint, weights_only=False)
            trainer = AutonomousTrainerV4(state["config"], 0, state["provenance"], state["width"])
            trainer.restore(checkpoint)
            model_hash = state_hash(trainer.model)
            expected = json.loads(
                checkpoint.with_name(checkpoint.stem + "-evaluation.json").read_text()
            )
            rows = []
            for entry in entries:
                for sampler in (9101, 9102):
                    _, row = episode(
                        trainer.model,
                        transport,
                        bank,
                        entry,
                        torch.Generator().manual_seed(sampler),
                        dict(state["config"], episode_limit=25),
                        verify=True,
                    )
                    row.update(sampler_seed=sampler)
                    enrich(row, transport.series, 500)
                    # Preserve completed/replayed measurements before any report assertion.
                    (args.output / f"{cell['id']}-episodes.json").write_text(
                        json.dumps([*rows, row], indent=2) + "\n"
                    )
                    short = prefix(row, 8)
                    original = expected[
                        next(i for i, r in enumerate(expected) if r["id"] == entry["id"])
                        + (sampler - 9101)
                    ]
                    for key in ("fingerprint", "welfare", "final_population"):
                        assert short[key] == original[key], (key, short[key], original[key])
                    for key in ("survived", "events", "elapsed_years"):
                        assert short[key] == original[key], (key, short[key], original[key])
                    assert (
                        float(np.mean([f["food_met"] for f in short["intervals"]]))
                        == original["food_met"]
                    )
                    row["original_eight_decision_summary_and_final_fingerprint_matches"] = True
                    rows.append(row)
                    (args.output / f"{cell['id']}-episodes.json").write_text(
                        json.dumps(rows, indent=2) + "\n"
                    )
            assert state_hash(trainer.model) == model_hash
            assert sha(checkpoint) == frame["checkpoint"]["sha256"]
            result["cells"].append(
                dict(
                    id=cell["id"],
                    checkpoint=frame["checkpoint"],
                    groups=[
                        dict(
                            limit=limit,
                            aggregate=aggregate([prefix(r, limit) for r in rows]),
                            families={
                                f: aggregate([prefix(r, limit) for r in rows if r["family"] == f])
                                for f in sorted({r["family"] for r in rows})
                            },
                            focal_events={
                                k: sum(prefix(r, limit)["events"][k] for r in rows)
                                for k in (
                                    "quotes",
                                    "focal_trades",
                                    "focal_fissions",
                                    "focal_migrants",
                                    "movement",
                                )
                            },
                        )
                        for limit in (8, 25)
                    ],
                    original_unchanged=True,
                )
            )
            print(json.dumps(result["cells"][-1]), flush=True)
        result["complete"] = True
        result["prefix_verification_scope"] = (
            "Original eight-decision aggregate reward, final population, food fulfillment, "
            "events, survival, elapsed years and final world fingerprint match. Original "
            "intermediate actions/population/QoL traces were not saved. Full new25-turn "
            "recorded-action replay is verified."
        )
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result.update(
            trajectory_steps=transport.total_steps,
            replay_steps=getattr(transport, "replay_steps", 0),
            cumulative_panel_steps=25 + transport.total_steps,
            cumulative_panel_replay_steps=25 + getattr(transport, "replay_steps", 0),
            wall_seconds=time.monotonic() - started,
        )
        transport.close()
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        assert result["cumulative_panel_steps"] <= 3225
        assert result["cumulative_panel_replay_steps"] <= 3225
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
