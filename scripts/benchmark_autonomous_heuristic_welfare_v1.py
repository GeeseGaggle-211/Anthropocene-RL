"""Matched native-heuristic headroom audit; no PPO or fitted evaluation statistics."""

import argparse
import json
import math
import signal
import time
from pathlib import Path

import torch
from report_autonomous_welfare_v4 import WelfareTransport, sha
from report_autonomous_welfare_v4 import aggregate as physical_aggregate

from anthropocene_rl.autonomous_v4.contracts import action_wire, observation
from anthropocene_rl.autonomous_v4.training import episode, market_peer
from anthropocene_rl.training.autonomous_ppo_v3 import AutonomousTrainerV3


class HeuristicTransport(WelfareTransport):
    def reset_heuristic(self, snapshot, focal, peer, limit):
        response = self._request(
            dict(
                op="heuristic_reset", snapshot=snapshot, focal=focal, peer=peer, decisionLimit=limit
            )
        )
        self.focal = str(focal)
        self.population = int(response["observations"][self.focal]["Context"][0])
        self.series = []
        return response

    def step_world(self, actions):
        response = self._request(
            dict(op="heuristic_step", actions={str(k): action_wire(v) for k, v in actions.items()})
        )
        result = response["result"]
        t = result["Telemetry"]
        quality, years = t["Quality"][self.focal], result["ElapsedYears"]
        raw = result["Rewards"][self.focal]
        assert math.isclose(raw, self.population * quality * years, rel_tol=1e-12, abs_tol=1e-9)
        self.series.append(
            dict(
                prior_population=self.population,
                final_population=t["Population"][self.focal],
                qol=quality,
                years=years,
                raw_welfare=raw,
                immigrants=t["Immigrated"][self.focal],
                emigrants=t["Emigrated"][self.focal],
                daughter_settlements=t["DaughterSettlements"][self.focal],
                fingerprint=result["Fingerprint"],
            )
        )
        self.population = t["Population"][self.focal]
        return response

    def verify_replay(self):
        response = self._request(dict(op="heuristic_verify"))
        self.replay_steps = getattr(self, "replay_steps", 0) + response["steps"]
        return response


def enrich(row, series, planned_years):
    person_years = sum(x["prior_population"] * x["years"] for x in series)
    raw = sum(x["raw_welfare"] for x in series)
    row.update(
        intervals=series,
        raw_welfare=raw,
        person_years=person_years,
        time_mean_population=person_years / planned_years,
        time_mean_qol=sum(x["qol"] * x["years"] for x in series) / planned_years,
        person_year_weighted_qol=raw / max(1, person_years),
        final_qol=series[-1]["qol"],
        elapsed_years=sum(x["years"] for x in series),
        planned_years=planned_years,
    )
    for horizon in (100, 400):
        elapsed, discounted = 0.0, 0.0
        for frame in series:
            discounted += (
                math.exp(-elapsed / horizon)
                * frame["raw_welfare"]
                / frame["years"]
                * horizon
                * -math.expm1(-frame["years"] / horizon)
            )
            elapsed += frame["years"]
        row[f"discounted_h{horizon}"] = discounted / row["initial_population"]
    return row


def aggregate(rows):
    result = physical_aggregate(rows)
    result["observed_elapsed_years"] = result.pop("elapsed_years")
    assert len({r["planned_years"] for r in rows}) == 1
    result["planned_years"] = rows[0]["planned_years"]
    families = sorted({r["family"] for r in rows})
    for horizon in (100, 400):
        key = f"discounted_h{horizon}"
        result[key] = sum(
            sum(r[key] for r in rows if r["family"] == f) / sum(r["family"] == f for r in rows)
            for f in families
        ) / len(families)
    return result


class RecordedWelfareTransport(WelfareTransport):
    def verify_replay(self):
        response = super().verify_replay()
        self.replay_steps = getattr(self, "replay_steps", 0) + response["steps"]
        return response

    def step_world(self, actions):
        response = super().step_world(actions)
        t = response["result"]["Telemetry"]
        self.series[-1].update(
            fingerprint=response["result"]["Fingerprint"],
            food_met=t["FoodMet"][self.focal],
            events=dict(
                movement=t["Moved"][self.focal],
                trades=t["Trades"],
                fissions=t["Fissions"],
                migrants=t["Migrants"],
                quotes=int(actions[int(self.focal)].get("quote") is not None),
                focal_trades=t["SettledTrades"][self.focal],
                focal_fissions=t["DaughterSettlements"][self.focal],
                focal_migrants=t["Immigrated"][self.focal] + t["Emigrated"][self.focal],
            ),
        )
        return response


def heuristic_episode(transport, bank, entry, limit, peer_mode):
    snapshot = (bank / entry["snapshot"]).read_bytes()
    assert sha(bank / entry["snapshot"]) == entry["sha256"]
    peer = entry["peer"] if peer_mode == "public-market" and entry["peer"] else -1
    response = transport.reset_heuristic(snapshot, entry["focal"], peer, limit)
    initial = response
    events = dict(
        movement=0,
        trades=0,
        fissions=0,
        migrants=0,
        quotes=0,
        focal_trades=0,
        focal_fissions=0,
        focal_migrants=0,
    )
    food, native_seconds = [], 0.0
    for _ in range(limit):
        actions = {}
        if peer >= 0 and str(peer) in response["observations"]:
            actions[peer] = market_peer(observation(response["observations"][str(peer)]))
        response = transport.step_world(actions)
        result, t = response["result"], response["result"]["Telemetry"]
        key = str(entry["focal"])
        events["movement"] += t["Moved"][key]
        for name, source in (
            ("trades", "Trades"),
            ("fissions", "Fissions"),
            ("migrants", "Migrants"),
        ):
            events[name] += t[source]
        events["focal_trades"] += t["SettledTrades"][key]
        events["focal_fissions"] += t["DaughterSettlements"][key]
        events["focal_migrants"] += t["Immigrated"][key] + t["Emigrated"][key]
        food.append(t["FoodMet"][key])
        native_seconds += response["nativeStepSeconds"]
        response["observations"] = result["Observations"]
        if result["Terminated"] or result["Truncated"]:
            break
    transport.verify_replay()
    row = dict(
        id=entry["id"],
        family=entry["family"],
        initial_population=entry["population"],
        final_population=transport.population,
        survived=not result["Terminated"],
        welfare=sum(x["raw_welfare"] for x in transport.series) / entry["population"],
        food_met=sum(food) / limit,
        events=dict(events, quotes=None),
        fingerprint=result["Fingerprint"],
        native_seconds=native_seconds,
        native_policy=initial["policy"],
        economic_allocation=initial["economicAllocation"],
        peer_mode=initial["peerMode"],
    )
    return enrich(row, transport.series, limit * 20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    assert cfg["maximum_steps"] == 2700 and cfg["maximum_wall_seconds"] == 600
    assert cfg["maximum_replay_steps"] == 2700 and cfg["maximum_native_resolves"] == 5400
    for pin in cfg["inputs"].values():
        assert sha(Path(pin["path"])) == pin["sha256"]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    bank = Path(cfg["inputs"]["bank"]["path"]).parent
    entries = [
        e
        for e in json.loads((bank / "manifest.json").read_text())["entries"]
        if e["partition"] == "development"
    ]
    assert len(entries) == 16
    started = time.monotonic()
    old_handler = signal.signal(
        signal.SIGALRM,
        lambda *_: (_ for _ in ()).throw(TimeoutError("Heuristic benchmark deadline")),
    )
    signal.alarm(600)
    result = dict(
        version="autonomous-heuristic-welfare-v1",
        config=cfg,
        config_sha256=sha(args.config),
        groups=[],
        trajectory_steps=0,
        training_updates=0,
    )
    heuristic = HeuristicTransport(
        [cfg["dotnet"], cfg["inputs"]["heuristic_bridge"]["path"]], deadline=started + 600
    )
    learned = RecordedWelfareTransport(
        [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]], deadline=started + 600
    )
    try:
        for mode in ("native", "public-market"):
            for limit in (8, 25):
                rows = [heuristic_episode(heuristic, bank, e, limit, mode) for e in entries]
                result["trajectory_steps"] += sum(len(r["intervals"]) for r in rows)
                group = dict(
                    policy=f"heuristic-{mode}",
                    limit=limit,
                    aggregate=aggregate(rows),
                    families={
                        f: aggregate([r for r in rows if r["family"] == f])
                        for f in sorted({e["family"] for e in entries})
                    },
                    episodes=rows,
                )
                result["groups"].append(group)
                if limit == 25:
                    prefix_group = next(
                        g
                        for g in result["groups"]
                        if g["policy"] == group["policy"] and g["limit"] == 8
                    )
                    for long, short in zip(rows, prefix_group["episodes"], strict=True):
                        assert long["id"] == short["id"]
                        assert long["intervals"][:8] == short["intervals"]
                        assert long["intervals"][:8][-1]["fingerprint"] == short["fingerprint"]
                print(
                    json.dumps(
                        {k: v for k, v in group.items() if k not in {"episodes", "families"}}
                    ),
                    flush=True,
                )
        prior = json.loads(Path(cfg["inputs"]["old_welfare"]["path"]).read_text())
        for seed in (29, 43):
            checkpoint = Path(cfg["inputs"][str(seed)]["path"])
            state = torch.load(checkpoint, weights_only=False)
            trainer = AutonomousTrainerV3(
                state["config"], seed, state["provenance"], state["width"]
            )
            trainer.restore(checkpoint)
            rows = []
            for entry in entries:
                for sampler in (9101, 9102):
                    _, row = episode(
                        trainer.model,
                        learned,
                        bank,
                        entry,
                        torch.Generator().manual_seed(sampler),
                        dict(state["config"], episode_limit=25),
                        verify=True,
                    )
                    enrich(row, learned.series, 500)
                    row["sampler_seed"] = sampler
                    rows.append(row)
                    result["trajectory_steps"] += len(learned.series)
                    old = next(
                        r for r in prior["reports"] if r["seed"] == seed and r["boundary"] == 24576
                    )
                    original = next(
                        r
                        for r in old["episodes"]
                        if r["id"] == entry["id"] and r["sampler_seed"] == sampler
                    )
                    for frame, old_frame in zip(
                        learned.series[:8], original["intervals"], strict=True
                    ):
                        assert {k: frame[k] for k in old_frame} == old_frame
                    assert learned.series[7]["fingerprint"] == original["fingerprint"]
                    assert result["trajectory_steps"] <= 2700
            for limit in (8, 25):
                subset = []
                for row in rows:
                    series = row["intervals"][:limit]
                    last = series[-1]
                    subset.append(
                        enrich(
                            dict(
                                row,
                                welfare=sum(x["raw_welfare"] for x in series)
                                / row["initial_population"],
                                final_population=last["final_population"],
                                survived=last["final_population"] > 0,
                                food_met=sum(x["food_met"] for x in series) / limit,
                                events={
                                    k: sum(x["events"][k] for x in series)
                                    for k in series[0]["events"]
                                },
                                fingerprint=last["fingerprint"],
                            ),
                            series,
                            limit * 20,
                        )
                    )
                result["groups"].append(
                    dict(
                        policy=f"learned-seed-{seed}",
                        limit=limit,
                        aggregate=aggregate(subset),
                        families={
                            f: aggregate([r for r in subset if r["family"] == f])
                            for f in sorted({e["family"] for e in entries})
                        },
                        episodes=subset,
                    )
                )
            assert sha(checkpoint) == cfg["inputs"][str(seed)]["sha256"]
        result.update(
            complete=True,
            wall_seconds=time.monotonic() - started,
            limitation=(
                "Native heuristic trade protocol differs; shared market-peer arm controls "
                "peer policy but cannot equalize quote interface. Native repeats are "
                "deterministic, not independent sampler replications."
            ),
        )
        result["replay_steps"] = heuristic.replay_steps + learned.replay_steps
        result["native_resolves"] = result["trajectory_steps"] + result["replay_steps"]
        assert result["replay_steps"] <= 2700 and result["native_resolves"] <= 5400
        result["comparisons"] = []
        for mode in ("native", "public-market"):
            for limit in (8, 25):
                reference = next(
                    g
                    for g in result["groups"]
                    if g["policy"] == f"heuristic-{mode}" and g["limit"] == limit
                )
                for seed in (29, 43):
                    candidate = next(
                        g
                        for g in result["groups"]
                        if g["policy"] == f"learned-seed-{seed}" and g["limit"] == limit
                    )
                    gain = reference["aggregate"]["welfare"] / candidate["aggregate"]["welfare"] - 1
                    result["comparisons"].append(
                        dict(
                            peer_mode=mode,
                            limit=limit,
                            learned_seed=seed,
                            relative_gain=gain,
                            substantial=gain >= cfg["substantial_relative_gain"],
                            family_relative_gain={
                                f: reference["families"][f]["welfare"]
                                / candidate["families"][f]["welfare"]
                                - 1
                                for f in reference["families"]
                            },
                        )
                    )
        result["substantial_full_native_advantage"] = all(
            c["substantial"] for c in result["comparisons"] if c["peer_mode"] == "native"
        )
    except BaseException as error:
        result.update(complete=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        heuristic.close()
        learned.close()
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
