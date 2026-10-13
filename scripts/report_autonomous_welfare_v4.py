"""Reconstruct population/QoL/welfare from unchanged evaluation trajectories; no updates."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from anthropocene_rl.autonomous_v4.collection import DeadlineTransport
from anthropocene_rl.autonomous_v4.training import episode
from anthropocene_rl.training.autonomous_ppo_v3 import AutonomousTrainerV3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WelfareTransport(DeadlineTransport):
    def reset_world(self, snapshot, controlled, limit):
        result = super().reset_world(snapshot, controlled, limit)
        self.focal = str(controlled[0])
        self.population = int(result["observations"][self.focal]["Context"][0])
        self.series = []
        return result

    def step_world(self, actions):
        response = super().step_world(actions)
        result = response["result"]
        telemetry = result["Telemetry"]
        quality = min(1, max(0, telemetry["Quality"][self.focal]))
        years = result["ElapsedYears"]
        raw = result["Rewards"][self.focal]
        assert math.isclose(raw, self.population * quality * years, rel_tol=1e-12, abs_tol=1e-9)
        self.series.append(
            dict(
                prior_population=self.population,
                final_population=telemetry["Population"][self.focal],
                qol=quality,
                years=years,
                raw_welfare=raw,
                immigrants=telemetry["Immigrated"][self.focal],
                emigrants=telemetry["Emigrated"][self.focal],
                daughter_settlements=telemetry["DaughterSettlements"][self.focal],
            )
        )
        self.population = telemetry["Population"][self.focal]
        return response


def aggregate(rows):
    families = sorted({r["family"] for r in rows})
    weights = [1 / len(families) / sum(x["family"] == r["family"] for x in rows) for r in rows]

    def mean(key):
        return sum(w * r[key] for w, r in zip(weights, rows, strict=True))

    fields = [
        "initial_population",
        "final_population",
        "elapsed_years",
        "time_mean_population",
        "time_mean_qol",
        "final_qol",
        "person_years",
        "raw_welfare",
        "welfare",
    ]
    result = {key: mean(key) for key in fields}
    # This pooled QoL gives an exact decomposition of the equally weighted raw welfare.
    result["person_year_weighted_qol"] = result["raw_welfare"] / result["person_years"]
    result["episodes"] = len(rows)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    campaign = json.loads((args.run / "summary.json").read_text())
    assert sorted(r["seed"] for r in campaign["seeds"]) == [29, 43]
    assert sha(args.bridge) == campaign["config"]["bridge_sha256"]
    assert sha(args.bank / "manifest.json") == campaign["config"]["bank_sha256"]
    entries = [
        r
        for r in json.loads((args.bank / "manifest.json").read_text())["entries"]
        if r["partition"] == "development"
    ]
    reports = []
    started = time.monotonic()
    transport = WelfareTransport(
        ["/workspace/Anthropocene/.tools/dotnet/dotnet", str(args.bridge.resolve())],
        deadline=started + 600,
    )
    try:
        for seed_result in campaign["seeds"]:
            seed = seed_result["seed"]
            for boundary in [0, seed_result["additional_transitions"]]:
                checkpoint = args.run / f"seed-{seed}-step-{boundary}.pt"
                digest = sha(checkpoint)
                state = torch.load(checkpoint, weights_only=False)
                trainer = AutonomousTrainerV3(
                    state["config"], seed, state["provenance"], state["width"]
                )
                trainer.restore(checkpoint)
                expected = json.loads(
                    (args.run / f"seed-{seed}-step-{boundary}-evaluation.json").read_text()
                )
                rows = []
                for entry in entries:
                    for sampler in state["config"]["evaluation_rng_seeds"]:
                        _, row = episode(
                            trainer.model,
                            transport,
                            args.bank,
                            entry,
                            torch.Generator().manual_seed(sampler),
                            state["config"],
                            verify=True,
                        )
                        previous = expected[len(rows)]
                        assert row["fingerprint"] == previous["fingerprint"]
                        assert row["welfare"] == previous["welfare"]
                        assert row["final_population"] == previous["final_population"]
                        series = transport.series
                        years = sum(r["years"] for r in series)
                        person_years = sum(r["prior_population"] * r["years"] for r in series)
                        raw = sum(r["raw_welfare"] for r in series)
                        row.update(
                            sampler_seed=sampler,
                            intervals=series,
                            raw_welfare=raw,
                            person_years=person_years,
                            time_mean_population=person_years / years,
                            time_mean_qol=sum(r["qol"] * r["years"] for r in series) / years,
                            person_year_weighted_qol=raw / person_years,
                            final_qol=series[-1]["qol"],
                        )
                        rows.append(row)
                assert sha(checkpoint) == digest
                reports.append(
                    dict(
                        seed=seed,
                        boundary=boundary,
                        checkpoint_sha256=digest,
                        aggregate=aggregate(rows),
                        families={
                            f: aggregate([r for r in rows if r["family"] == f])
                            for f in sorted({r["family"] for r in rows})
                        },
                        episodes=rows,
                        exact_original_evaluations=True,
                    )
                )
                print(json.dumps(dict(seed=seed, boundary=boundary, **aggregate(rows))), flush=True)
    finally:
        transport.close()
    args.output.write_text(
        json.dumps(
            dict(
                version="autonomous-welfare-breakdown-v1",
                script_sha256=sha(Path(__file__)),
                campaign_sha256=sha(args.run / "summary.json"),
                wall_seconds=time.monotonic() - started,
                native_training_updates=0,
                reports=reports,
                definition=(
                    "sum(prior_population * end_interval_QoL * years); displayed score "
                    "divides each episode by its initial population, then averages families equally"
                ),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
