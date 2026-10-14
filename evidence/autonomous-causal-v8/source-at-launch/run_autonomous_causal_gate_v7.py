"""Frozen-source, cumulative-budget critic calibration and causal PPO comparison."""

import argparse
import copy
import hashlib
import json
import math
import signal
import time
from pathlib import Path

import numpy as np
import torch
from run_autonomous_continuation_v4 import measured_update

from anthropocene_rl.autonomous_v4.collection import episode_seed
from anthropocene_rl.autonomous_v5.causal import fit_critic, n_step_targets, protected_state
from anthropocene_rl.autonomous_v5.causal_budget import BudgetLedger, CausalCollector
from anthropocene_rl.autonomous_v5.causal_diagnostics import (
    reversal_probe,
    signature,
    signature_change,
    snapshot_health,
)
from anthropocene_rl.autonomous_v6.contracts import observation
from anthropocene_rl.autonomous_v6.policy import inputs_with_mode as inputs
from anthropocene_rl.training.autonomous_ppo_v6 import AutonomousTrainerV6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fork_parent(pin, provenance):
    path = Path(pin["path"])
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(
        state["config"],
        version="autonomous-ppo-direct-control-v6",
        model_version="autonomous-ppo-model-v6",
        observation_version=6,
    )
    trainer = AutonomousTrainerV6(config, 0, provenance, state["width"])
    trainer.warm_start(path, expected_sha256=pin["sha256"])
    if not all(
        trainer.optimizer.state.get(p)
        for p in trainer.model.actor.parameters()
        if p is not trainer.model.actor[0].history_weight
    ):
        raise ValueError("Parent actor Adam is not populated")
    return trainer


def summary_without_tensors(summary):
    summary = dict(summary)
    summary.pop("final_obs", None)
    return summary


def combine_inputs(rows):
    return {k: torch.cat([r["obs"][k] for r in rows]) for k in rows[0]["obs"]}


def summarize_episodes(episodes):
    traces = [r for e in episodes for r in e["trace"]]
    elapsed = sum(r["elapsed"] for r in traces)
    endpoints = []
    for i in range(3):
        values = [r["requested_controls"][i] for r in traces if r["control_mask"][i]]
        endpoints.append(
            dict(
                eligible=len(values),
                zero=sum(v == 0 for v in values),
                one=sum(v == 1 for v in values),
                mean=float(np.mean(values)) if values else None,
            )
        )
    return dict(
        episodes=len(episodes),
        physical_control_endpoints=endpoints,
        steps=len(traces),
        average_qol=sum(r["qol"] * r["elapsed"] for r in traces) / elapsed,
        average_population=sum(r["population_start"] * r["elapsed"] for r in traces) / elapsed,
        annual_welfare=sum(r["reward"] for r in traces) / elapsed,
        integrated_welfare=sum(r["reward"] for r in traces),
        final_population_mean=float(np.mean([e["final_population"] for e in episodes])),
        terminal_episodes=sum(e["terminal"] for e in episodes),
        events={k: sum(e["events"][k] for e in episodes) for k in episodes[0]["events"]},
        native_step_seconds_mean=float(np.mean([r["native_step_seconds"] for r in traces])),
        family_counts={
            f: sum(e["family"] == f for e in episodes)
            for f in sorted({e["family"] for e in episodes})
        },
        mean_requested_workforce=np.mean(
            [r["requested_workforce"] for r in traces], axis=0
        ).tolist(),
        mean_compiled_staffing=np.mean([r["staffing"] for r in traces], axis=0).tolist(),
        total_production=np.sum([r["production"] for r in traces], axis=0).tolist(),
        total_extraction=np.sum([r["extraction"] for r in traces], axis=0).tolist(),
    )


def validate_config(cfg, entries, stage):
    if (
        cfg["horizons"] != [100, 400]
        or cfg["feature_modes"] != ["full", "receipt_none"]
        or cfg["episode_limits"] != [8, 25]
        or cfg["seeds"] != [11, 23, 61]
        or cfg["transitions_per_cell"] != 400
        or cfg["workers"] != 4
        or cfg["torch_threads"] != 1
        or cfg["maximum_wall_seconds"] != 1800
    ):
        raise ValueError("Frozen causal matrix/resource envelope changed")
    if set(cfg["fit_worlds"]) & set(cfg["validation_worlds"]):
        raise ValueError("Calibration world leakage")
    if (
        len(set(cfg["fit_worlds"])) != 8
        or len(cfg["fit_worlds"]) != 8
        or len(set(cfg["validation_worlds"])) != 4
        or len(cfg["validation_worlds"]) != 4
    ):
        raise ValueError("Expected eight unique fit and four unique validation worlds")
    if set(cfg["ppo_worlds"]) & set(cfg["validation_worlds"]):
        raise ValueError("PPO uses critic validation worlds")
    for role in ["calibration_entries", "ppo_entries", "development_entries"]:
        if not cfg[role] or len(set(cfg[role])) != len(cfg[role]):
            raise ValueError("Empty/duplicate reset cohort")
        for id_ in cfg[role]:
            entry = entries[id_]
            expected = "development" if role == "development_entries" else "train"
            if entry["partition"] != expected or (
                not entry["neural_training_eligible"]
                and entry["family"] not in ("husbandry", "fishing")
            ):
                raise ValueError(f"Wrong source partition/admission: {id_}")
            if role == "ppo_entries" and (
                entry["world_seed"] not in cfg["ppo_worlds"]
                or entry["world_seed"] in cfg["validation_worlds"]
            ):
                raise ValueError("Actual PPO reset world violates fitting partition")
    calibration = [entries[i] for i in cfg["calibration_entries"]]
    if len(calibration) != 12 or {e["world_seed"] for e in calibration} != set(
        cfg["fit_worlds"] + cfg["validation_worlds"]
    ):
        raise ValueError("Expected eight fit and four validation worlds")
    if len(cfg["development_entries"]) != 2:
        raise ValueError("Frozen common DEV panel changed")
    if stage == "train" and "calibration_summary" not in cfg["inputs"]:
        raise ValueError("Training requires pinned calibration results")


def calibrate(cfg, collector, bank, entries, probe, output, result, save):
    collector.transports[0].ledger.phase = "calibration"
    provenance = dict(config_sha256=result["config_sha256"], experiment="causal-calibration-v8")
    for horizon in cfg["horizons"]:
        parent = fork_parent(cfg["inputs"][f"parent_h{horizon}"], provenance)
        parent.config.update(episode_limit=25, feature_mode="full", require_core_action_faces=False)
        before_collection = protected_state(parent)
        jobs = [
            dict(entry=entries[id_], sampler_seed=episode_seed(7700 + horizon, i), limit=25)
            for i, id_ in enumerate(cfg["calibration_entries"])
        ]
        rows, episodes = collector.collect(
            parent.model, bank, jobs, parent.config, collect=False, verify=True
        )
        if protected_state(parent) != before_collection:
            raise ValueError("Frozen-policy collection changed protected parent state")
        offset = 0
        targets, weights, zeros, doubles, fitting = [], [], [], [], []
        for e in episodes:
            trace = e["trace"]
            nstep = n_step_targets(
                [r["reward"] for r in trace],
                [r["elapsed"] for r in trace],
                [r["terminal"] for r in trace],
                e["tail_value"],
                horizon,
                parent.config["reward_scale"],
            )
            targets.extend(nstep["target"])
            weights.extend(nstep["tail_weight"])
            zeros.extend(nstep["zero_tail"])
            doubles.extend(nstep["doubled_tail"])
            fitting.extend([e["world_seed"] in cfg["fit_worlds"]] * len(trace))
            offset += len(trace)
        x, y = combine_inputs(rows), torch.tensor(targets, dtype=torch.float32)
        fit = torch.tensor(fitting, dtype=torch.bool)
        if not fit.any() or not (~fit).any():
            raise ValueError("Empty fitting/validation native target cohort")
        train = ({k: v[fit] for k, v in x.items()}, y[fit])
        validation = ({k: v[~fit] for k, v in x.items()}, y[~fit])
        torch.save(
            dict(
                obs=x,
                targets=y,
                fit=fit,
                zero_tail=zeros,
                doubled_tail=doubles,
                tail_weight=weights,
                episodes=[summary_without_tensors(e) for e in episodes],
            ),
            output / f"native-targets-h{horizon}.pt",
        )
        result["target_cohorts"].append(
            dict(
                horizon=horizon,
                fitting_rows=int(fit.sum()),
                validation_rows=int((~fit).sum()),
                protected_state=before_collection,
                native_summary=summarize_episodes(episodes),
                episodes=[summary_without_tensors(e) for e in episodes],
                tail_weight_mean=float(np.mean(weights)),
                tail_weight_at_reset=[
                    float(weights[sum(p["steps"] for p in episodes[:i])])
                    for i in range(len(episodes))
                ],
                target_mean=float(y.mean()),
                zero_tail_mean=float(np.mean(zeros)),
                doubled_tail_mean=float(np.mean(doubles)),
            )
        )
        save()
        for mode in cfg["feature_modes"]:
            trainer = copy.deepcopy(parent)
            trainer.config["feature_mode"] = mode
            report = fit_critic(
                trainer,
                train,
                validation,
                mode=mode,
                max_epochs=cfg["critic_max_epochs"],
                seed=711 + horizon,
            )
            checkpoint = output / f"calibrated-h{horizon}-{mode}.pt"
            trainer.checkpoint(checkpoint)
            result["calibrations"].append(
                dict(
                    horizon=horizon,
                    feature_mode=mode,
                    report=report,
                    checkpoint=str(checkpoint),
                    checkpoint_sha256=sha(checkpoint),
                    health=snapshot_health(trainer.model, probe, mode),
                )
            )
            save()
    result["passed"] = all(c["report"]["passed"] for c in result["calibrations"])
    result["status"] = "READY" if result["passed"] else "NO-GO_CRITIC"


def load_calibrated(pin):
    state = torch.load(pin["path"], map_location="cpu", weights_only=False)
    trainer = AutonomousTrainerV6(state["config"], 0, state["provenance"], state["width"])
    trainer.restore(Path(pin["path"]))
    return trainer


def train_matrix(cfg, collector, bank, entries, probe, output, result, save):
    calibration = json.loads(Path(cfg["inputs"]["calibration_summary"]["path"]).read_text())
    if not calibration.get("passed") or calibration["status"] != "READY":
        raise ValueError("Frozen critic calibration is not ready")
    ledger = collector.transports[0].ledger
    pool = [entries[id_] for id_ in cfg["ppo_entries"]]
    development = [entries[id_] for id_ in cfg["development_entries"]]
    for horizon in cfg["horizons"]:
        parent = load_calibrated(cfg["inputs"][f"calibrated_h{horizon}_full"])
        for seed in cfg["seeds"]:
            ledger.phase = "evaluation"
            jobs = [
                dict(entry=e, sampler_seed=episode_seed(seed + 90000, i), limit=25)
                for i, e in enumerate(development)
            ]
            _, baseline = collector.collect(
                parent.model,
                bank,
                jobs,
                dict(parent.config, episode_limit=25),
                collect=False,
                verify=True,
            )
            result["baselines"].append(
                dict(
                    horizon=horizon,
                    seed=seed,
                    summary=summarize_episodes(baseline),
                    episodes=[summary_without_tensors(e) for e in baseline],
                )
            )
            save()
            for mode in cfg["feature_modes"]:
                for limit in cfg["episode_limits"]:
                    trainer = load_calibrated(cfg["inputs"][f"calibrated_h{horizon}_{mode}"])
                    trainer.config.update(
                        episode_limit=limit,
                        feature_mode=mode,
                        require_core_action_faces=False,
                        episodes_per_rollout=math.ceil(200 / limit),
                    )
                    trainer.provenance = dict(
                        trainer.provenance,
                        causal_gate_sha256=result["config_sha256"],
                        continuation_seed=seed,
                        episode_limit=limit,
                        feature_mode=mode,
                        sampler_reseed=seed + 1000,
                        shuffle_reseed=seed + 2000,
                    )
                    trainer.sampler.manual_seed(seed + 1000)
                    trainer.shuffle.manual_seed(seed + 2000)
                    obs = {k: v.clone() for k, v in probe.items()}
                    if mode == "receipt_none":
                        obs["x"][:, 761:] = 0
                    opening_signature = signature(trainer.model, obs)
                    cell = dict(
                        id=f"h{horizon}-{mode}-limit{limit}-seed{seed}",
                        horizon=horizon,
                        feature_mode=mode,
                        episode_limit=limit,
                        seed=seed,
                        status="RUNNING",
                        transitions=0,
                        updates=[],
                        health_before=snapshot_health(trainer.model, probe, mode),
                    )
                    result["cells"].append(cell)
                    all_episodes = []
                    ledger.phase = "training"
                    while cell["transitions"] < cfg["transitions_per_cell"]:
                        remaining = min(200, cfg["transitions_per_cell"] - cell["transitions"])
                        rows, episodes = collector.rollout(trainer, bank, pool, seed, remaining)
                        if not rows:
                            raise ValueError("Empty native PPO rollout")
                        cell["transitions"] += len(rows)
                        all_episodes.extend(episodes)
                        cell["updates"].append(measured_update(trainer, rows))
                        save()
                    cell["training_summary"] = summarize_episodes(all_episodes)
                    (output / f"{cell['id']}-training-traces.json").write_text(
                        json.dumps([summary_without_tensors(e) for e in all_episodes]) + "\n"
                    )
                    cell["health_after"] = snapshot_health(trainer.model, probe, mode)
                    cell["distribution_change"] = signature_change(
                        opening_signature, signature(trainer.model, obs)
                    )
                    cell["reversal"] = reversal_probe(
                        trainer.model, obs, steps=cfg["reversal_steps"], lr=cfg["reversal_lr"]
                    )
                    checkpoint = output / f"{cell['id']}-step-{cell['transitions']}.pt"
                    trainer.checkpoint(checkpoint)
                    cell.update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))
                    ledger.phase = "evaluation"
                    _, final = collector.collect(
                        trainer.model,
                        bank,
                        jobs,
                        dict(trainer.config, episode_limit=25),
                        collect=False,
                        verify=True,
                    )
                    cell["evaluation"] = summarize_episodes(final)
                    cell["evaluation_episodes"] = [summary_without_tensors(e) for e in final]
                    cell["baseline_annual_welfare"] = result["baselines"][-1]["summary"][
                        "annual_welfare"
                    ]
                    cell["status"] = "COMPLETE"
                    save()
    result["status"] = "COMPLETE"
    result["scale_out"] = "NO-GO pending independent actual-result review and promotion gates"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["calibrate", "train"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads(args.config.read_text())
    for pin in cfg["inputs"].values():
        if sha(pin["path"]) != pin["sha256"]:
            raise ValueError(f"Frozen causal input changed: {pin['path']}")
    bank = Path(cfg["inputs"]["bank"]["path"]).parent
    entries = {e["id"]: e for e in json.loads((bank / "manifest.json").read_text())["entries"]}
    validate_config(cfg, entries, args.stage)
    prior = json.loads(Path(cfg["inputs"]["prior_stage"]["path"]).read_text())
    if args.stage == "calibrate":
        if prior["status"] != "PASS" or not prior["physical_growth_proved"]:
            raise ValueError("Head stage must complete before calibration")
        previous = json.loads(Path(cfg["inputs"]["prior_budget"]["path"]).read_text())
        prior_resolves, prior_wall = previous["total_resolves"], previous["cumulative_wall_seconds"]
        caps = dict(calibration=600, calibration_replay=600)
    else:
        if prior["status"] != "READY" or not prior["passed"]:
            raise ValueError("Critic stage is not ready")
        previous = json.loads(Path(cfg["inputs"]["prior_budget"]["path"]).read_text())
        prior_resolves = previous["total_resolves"]
        prior_wall = previous["cumulative_wall_seconds"]
        caps = dict(training=9600, evaluation=1500, evaluation_replay=1500)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    probe = inputs(
        [observation(o) for o in json.loads(Path(cfg["inputs"]["probe"]["path"]).read_text())]
    )
    args.output.mkdir(parents=True)
    started = time.monotonic()
    remaining_seconds = cfg["maximum_wall_seconds"] - prior_wall
    if remaining_seconds <= 0:
        raise TimeoutError("Prior causal stages exhausted the CPU envelope")
    deadline = started + remaining_seconds
    ledger = BudgetLedger(
        args.output / "budget.json",
        caps,
        prior_resolves=prior_resolves,
        prior_wall_seconds=prior_wall,
        deadline=deadline,
        native_version=6,
    )
    result = dict(
        version="autonomous-causal-gate-v8",
        stage=args.stage,
        status="RUNNING",
        config_sha256=sha(args.config),
        config=cfg,
        target_cohorts=[],
        calibrations=[],
        cells=[],
        baselines=[],
    )

    def save():
        result["wall_seconds"] = time.monotonic() - started
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    def expired(*_):
        raise TimeoutError("Cumulative causal campaign CPU deadline")

    previous_handler = signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, int(remaining_seconds)))
    save()
    try:
        with CausalCollector(
            [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]],
            ledger=ledger,
            workers=4,
            deadline=deadline,
            request_timeout=60,
        ) as collector:
            if args.stage == "calibrate":
                calibrate(cfg, collector, bank, entries, probe, args.output, result, save)
            else:
                train_matrix(cfg, collector, bank, entries, probe, args.output, result, save)
        save()
    except BaseException as error:
        result.update(status="FAILED", error=str(error))
        save()
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        with ledger.lock:
            ledger.save()
    print(json.dumps({k: result[k] for k in ["version", "stage", "status", "wall_seconds"]}))


if __name__ == "__main__":
    main()
