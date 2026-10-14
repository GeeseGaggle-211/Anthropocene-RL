"""Matched, bounded objective-specific calibration and retained PPO comparisons."""

import argparse
import copy
import json
import time
from functools import partial
from pathlib import Path

import torch
from run_autonomous_causal_gate_v7 import (
    combine_inputs,
    summarize_episodes,
    summary_without_tensors,
)
from run_autonomous_continuation_v4 import measured_update
from run_autonomous_learning_v10 import load_bank, restore, sha
from train_autonomous_opportunity_v9 import OpportunityCollector, pair_hash, reversal_with_adam

from anthropocene_rl.autonomous_v4.collection import episode_seed
from anthropocene_rl.autonomous_v5.causal_budget import BudgetLedger, CausalCollector
from anthropocene_rl.autonomous_v5.causal_diagnostics import snapshot_health
from anthropocene_rl.autonomous_v6.canary_recovery import persist_rollout, validate_probe
from anthropocene_rl.autonomous_v6.contracts import observation
from anthropocene_rl.autonomous_v6.learning_guards import endpoint_summary
from anthropocene_rl.autonomous_v6.objective_collection import (
    context,
    objective_episode,
    revalue_episode,
    targets,
)
from anthropocene_rl.autonomous_v6.objective_rollout_audit import save_rollout_audit
from anthropocene_rl.autonomous_v6.opportunity import market_peer
from anthropocene_rl.autonomous_v6.policy import inputs_with_mode
from anthropocene_rl.autonomous_v6.readiness import (
    fit_critic,
    metrics,
    protected_calibration,
    ready,
)
from anthropocene_rl.autonomous_v6.reward_objectives import EpisodeObjectiveState, Objective
from anthropocene_rl.training.autonomous_ppo_v8 import AutonomousTrainerV8


class ObjectiveCollector(CausalCollector):
    episode_runner = staticmethod(partial(objective_episode, peer_policy=market_peer))


def fixed_rollout(collector, trainer, bank, pool, seed, count):
    """Equal rows/update calls per arm despite different terminal episode lengths."""
    rows, episodes = [], []
    while len(rows) < count:
        batch, eps = collector.rollout(trainer, bank, pool, seed, count - len(rows))
        if not batch:
            raise ValueError("Empty physical rollout")
        rows.extend(batch)
        episodes.extend(eps)
    if len(rows) != count:
        raise ValueError("Physical collector exceeded the fixed-row PPO boundary")
    return rows, episodes


def fork(parent, pin, objective, config_sha):
    config = dict(
        parent.config,
        model_version="autonomous-ppo-model-v8",
        training_objective=objective.wire(),
        critic_objective_loss="raw-objective-units-fixed-parent-scale",
    )
    provenance = dict(
        parent.provenance,
        objective_parent_sha256=pin["sha256"],
        objective_calibration_config_sha256=config_sha,
    )
    trainer = AutonomousTrainerV8(config, 0, provenance, parent.model.width)
    trainer.import_v7(Path(pin["path"]), expected_sha256=pin["sha256"])
    return trainer


@torch.no_grad()
def fit_objective_intercept(trainer, fitting):
    protected = protected_calibration(trainer)
    x, y = fitting
    values = trainer.model.value(x)
    derivative = trainer.model.value_scale * x["ValueFactor"]
    delta = ((y - values) * derivative).sum() / derivative.square().sum()
    trainer.model.critic[1].bias.add_(delta)
    if protected_calibration(trainer) != protected:
        raise ValueError("Objective intercept changed protected state")
    return dict(
        normalized_head_delta=float(delta),
        fitting_rows=len(y),
        raw_mean_applied_delta=float((trainer.model.value(x) - values).mean()),
    )


@torch.no_grad()
def strata(trainer, validation):
    x, y = validation
    prediction = trainer.model.value(x)
    p0 = x["InitialPopulation"]
    groups = {"below100": p0 < 100, "100to249": (p0 >= 100) & (p0 < 250), "250plus": p0 >= 250}
    result = {}
    for name, mask in groups.items():
        if not mask.any():
            result[name] = dict(rows=0)
            continue
        error, target = prediction[mask] - y[mask], y[mask]
        variance = float(target.var(unbiased=False))
        result[name] = dict(
            rows=int(mask.sum()),
            mse=float(error.square().mean()),
            mean_value_bias=float(error.mean()),
            return_variance=variance,
            explained_variance=1 - float(error.var(unbiased=False)) / variance
            if variance > 1e-8
            else None,
            mse_over_return_variance=float(error.square().mean()) / variance
            if variance > 1e-8
            else None,
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads(args.config.read_text())
    for pin in cfg["inputs"].values():
        if sha(pin["path"]) != pin["sha256"]:
            raise ValueError("Changed frozen pin: " + pin["path"])
    if cfg["arms"] != ["original", "midpoint", "superlinear"] or cfg["seeds"] != [101, 211, 307]:
        raise ValueError("Frozen matched objectives/seeds required")
    expected_caps = {
        "calibrate": dict(calibration=2250, calibration_replay=2250),
        "train": dict(training=9216, evaluation=2700, evaluation_replay=2700),
    }
    if cfg["stage"] not in expected_caps or cfg["caps"] != expected_caps[cfg["stage"]]:
        raise ValueError("Undeclared calibration/training allowance")
    if cfg["stage"] == "train" and cfg["transitions_per_cell"] != 1024:
        raise ValueError("First reward-comparison gate is exactly1024 transitions per cell")
    prior = json.loads(Path(cfg["inputs"]["prior_budget"]["path"]).read_text())
    envelope = json.loads(Path(cfg["inputs"]["envelope"]["path"]).read_text())
    if prior["total_resolves"] + sum(cfg["caps"].values()) > envelope["total_resolve_ceiling"]:
        raise ValueError("Prospective V11 resolve envelope exhausted")
    remaining = envelope["total_active_second_ceiling"] - prior["cumulative_wall_seconds"]
    if remaining <= 0:
        raise TimeoutError("Prospective V11 elapsed envelope exhausted")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    ledger = BudgetLedger(
        args.output / "budget.json",
        cfg["caps"],
        prior_resolves=prior["total_resolves"],
        prior_wall_seconds=prior["cumulative_wall_seconds"],
        deadline=time.monotonic() + remaining,
        native_version=6,
    )
    bank, entries = load_bank(cfg["inputs"]["bank"])
    admission = json.loads(Path(cfg["inputs"]["admission"]["path"]).read_text())
    if admission["status"] != "PASS":
        raise ValueError("Physical admission not ready")
    selected = admission["selected"]
    fitting_ids, validation_ids = cfg["fitting_entries"], cfg["validation_entries"]
    if (
        len(fitting_ids) != 23
        or fitting_ids != selected["ppo"]
        or len(validation_ids) != 6
        or len(set(validation_ids)) != 6
        or {entries[i]["family"] for i in validation_ids}
        != {"survival", "trade", "migration", "fission", "husbandry", "fishing"}
        or any(entries[i]["world_seed"] != 886006 for i in validation_ids)
        or any(
            entries[i]["world_seed"] not in [886001, 886002, 886003, 886004] for i in fitting_ids
        )
        or any(entries[i]["partition"] != "train" for i in fitting_ids + validation_ids)
        or any(entries[i]["world_seed"] not in [886007, 886008] for i in selected["development"])
    ):
        raise ValueError("Actual TRAIN/validation/DEV cohort leakage or coverage")
    result = dict(
        version=cfg["version"],
        stage=cfg["stage"],
        status="RUNNING",
        config_sha256=sha(args.config),
        calibrations=[],
        cells=[],
    )

    def save():
        ledger.save()
        result["budget"] = json.loads(ledger.path.read_text())
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    try:
        if cfg["stage"] == "calibrate":
            parent_gate = json.loads(Path(cfg["inputs"]["parent_gate"]["path"]).read_text())
            with OpportunityCollector(
                [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]],
                ledger=ledger,
                workers=4,
                deadline=ledger.deadline,
            ) as collector:
                ledger.phase = "calibration"
                for seed in cfg["seeds"]:
                    pin = cfg["inputs"][f"parent_seed{seed}"]
                    parent = restore(pin)
                    cell = next(c for c in parent_gate["cells"] if c["seed"] == seed)
                    if (
                        cell["status"] != "COMPLETE"
                        or cell["durable_transitions"] != 4096
                        or cell["checkpoint_sha256"] != pin["sha256"]
                        or cell["checkpoint_counters"]
                        != dict(
                            episodes=parent.episodes,
                            transitions=parent.transitions,
                            updates=parent.updates,
                        )
                    ):
                        raise ValueError("Exact current4096 parent/counters required")
                    protected = pair_hash(parent)
                    jobs = [
                        dict(
                            entry=entries[i], sampler_seed=episode_seed(seed + 410000, j), limit=25
                        )
                        for j, i in enumerate(fitting_ids + validation_ids)
                    ]
                    rows, episodes = collector.collect(
                        parent.model, bank, jobs, parent.config, collect=True, verify=True
                    )
                    if pair_hash(parent) != protected:
                        raise ValueError("Shared native calibration changed original parent")
                    physical = args.output / f"physical-seed{seed}.pt"
                    torch.save(dict(rows=rows, episodes=episodes), physical)
                    for kind in cfg["arms"]:
                        objective = Objective(kind)
                        trainer = fork(parent, pin, objective, sha(args.config))
                        trainer.deadline = ledger.deadline
                        protected = protected_calibration(trainer)
                        objective_rows, objective_episodes, targets_by_name, fit_mask = (
                            [],
                            [],
                            {},
                            [],
                        )
                        offset = 0
                        for index, e in enumerate(copy.deepcopy(episodes)):
                            native_rows = copy.deepcopy(rows[offset : offset + e["steps"]])
                            offset += e["steps"]
                            state = EpisodeObjectiveState(entries[e["id"]]["population"], objective)
                            native_rows, e = revalue_episode(
                                trainer.model, native_rows, e, state, trainer.config
                            )
                            objective_rows.extend(native_rows)
                            objective_episodes.append(e)
                            y = targets(native_rows, e["tail_value"], objective.horizon)
                            for key, values in y.items():
                                targets_by_name.setdefault(key, []).extend(values)
                            fit_mask.extend([index < len(fitting_ids)] * len(native_rows))
                        x = combine_inputs(objective_rows)
                        y = torch.tensor(targets_by_name["target"], dtype=torch.float32)
                        mask = torch.tensor(fit_mask)
                        fitting = ({k: v[mask] for k, v in x.items()}, y[mask])
                        validation = ({k: v[~mask] for k, v in x.items()}, y[~mask])
                        cache = args.output / f"targets-{kind}-seed{seed}.pt"
                        torch.save(
                            dict(
                                rows=objective_rows,
                                targets=targets_by_name,
                                fitting=fitting,
                                validation=validation,
                                fit_mask=mask,
                                episodes=[summary_without_tensors(e) for e in objective_episodes],
                            ),
                            cache,
                        )
                        entry = metrics(trainer, fitting, validation)
                        intercept_before = fit_objective_intercept(trainer, fitting)
                        fitted = fit_critic(
                            trainer,
                            fitting,
                            validation,
                            original_mse=entry["mse"],
                            max_epochs=cfg["max_calibration_epochs"],
                            seed=seed + 411000,
                            deadline=ledger.deadline,
                        )
                        intercept_after = fit_objective_intercept(trainer, fitting)
                        final = metrics(trainer, fitting, validation)
                        if protected_calibration(trainer) != protected:
                            raise ValueError("Objective calibration changed actor/Adam/RNG/scale")
                        path = args.output / f"calibrated-{kind}-seed{seed}.pt"
                        trainer.checkpoint(path)
                        result["calibrations"].append(
                            dict(
                                seed=seed,
                                kind=kind,
                                passed=ready(final, entry["mse"]),
                                entry=entry,
                                final=final,
                                fit=fitted,
                                strata=strata(trainer, validation),
                                intercept_before=intercept_before,
                                intercept_after=intercept_after,
                                checkpoint=str(path),
                                checkpoint_sha256=sha(path),
                                checkpoint_counters=dict(
                                    episodes=trainer.episodes,
                                    transitions=trainer.transitions,
                                    updates=trainer.updates,
                                ),
                                physical_cache=str(physical),
                                physical_sha256=sha(physical),
                                objective_cache=str(cache),
                                objective_cache_sha256=sha(cache),
                                actor_rng_adam_scale_protected=True,
                            )
                        )
                        save()
            result["status"] = (
                "READY"
                if all(c["passed"] for c in result["calibrations"])
                else "REVIEW_REQUIRED_CRITIC"
            )
        elif cfg["stage"] == "train":
            gate = json.loads(Path(cfg["inputs"]["calibration_gate"]["path"]).read_text())
            if (
                gate["status"] != "READY"
                or len(gate["calibrations"]) != 9
                or not all(c["passed"] for c in gate["calibrations"])
            ):
                raise ValueError("All-nine objective-specific READY critics required")
            probe = inputs_with_mode(
                [
                    observation(w)
                    for w in json.loads(Path(cfg["inputs"]["adaptation_probe"]["path"]).read_text())
                ]
            )
            validate_probe(probe)
            with ObjectiveCollector(
                [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]],
                ledger=ledger,
                workers=4,
                deadline=ledger.deadline,
            ) as collector:
                for seed in cfg["seeds"]:
                    for kind in cfg["arms"]:
                        pin = cfg["inputs"][f"parent_{kind}_seed{seed}"]
                        state = torch.load(pin["path"], map_location="cpu", weights_only=False)
                        trainer = AutonomousTrainerV8(
                            state["config"], 0, state["provenance"], state["width"]
                        )
                        trainer.restore(pin["path"])
                        if Objective.from_wire(trainer.config["training_objective"]).kind != kind:
                            raise ValueError("Checkpoint reward arm differs")
                        parent = next(
                            c
                            for c in gate["calibrations"]
                            if c["seed"] == seed and c["kind"] == kind
                        )
                        if parent["checkpoint_sha256"] != pin["sha256"] or parent[
                            "checkpoint_counters"
                        ] != dict(
                            episodes=trainer.episodes,
                            transitions=trainer.transitions,
                            updates=trainer.updates,
                        ):
                            raise ValueError("Wrong matched calibrated parent/counters")
                        trainer.deadline = ledger.deadline
                        objective = Objective.from_wire(trainer.config["training_objective"])
                        adapted = context(probe, EpisodeObjectiveState(100, objective), 100)
                        cell = dict(
                            id=f"{kind}-seed{seed}",
                            seed=seed,
                            kind=kind,
                            status="RUNNING",
                            transitions=0,
                            updates=[],
                            health_before=snapshot_health(trainer.model, adapted, "full"),
                        )
                        result["cells"].append(cell)
                        start_counter = trainer.transitions
                        episodes = []
                        while trainer.transitions - start_counter < cfg["transitions_per_cell"]:
                            ledger.phase = "training"
                            remaining = min(
                                224,
                                cfg["transitions_per_cell"] - (trainer.transitions - start_counter),
                            )
                            trainer.set_rollout_mix()
                            rows, eps = fixed_rollout(
                                collector,
                                trainer,
                                bank,
                                [entries[i] for i in selected["ppo"]],
                                seed + 420000,
                                remaining,
                            )
                            if not rows:
                                raise ValueError("Empty physical rollout")
                            update = measured_update(trainer, rows)
                            episodes.extend(eps)
                            cell["transitions"] = trainer.transitions - start_counter
                            cell["updates"].append(update)
                            checkpoint = persist_rollout(trainer, args.output, cell, episodes)
                            cell["checkpoint_counters"] = dict(
                                episodes=trainer.episodes,
                                transitions=trainer.transitions,
                                updates=trainer.updates,
                            )
                            save()
                            protected = pair_hash(trainer)
                            audit = save_rollout_audit(
                                trainer,
                                rows,
                                args.output / f"{cell['id']}-rollout{trainer.updates}.pt",
                                rollout_provenance=dict(
                                    config_sha256=sha(args.config),
                                    checkpoint_sha256=sha(checkpoint),
                                    episode_ids=[e["id"] for e in eps],
                                    sampler_scheme="episode_seed(seed+420000,absolute_episode)",
                                ),
                            )
                            if pair_hash(trainer) != protected:
                                raise ValueError("Objective rollout audit changed retained state")
                            cell.setdefault("rollout_audits", []).append(audit)
                            save()
                        cell["training_summary"] = summarize_episodes(episodes)
                        cell["training_endpoints"] = endpoint_summary(episodes)
                        protected = pair_hash(trainer)
                        cell["reversal"] = reversal_with_adam(
                            trainer, adapted, steps=64, deadline=ledger.deadline
                        )
                        cell["reversal"]["full_signed_switch"] = all(
                            p["passed"] and p["after"][1] > 0 for p in cell["reversal"]["phases"]
                        )
                        cell["health_after"] = snapshot_health(trainer.model, adapted, "full")
                        if pair_hash(trainer) != protected:
                            raise ValueError("Diagnostic changed retained objective model")
                        ledger.phase = "evaluation"
                        jobs = [
                            dict(
                                entry=entries[i],
                                sampler_seed=episode_seed(seed + 290000, j),
                                limit=25,
                            )
                            for j, i in enumerate(selected["development"])
                        ]
                        _, eps = collector.collect(
                            trainer.model, bank, jobs, trainer.config, collect=False, verify=True
                        )
                        cell["evaluation"] = summarize_episodes(eps)
                        cell["evaluation_endpoints"] = endpoint_summary(eps)
                        cell["evaluation_episodes"] = [summary_without_tensors(e) for e in eps]
                        cell["status"] = "COMPLETE"
                        save()
            result["status"] = "STAGED_REVIEW_REQUIRED"
        else:
            raise ValueError("Unknown bounded stage")
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        print(json.dumps({k: result[k] for k in ["stage", "status", "budget"]}))


if __name__ == "__main__":
    main()
