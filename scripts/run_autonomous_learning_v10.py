"""Fresh-world calibration and auditable low-LR continuation, staged at1024/4096."""

import argparse
import json
import time
from pathlib import Path

import torch
from run_autonomous_causal_gate_v7 import (
    combine_inputs,
    summarize_episodes,
    summary_without_tensors,
)
from run_autonomous_continuation_v4 import measured_update
from train_autonomous_opportunity_v9 import OpportunityCollector, pair_hash, reversal_with_adam

from anthropocene_rl.autonomous_v4.collection import episode_seed
from anthropocene_rl.autonomous_v5.causal import n_step_targets, protected_state
from anthropocene_rl.autonomous_v5.causal_budget import BudgetLedger
from anthropocene_rl.autonomous_v5.causal_diagnostics import snapshot_health
from anthropocene_rl.autonomous_v6.canary_recovery import persist_rollout, validate_probe
from anthropocene_rl.autonomous_v6.contracts import observation
from anthropocene_rl.autonomous_v6.learning_guards import (
    endpoint_summary,
    validate_cohorts,
    validate_parent,
)
from anthropocene_rl.autonomous_v6.policy import inputs_with_mode
from anthropocene_rl.autonomous_v6.readiness import fit_critic, fit_intercept, metrics, ready
from anthropocene_rl.autonomous_v6.rollout_audit import save_rollout_audit
from anthropocene_rl.training.autonomous_ppo_v7 import AutonomousTrainerV7


def sha(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore(pin):
    state = torch.load(pin["path"], map_location="cpu", weights_only=False)
    trainer = AutonomousTrainerV7(state["config"], 0, state["provenance"], state["width"])
    trainer.restore(pin["path"])
    if trainer.config["actor_lr"] != 3e-5 or trainer.config["horizon_years"] != 400:
        raise ValueError("Wrong fixed low-LR H400 parent")
    return trainer


def load_bank(pin):
    path = Path(pin["path"])
    return path.parent, {e["id"]: e for e in json.loads(path.read_text())["entries"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["calibrate", "train"], required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads(args.config.read_text())
    if cfg["stage"] != args.stage:
        raise ValueError("Config/stage mismatch")
    for pin in cfg["inputs"].values():
        if sha(pin["path"]) != pin["sha256"]:
            raise ValueError("Changed frozen pin: " + pin["path"])
    prior = json.loads(Path(cfg["inputs"]["prior_budget"]["path"]).read_text())
    if prior["total_resolves"] + sum(cfg["caps"].values()) > 27951 + 24000:
        raise ValueError("V10 cumulative resolve ceiling")
    remaining_time = 912.8917767360253 + 900 - prior["cumulative_wall_seconds"]
    if remaining_time <= 0:
        raise TimeoutError("V10 cumulative elapsed ceiling")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    ledger = BudgetLedger(
        args.output / "budget.json",
        cfg["caps"],
        prior_resolves=prior["total_resolves"],
        prior_wall_seconds=prior["cumulative_wall_seconds"],
        deadline=time.monotonic() + remaining_time,
        native_version=6,
    )
    bank, entries = load_bank(cfg["inputs"]["bank"])
    old_bank, old_entries = load_bank(cfg["inputs"]["old_bank"])
    selection = json.loads(Path(cfg["inputs"]["selection"]["path"]).read_text())
    admission = json.loads(Path(cfg["inputs"]["admission"]["path"]).read_text())
    selected = admission["selected"]
    if admission["status"] != "PASS":
        raise ValueError("Physical admission failed")
    validate_cohorts(
        selection, selected, entries, old_entries, cfg["old_fit_entries"], cfg["old_fit_worlds"]
    )
    worlds = [set(selection[k]) for k in ["fit_worlds", "validation_worlds", "development_worlds"]]
    if any(worlds[i] & worlds[j] for i in range(3) for j in range(i)):
        raise ValueError("World leakage")
    for role, expected in [("ppo", worlds[0]), ("development", worlds[2])]:
        if any(entries[i]["world_seed"] not in expected for i in selected[role]):
            raise ValueError("Actual reset partition leakage")
    adaptation = inputs_with_mode(
        [
            observation(o)
            for o in json.loads(Path(cfg["inputs"]["adaptation_probe"]["path"]).read_text())
        ]
    )
    probe_check = validate_probe(adaptation)
    result = dict(
        version="autonomous-learning-v10",
        stage=args.stage,
        status="RUNNING",
        config_sha256=sha(args.config),
        calibrations=[],
        baselines=[],
        cells=[],
        probe=probe_check,
    )

    def save():
        ledger.save()
        result["budget"] = json.loads(ledger.path.read_text())
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    try:
        with OpportunityCollector(
            [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]],
            ledger=ledger,
            workers=4,
            deadline=ledger.deadline,
        ) as collector:
            for seed in [101, 211, 307]:
                trainer = restore(cfg["inputs"][f"parent_seed{seed}"])
                trainer.deadline = ledger.deadline
                if args.stage == "train":
                    parent_gate = json.loads(Path(cfg["inputs"]["parent_gate"]["path"]).read_text())
                    validate_parent(
                        parent_gate,
                        cfg["inputs"][f"parent_seed{seed}"],
                        trainer,
                        seed,
                        cfg.get("prior_additional_transitions", 0),
                    )
                if args.stage == "calibrate":
                    protected = protected_state(trainer)
                    buffers = {
                        k: trainer.model.state_dict()[k].clone()
                        for k in ["value_offset", "value_scale"]
                    }
                    cal = selected["calibration"]
                    if len(cal) != 24 or len(cfg["old_fit_entries"]) != 12:
                        raise ValueError("Frozen36 episode calibration count")
                    if any(
                        old_entries[i]["world_seed"] not in cfg["old_fit_worlds"]
                        for i in cfg["old_fit_entries"]
                    ):
                        raise ValueError("Old fitting leakage")
                    ledger.phase = "calibration"
                    all_rows, episodes = [], []
                    for source_bank, source_entries, ids, offset in [
                        (old_bank, old_entries, cfg["old_fit_entries"], 0),
                        (bank, entries, cal, 12),
                    ]:
                        jobs = [
                            dict(
                                entry=source_entries[i],
                                sampler_seed=episode_seed(seed + 210000, j + offset),
                                limit=25,
                            )
                            for j, i in enumerate(ids)
                        ]
                        rows, eps = collector.collect(
                            trainer.model,
                            source_bank,
                            jobs,
                            trainer.config,
                            collect=True,
                            verify=True,
                        )
                        all_rows.extend(rows)
                        episodes.extend(eps)
                    if protected_state(trainer) != protected:
                        raise ValueError("Frozen actor collection changed protected state")
                    targets, zeros, doubles, tail_weights, fit_mask = [], [], [], [], []
                    for index, e in enumerate(episodes):
                        trace = e["trace"]
                        y = n_step_targets(
                            [r["reward"] for r in trace],
                            [r["elapsed"] for r in trace],
                            [r["terminal"] for r in trace],
                            e["tail_value"],
                            400,
                            120,
                        )
                        targets.extend(y["target"])
                        zeros.extend(y["zero_tail"])
                        doubles.extend(y["doubled_tail"])
                        tail_weights.extend(y["tail_weight"])
                        is_fit = index < 12 or entries[e["id"]]["world_seed"] in worlds[0]
                        fit_mask.extend([is_fit] * len(trace))
                    x = combine_inputs(all_rows)
                    y = torch.tensor(targets, dtype=torch.float32)
                    mask = torch.tensor(fit_mask)
                    fitting = ({k: v[mask] for k, v in x.items()}, y[mask])
                    validation = ({k: v[~mask] for k, v in x.items()}, y[~mask])
                    cache = args.output / f"targets-seed{seed}.pt"
                    torch.save(
                        dict(
                            fitting=fitting,
                            validation=validation,
                            rows=all_rows,
                            zero_tail=zeros,
                            doubled_tail=doubles,
                            tail_weights=tail_weights,
                            episodes=[summary_without_tensors(e) for e in episodes],
                        ),
                        cache,
                    )
                    entry = metrics(trainer, fitting, validation)
                    intercept_before = fit_intercept(trainer, fitting)
                    fitted = fit_critic(
                        trainer,
                        fitting,
                        validation,
                        original_mse=entry["mse"],
                        max_epochs=256,
                        seed=seed + 211000,
                        deadline=ledger.deadline,
                    )
                    intercept_after = fit_intercept(trainer, fitting)
                    final = metrics(trainer, fitting, validation)
                    if protected_state(trainer) != protected or any(
                        not torch.equal(v, trainer.model.state_dict()[k])
                        for k, v in buffers.items()
                    ):
                        raise ValueError("Calibration changed protected actor/value coordinates")
                    trainer.provenance = dict(
                        trainer.provenance,
                        v10_calibration_config_sha256=sha(args.config),
                        v10_targets_sha256=sha(cache),
                    )
                    path = args.output / f"calibrated-seed{seed}.pt"
                    trainer.checkpoint(path)
                    result["calibrations"].append(
                        dict(
                            seed=seed,
                            passed=ready(final, entry["mse"]),
                            entry=entry,
                            final=final,
                            fit=fitted,
                            intercept_before=intercept_before,
                            intercept_after=intercept_after,
                            checkpoint=str(path),
                            checkpoint_sha256=sha(path),
                            checkpoint_counters=dict(
                                episodes=trainer.episodes,
                                transitions=trainer.transitions,
                                updates=trainer.updates,
                            ),
                            actor_protected=True,
                            value_buffers_unchanged=True,
                            episodes=[summary_without_tensors(e) for e in episodes],
                        )
                    )
                    save()
                    continue
                target = cfg["target_additional_transitions"]
                if target not in (1024, 4096):
                    raise ValueError("Undeclared retained checkpoint")
                initial_additional = cfg.get("prior_additional_transitions", 0)
                initial_counter = trainer.transitions - initial_additional
                cell = dict(
                    id=f"seed{seed}",
                    seed=seed,
                    status="RUNNING",
                    transitions=initial_additional,
                    updates=[],
                    health_before=snapshot_health(trainer.model, adaptation, "full"),
                )
                result["cells"].append(cell)
                jobs = [
                    dict(entry=entries[i], sampler_seed=episode_seed(seed + 290000, j), limit=25)
                    for j, i in enumerate(selected["development"])
                ]
                if cfg.get("evaluate_baseline", False):
                    ledger.phase = "evaluation"
                    _, eps = collector.collect(
                        trainer.model, bank, jobs, trainer.config, collect=False, verify=True
                    )
                    result["baselines"].append(
                        dict(
                            seed=seed,
                            summary=summarize_episodes(eps),
                            endpoints=endpoint_summary(eps),
                            episodes=[summary_without_tensors(e) for e in eps],
                        )
                    )
                    save()
                all_episodes = []
                pool = [entries[i] for i in selected["ppo"]]
                while trainer.transitions - initial_counter < target:
                    ledger.phase = "training"
                    remaining = min(224, target - (trainer.transitions - initial_counter))
                    trainer.set_rollout_mix()
                    rows, eps = collector.rollout(trainer, bank, pool, seed + 200000, remaining)
                    if not rows:
                        raise ValueError("Empty native rollout")
                    update = measured_update(trainer, rows)
                    all_episodes.extend(eps)
                    cell["transitions"] = trainer.transitions - initial_counter
                    cell["updates"].append(update)
                    checkpoint = persist_rollout(trainer, args.output, cell, all_episodes)
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
                        args.output / f"seed{seed}-rollout{trainer.updates}.pt",
                        rollout_provenance=dict(
                            config_sha256=sha(args.config),
                            checkpoint_sha256=sha(checkpoint),
                            episode_ids=[e["id"] for e in eps],
                            sampler_scheme="episode_seed(seed+200000,absolute_episode)",
                        ),
                    )
                    if pair_hash(trainer) != protected:
                        raise ValueError("Audit changed retained state")
                    cell.setdefault("rollout_audits", []).append(audit)
                    save()
                cell["training_summary"] = summarize_episodes(all_episodes)
                cell["training_endpoints"] = endpoint_summary(all_episodes)
                protected = pair_hash(trainer)
                cell["reversal"] = reversal_with_adam(
                    trainer, adaptation, steps=64, deadline=ledger.deadline
                )
                cell["reversal"]["full_signed_switch"] = all(
                    p["passed"] and p["after"][1] > 0 for p in cell["reversal"]["phases"]
                )
                cell["health_after"] = snapshot_health(trainer.model, adaptation, "full")
                if pair_hash(trainer) != protected:
                    raise ValueError("Probe changed retained state")
                if cfg.get("evaluate_final", False):
                    ledger.phase = "evaluation"
                    _, eps = collector.collect(
                        trainer.model, bank, jobs, trainer.config, collect=False, verify=True
                    )
                    cell["evaluation"] = summarize_episodes(eps)
                    cell["evaluation_endpoints"] = endpoint_summary(eps)
                    cell["evaluation_episodes"] = [summary_without_tensors(e) for e in eps]
                cell["status"] = "COMPLETE"
                save()
        if args.stage == "calibrate":
            result["status"] = (
                "READY" if all(c["passed"] for c in result["calibrations"]) else "NO-GO_CRITIC"
            )
        else:
            result["status"] = "STAGED_REVIEW_REQUIRED"
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        print(json.dumps({k: result[k] for k in ["stage", "status", "budget"]}))


if __name__ == "__main__":
    main()
