"""One fixed TRAIN-only live-critic repair; frozen validation and actors preserved."""

import argparse
import json
import time
from pathlib import Path

import torch
from run_autonomous_causal_gate_v7 import combine_inputs, summary_without_tensors
from run_autonomous_learning_v10 import load_bank, sha
from run_autonomous_reward_gate_v11 import ObjectiveCollector, fit_objective_intercept, strata

from anthropocene_rl.autonomous_v4.collection import episode_seed
from anthropocene_rl.autonomous_v5.causal_budget import BudgetLedger
from anthropocene_rl.autonomous_v6.objective_collection import targets
from anthropocene_rl.autonomous_v6.readiness import metrics, protected_calibration, ready
from anthropocene_rl.training.autonomous_ppo_v8 import AutonomousTrainerV8


def fixed_fit(trainer, fitting, *, epochs, seed, deadline):
    before = protected_calibration(trainer)
    rng = torch.Generator().manual_seed(seed)
    x, y = fitting
    actor = [*trainer.model.actor.parameters(), *trainer.model.heads.parameters()]
    flags = [p.requires_grad for p in actor]
    for p in actor:
        p.requires_grad_(False)
    records = []
    try:
        for epoch in range(epochs):
            losses = []
            for ix in torch.randperm(len(y), generator=rng).split(32):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Fixed critic-only repair deadline")
                loss = trainer.critic_loss(
                    trainer.model.value({k: v[ix] for k, v in x.items()}), y[ix]
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite fixed critic loss")
                trainer.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    trainer.model.critic.parameters(), 0.5, error_if_nonfinite=True
                )
                trainer.optimizer.step()
                losses.append(float(loss.detach()))
            records.append(
                dict(
                    epoch=epoch + 1,
                    mean_training_loss=sum(losses) / len(losses),
                    gradient=float(gradient),
                )
            )
    finally:
        for p, flag in zip(actor, flags, strict=True):
            p.requires_grad_(flag)
    if protected_calibration(trainer) != before:
        raise ValueError("Fixed critic fit changed protected actor/Adam/RNG/anchors/scale")
    return records


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
    if (
        cfg["seeds"] != [101, 211]
        or cfg["epochs"] != 64
        or cfg["caps"] != dict(repair=450, repair_replay=450)
    ):
        raise ValueError("Fixed failed-critic-only dose differs")
    prior = json.loads(Path(cfg["inputs"]["prior_budget"]["path"]).read_text())
    envelope = json.loads(Path(cfg["inputs"]["envelope"]["path"]).read_text())
    remaining = envelope["total_active_second_ceiling"] - prior["cumulative_wall_seconds"]
    if remaining <= 0 or prior["total_resolves"] + 900 > envelope["total_resolve_ceiling"]:
        raise ValueError("V11 prospective envelope exhausted")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    ledger = BudgetLedger(
        args.output / "budget.json",
        cfg["caps"],
        prior_resolves=prior["total_resolves"],
        prior_wall_seconds=prior["cumulative_wall_seconds"],
        deadline=time.monotonic() + min(remaining, cfg["max_active_seconds"]),
        native_version=6,
    )
    bank, entries = load_bank(cfg["inputs"]["fitting_bank"])
    ids = cfg["fitting_entries"]
    if (
        len(ids) != 9
        or len(set(ids)) != 9
        or any(
            entries[i]["world_seed"] not in [886001, 886002, 886003, 886004]
            or entries[i]["partition"] != "train"
            for i in ids
        )
        or sum(entries[i]["family"] == "trade" for i in ids) != 4
        or {entries[i]["family"] for i in ids}
        != {"survival", "trade", "migration", "fission", "husbandry", "fishing"}
    ):
        raise ValueError("Fixed nine TRAIN fitting starts differ")
    gate = json.loads(Path(cfg["inputs"]["parent_gate"]["path"]).read_text())
    result = dict(
        version=cfg["version"],
        status="RUNNING",
        config_sha256=sha(args.config),
        repairs=[],
        unchanged_seed307=cfg["inputs"]["checkpoint_superlinear_seed307"],
    )

    def save():
        ledger.save()
        result["budget"] = json.loads(ledger.path.read_text())
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    try:
        with ObjectiveCollector(
            [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]],
            ledger=ledger,
            workers=4,
            deadline=ledger.deadline,
        ) as collector:
            ledger.phase = "repair"
            for seed in cfg["seeds"]:
                pin = cfg["inputs"][f"checkpoint_superlinear_seed{seed}"]
                state = torch.load(pin["path"], map_location="cpu", weights_only=False)
                trainer = AutonomousTrainerV8(
                    state["config"], 0, state["provenance"], state["width"]
                )
                trainer.restore(pin["path"])
                parent = next(
                    c for c in gate["cells"] if c["seed"] == seed and c["kind"] == "superlinear"
                )
                if (
                    gate["status"] != "STAGED_REVIEW_REQUIRED"
                    or parent["status"] != "COMPLETE"
                    or parent["checkpoint_sha256"] != pin["sha256"]
                    or parent["checkpoint_counters"]
                    != dict(
                        episodes=trainer.episodes,
                        transitions=trainer.transitions,
                        updates=trainer.updates,
                    )
                ):
                    raise ValueError("Wrong exact retained actor/critic parent")
                protected = protected_calibration(trainer)
                frozen = torch.load(
                    cfg["inputs"][f"fresh_cache_seed{seed}"]["path"],
                    map_location="cpu",
                    weights_only=False,
                )
                xv = {
                    k: torch.cat([r["obs"][k] for r in frozen["rows"]])
                    for k in frozen["rows"][0]["obs"]
                }
                validation = (xv, torch.tensor(frozen["targets"]["target"], dtype=torch.float32))
                jobs = [
                    dict(entry=entries[i], sampler_seed=episode_seed(seed + 610000, j), limit=25)
                    for j, i in enumerate(ids)
                ]
                rows, episodes = collector.collect(
                    trainer.model, bank, jobs, trainer.config, collect=True, verify=True
                )
                target, offset = [], 0
                for e in episodes:
                    batch = rows[offset : offset + e["steps"]]
                    offset += e["steps"]
                    target.extend(targets(batch, e["tail_value"], 400)["target"])
                fitting = (combine_inputs(rows), torch.tensor(target, dtype=torch.float32))
                cache = args.output / f"fitting-seed{seed}.pt"
                torch.save(
                    dict(
                        rows=rows,
                        episodes=[summary_without_tensors(e) for e in episodes],
                        fitting=fitting,
                    ),
                    cache,
                )
                if protected_calibration(trainer) != protected:
                    raise ValueError("TRAIN collection changed retained state")
                entry = metrics(trainer, fitting, validation)
                before = fit_objective_intercept(trainer, fitting)
                records = fixed_fit(
                    trainer, fitting, epochs=64, seed=seed + 611000, deadline=ledger.deadline
                )
                after = fit_objective_intercept(trainer, fitting)
                final = metrics(trainer, fitting, validation)
                predictions_path = args.output / f"validation-predictions-seed{seed}.pt"
                with torch.no_grad():
                    values = trainer.model.value(xv).cpu()
                torch.save(dict(values=values, frozen_targets=validation[1]), predictions_path)
                if protected_calibration(trainer) != protected:
                    raise ValueError("Critic-only repair changed protected state")
                path = args.output / f"critic-repaired-superlinear-seed{seed}.pt"
                trainer.provenance = dict(
                    trainer.provenance,
                    critic_repair_config_sha256=sha(args.config),
                    frozen_validation_cache_sha256=cfg["inputs"][f"fresh_cache_seed{seed}"][
                        "sha256"
                    ],
                )
                trainer.checkpoint(path)
                result["repairs"].append(
                    dict(
                        seed=seed,
                        entry=entry,
                        final=final,
                        passed=ready(final, entry["mse"]),
                        dose=records,
                        intercept_before=before,
                        intercept_after=after,
                        strata=strata(trainer, validation),
                        checkpoint=str(path),
                        checkpoint_sha256=sha(path),
                        fitting_cache=str(cache),
                        fitting_cache_sha256=sha(cache),
                        validation_predictions=str(predictions_path),
                        validation_predictions_sha256=sha(predictions_path),
                        protected_before=protected,
                        protected_after=protected_calibration(trainer),
                        actor_heads_rng_adam_anchors_scale_counters_unchanged=True,
                        checkpoint_counters=dict(
                            episodes=trainer.episodes,
                            transitions=trainer.transitions,
                            updates=trainer.updates,
                        ),
                    )
                )
                save()
        result["status"] = (
            "READY_REUSED_VALIDATION"
            if all(c["passed"] for c in result["repairs"])
            else "NO-GO_CRITIC"
        )
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        print(json.dumps({k: result[k] for k in ["status", "budget"]}))


if __name__ == "__main__":
    main()
