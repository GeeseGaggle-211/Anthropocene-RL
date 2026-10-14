"""Expanded native TRAIN coverage, unchanged strict gate, and smaller paired canary."""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from run_autonomous_causal_gate_v7 import (
    combine_inputs,
    sha,
    summarize_episodes,
    summary_without_tensors,
)
from run_autonomous_continuation_v4 import measured_update
from train_autonomous_opportunity_v9 import (
    OpportunityCollector,
    pair_hash,
    reversal_with_adam,
    tensor_hash,
)

from anthropocene_rl.autonomous_v4.collection import episode_seed
from anthropocene_rl.autonomous_v5.causal import n_step_targets, protected_state
from anthropocene_rl.autonomous_v5.causal_budget import BudgetLedger
from anthropocene_rl.autonomous_v5.causal_diagnostics import (
    signature,
    signature_change,
    snapshot_health,
)
from anthropocene_rl.autonomous_v6.contracts import observation
from anthropocene_rl.autonomous_v6.coverage import reset_critic_for_collection
from anthropocene_rl.autonomous_v6.policy import inputs_with_mode
from anthropocene_rl.autonomous_v6.readiness import (
    fit_critic,
    fit_intercept,
    metrics,
    protected_calibration,
    ready,
)
from anthropocene_rl.training.autonomous_ppo_v7 import AutonomousTrainerV7


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
            raise ValueError(f"Changed frozen input: {pin['path']}")
    if cfg["transitions_per_cell"] != 448 or cfg["seeds"] != [101, 211, 307]:
        raise ValueError("Frozen coverage repair resource protocol")
    old = json.loads(Path(cfg["inputs"]["failed_summary"]["path"]).read_text())
    admission = json.loads(Path(cfg["inputs"]["admission"]["path"]).read_text())
    if old["status"] != "NO-GO_CRITIC" or old["cells"] or admission["status"] != "PASS":
        raise ValueError("Repair requires preserved failed calibration and native admission")
    selection = json.loads(Path(cfg["inputs"]["selection"]["path"]).read_text())
    sets = [set(selection[k]) for k in ["fit_worlds", "validation_worlds", "development_worlds"]]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("World partition leakage")
    selected = admission["selected"]
    bank = Path(cfg["inputs"]["bank"]["path"]).parent
    entries = {e["id"]: e for e in json.loads((bank / "manifest.json").read_text())["entries"]}
    if len(selected["additional_fitting"]) != 23 or any(
        entries[i]["world_seed"] not in sets[0] or entries[i]["partition"] != "train"
        for i in selected["additional_fitting"] + selected["ppo"]
    ):
        raise ValueError("Fitting cohort count/partition differs")
    if any(entries[i]["world_seed"] not in sets[2] for i in selected["development"]):
        raise ValueError("DEV world mismatch")
    prior = json.loads(Path(cfg["inputs"]["prior_budget"]["path"]).read_text())
    if prior["total_resolves"] + sum(cfg["caps"].values()) > 27991:
        raise ValueError("Cumulative resource envelope")
    torch.set_num_threads(1)
    probe = inputs_with_mode(
        [observation(o) for o in json.loads(Path(cfg["inputs"]["probe"]["path"]).read_text())]
    )
    adaptation = inputs_with_mode(
        [
            observation(o)
            for o in json.loads(Path(cfg["inputs"]["adaptation_probe"]["path"]).read_text())
        ]
    )
    args.output.mkdir(parents=True)
    ledger = BudgetLedger(
        args.output / "budget.json",
        cfg["caps"],
        prior_resolves=prior["total_resolves"],
        prior_wall_seconds=prior["cumulative_wall_seconds"],
        deadline=time.monotonic() + 2369.4072460189927 - prior["cumulative_wall_seconds"],
        native_version=6,
    )
    result = dict(
        version=cfg["version"],
        status="RUNNING",
        config_sha256=sha(args.config),
        calibrations=[],
        baselines=[],
        cells=[],
        validation_semantics="reused frozen TRAIN gate; no DEV fitting",
    )

    def save():
        ledger.save()
        result["budget"] = json.loads(ledger.path.read_text())
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    parents = []
    try:
        with OpportunityCollector(
            [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]],
            ledger=ledger,
            workers=4,
            deadline=ledger.deadline,
        ) as collector:
            for seed in cfg["seeds"]:
                state = torch.load(
                    cfg["inputs"][f"failed_seed{seed}"]["path"],
                    map_location="cpu",
                    weights_only=False,
                )
                trainer = AutonomousTrainerV7(
                    state["config"], 0, state["provenance"], width=state["width"]
                )
                trainer.restore(cfg["inputs"][f"failed_seed{seed}"]["path"])
                trainer.deadline = ledger.deadline
                protected = protected_state(trainer)
                reset = reset_critic_for_collection(trainer, adaptation)
                cached = torch.load(
                    cfg["inputs"][f"cached_seed{seed}"]["path"],
                    map_location="cpu",
                    weights_only=False,
                )
                mask = cached["fit"]
                validation = (
                    {k: v[~mask] for k, v in cached["obs"].items()},
                    cached["targets"][~mask],
                )
                original_training = (
                    {k: v[mask] for k, v in cached["obs"].items()},
                    cached["targets"][mask],
                )
                ledger.phase = "additional_fitting"
                jobs = [
                    dict(entry=entries[id_], sampler_seed=episode_seed(seed + 166000, i), limit=25)
                    for i, id_ in enumerate(selected["additional_fitting"])
                ]
                rows, episodes = collector.collect(
                    trainer.model, bank, jobs, trainer.config, collect=False, verify=True
                )
                if protected_state(trainer) != protected:
                    raise ValueError("Expanded native collection changed protected actor state")
                targets, zero, doubled, h100, tails = [], [], [], [], []
                for e in episodes:
                    trace = e["trace"]
                    rewards, years, terminal = (
                        [r[k] for r in trace] for k in ["reward", "elapsed", "terminal"]
                    )
                    y = n_step_targets(rewards, years, terminal, e["tail_value"], 400, 120)
                    targets.extend(y["target"])
                    zero.extend(y["zero_tail"])
                    doubled.extend(y["doubled_tail"])
                    tails.extend(y["tail_weight"])
                    h100.extend(n_step_targets(rewards, years, terminal, 0, 100, 120)["zero_tail"])
                new_x = combine_inputs(rows)
                fitting = (
                    {k: torch.cat([original_training[0][k], new_x[k]]) for k in new_x},
                    torch.cat([original_training[1], torch.tensor(targets, dtype=torch.float32)]),
                )
                torch.save(
                    dict(
                        fitting=fitting,
                        validation=validation,
                        zero_tail_targets=torch.tensor(cached["zero_tail_targets"])[mask].tolist()
                        + zero,
                        doubled_tail_targets=torch.tensor(cached["doubled_tail_targets"])[
                            mask
                        ].tolist()
                        + doubled,
                        h100_zero_tail_targets=torch.tensor(cached["h100_zero_tail_targets"])[
                            mask
                        ].tolist()
                        + h100,
                        tail_weights=torch.tensor(cached["tail_weights"])[mask].tolist() + tails,
                        row_mapping=dict(
                            original_fitting_rows=len(original_training[1]),
                            additional_rows=len(rows),
                            order="original fitting rows then canonical added episodes",
                        ),
                        episodes=[summary_without_tensors(e) for e in episodes],
                    ),
                    args.output / f"expanded-targets-seed{seed}.pt",
                )
                expanded = args.output / f"expanded-targets-seed{seed}.pt"
                trainer.provenance = dict(
                    trainer.provenance,
                    coverage_gate_config_sha256=sha(args.config),
                    failed_parent_checkpoint_sha256=cfg["inputs"][f"failed_seed{seed}"]["sha256"],
                    expanded_targets_path=str(expanded),
                    expanded_targets_sha256=sha(expanded),
                )
                entry = metrics(trainer, fitting, validation)
                normalization = trainer.model.fit_value_scale(fitting[1])
                normalization_state = protected_calibration(trainer)
                first = fit_intercept(trainer, fitting)
                fit = fit_critic(
                    trainer,
                    fitting,
                    validation,
                    original_mse=entry["mse"],
                    max_epochs=256,
                    seed=seed + 188000,
                    deadline=ledger.deadline,
                )
                second = fit_intercept(trainer, fitting)
                final = metrics(trainer, fitting, validation)
                checkpoint = args.output / f"calibrated-seed{seed}.pt"
                trainer.checkpoint(checkpoint)
                result["calibrations"].append(
                    dict(
                        seed=seed,
                        passed=ready(final, entry["mse"]),
                        entry=entry,
                        final=final,
                        fitting_rows=len(fitting[1]),
                        validation_rows=len(validation[1]),
                        critic_reset=reset,
                        normalization=normalization,
                        before_intercept=first,
                        after_intercept=second,
                        gradient_fit=fit,
                        actor_protected=protected_state(trainer) == protected,
                        normalized_state_protected=protected_calibration(trainer)
                        == normalization_state,
                        native_summary=summarize_episodes(episodes),
                        checkpoint=str(checkpoint),
                        checkpoint_sha256=sha(checkpoint),
                    )
                )
                parents.append(trainer)
                save()
            if not all(
                c["passed"] and c["actor_protected"] and c["normalized_state_protected"]
                for c in result["calibrations"]
            ):
                result["status"] = "NO-GO_CRITIC"
                save()
                return
            for seed, parent in zip(cfg["seeds"], parents, strict=True):
                ledger.phase = "evaluation"
                jobs = [
                    dict(entry=entries[id_], sampler_seed=episode_seed(seed + 90000, i), limit=25)
                    for i, id_ in enumerate(selected["development"])
                ]
                _, baseline = collector.collect(
                    parent.model, bank, jobs, parent.config, collect=False, verify=True
                )
                result["baselines"].append(
                    dict(
                        seed=seed,
                        summary=summarize_episodes(baseline),
                        episodes=[summary_without_tensors(e) for e in baseline],
                    )
                )
                reference = pair_hash(parent)
                for lr in cfg["actor_lrs"]:
                    trainer = copy.deepcopy(parent)
                    trainer.config = dict(trainer.config, actor_lr=lr)
                    trainer.optimizer.param_groups[0]["lr"] = lr
                    if pair_hash(trainer) != reference:
                        raise ValueError("Learning-rate pair differs beyond LR")
                    trainer.provenance = dict(
                        trainer.provenance, coverage_gate_sha256=sha(args.config), paired_lr=lr
                    )
                    cell = dict(
                        id=f"seed{seed}-lr{lr}",
                        seed=seed,
                        actor_lr=lr,
                        status="RUNNING",
                        transitions=0,
                        updates=[],
                        pair_state_sha256=reference,
                        health_before=snapshot_health(trainer.model, probe, "full"),
                    )
                    result["cells"].append(cell)
                    opening = signature(trainer.model, probe)
                    value_buffers = tensor_hash(
                        {k: trainer.model.state_dict()[k] for k in ["value_offset", "value_scale"]}
                    )
                    all_episodes = []
                    ledger.phase = "training"
                    while cell["transitions"] < 448:
                        trainer.set_rollout_mix()
                        rows, eps = collector.rollout(
                            trainer,
                            bank,
                            [entries[id_] for id_ in selected["ppo"]],
                            seed,
                            min(224, 448 - cell["transitions"]),
                        )
                        if not rows:
                            raise ValueError("Empty native rollout")
                        cell["transitions"] += len(rows)
                        all_episodes.extend(eps)
                        cell["updates"].append(measured_update(trainer, rows))
                        if (
                            tensor_hash(
                                {
                                    k: trainer.model.state_dict()[k]
                                    for k in ["value_offset", "value_scale"]
                                }
                            )
                            != value_buffers
                        ):
                            raise ValueError("PPO altered fixed value normalization")
                        save()
                    cell.update(
                        training_summary=summarize_episodes(all_episodes),
                        health_after=snapshot_health(trainer.model, probe, "full"),
                        distribution_change=signature_change(
                            opening, signature(trainer.model, probe)
                        ),
                        reversal=reversal_with_adam(trainer, adaptation, deadline=ledger.deadline),
                    )
                    (args.output / f"{cell['id']}-training-traces.json").write_text(
                        json.dumps([summary_without_tensors(e) for e in all_episodes]) + "\n"
                    )
                    checkpoint = args.output / f"{cell['id']}-step448.pt"
                    trainer.checkpoint(checkpoint)
                    cell.update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))
                    ledger.phase = "evaluation"
                    _, evaluation = collector.collect(
                        trainer.model, bank, jobs, trainer.config, collect=False, verify=True
                    )
                    cell.update(
                        evaluation=summarize_episodes(evaluation),
                        evaluation_episodes=[summary_without_tensors(e) for e in evaluation],
                        status="COMPLETE",
                    )
                    save()
            result["status"] = "COMPLETE"
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        print(
            json.dumps(
                dict(status=result["status"], cells=len(result["cells"]), budget=result["budget"])
            )
        )


if __name__ == "__main__":
    main()
