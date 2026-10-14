"""Fresh-world admission then frozen-policy endpoints/current-value comparison."""

import argparse
import json
import time
from pathlib import Path

import torch
from audit_autonomous_current_value_v10 import score
from run_autonomous_causal_gate_v7 import summarize_episodes, summary_without_tensors
from run_autonomous_learning_v10 import load_bank, sha
from run_autonomous_reward_gate_v11 import ObjectiveCollector, strata
from train_autonomous_opportunity_v9 import pair_hash

from anthropocene_rl.autonomous_v4.collection import episode_seed
from anthropocene_rl.autonomous_v5.causal_budget import BudgetLedger, BudgetTransport
from anthropocene_rl.autonomous_v6.contracts import observation, validate_schema
from anthropocene_rl.autonomous_v6.learning_guards import endpoint_summary
from anthropocene_rl.autonomous_v6.objective_collection import targets
from anthropocene_rl.autonomous_v6.opportunity import own_address
from anthropocene_rl.training.autonomous_ppo_v8 import AutonomousTrainerV8


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
    expected = dict(admit={}, evaluate=dict(evaluation=1800, evaluation_replay=1800))
    if cfg["stage"] not in expected or cfg["caps"] != expected[cfg["stage"]]:
        raise ValueError("Unapproved fresh-validation scope")
    prior = json.loads(Path(cfg["inputs"]["prior_budget"]["path"]).read_text())
    envelope = json.loads(Path(cfg["inputs"]["envelope"]["path"]).read_text())
    remaining = envelope["total_active_second_ceiling"] - prior["cumulative_wall_seconds"]
    if (
        remaining <= 0
        or prior["total_resolves"] + sum(cfg["caps"].values()) > envelope["total_resolve_ceiling"]
    ):
        raise ValueError("Prospective V11 envelope exhausted")
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
    bank, entries = load_bank(cfg["inputs"]["bank"])
    slots = cfg["slots"]
    families = {"survival", "trade", "migration", "fission", "husbandry", "fishing"}
    if (
        len(slots) != 12
        or {s["world_seed"] for s in slots} != {886009, 886010}
        or any(
            {s["family"] for s in slots if s["world_seed"] == world} != families
            for world in [886009, 886010]
        )
    ):
        raise ValueError("Two fixed six-family fresh worlds required")
    if any(
        e["partition"] != "validation" or e["world_seed"] not in [886009, 886010]
        for e in entries.values()
    ):
        raise ValueError("Fresh validation bank role/world differs")
    result = dict(
        version=cfg["version"],
        stage=cfg["stage"],
        status="RUNNING",
        config_sha256=sha(args.config),
        selected=[],
        checks=[],
        cells=[],
        retained_updates=0,
    )

    def save():
        ledger.save()
        result["budget"] = json.loads(ledger.path.read_text())
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    try:
        if cfg["stage"] == "admit":
            cls = type("FreshValidationAdmission", (BudgetTransport,), {"ledger": ledger})
            with cls(
                [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]], deadline=ledger.deadline
            ) as tr:
                validate_schema(tr.schema())
                for slot in slots:
                    accepted = None
                    for age in slot["ages"]:
                        id_ = f"validation-{slot['world_seed']}-{slot['family']}-d{age}"
                        if id_ not in entries:
                            continue
                        e = entries[id_]
                        blob = (bank / e["snapshot"]).read_bytes()
                        if sha(bank / e["snapshot"]) != e["sha256"]:
                            raise ValueError("Source snapshot differs")
                        ids = [e["focal"]] + ([e["peer"]] if e["peer"] is not None else [])
                        reset = tr.reset_world(blob, ids, 25)
                        own = observation(reset["observations"][str(e["focal"])])
                        valid = True
                        reciprocal = False
                        if slot["family"] == "husbandry":
                            valid = e["owned_animals"] > 0 and bool(own["ControlMask"][1:].all())
                        elif slot["family"] == "fishing":
                            valid = bool(own["WorkforceMask"][3])
                        elif slot["family"] == "trade":
                            if e["peer"] is None:
                                valid = False
                            else:
                                peer = observation(reset["observations"][str(e["peer"])])
                                give = e["market_focal_give"]
                                reciprocal = (
                                    own_address(tr, e["focal"], e["peer"]) >= 0
                                    and own_address(tr, e["peer"], e["focal"]) >= 0
                                    and own["Stocks"][give] >= 1e-6
                                    and peer["Stocks"][9 - give] >= 1e-6
                                    and bool(own["GoodMask"][[3, 6]].all())
                                    and bool(peer["GoodMask"][[3, 6]].all())
                                )
                                valid = reciprocal
                        result["checks"].append(
                            dict(
                                id=id_,
                                accepted=bool(valid),
                                reciprocal_opportunity=bool(reciprocal),
                                workforce_mask=own["WorkforceMask"].tolist(),
                                control_mask=own["ControlMask"].tolist(),
                            )
                        )
                        if valid:
                            accepted = id_
                            break
                    if accepted is None:
                        raise ValueError("Missing fixed physical validation family: " + str(slot))
                    result["selected"].append(accepted)
                    save()
            result["status"] = "PASS"
        else:
            admission = json.loads(Path(cfg["inputs"]["fresh_admission"]["path"]).read_text())
            if admission["status"] != "PASS" or len(admission["selected"]) != 12:
                raise ValueError("Full twelve-slot physical admission required")
            result["selected"] = admission["selected"]
            for id_, slot in zip(result["selected"], slots, strict=True):
                e = entries[id_]
                if (
                    e["world_seed"] != slot["world_seed"]
                    or e["family"] != slot["family"]
                    or e["decision_index"] not in slot["ages"]
                ):
                    raise ValueError("Actual validation start differs from fixed slot")
            gate = json.loads(Path(cfg["inputs"]["parent_gate"]["path"]).read_text())
            with ObjectiveCollector(
                [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]],
                ledger=ledger,
                workers=4,
                deadline=ledger.deadline,
            ) as collector:
                ledger.phase = "evaluation"
                for seed in [101, 211, 307]:
                    for kind in ["original", "superlinear"]:
                        pin = cfg["inputs"][f"checkpoint_{kind}_seed{seed}"]
                        state = torch.load(pin["path"], map_location="cpu", weights_only=False)
                        trainer = AutonomousTrainerV8(
                            state["config"], 0, state["provenance"], state["width"]
                        )
                        trainer.restore(pin["path"])
                        parent = next(
                            c for c in gate["cells"] if c["seed"] == seed and c["kind"] == kind
                        )
                        if (
                            gate["status"] != "STAGED_REVIEW_REQUIRED"
                            or parent["status"] != "COMPLETE"
                            or parent["durable_transitions"] != 1024
                            or parent["checkpoint_sha256"] != pin["sha256"]
                            or parent["checkpoint_counters"]
                            != dict(
                                episodes=trainer.episodes,
                                transitions=trainer.transitions,
                                updates=trainer.updates,
                            )
                        ):
                            raise ValueError("Wrong retained matched policy parent")
                        protected = pair_hash(trainer)
                        jobs = [
                            dict(
                                entry=entries[i],
                                sampler_seed=episode_seed(seed + 510000, j),
                                limit=25,
                            )
                            for j, i in enumerate(result["selected"])
                        ]
                        rows, eps = collector.collect(
                            trainer.model, bank, jobs, trainer.config, collect=True, verify=True
                        )
                        if any(r["elapsed"] != 20 for r in rows):
                            raise ValueError(
                                "Fixed500-year annualization requires20-year intervals"
                            )
                        ys = {k: [] for k in ["target", "zero_tail", "doubled_tail", "tail_weight"]}
                        offset = 0
                        for e in eps:
                            batch = rows[offset : offset + e["steps"]]
                            offset += e["steps"]
                            y = targets(batch, e["tail_value"], 400)
                            for key, values in y.items():
                                ys[key].extend(values)
                        prediction = torch.tensor([r["value"] for r in rows], dtype=torch.float32)
                        metrics = {
                            k: score(prediction, torch.tensor(ys[k], dtype=torch.float32))
                            for k in ["target", "zero_tail", "doubled_tail"]
                        }
                        x = {k: torch.cat([r["obs"][k] for r in rows]) for k in rows[0]["obs"]}
                        y = torch.tensor(ys["target"], dtype=torch.float32)
                        cache = args.output / f"audit-{kind}-seed{seed}.pt"
                        torch.save(
                            dict(
                                rows=rows,
                                targets=ys,
                                prediction=prediction,
                                episodes=[summary_without_tensors(e) for e in eps],
                            ),
                            cache,
                        )
                        if pair_hash(trainer) != protected:
                            raise ValueError("Frozen validation changed retained state")
                        result["cells"].append(
                            dict(
                                seed=seed,
                                kind=kind,
                                summary=summarize_episodes(eps),
                                endpoints=endpoint_summary(eps),
                                episodes=[summary_without_tensors(e) for e in eps],
                                mean_integrated_original_welfare=sum(
                                    e["integrated_welfare"] for e in eps
                                )
                                / len(eps),
                                scheduled_500year_annual_welfare=sum(
                                    e["integrated_welfare"] for e in eps
                                )
                                / (len(eps) * 500),
                                metrics=metrics,
                                strata=strata(trainer, (x, y)),
                                mean_tail_weight=float(torch.tensor(ys["tail_weight"]).mean()),
                                cache=str(cache),
                                cache_sha256=sha(cache),
                                protected_hash=protected,
                                checkpoint_sha256=pin["sha256"],
                                retained_state_unchanged=True,
                            )
                        )
                        save()
            result["status"] = "REVIEW_REQUIRED"
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        print(json.dumps({k: result[k] for k in ["stage", "status", "budget"]}))


if __name__ == "__main__":
    main()
