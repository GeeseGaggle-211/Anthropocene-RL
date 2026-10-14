"""Fixed TRAIN coverage, protected critic fit, and untouched balanced confirmation."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from audit_autonomous_current_value_v10 import score
from repair_autonomous_current_critic_v11 import fixed_fit
from run_autonomous_causal_gate_v7 import (
    combine_inputs,
    summarize_episodes,
    summary_without_tensors,
)
from run_autonomous_learning_v10 import load_bank, sha
from run_autonomous_reward_gate_v11 import ObjectiveCollector, fit_objective_intercept, strata

from anthropocene_rl.autonomous_v4.collection import episode_seed
from anthropocene_rl.autonomous_v5.causal_budget import BudgetLedger, BudgetTransport
from anthropocene_rl.autonomous_v6.contracts import observation, validate_schema
from anthropocene_rl.autonomous_v6.learning_guards import endpoint_summary
from anthropocene_rl.autonomous_v6.objective_collection import targets
from anthropocene_rl.autonomous_v6.opportunity import own_address
from anthropocene_rl.autonomous_v6.readiness import protected_calibration, ready
from anthropocene_rl.training.autonomous_ppo_v8 import AutonomousTrainerV8

FAMILIES = {"survival", "trade", "migration", "fission", "husbandry", "fishing"}
CAPS = {
    "generate": dict(generation=900, generation_replay=900),
    "admit": {},
    "fit": dict(fitting=4200, fitting_replay=4200),
    "confirm": dict(confirmation=1350, confirmation_replay=1350),
}


def restore(pin):
    state = torch.load(pin["path"], map_location="cpu", weights_only=False)
    trainer = AutonomousTrainerV8(state["config"], 0, state["provenance"], state["width"])
    trainer.restore(pin["path"])
    return trainer


def repeated_openings(rows, episodes, ys):
    """Descriptive reset-level variance, never a readiness denominator."""
    groups, initial_inputs, offset = {}, {}, 0
    for e in episodes:
        own = rows[offset]["obs"]
        if e["id"] in initial_inputs:
            prior = initial_inputs[e["id"]]
            if prior.keys() != own.keys() or any(not torch.equal(prior[k], own[k]) for k in own):
                raise ValueError("Replicated reset inputs differ")
        else:
            initial_inputs[e["id"]] = own
        record = dict(value=rows[offset]["value"], target=ys["target"][offset])
        groups.setdefault(e["id"], []).append(record)
        offset += e["steps"]
    return {
        id_: dict(
            draws=values,
            count=len(values),
            target_mean=float(np.mean([v["target"] for v in values])),
            target_sample_variance=(
                float(np.var([v["target"] for v in values], ddof=1)) if len(values) > 1 else None
            ),
        )
        for id_, values in groups.items()
    }


def returns_for(rows, episodes):
    ys = {k: [] for k in ["target", "zero_tail", "doubled_tail", "tail_weight"]}
    offset = 0
    for e in episodes:
        part = rows[offset : offset + e["steps"]]
        offset += e["steps"]
        for key, values in targets(part, e["tail_value"], 400).items():
            ys[key].extend(values)
    if offset != len(rows):
        raise ValueError("Episode offsets do not reconcile")
    return ys


def admit(tr, bank, entries, slots, role):
    selected, checks = [], []
    for slot in slots:
        accepted = None
        for age in slot["ages"]:
            id_ = f"{role}-{slot['world_seed']}-{slot['family']}-d{age}"
            if id_ not in entries:
                continue
            e = entries[id_]
            if e["partition"] != role or sha(bank / e["snapshot"]) != e["sha256"]:
                raise ValueError("Source role/digest differs")
            ids = [e["focal"]] + ([e["peer"]] if e["peer"] is not None else [])
            reset = tr.reset_world((bank / e["snapshot"]).read_bytes(), ids, 25)
            own = observation(reset["observations"][str(e["focal"])])
            valid = True
            if slot["family"] == "husbandry":
                valid = e["owned_animals"] > 0 and bool(own["ControlMask"][1:].all())
            elif slot["family"] == "fishing":
                valid = bool(own["WorkforceMask"][3])
            elif slot["family"] == "trade":
                valid = False
                if e["peer"] is not None:
                    peer = observation(reset["observations"][str(e["peer"])])
                    give = e["market_focal_give"]
                    valid = (
                        own_address(tr, e["focal"], e["peer"]) >= 0
                        and own_address(tr, e["peer"], e["focal"]) >= 0
                        and own["Stocks"][give] >= 1e-6
                        and peer["Stocks"][9 - give] >= 1e-6
                        and bool(own["GoodMask"][[3, 6]].all())
                        and bool(peer["GoodMask"][[3, 6]].all())
                    )
            checks.append(dict(id=id_, accepted=bool(valid)))
            if valid:
                accepted = id_
                break
        if accepted is None:
            raise ValueError("Unavailable fixed family slot: " + str(slot))
        selected.append(accepted)
    return dict(selected=selected, checks=checks)


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
            raise ValueError("Changed prospective pin: " + pin["path"])
    stage = cfg["stage"]
    if stage not in CAPS or cfg["caps"] != CAPS[stage]:
        raise ValueError("Frozen scope/caps differ")
    prior = json.loads(Path(cfg["inputs"]["prior_budget"]["path"]).read_text())
    envelope = json.loads(Path(cfg["inputs"]["envelope"]["path"]).read_text())
    remaining = envelope["total_active_second_ceiling"] - prior["cumulative_wall_seconds"]
    if (
        remaining <= 0
        or prior["total_resolves"] + sum(cfg["caps"].values()) > envelope["total_resolve_ceiling"]
    ):
        raise ValueError("Prospective V12 envelope exhausted")
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
    result = dict(version=cfg["version"], stage=stage, status="RUNNING", cells=[])

    def save():
        ledger.save()
        result["budget"] = json.loads(ledger.path.read_text())
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    try:
        if stage == "generate":
            result["banks"] = []
            for j, output in enumerate(cfg["outputs"]):
                bank = Path(output)
                if bank.exists():
                    raise FileExistsError(bank)
                ledger.reserve("generation", 300)
                ledger.reserve("generation_replay", 300)
                try:
                    p = subprocess.run(
                        [
                            cfg["dotnet"],
                            cfg["inputs"]["generator"]["path"],
                            cfg["inputs"][f"plan{j}"]["path"],
                            str(bank),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=ledger.deadline - time.monotonic(),
                    )
                except subprocess.TimeoutExpired as error:
                    for name, value in [("stdout", error.stdout), ("stderr", error.stderr)]:
                        (args.output / f"{j}-{name}").write_text(
                            value.decode(errors="replace")
                            if isinstance(value, bytes)
                            else value or ""
                        )
                    raise
                (args.output / f"{j}-stdout").write_text(p.stdout)
                (args.output / f"{j}-stderr").write_text(p.stderr)
                p.check_returncode()
                path = bank / "manifest.json"
                manifest = json.loads(path.read_text())
                if (
                    manifest["status"] not in ("PASS", "PASS_WITH_GAPS")
                    or manifest["generation_resolves"] > 300
                    or manifest["replay_resolves"] != manifest["generation_resolves"]
                ):
                    raise ValueError("Generation/replay failed")
                for e in manifest["entries"]:
                    if sha(bank / e["snapshot"]) != e["sha256"]:
                        raise ValueError("Generated snapshot changed")
                result["banks"].append(
                    dict(
                        path=str(path),
                        sha256=sha(path),
                        snapshots=len(manifest["entries"]),
                        unavailable=manifest["unavailable"],
                    )
                )
                save()
            result["status"] = "PASS"
        elif stage == "admit":
            result["admissions"] = []
            cls = type("CoverageAdmission", (BudgetTransport,), {"ledger": ledger})
            with cls(
                [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]], deadline=ledger.deadline
            ) as tr:
                validate_schema(tr.schema())
                for j, group in enumerate(cfg["slot_groups"]):
                    bank, entries = load_bank(cfg["inputs"][f"bank{j}"])
                    if (
                        len(group["slots"]) != 12
                        or any(
                            {s["family"] for s in group["slots"] if s["world_seed"] == w}
                            != FAMILIES
                            for w in group["worlds"]
                        )
                        or set(group["worlds"]) != {s["world_seed"] for s in group["slots"]}
                    ):
                        raise ValueError("Six-family two-world slots required")
                    result["admissions"].append(
                        admit(tr, bank, entries, group["slots"], group["role"])
                    )
                    save()
            result["status"] = "PASS"
        else:
            admission = json.loads(Path(cfg["inputs"]["admission"]["path"]).read_text())
            if admission["status"] != "PASS" or len(admission["admissions"]) != 3:
                raise ValueError("Full fixed physical admission required")
            with ObjectiveCollector(
                [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]],
                ledger=ledger,
                workers=4,
                deadline=ledger.deadline,
            ) as collector:
                ledger.phase = "fitting" if stage == "fit" else "confirmation"
                seeds = [101, 211] if stage == "fit" else [101, 211, 307]
                for seed in seeds:
                    trainer = restore(cfg["inputs"][f"checkpoint_seed{seed}"])
                    protected = protected_calibration(trainer)
                    if (
                        trainer.config["horizon_years"] != 400
                        or trainer.config["training_objective"]["kind"] != "superlinear"
                    ):
                        raise ValueError("Exact alpha H400 parent required")
                    rows, eps = [], []
                    if stage == "fit":
                        if cfg["epochs"] != 32 or len(cfg["old_fitting_entries"]) != 9:
                            raise ValueError("Fixed dose/old fitting slots differ")
                        sources = [
                            ("old_bank", cfg["old_fitting_entries"], 4),
                            ("bank0", admission["admissions"][0]["selected"], 2),
                            ("bank1", admission["admissions"][1]["selected"], 2),
                        ]
                    else:
                        sources = [("bank2", admission["admissions"][2]["selected"], 1)]
                    offset = 0
                    for bank_key, ids, repeats in sources:
                        bank, entries = load_bank(cfg["inputs"][bank_key])
                        expected_role = "train" if stage == "fit" else "validation"
                        worlds = {886001, 886002, 886003, 886004, 886011, 886012, 886013, 886014}
                        if stage != "fit":
                            worlds = {886015, 886016}
                        if any(
                            entries[i]["partition"] != expected_role
                            or entries[i]["world_seed"] not in worlds
                            for i in ids
                        ):
                            raise ValueError("Actual source world/role leakage")
                        jobs = [
                            dict(
                                entry=entries[i],
                                limit=25,
                                sampler_seed=episode_seed(seed + cfg["sampler_offset"], offset + j),
                            )
                            for j, i in enumerate(ids * repeats)
                        ]
                        if stage == "confirm":
                            primary_count = len(jobs)
                            trade = [i for i in ids if entries[i]["family"] == "trade"]
                            if len(trade) != 2 or any(
                                entries[i]["decision_index"] != 0 for i in trade
                            ):
                                raise ValueError("Trade replicas require both fixed d0 starts")
                            jobs.extend(
                                dict(
                                    entry=entries[i],
                                    limit=25,
                                    sampler_seed=episode_seed(
                                        seed + cfg["sampler_offset"], primary_count + j
                                    ),
                                )
                                for j, i in enumerate(trade * 3)
                            )
                        offset += len(jobs)
                        rr, ee = collector.collect(
                            trainer.model, bank, jobs, trainer.config, collect=True, verify=True
                        )
                        rows.extend(rr)
                        eps.extend(ee)
                    if protected_calibration(trainer) != protected:
                        raise ValueError("Collection changed protected state")
                    if len(eps) != (84 if stage == "fit" else 18):
                        raise ValueError("Fixed replicated episode count differs")
                    ys = returns_for(rows, eps)
                    x = combine_inputs(rows)
                    y = torch.tensor(ys["target"], dtype=torch.float32)
                    cache = args.output / f"actual-seed{seed}.pt"
                    torch.save(dict(rows=rows, episodes=eps, targets=ys), cache)
                    cell = dict(
                        seed=seed,
                        cache=str(cache),
                        cache_sha256=sha(cache),
                        repeated_openings=repeated_openings(rows, eps, ys),
                    )
                    result["cells"].append(cell)
                    if stage == "fit":
                        cell.update(
                            coverage=dict(
                                rows=len(rows),
                                episodes=len(eps),
                                target_min=float(y.min()),
                                target_max=float(y.max()),
                                target_mean=float(y.mean()),
                                target_variance=float(y.var(unbiased=False)),
                            )
                        )
                        save()
                        before = fit_objective_intercept(trainer, (x, y))
                        dose = fixed_fit(
                            trainer, (x, y), epochs=32, seed=seed + 711000, deadline=ledger.deadline
                        )
                        after = fit_objective_intercept(trainer, (x, y))
                        with torch.no_grad():
                            train_metrics = score(trainer.model.value(x), y)
                        trainer.provenance = dict(
                            trainer.provenance,
                            critic_coverage_config_sha256=sha(args.config),
                            critic_coverage_cache_sha256=sha(cache),
                        )
                        checkpoint = args.output / f"coverage-superlinear-seed{seed}.pt"
                        trainer.checkpoint(checkpoint)
                        cell.update(
                            dose=dose,
                            intercept_before=before,
                            intercept_after=after,
                            training_metrics=train_metrics,
                            checkpoint=str(checkpoint),
                            checkpoint_sha256=sha(checkpoint),
                        )
                    else:
                        primary_rows = sum(e["steps"] for e in eps[:12])
                        parent = restore(cfg["inputs"][f"entry_seed{seed}"])
                        with torch.no_grad():
                            prediction = trainer.model.value(x).cpu()
                            entry_prediction = parent.model.value(x).cpu()
                        primary = score(prediction[:primary_rows], y[:primary_rows])
                        entry = score(entry_prediction[:primary_rows], y[:primary_rows])
                        pp = args.output / f"predictions-seed{seed}.pt"
                        torch.save(
                            dict(
                                prediction=prediction,
                                entry_prediction=entry_prediction,
                                primary_rows=primary_rows,
                            ),
                            pp,
                        )
                        cell.update(
                            primary=primary,
                            entry=entry,
                            passed=ready(primary, entry["mse"]),
                            primary_rows=primary_rows,
                            primary_episodes=12,
                            secondary=score(prediction, y),
                            tail_variants={
                                k: score(
                                    prediction[:primary_rows],
                                    torch.tensor(ys[k][:primary_rows], dtype=torch.float32),
                                )
                                for k in ["zero_tail", "doubled_tail"]
                            },
                            strata=strata(
                                trainer,
                                ({k: v[:primary_rows] for k, v in x.items()}, y[:primary_rows]),
                            ),
                            endpoints=endpoint_summary(eps[:12]),
                            episodes=[summary_without_tensors(e) for e in eps],
                            summary=summarize_episodes(eps[:12]),
                            predictions=str(pp),
                            predictions_sha256=sha(pp),
                        )
                    if protected_calibration(trainer) != protected:
                        raise ValueError("Actor/heads/Adam/RNG/counters/buffers changed")
                    cell.update(
                        protected_before=protected, protected_after=protected_calibration(trainer)
                    )
                    save()
            result["status"] = "REVIEW_REQUIRED"
            if stage == "confirm":
                result["status"] = (
                    "PASS_CRITIC" if all(c["passed"] for c in result["cells"]) else "NO-GO_CRITIC"
                )
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        print(json.dumps(dict(status=result["status"], budget=result["budget"])))


if __name__ == "__main__":
    main()
