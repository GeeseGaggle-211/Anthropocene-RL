"""Reject partition leakage and wrong-parent/counter continuation before collection."""


def validate_cohorts(selection, selected, entries, old_entries, old_ids, old_worlds):
    worlds = [set(selection[k]) for k in ["fit_worlds", "validation_worlds", "development_worlds"]]
    if any(worlds[i] & worlds[j] for i in range(3) for j in range(i)):
        raise ValueError("World partition overlap")
    if set(old_worlds) & set.union(*worlds):
        raise ValueError("Old fitting worlds overlap fresh partitions")
    if (
        len(old_ids) != 12
        or len(set(old_ids)) != 12
        or any(
            old_entries[i]["world_seed"] not in old_worlds or old_entries[i]["partition"] != "train"
            for i in old_ids
        )
    ):
        raise ValueError("Old fitting membership/count")
    counts = dict(fit=0, validation=0)
    for role in ["calibration", "ppo", "development"]:
        ids, slots = selected[role], selection[role]
        if len(ids) != len(slots) or len(set(ids)) != len(ids):
            raise ValueError("Duplicate or unmatched actual cohort")
        for id_, slot in zip(ids, slots, strict=True):
            entry = entries[id_]
            if (
                entry["world_seed"] != slot["world_seed"]
                or entry["family"] != slot["family"]
                or entry["decision_index"] not in slot["ages"]
            ):
                raise ValueError("Actual reset differs from frozen slot")
            partition = "development" if role == "development" else "train"
            if entry["partition"] != partition:
                raise ValueError("Wrong source partition")
            cohort = slot["role"] if role == "calibration" else role
            allowed = dict(
                fit=worlds[0], validation=worlds[1], ppo=worlds[0], development=worlds[2]
            )
            if cohort not in allowed or entry["world_seed"] not in allowed[cohort]:
                raise ValueError("Calibration/PPO/DEV membership leakage")
            if role == "calibration":
                counts[cohort] += 1
    if counts != dict(fit=12, validation=12):
        raise ValueError("Twelve unique fresh fitting and validation starts required")


def validate_parent(summary, pin, trainer, seed, prior_additional):
    if prior_additional == 0:
        if (
            summary["status"] != "READY"
            or len(summary["calibrations"]) != 3
            or not all(c["passed"] for c in summary["calibrations"])
        ):
            raise ValueError("All-three-critic READY gate required")
        parent = next(c for c in summary["calibrations"] if c["seed"] == seed)
    elif prior_additional == 1024:
        if summary["status"] != "STAGED_REVIEW_REQUIRED":
            raise ValueError("Expected preserved1024 stage")
        parent = next(c for c in summary["cells"] if c["seed"] == seed)
        if parent["status"] != "COMPLETE" or parent["durable_transitions"] != 1024:
            raise ValueError("Uncompleted prior retained stage")
    else:
        raise ValueError("Undeclared continuation offset")
    if parent["checkpoint_sha256"] != pin["sha256"]:
        raise ValueError("Exact approved parent hash required")
    if parent["checkpoint_counters"] != dict(
        episodes=trainer.episodes, transitions=trainer.transitions, updates=trainer.updates
    ):
        raise ValueError("Checkpoint continuation counters differ")


def endpoint_summary(episodes):
    rows = [
        dict(
            id=e["id"],
            family=e["family"],
            final_population=e["final_population"],
            final_realized_qol=e["trace"][-1]["qol"],
            living_endpoint_qol=e["trace"][-1]["qol"] if e["final_population"] > 0 else 0.0,
        )
        for e in episodes
    ]
    return dict(
        mean_final_population=sum(r["final_population"] for r in rows) / len(rows),
        mean_final_realized_qol=sum(r["final_realized_qol"] for r in rows) / len(rows),
        mean_living_endpoint_qol=sum(r["living_endpoint_qol"] for r in rows) / len(rows),
        convention="Last native closed-interval QoL; separately zero when focal endpoint extinct",
        episodes=rows,
    )
