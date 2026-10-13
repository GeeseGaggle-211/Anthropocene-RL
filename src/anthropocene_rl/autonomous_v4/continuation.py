"""Cumulative CPU continuation limits and deterministic technical checkpoint selection."""

import math


def validate_budget(config, histories):
    counts = {}
    seconds = 0.0
    for history in histories:
        amount = sum(row["additional_transitions"] for row in history["seeds"])
        if amount != history["additional_transitions"] or amount < 0:
            raise ValueError("Invalid historical transition accounting")
        duration = history["wall_seconds"]
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid historical duration")
        seconds += duration
        for row in history["seeds"]:
            if row["additional_transitions"] < 0:
                raise ValueError("Negative historical seed budget")
            counts[row["seed"]] = counts.get(row["seed"], 0) + row["additional_transitions"]
    if counts != {17: 32768, 29: 8192, 43: 8192}:
        raise ValueError("The complete frozen v2/v3 history is required")
    if not (
        config["seeds"] == [29, 43]
        and 0 < config["additional_transitions_per_seed"] <= 24576
        and config["maximum_total_transitions"] == 2 * config["additional_transitions_per_seed"]
        and config["maximum_total_transitions"] + sum(counts.values()) <= 98304
        and 0 < config["maximum_wall_seconds"] <= math.floor(14400 - seconds)
        and config["workers"] == 4
        and config["torch_threads"] == 1
        and config["evaluate_every_transitions"] == 4096
    ):
        raise ValueError("Cumulative continuation envelope exceeded")
    return dict(
        prior_transitions=sum(counts.values()),
        prior_by_seed=counts,
        prior_wall_seconds=seconds,
        remaining_wall_seconds=14400 - seconds,
    )


def select_checkpoints(frames):
    """Final + maximum single-layer saturation + maximum legal categorical confidence."""
    if not frames:
        raise ValueError("No checkpoint health frames")
    ordered = sorted(frames, key=lambda f: f["additional_transitions"])
    if len({f["additional_transitions"] for f in ordered}) != len(ordered):
        raise ValueError("Duplicate checkpoint boundary")

    def saturation(frame):
        h = frame["health"]
        values = [r["saturated_fraction"] for r in h["actor_layers"] + h["critic_layers"]]
        if not values or not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
            raise ValueError("Invalid saturation metric")
        return max(values)

    def confidence(frame):
        values = [
            r["maximum_probability"]
            for r in frame["health"]["categorical"].values()
            if r.get("eligible", 0) > 0
        ]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
            raise ValueError("Invalid categorical metric")
        return max(values, default=0.0)

    picks = {}
    for reason, frame in [
        ("final", ordered[-1]),
        ("maximum_actor_or_critic_layer_saturation", max(ordered, key=saturation)),
        ("maximum_legal_categorical_confidence", max(ordered, key=confidence)),
    ]:
        picks.setdefault(frame["additional_transitions"], []).append(reason)
    return [
        dict(transitions=boundary, reasons=reasons) for boundary, reasons in sorted(picks.items())
    ]
