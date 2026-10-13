from __future__ import annotations

import json
from collections import Counter

from anthropocene_rl.training.overfit import _write_json
from anthropocene_rl.training.pilot import (
    PilotConfig,
    _advance_health_tripwire_streaks,
    _evaluation_stop_reason,
    _health_tripwire_evaluation,
)


def _evaluated(
    *,
    endpoint_rate: float = 0.0,
    critic_ev: float = 0.4,
    critic_nmse: float = 0.6,
) -> dict:
    names = (
        "field_operation",
        "herd_retention",
        "development_budget",
        "field_development_share",
    )
    return {
        "summary": {
            "actor_mean_endpoint_health_by_head": {
                name: {
                    "endpoint_count": int(endpoint_rate * 100),
                    "active_decision_count": 100,
                    "active_episode_count": 5,
                    "rate": endpoint_rate,
                }
                for name in names
            },
            "critic_initial_value_vs_realized_horizon": {
                "evaluable": True,
                "explained_variance": critic_ev,
                "normalized_mse": critic_nmse,
            },
        }
    }


def _learning_curve(
    *,
    kl_stops: int = 0,
    actor_epochs: float = 3.0,
    clip_fraction: float = 0.2,
) -> list[dict]:
    return [
        {
            "update": update,
            "ppo": {
                "pre_update_critic_explained_variance": 0.2,
                "pre_update_critic_normalized_mse": 0.8,
                "pre_update_critic_normalized_bias": 0.0,
                "target_kl_early_stop": float(index < kl_stops),
                "optimization_actor_epochs_completed": actor_epochs,
                "post_update_clip_fraction": clip_fraction,
            },
        }
        for index, update in enumerate(range(11, 21))
    ]


def _inactive_audit(*, gradient: float = 0.0) -> dict:
    per_head = {
        f"continuous_{head}": {
            "inactive_rows": 1,
            "max_abs_gradient": gradient,
        }
        for head in range(4)
    }
    per_head["continuous_2"]["inactive_rows"] = 0
    return {
        "max_abs_gradient": gradient,
        "threshold": 1e-12,
        "per_head": {
            **per_head,
            "movement": {"inactive_rows": 0, "max_abs_gradient": None},
        },
    }


def test_health_tripwire_boundaries_match_launch_contract() -> None:
    report = _health_tripwire_evaluation(
        update=20,
        evaluated=_evaluated(endpoint_rate=0.95, critic_ev=0.0, critic_nmse=1.0),
        learning_curve=_learning_curve(
            kl_stops=8,
            actor_epochs=1.5,
            clip_fraction=0.60,
        ),
        inactive_head_audit=_inactive_audit(gradient=1e-12),
        campaign_scope="broad-domain",
        ppo_epochs=3,
        config=PilotConfig(),
        last_operational_window_end=None,
    )

    conditions = report["conditions"]
    assert conditions["actor_endpoint_pinning_field_operation"]["status"] == "fail"
    assert conditions["development_critic_collapse"]["status"] == "fail"
    assert conditions["rolling_training_critic_health"]["status"] == "pass"
    assert conditions["widespread_target_kl_truncation"]["status"] == "fail"
    assert conditions["excessive_clipping"]["status"] == "pass"
    assert conditions["inactive_head_gradient_inert"]["status"] == "pass"
    inactive = conditions["inactive_head_gradient_inert"]["measurements"]
    assert inactive["not_applicable_zero_row_heads"] == ["continuous_2", "movement"]


def test_final_epoch_kl_flag_does_not_count_as_discarded_learning() -> None:
    report = _health_tripwire_evaluation(
        update=20,
        evaluated=_evaluated(),
        learning_curve=_learning_curve(kl_stops=10, actor_epochs=3.0),
        inactive_head_audit=_inactive_audit(),
        campaign_scope="broad-domain",
        ppo_epochs=3,
        config=PilotConfig(),
        last_operational_window_end=None,
    )

    assert report["conditions"]["widespread_target_kl_truncation"]["status"] == "pass"


def test_health_tripwire_patience_resets_after_pass() -> None:
    failure_streaks: Counter[str] = Counter()
    unavailable_streaks: Counter[str] = Counter()

    def report(status: str) -> dict:
        return {"conditions": {"screen": {"status": status, "immediate": False}}}

    assert (
        _advance_health_tripwire_streaks(
            report("fail"),
            failure_streaks=failure_streaks,
            unavailable_streaks=unavailable_streaks,
            patience=2,
        )
        is None
    )
    assert (
        _advance_health_tripwire_streaks(
            report("pass"),
            failure_streaks=failure_streaks,
            unavailable_streaks=unavailable_streaks,
            patience=2,
        )
        is None
    )
    assert (
        _advance_health_tripwire_streaks(
            report("fail"),
            failure_streaks=failure_streaks,
            unavailable_streaks=unavailable_streaks,
            patience=2,
        )
        is None
    )
    assert _advance_health_tripwire_streaks(
        report("fail"),
        failure_streaks=failure_streaks,
        unavailable_streaks=unavailable_streaks,
        patience=2,
    ).startswith("health-tripwire-failed-with-patience:")


def test_unavailable_health_evidence_and_inactive_gradient_latch() -> None:
    failure_streaks: Counter[str] = Counter()
    unavailable_streaks: Counter[str] = Counter()
    unavailable = {"conditions": {"screen": {"status": "unavailable"}}}
    assert (
        _advance_health_tripwire_streaks(
            unavailable,
            failure_streaks=failure_streaks,
            unavailable_streaks=unavailable_streaks,
            patience=2,
        )
        is None
    )
    assert _advance_health_tripwire_streaks(
        unavailable,
        failure_streaks=failure_streaks,
        unavailable_streaks=unavailable_streaks,
        patience=2,
    ).startswith("health-evidence-unavailable-with-patience:")

    report = _health_tripwire_evaluation(
        update=20,
        evaluated=_evaluated(),
        learning_curve=_learning_curve(),
        inactive_head_audit=_inactive_audit(gradient=2e-12),
        campaign_scope="broad-domain",
        ppo_epochs=3,
        config=PilotConfig(),
        last_operational_window_end=None,
    )
    failure_streaks.clear()
    unavailable_streaks.clear()
    assert _advance_health_tripwire_streaks(
        report,
        failure_streaks=failure_streaks,
        unavailable_streaks=unavailable_streaks,
        patience=2,
    ).startswith("health-tripwire-immediate:inactive_head_gradient_inert")


def test_nonfinite_health_is_immediate_before_warmup_and_json_safe(tmp_path) -> None:
    evaluated = _evaluated()
    endpoint = evaluated["summary"]["actor_mean_endpoint_health_by_head"]
    endpoint["field_operation"]["rate"] = float("nan")
    evaluated["summary"]["critic_initial_value_vs_realized_horizon"] = {
        "evaluable": False,
        "explained_variance": float("inf"),
        "normalized_mse": None,
    }
    learning_curve = _learning_curve()
    for update, row in enumerate(learning_curve, start=1):
        row["update"] = update

    report = _health_tripwire_evaluation(
        update=10,
        evaluated=evaluated,
        learning_curve=learning_curve,
        inactive_head_audit=_inactive_audit(),
        campaign_scope="broad-domain",
        ppo_epochs=3,
        config=PilotConfig(),
        last_operational_window_end=None,
    )

    assert report["conditions"]["actor_endpoint_pinning_field_operation"] == {
        "status": "fail",
        "immediate": True,
        "reason": "non-finite endpoint rate",
        "measurements": {
            "endpoint_count": 0,
            "active_decision_count": 100,
            "active_episode_count": 5,
            "rate": {"non_finite": "NaN"},
        },
    }
    critic = report["conditions"]["development_critic_collapse"]
    assert critic["status"] == "fail"
    assert critic["immediate"] is True
    assert critic["measurements"]["explained_variance"] == {
        "non_finite": "Infinity"
    }

    failure_streaks: Counter[str] = Counter()
    unavailable_streaks: Counter[str] = Counter()
    reason = _advance_health_tripwire_streaks(
        report,
        failure_streaks=failure_streaks,
        unavailable_streaks=unavailable_streaks,
        patience=2,
    )
    assert reason is not None and reason.startswith("health-tripwire-immediate:")
    latched_failures = [reason]
    assert (not latched_failures) is False

    path = tmp_path / "health.json"
    _write_json(path, report)
    assert json.loads(path.read_text(encoding="utf-8"))["update"] == 10


def test_health_stop_precedes_simultaneous_parity_success() -> None:
    reason = _evaluation_stop_reason(
        health_stop_reason="health-tripwire-immediate:inactive_head_gradient_inert",
        activation_failure_streak=0,
        log_std_failure_streak=0,
        heuristic_parity_patience=2,
        parity_streak=2,
        parity_plateau_reached=True,
    )

    assert reason == "health-tripwire-immediate:inactive_head_gradient_inert"
