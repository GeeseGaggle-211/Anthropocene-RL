"""TRAIN-only paired action-credit and gradient-clipping audit.

This module is deliberately diagnostic.  It never saves the shadow policies and
never reads DEVELOPMENT or HELDOUT records.  Native requested, accepted, and
realized actions remain environment outcomes; PPO likelihoods continue to use
the sampled requested actions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from anthropocene_rl.datasets.models import Partition, SnapshotRecord
from anthropocene_rl.policies.distributions import evaluate_hybrid
from anthropocene_rl.policies.elder_mlp import ElderActorCritic, ElderActorCriticConfig
from anthropocene_rl.training.overfit import (
    _all_finite,
    _mapping,
    _pick,
    _plain_mapping,
    _raw_observation,
    _requested_matches_sample,
    _restore_rng_state,
    _sequence,
    _write_json,
)
from anthropocene_rl.training.pilot import (
    PILOT_FAMILIES,
    BalancedFamilySampler,
    CampaignDataset,
    PilotConfig,
    _assert_cultivation_scope,
    _make_environment,
    _reset_request,
    _validate_campaign_reset,
)
from anthropocene_rl.training.ppo import (
    PPOConfig,
    PPOTrainer,
    RolloutBatch,
    critic_fit_diagnostics,
)
from anthropocene_rl.training.returns import (
    discounted_interval_reward,
    generalized_advantage_estimate,
    physical_gamma,
)

ACTIVE_HEADS = {"field_operation": 0, "development_budget": 2}
PERTURBATION = 0.10
ACCEPTED_SPAN_FLOOR = 0.01
PHYSICAL_RELATIVE_SPREAD_FLOOR = 0.001


def run_action_credit_audit(
    *,
    checkpoint_directory: str | Path,
    evidence_directory: str | Path,
    bridge_dll: str | Path,
    dotnet: str,
    manifest_path: str | Path,
    blob_root: str | Path,
    normalizer_path: str | Path,
    snapshot_store_config: str | Path,
    dataset_manifest: str | Path | None,
    dataset_store_config: str | Path,
    seeds: tuple[int, ...] = (1701, 1901, 2101),
) -> dict[str, Any]:
    """Run the bounded audit and persist raw per-seed evidence plus a summary."""

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("The action-credit audit requires distinct seeds")
    checkpoint_dir = Path(checkpoint_directory)
    evidence_dir = Path(evidence_directory)
    bridge_path = Path(bridge_dll).resolve()
    dataset = CampaignDataset(
        manifest_path=manifest_path,
        blob_root=blob_root,
        normalizer_path=normalizer_path,
        snapshot_store_config=snapshot_store_config,
        dataset_manifest=dataset_manifest,
        dataset_store_config=dataset_store_config,
    )
    train = dataset.by_partition[Partition.TRAIN]
    reports: list[dict[str, Any]] = []
    for seed in seeds:
        report = _run_seed_audit(
            seed=seed,
            checkpoint_path=checkpoint_dir / f"seed-{seed}.pt",
            dataset=dataset,
            train=train,
            bridge_path=bridge_path,
            dotnet=dotnet,
        )
        reports.append(report)
        _write_json(evidence_dir / f"seed-{seed}.json", report)

    assessment = _campaign_assessment(reports)
    result = {
        "status": assessment["status"],
        "scope": "train-only-paired-action-credit-audit",
        "partitions_read": [Partition.TRAIN.value],
        "heldout_read": False,
        "development_read": False,
        "perturbation": PERTURBATION,
        "thresholds": {
            "request_span_minimum": PERTURBATION,
            "accepted_span_unresponsive_maximum": ACCEPTED_SPAN_FLOOR,
            "physical_relative_spread_unresponsive_maximum": (
                PHYSICAL_RELATIVE_SPREAD_FLOOR
            ),
            "acceptance_failure_f03_fraction": 0.50,
            "credit_failure_agreement_maximum": 0.50,
            "credit_failure_minimum_responsive_observations": 12,
            "clear_pass_acceptance_responsive_fraction": 0.80,
            "clear_pass_directional_agreement_minimum": 0.70,
            "clear_pass_separate_actor_displacement_change_maximum": 0.20,
            "separate_clip_actor_displacement_ratio_minimum": 1.50,
            "separate_clip_post_update_kl_maximum": 0.05,
            "separate_clip_critic_nmse_deterioration_maximum": 0.10,
        },
        "dataset": {
            "manifest_sha256": dataset.manifest_sha256,
            "normalizer_partition_id": dataset.normalizer.manifest.partition_id,
            "train_records": len(train),
        },
        "seeds": reports,
        "assessment": assessment,
    }
    result["evidence_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    _write_json(evidence_dir / "summary.json", result)
    return result


def _run_seed_audit(
    *,
    seed: int,
    checkpoint_path: Path,
    dataset: CampaignDataset,
    train: tuple[SnapshotRecord, ...],
    bridge_path: Path,
    dotnet: str,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _validate_checkpoint(payload, dataset, seed)
    actor_config = ElderActorCriticConfig(**payload["actor_critic_config"])
    ppo_config = PPOConfig(**payload["ppo_config"])
    pilot_config = PilotConfig(**payload["pilot_config"])
    model = ElderActorCritic(actor_config)
    model.load_state_dict(payload["model_state_dict"])
    trainer = PPOTrainer(model, ppo_config)
    trainer.optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state_dict"]))
    _restore_rng_state(payload["rng"])

    sampler = BalancedFamilySampler(train, seed + 1_000_003)
    records = sampler.sample(32)
    family_counts = Counter(record.scenario_family for record in records)
    if family_counts != Counter({family: 4 for family in PILOT_FAMILIES}):
        raise RuntimeError("Audit batch is not exactly balanced by family")

    selection = dataset.selection(records[0])
    env = _make_environment(
        bridge_path=bridge_path,
        dotnet=dotnet,
        selection=selection,
        episode_limit=pilot_config.episode_steps,
    )
    try:
        batch, episodes, contract = _collect_fixed_batch(
            env=env,
            dataset=dataset,
            records=records,
            model=model,
            config=pilot_config,
        )
        credit = _action_credit(model, batch)
        probes = _counterfactual_probes(
            env=env,
            dataset=dataset,
            episodes=episodes,
            model=model,
            config=pilot_config,
            credit=credit,
        )
    finally:
        env.close()

    clipping = _shadow_clipping_control(payload, batch)
    probe_summary = _probe_summary(probes)
    return {
        "seed": seed,
        "checkpoint": {
            "path": str(checkpoint_path),
            "selected_update": int(payload["identities"]["selected_update"]),
            "learner_commit": payload["identities"]["learner_commit"],
            "simulator_commit": payload["identities"]["simulator_commit"],
        },
        "batch": {
            "episodes": len(episodes),
            "transitions": int(batch.observations.shape[0]),
            "family_counts": dict(sorted(family_counts.items())),
            "snapshot_ids": [record.snapshot_id for record in records],
            "contract": contract,
        },
        "probe_summary": probe_summary,
        "shadow_clipping_control": clipping,
        "probes": probes,
    }


def _validate_checkpoint(payload: dict[str, Any], dataset: CampaignDataset, seed: int) -> None:
    identities = payload["identities"]
    if identities["neural_seed"] != seed:
        raise ValueError("Checkpoint neural seed does not match the audit seed")
    if identities["snapshot_manifest_sha256"] != dataset.manifest_sha256:
        raise ValueError("Checkpoint and audit dataset manifests differ")
    if identities["normalizer_partition_id"] != dataset.normalizer.manifest.partition_id:
        raise ValueError("Checkpoint and audit normalizers differ")
    if payload["pilot_config"].get("evaluate_heldout", True):
        raise ValueError("Audit requires a tuning checkpoint that kept HELDOUT sealed")


def _collect_fixed_batch(
    *,
    env: Any,
    dataset: CampaignDataset,
    records: tuple[SnapshotRecord, ...],
    model: ElderActorCritic,
    config: PilotConfig,
) -> tuple[RolloutBatch, list[dict[str, Any]], dict[str, Any]]:
    observations: list[Tensor] = []
    actions: list[Tensor] = []
    movements: list[Tensor] = []
    log_probabilities: list[Tensor] = []
    advantages: list[np.ndarray] = []
    returns: list[np.ndarray] = []
    episodes: list[dict[str, Any]] = []
    replay_failures = 0
    request_mismatches = 0
    scope_failures = 0

    for record in records:
        selection = dataset.selection(record)
        observation, info = env.reset(options=_reset_request(selection, config.episode_steps))
        _validate_campaign_reset(_plain_mapping(info["native_result"]), selection)
        try:
            _assert_cultivation_scope(
                _raw_observation(info), snapshot_id=record.snapshot_id, phase="audit-reset"
            )
        except Exception:
            scope_failures += 1
            raise

        steps: list[dict[str, Any]] = []
        values: list[float] = []
        next_values: list[float] = []
        rewards: list[float] = []
        elapsed_years: list[float] = []
        terminated_flags: list[bool] = []
        truncated_flags: list[bool] = []
        for decision in range(config.episode_steps):
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
            with torch.no_grad():
                sampled = model.actor.sample(observation_tensor.unsqueeze(0))
                value = model.critic(observation_tensor.unsqueeze(0))[0]
            continuous = sampled.continuous[0].detach()
            movement = sampled.movement_index[0].detach()
            old_log_probability = sampled.log_prob[0].detach()
            if not _all_finite(continuous, old_log_probability, value):
                raise RuntimeError("Audit sampled a non-finite policy value")
            movement_value = int(movement)
            if not info["action_mask"].movement_candidates[movement_value]:
                raise RuntimeError("Audit sampled an illegal movement action")
            next_observation, _, terminated, truncated, next_info = env.step(
                {
                    "continuous": continuous.numpy(),
                    "movement_index": movement_value,
                }
            )
            result = _plain_mapping(next_info["native_result"])
            if not _requested_matches_sample(
                result["Requested"], continuous.numpy(), movement_value
            ):
                request_mismatches += 1
                raise RuntimeError("Native requested action differs from the PPO sample")
            if int(_pick(result["Requested"], "MovementCandidate")) != -1:
                raise RuntimeError("Audit escaped fixed-location scope")
            reward = _mapping(result, "Reward")
            elapsed = float(_pick(result, "ElapsedYears"))
            physical_reward = discounted_interval_reward(
                float(_pick(reward, "WelfareRate")), elapsed, config.horizon_years
            )
            scaled_reward = physical_reward * config.reward_scale
            next_tensor = torch.as_tensor(next_observation, dtype=torch.float32)
            with torch.no_grad():
                next_value = 0.0 if terminated else float(
                    model.critic(next_tensor.unsqueeze(0))[0]
                )
            try:
                _assert_cultivation_scope(
                    _raw_observation(next_info),
                    snapshot_id=record.snapshot_id,
                    phase=f"audit-post-step-{decision + 1}",
                )
            except Exception:
                scope_failures += 1
                raise
            steps.append(
                {
                    "decision": decision,
                    "observation": observation_tensor.detach().clone(),
                    "observation_sha256": hashlib.sha256(
                        observation_tensor.numpy().tobytes()
                    ).hexdigest(),
                    "continuous": continuous.detach().clone(),
                    "movement": movement.detach().clone(),
                    "log_probability": old_log_probability.detach().clone(),
                    "value": float(value),
                    "next_value": next_value,
                    "physical_reward": physical_reward,
                    "scaled_reward": scaled_reward,
                    "elapsed_years": elapsed,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "requested": _action_values(_mapping(result, "Requested")),
                    "accepted": _action_values(_mapping(result, "Accepted")),
                    "realized": _action_values(_mapping(result, "Realized")),
                    "interventions": list(_sequence(result, "Interventions")),
                }
            )
            observations.append(observation_tensor.detach())
            actions.append(continuous)
            movements.append(movement)
            log_probabilities.append(old_log_probability)
            values.append(float(value))
            next_values.append(next_value)
            rewards.append(scaled_reward)
            elapsed_years.append(elapsed)
            terminated_flags.append(bool(terminated))
            truncated_flags.append(bool(truncated))
            observation = next_observation
            info = next_info
            if terminated or truncated:
                break

        replay = env.verify_replay()
        if int(replay.get("steps", 0)) != len(steps) or not steps:
            replay_failures += 1
            raise RuntimeError("Baseline audit trajectory failed native replay")
        episode_advantages, episode_returns = generalized_advantage_estimate(
            np.asarray(rewards, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
            np.asarray(next_values, dtype=np.float64),
            np.asarray(elapsed_years, dtype=np.float64),
            np.asarray(terminated_flags, dtype=bool),
            np.asarray(truncated_flags, dtype=bool),
            horizon_years=config.horizon_years,
            lambda_ref=config.lambda_ref,
            lambda_reference_years=config.lambda_reference_years,
        )
        advantages.append(episode_advantages)
        returns.append(episode_returns)
        episodes.append(
            {
                "record": record,
                "selection": selection,
                "steps": steps,
                "advantages": episode_advantages,
                "returns": episode_returns,
            }
        )

    batch = RolloutBatch(
        observations=torch.stack(observations),
        continuous_actions=torch.stack(actions),
        movement_indices=torch.stack(movements).to(torch.long),
        old_log_prob=torch.stack(log_probabilities),
        advantages=torch.as_tensor(np.concatenate(advantages), dtype=torch.float32),
        returns=torch.as_tensor(np.concatenate(returns), dtype=torch.float32),
    )
    batch.validate()
    return batch, episodes, {
        "native_replay_failures": replay_failures,
        "requested_sample_mismatches": request_mismatches,
        "scope_failures": scope_failures,
        "status": "PASS",
    }


def _action_credit(model: ElderActorCritic, batch: RolloutBatch) -> np.ndarray:
    advantages = (batch.advantages - batch.advantages.mean()) / (
        batch.advantages.std(unbiased=False) + 1e-8
    )
    mean, log_std, movement = model.actor.parameters_for(batch.observations)
    evaluated = evaluate_hybrid(
        mean,
        log_std,
        movement,
        batch.observations,
        batch.continuous_actions,
        batch.movement_indices,
    )
    objective = torch.sum(evaluated.log_prob * advantages.detach())
    gradient = torch.autograd.grad(objective, mean)[0]
    return gradient.detach().cpu().numpy()


def _counterfactual_probes(
    *,
    env: Any,
    dataset: CampaignDataset,
    episodes: list[dict[str, Any]],
    model: ElderActorCritic,
    config: PilotConfig,
    credit: np.ndarray,
) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    flat_index = 0
    for episode in episodes:
        steps = episode["steps"]
        for decision, step in enumerate(steps):
            for action_name, head in ACTIVE_HEADS.items():
                baseline = _baseline_outcome(steps, decision, config)
                outcomes: dict[str, dict[str, Any]] = {"baseline": baseline}
                for label, delta in (("minus", -PERTURBATION), ("plus", PERTURBATION)):
                    outcomes[label] = _run_branch(
                        env=env,
                        dataset=dataset,
                        episode=episode,
                        decision=decision,
                        head=head,
                        delta=delta,
                        model=model,
                        config=config,
                    )
                requested = [
                    outcomes[label]["decision_action"]["requested"][action_name]
                    for label in ("minus", "baseline", "plus")
                ]
                accepted = [
                    outcomes[label]["decision_action"]["accepted"][action_name]
                    for label in ("minus", "baseline", "plus")
                ]
                physical = [
                    outcomes[label]["physical_return"]
                    for label in ("minus", "baseline", "plus")
                ]
                combined = [
                    outcomes[label]["combined_scaled_return"]
                    for label in ("minus", "baseline", "plus")
                ]
                requested_span = max(requested) - min(requested)
                accepted_span = max(accepted) - min(accepted)
                physical_spread = max(physical) - min(physical)
                physical_scale = max(max(abs(value) for value in physical), 1e-12)
                physical_relative_spread = physical_spread / physical_scale
                physical_direction = _direction(physical[2] - physical[0])
                combined_direction = _direction(combined[2] - combined[0])
                credit_value = float(credit[flat_index, head])
                credit_direction = _direction(credit_value)
                responsive = physical_relative_spread > PHYSICAL_RELATIVE_SPREAD_FLOOR
                probes.append(
                    {
                        "snapshot_id": episode["record"].snapshot_id,
                        "family": episode["record"].scenario_family,
                        "decision": decision,
                        "observation_sha256": step["observation_sha256"],
                        "action": action_name,
                        "head": head,
                        "policy_credit_gradient": credit_value,
                        "policy_credit_direction": credit_direction,
                        "requested_span": requested_span,
                        "accepted_span": accepted_span,
                        "physical_return_spread": physical_spread,
                        "physical_relative_spread": physical_relative_spread,
                        "physical_direction": physical_direction,
                        "combined_scaled_direction": combined_direction,
                        "physical_reward_responsive": responsive,
                        "credit_agrees_with_physical_direction": (
                            responsive
                            and credit_direction != 0
                            and credit_direction == physical_direction
                        ),
                        "credit_agrees_with_bootstrapped_direction": (
                            credit_direction != 0 and credit_direction == combined_direction
                        ),
                        "physical_and_bootstrapped_directions_agree": (
                            physical_direction != 0
                            and physical_direction == combined_direction
                        ),
                        "outcomes": outcomes,
                    }
                )
            flat_index += 1
    if flat_index != credit.shape[0]:
        raise RuntimeError("Counterfactual probes and fixed batch disagree")
    return probes


def _run_branch(
    *,
    env: Any,
    dataset: CampaignDataset,
    episode: dict[str, Any],
    decision: int,
    head: int,
    delta: float,
    model: ElderActorCritic,
    config: PilotConfig,
) -> dict[str, Any]:
    selection = episode["selection"]
    observation, info = env.reset(options=_reset_request(selection, config.episode_steps))
    _validate_campaign_reset(_plain_mapping(info["native_result"]), selection)
    physical_return = 0.0
    discount = 1.0
    decision_action: dict[str, Any] | None = None
    final_terminated = False
    final_observation = observation
    steps_executed = 0
    for index, baseline_step in enumerate(episode["steps"]):
        if index <= decision and not np.array_equal(
            observation, baseline_step["observation"].numpy()
        ):
            raise RuntimeError("Counterfactual prefix failed deterministic replay")
        continuous = baseline_step["continuous"].numpy().copy()
        if index == decision:
            continuous[head] = np.clip(continuous[head] + delta, 0.0, 1.0)
        movement = int(baseline_step["movement"])
        next_observation, _, terminated, truncated, next_info = env.step(
            {"continuous": continuous, "movement_index": movement}
        )
        result = _plain_mapping(next_info["native_result"])
        if not _requested_matches_sample(result["Requested"], continuous, movement):
            raise RuntimeError("Counterfactual native request differs from the perturbation")
        _assert_cultivation_scope(
            _raw_observation(next_info),
            snapshot_id=episode["record"].snapshot_id,
            phase=f"audit-counterfactual-{decision}-{head}-{index}",
        )
        if index == decision:
            decision_action = {
                "requested": _action_values(_mapping(result, "Requested")),
                "accepted": _action_values(_mapping(result, "Accepted")),
                "realized": _action_values(_mapping(result, "Realized")),
                "interventions": list(_sequence(result, "Interventions")),
            }
        if index >= decision:
            reward = _mapping(result, "Reward")
            elapsed = float(_pick(result, "ElapsedYears"))
            interval = discounted_interval_reward(
                float(_pick(reward, "WelfareRate")), elapsed, config.horizon_years
            )
            physical_return += discount * interval
            discount *= physical_gamma(elapsed, config.horizon_years)
        steps_executed += 1
        final_terminated = bool(terminated)
        final_observation = next_observation
        observation = next_observation
        info = next_info
        if terminated or truncated:
            break
    replay = env.verify_replay()
    if int(replay.get("steps", 0)) != steps_executed:
        raise RuntimeError("Counterfactual branch failed native replay")
    if decision_action is None:
        raise RuntimeError("Counterfactual branch ended before the perturbed decision")
    with torch.no_grad():
        terminal_bootstrap = 0.0 if final_terminated else float(
            model.critic(torch.as_tensor(final_observation).unsqueeze(0))[0]
        )
    return {
        "physical_return": physical_return,
        "scaled_physical_return": physical_return * config.reward_scale,
        "terminal_bootstrap": discount * terminal_bootstrap,
        "combined_scaled_return": (
            physical_return * config.reward_scale + discount * terminal_bootstrap
        ),
        "decision_action": decision_action,
        "native_replay": replay,
    }


def _baseline_outcome(
    steps: list[dict[str, Any]], decision: int, config: PilotConfig
) -> dict[str, Any]:
    physical_return = 0.0
    discount = 1.0
    for step in steps[decision:]:
        physical_return += discount * step["physical_reward"]
        discount *= physical_gamma(step["elapsed_years"], config.horizon_years)
    last = steps[-1]
    terminal_bootstrap = 0.0 if last["terminated"] else discount * last["next_value"]
    step = steps[decision]
    return {
        "physical_return": physical_return,
        "scaled_physical_return": physical_return * config.reward_scale,
        "terminal_bootstrap": terminal_bootstrap,
        "combined_scaled_return": physical_return * config.reward_scale
        + terminal_bootstrap,
        "decision_action": {
            "requested": step["requested"],
            "accepted": step["accepted"],
            "realized": step["realized"],
            "interventions": step["interventions"],
        },
        "native_replay": {"baseline_fixed_batch": True},
    }


def _shadow_clipping_control(
    payload: dict[str, Any], batch: RolloutBatch
) -> dict[str, Any]:
    actor_config = ElderActorCriticConfig(**payload["actor_critic_config"])
    base_config = PPOConfig(**payload["ppo_config"])
    if base_config.gradient_clip_mode != "joint":
        raise ValueError("The source checkpoint must use joint clipping")

    def build(config: PPOConfig) -> tuple[ElderActorCritic, PPOTrainer]:
        model = ElderActorCritic(actor_config)
        model.load_state_dict(payload["model_state_dict"])
        trainer = PPOTrainer(model, config)
        trainer.optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state_dict"]))
        return model, trainer

    joint_model, joint_trainer = build(base_config)
    separate_config = replace(base_config, gradient_clip_mode="separate")
    separate_model, separate_trainer = build(separate_config)
    before = _batch_objectives(joint_model, batch)
    rng = copy.deepcopy(payload["rng"])
    _restore_rng_state(rng)
    joint_metrics = joint_trainer.update(batch)
    _restore_rng_state(rng)
    separate_metrics = separate_trainer.update(batch)
    joint_after = _batch_objectives(joint_model, batch)
    separate_after = _batch_objectives(separate_model, batch)
    actor_ratio = separate_metrics["actor_parameter_update_l2"] / max(
        joint_metrics["actor_parameter_update_l2"], 1e-12
    )
    critic_nmse_ratio = separate_metrics["post_update_critic_normalized_mse"] / max(
        joint_metrics["post_update_critic_normalized_mse"], 1e-12
    )
    return {
        "before": before,
        "joint": {
            "metrics": joint_metrics,
            "after": joint_after,
        },
        "separate": {
            "metrics": separate_metrics,
            "after": separate_after,
        },
        "comparison": {
            "actor_parameter_displacement_ratio": actor_ratio,
            "separate_surrogate_improvement_over_joint": (
                separate_after["clipped_policy_surrogate"]
                - joint_after["clipped_policy_surrogate"]
            ),
            "separate_post_update_kl": separate_after["approx_kl"],
            "critic_normalized_mse_ratio_separate_to_joint": critic_nmse_ratio,
            "joint_clipping_escalation_seed_pass": (
                actor_ratio >= 1.50
                and separate_after["clipped_policy_surrogate"]
                > joint_after["clipped_policy_surrogate"]
                and separate_after["approx_kl"] <= 0.05
                and critic_nmse_ratio <= 1.10
            ),
        },
    }


@torch.no_grad()
def _batch_objectives(model: ElderActorCritic, batch: RolloutBatch) -> dict[str, float]:
    advantages = (batch.advantages - batch.advantages.mean()) / (
        batch.advantages.std(unbiased=False) + 1e-8
    )
    evaluated = model.actor.evaluate_actions(
        batch.observations,
        batch.continuous_actions,
        batch.movement_indices,
    )
    ratio = torch.exp(evaluated.log_prob - batch.old_log_prob)
    clipped = torch.clamp(ratio, 0.8, 1.2)
    surrogate = torch.minimum(ratio * advantages, clipped * advantages).mean()
    log_ratio = evaluated.log_prob - batch.old_log_prob
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    critic = critic_fit_diagnostics(model.critic(batch.observations), batch.returns)
    return {
        "clipped_policy_surrogate": float(surrogate),
        "approx_kl": float(approx_kl),
        "critic_normalized_mse": critic["normalized_mse"],
        "critic_explained_variance": critic["explained_variance"],
    }


def _probe_summary(probes: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for action_name in ACTIVE_HEADS:
        selected = [probe for probe in probes if probe["action"] == action_name]
        responsive = [probe for probe in selected if probe["physical_reward_responsive"]]
        request_eligible = [
            probe for probe in selected if probe["requested_span"] >= PERTURBATION - 1e-9
        ]
        acceptance_responsive = [
            probe for probe in request_eligible if probe["accepted_span"] > ACCEPTED_SPAN_FLOOR
        ]
        result[action_name] = {
            "probes": len(selected),
            "request_span_eligible": len(request_eligible),
            "acceptance_responsive_fraction": _fraction(
                len(acceptance_responsive), len(request_eligible)
            ),
            "physical_reward_responsive": len(responsive),
            "physical_directional_credit_agreement": _fraction(
                sum(probe["credit_agrees_with_physical_direction"] for probe in responsive),
                len(responsive),
            ),
            "bootstrapped_directional_credit_agreement": _fraction(
                sum(
                    probe["credit_agrees_with_bootstrapped_direction"]
                    for probe in selected
                ),
                len(selected),
            ),
            "physical_vs_bootstrapped_direction_agreement": _fraction(
                sum(
                    probe["physical_and_bootstrapped_directions_agree"]
                    for probe in responsive
                ),
                len(responsive),
            ),
        }

    f03 = [
        probe
        for probe in probes
        if probe["family"] == "F03" and probe["action"] == "development_budget"
    ]
    f03_eligible = [
        probe for probe in f03 if probe["requested_span"] >= PERTURBATION - 1e-9
    ]
    f03_acceptance_failures = [
        probe
        for probe in f03_eligible
        if probe["accepted_span"] <= ACCEPTED_SPAN_FLOOR
        and probe["physical_relative_spread"] <= PHYSICAL_RELATIVE_SPREAD_FLOOR
    ]
    result["f03_development"] = {
        "probes": len(f03),
        "request_span_eligible": len(f03_eligible),
        "acceptance_and_physical_unresponsive": len(f03_acceptance_failures),
        "acceptance_and_physical_unresponsive_fraction": _fraction(
            len(f03_acceptance_failures), len(f03_eligible)
        ),
    }
    return result


def _campaign_assessment(reports: list[dict[str, Any]]) -> dict[str, Any]:
    acceptance_failure_seeds: list[int] = []
    credit_failure_seeds: list[int] = []
    clear_pass_seeds: list[int] = []
    clipping_escalation_seeds: list[int] = []
    for report in reports:
        seed = report["seed"]
        development = report["probe_summary"]["development_budget"]
        f03 = report["probe_summary"]["f03_development"]
        clipping = report["shadow_clipping_control"]["comparison"]
        if f03["acceptance_and_physical_unresponsive_fraction"] >= 0.50:
            acceptance_failure_seeds.append(seed)
        if (
            development["physical_reward_responsive"] >= 12
            and development["physical_directional_credit_agreement"] <= 0.50
        ):
            credit_failure_seeds.append(seed)
        if clipping["joint_clipping_escalation_seed_pass"]:
            clipping_escalation_seeds.append(seed)
        if (
            development["acceptance_responsive_fraction"] >= 0.80
            and development["physical_reward_responsive"] >= 12
            and development["physical_directional_credit_agreement"] >= 0.70
            and abs(clipping["actor_parameter_displacement_ratio"] - 1.0) < 0.20
        ):
            clear_pass_seeds.append(seed)

    if len(clipping_escalation_seeds) >= 2:
        status = "JOINT_CLIPPING_ABLATION_JUSTIFIED"
    elif len(acceptance_failure_seeds) >= 2 or len(credit_failure_seeds) >= 2:
        status = "ACTION_CREDIT_FAILURE"
    elif len(clear_pass_seeds) >= 2:
        status = "CLEAR_DIAGNOSTIC_PASS"
    else:
        status = "INCONCLUSIVE"
    return {
        "status": status,
        "acceptance_failure_seeds": acceptance_failure_seeds,
        "credit_failure_seeds": credit_failure_seeds,
        "joint_clipping_escalation_seeds": clipping_escalation_seeds,
        "clear_pass_seeds": clear_pass_seeds,
        "mlp_representational_failure_demonstrated": False,
        "architecture_change_justified": False,
        "heldout_remains_sealed": True,
    }


def _action_values(mapping: dict[str, Any]) -> dict[str, float]:
    return {
        "field_operation": float(_pick(mapping, "RetainedFieldOperation")),
        "herd_retention": float(_pick(mapping, "RetainedHerdFraction")),
        "development_budget": float(_pick(mapping, "DevelopmentBudget")),
        "field_development_share": float(_pick(mapping, "FieldDevelopmentShare")),
    }


def _direction(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan
