"""Bounded native traces, censored targets and protected frozen-actor calibration."""

import hashlib

import numpy as np
import torch

from anthropocene_rl.autonomous_v4.training import market_peer
from anthropocene_rl.autonomous_v5.contracts import SLICES, observation
from anthropocene_rl.autonomous_v5.policy import inputs_with_mode, to_native
from anthropocene_rl.training.returns import (
    discounted_interval_reward,
    generalized_advantage_estimate,
)


def protected_state(trainer):
    """Hash actor/heads/anchors, actor Adam, and both training RNG streams."""
    digest = hashlib.sha256()
    for name, tensor in trainer.model.state_dict().items():
        if name.startswith(("actor.", "heads.", "reference_critic.", "initial_normalized_critic.")):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    for p in trainer.optimizer.param_groups[0]["params"]:
        for key, value in sorted(trainer.optimizer.state.get(p, {}).items()):
            digest.update(key.encode())
            digest.update(
                value.cpu().numpy().tobytes()
                if isinstance(value, torch.Tensor)
                else repr(value).encode()
            )
    digest.update(
        repr(
            sorted((k, v) for k, v in trainer.optimizer.param_groups[0].items() if k != "params")
        ).encode()
    )
    for rng in (trainer.sampler, trainer.shuffle):
        digest.update(rng.get_state().numpy().tobytes())
    digest.update(repr((trainer.episodes, trainer.transitions, trainer.updates)).encode())
    return digest.hexdigest()


def n_step_targets(rewards, elapsed, terminal, tail, horizon, reward_scale):
    if len(rewards) != len(elapsed) or len(rewards) != len(terminal):
        raise ValueError("Return trajectory dimensions differ")
    if horizon <= 0 or reward_scale <= 0 or any(x <= 0 for x in elapsed):
        raise ValueError("Physical return units must be positive")
    if not np.isfinite([*rewards, *elapsed, tail, horizon, reward_scale]).all():
        raise ValueError("Nonfinite physical return")
    target = np.zeros(len(rewards))
    weight = np.zeros(len(rewards))
    zero_tail = np.zeros(len(rewards))
    value = float(tail)
    zeros = 0.0
    coefficient = 1.0
    for i in reversed(range(len(rewards))):
        if terminal[i]:
            value = zeros = coefficient = 0.0
        discount = np.exp(-elapsed[i] / horizon)
        reward = discounted_interval_reward(
            rewards[i] / elapsed[i] / reward_scale, elapsed[i], horizon
        )
        value = reward + discount * value
        zeros = reward + discount * zeros
        coefficient *= discount
        target[i] = value
        zero_tail[i] = zeros
        weight[i] = coefficient
    return dict(
        target=target, zero_tail=zero_tail, doubled_tail=target + weight * tail, tail_weight=weight
    )


def episode(model, transport, bank, entry, generator, config, *, collect=True, verify=False):
    decode, tensor_inputs, convert = observation, inputs_with_mode, to_native
    if config.get("observation_version") == 6:
        from anthropocene_rl.autonomous_v6.contracts import observation as decode
        from anthropocene_rl.autonomous_v6.policy import inputs_with_mode as tensor_inputs
        from anthropocene_rl.autonomous_v6.policy import to_native as convert
    blob = (bank / entry["snapshot"]).read_bytes()
    if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
        raise ValueError("Source snapshot digest mismatch")
    ids = [entry["focal"]] + ([entry["peer"]] if entry["peer"] is not None else [])
    reset = transport.reset_world(blob, ids, config["episode_limit"])
    current = {int(k): decode(v) for k, v in reset["observations"].items()}
    focal = entry["focal"]
    mode = config.get("feature_mode", "full")
    require_core = config.get("require_core_action_faces", True)
    if not isinstance(require_core, bool):
        raise ValueError("Core admission must be boolean")

    def encode(obs):
        if require_core and (
            obs["WorkforceMask"][[2, 3]].any() or obs["ControlMask"][[1, 2]].any()
        ):
            raise ValueError("Unvalidated husbandry/fishing face")
        return tensor_inputs([obs], mode)

    rows = []
    trace = []
    events = dict(movement=0, trades=0, migrants=0, daughters=0, quotes=0)
    for _ in range(config["episode_limit"]):
        before = current[focal]
        x = encode(before)
        with torch.no_grad():
            action, lp, _, value = model(x, generator=generator)
        native = convert(action.wire_action(), before)
        submitted = {focal: native}
        for peer in ids[1:]:
            if peer in current:
                submitted[peer] = market_peer(current[peer])
        response = transport.step_world(submitted)
        result = response["result"]
        current = {int(k): decode(v) for k, v in result["Observations"].items()}
        terminal = focal not in current
        truncated = result["Truncated"] and not terminal
        years = result["ElapsedYears"]
        raw_reward = result["Rewards"][str(focal)]
        h = np.asarray(result["CompletedHistory"][str(focal)])
        with torch.no_grad():
            following = 0.0 if terminal else float(model.value(encode(current[focal])))
        row = dict(
            obs=x,
            action=action,
            logp=lp.detach(),
            value=float(value),
            next_value=following,
            elapsed=years,
            terminal=terminal,
            truncated=truncated,
            reward=discounted_interval_reward(
                raw_reward / years / config["reward_scale"], years, config["horizon_years"]
            ),
        )
        rows.append(row)
        events["movement"] += int(h[SLICES["moved"]][0])
        events["trades"] += int(h[SLICES["settled_trade_count"]][0])
        events["migrants"] += int(h[SLICES["immigrated"]][0] + h[SLICES["emigrated"]][0])
        events["daughters"] += int(h[SLICES["daughters"]][0])
        events["quotes"] += int(native.get("quote") is not None)
        trace.append(
            dict(
                reward=raw_reward,
                elapsed=years,
                terminal=terminal,
                population_start=float(h[SLICES["population_start"]][0]),
                population_end=float(h[SLICES["population_end"]][0]),
                qol=float(h[SLICES["qol"]][0]),
                food_met=float(h[SLICES["food_met"]][0]),
                staffing=result["Staffing"][str(focal)],
                production=h[SLICES["production_output"]].tolist(),
                extraction=h[SLICES["material_extraction"]].tolist(),
                capital_end=float(h[SLICES["ending_field_capital"]][0]),
                animals_end=float(h[SLICES["ending_animals"]][0]),
                requested_workforce=native["workforce"].tolist(),
                requested_controls=native["controls"].tolist(),
                latent_controls=action.controls[0].detach().tolist(),
                control_mask=before["ControlMask"].tolist(),
                quote=native.get("quote"),
                quote_settled_give=float(h[SLICES["settled_give"]][0]),
                quote_settled_want=float(h[SLICES["settled_want"]][0]),
                native_step_seconds=response["nativeStepSeconds"],
            )
        )
        if terminal or truncated:
            break
    if verify:
        transport.verify_replay()
    if collect:
        adv, returns = generalized_advantage_estimate(
            np.asarray([r["reward"] for r in rows], dtype=np.float64),
            np.asarray([r["value"] for r in rows], dtype=np.float64),
            np.asarray([r["next_value"] for r in rows], dtype=np.float64),
            np.asarray([r["elapsed"] for r in rows], dtype=np.float64),
            np.asarray([r["terminal"] for r in rows], dtype=np.bool_),
            np.asarray([r["truncated"] for r in rows], dtype=np.bool_),
            horizon_years=config["horizon_years"],
            lambda_ref=config["lambda_ref"],
        )
        for row, a, ret in zip(rows, adv, returns, strict=True):
            row.update(advantage=float(a), **{"return": float(ret)})
    welfare = sum(r["reward"] for r in trace)
    time = sum(r["elapsed"] for r in trace)
    summary = dict(
        id=entry["id"],
        family=entry["family"],
        world_seed=entry["world_seed"],
        reset_decision=entry.get("decision_index", 0),
        steps=len(rows),
        integrated_welfare=welfare,
        annual_welfare=welfare / time,
        welfare_per_initial_resident=welfare / entry["population"],
        elapsed_years=time,
        average_qol=sum(r["qol"] * r["elapsed"] for r in trace) / time,
        average_population=sum(r["population_start"] * r["elapsed"] for r in trace) / time,
        final_population=trace[-1]["population_end"],
        terminal=terminal,
        events=events,
        trace=trace,
        fingerprint=result["Fingerprint"],
        tail_value=following,
        final_obs=None if terminal else encode(current[focal]),
    )
    return rows, summary


def fit_critic(trainer, training, validation, *, mode, max_epochs=256, seed=711):
    before = protected_state(trainer)
    rng = torch.Generator().manual_seed(seed)
    xt, yt = training
    xv, yv = validation
    xt = {k: v.clone() for k, v in xt.items()}
    xv = {k: v.clone() for k, v in xv.items()}
    if mode == "receipt_none":
        xt["x"][:, 761:] = 0
        xv["x"][:, 761:] = 0
    elif mode != "full":
        raise ValueError("Unknown calibration feature mode")
    parameters = list(trainer.model.critic.parameters())
    constant = float(yt.mean())
    constant_mse = float((yv - constant).square().mean())
    scale = max(float(yv.std(unbiased=False)), 2.0)

    def metrics():
        with torch.no_grad():
            e = trainer.model.value(xv) - yv
            return dict(
                mse=float(e.square().mean()),
                normalized_mse=float(e.square().mean()) / max(constant_mse, 1.0),
                normalized_bias=float(e.mean()) / scale,
            )

    entry = metrics()
    records = []
    steps = 0
    passed = False
    for epoch in range(max_epochs):
        for ix in torch.randperm(len(yt), generator=rng).split(32):
            loss = (
                0.5
                * (trainer.model.value({k: v[ix] for k, v in xt.items()}) - yt[ix]).square().mean()
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite native-target critic loss")
            trainer.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 0.5, error_if_nonfinite=True)
            trainer.optimizer.step()
            steps += 1
        if (epoch + 1) % 32 == 0:
            value = metrics()
            passed = critic_ready(value, entry, constant_mse)
            records.append(dict(epoch=epoch + 1, gradient_norm=float(norm), **value, ready=passed))
            if passed:
                break
    after = protected_state(trainer)
    if before != after:
        raise ValueError("Critic calibration changed protected actor/anchor/optimizer/RNG state")
    return dict(
        passed=passed,
        entry=entry,
        final=metrics(),
        constant_mse=constant_mse,
        epochs=epoch + 1,
        optimizer_steps=steps,
        trace=records,
        protected_state_before=before,
        protected_state_after=after,
        target_semantics="censored n-step plus frozen tail",
        readiness_rule="MSE <= .95 * TRAIN-constant validation MSE; bias <= .5; nonworse entry",
        degenerate_constant=constant_mse < 1e-8,
    )


def critic_ready(value, entry, constant_mse):
    """Reporting floors must never allow a critic worse than the constant to pass."""
    return (
        constant_mse >= 1e-8
        and value["mse"] <= 0.95 * constant_mse
        and abs(value["normalized_bias"]) <= 0.5
        and value["mse"] <= entry["mse"]
    )
