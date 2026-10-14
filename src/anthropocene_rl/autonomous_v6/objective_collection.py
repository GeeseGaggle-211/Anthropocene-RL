"""Owner-only reward context; reuse the exact native episode/market chronology."""

import numpy as np
import torch

from anthropocene_rl.autonomous_v5.causal import episode as native_episode
from anthropocene_rl.autonomous_v5.contracts import SLICES
from anthropocene_rl.autonomous_v6.reward_objectives import EpisodeObjectiveState
from anthropocene_rl.training.returns import generalized_advantage_estimate


def context(observations, state, population):
    n = len(observations["x"])
    result = dict(observations)
    population = torch.as_tensor(population, dtype=torch.float64).expand(n).clone()
    result.update(
        InitialPopulation=torch.full((n,), state.initial_population, dtype=torch.float64),
        OwnPopulation=population,
        ValueFactor=torch.full((n,), state.objective.value_factor(state.initial_population)),
        ValuePotential=torch.tensor([state.objective.potential(float(p)) for p in population]),
    )
    return result


def targets(rows, tail, horizon):
    """Rows already contain integrated/discounted learner reward, including potential."""
    ys = {k: np.zeros(len(rows)) for k in ["target", "zero_tail", "doubled_tail", "tail_weight"]}
    value, zero, weight = float(tail), 0.0, 1.0
    for i in reversed(range(len(rows))):
        r = rows[i]
        if r["terminal"]:
            value = zero = weight = 0.0
        gamma = np.exp(-r["elapsed"] / horizon)
        value = r["reward"] + gamma * value
        zero = r["reward"] + gamma * zero
        weight *= gamma
        ys["target"][i], ys["zero_tail"][i], ys["tail_weight"][i] = value, zero, weight
        ys["doubled_tail"][i] = value + weight * tail
    return ys


def assign_learning_rewards(rows, summary, state):
    if summary["trace"][0]["population_start"] != state.initial_population:
        raise ValueError("Immutable reset population differs from native opening receipt")
    for row, receipt in zip(rows, summary["trace"], strict=True):
        learning = state.objective.interval(receipt, state.initial_population)
        row["reward"] = learning["reward"]
        receipt["learning"] = learning
        row["obs"] = context(row["obs"], state, receipt["population_start"])
    if summary["final_obs"] is not None:
        summary["final_obs"] = context(summary["final_obs"], state, summary["final_population"])
    summary["objective_episode_state"] = state.wire()
    summary["training_welfare"] = sum(
        r["learning"]["raw_training_welfare"] for r in summary["trace"]
    )
    summary["discounted_learning_reward_sum"] = sum(r["reward"] for r in rows)


def assign_gae(rows, config):
    adv, ret = generalized_advantage_estimate(
        np.asarray([r["reward"] for r in rows]),
        np.asarray([r["value"] for r in rows]),
        np.asarray([r["next_value"] for r in rows]),
        np.asarray([r["elapsed"] for r in rows]),
        np.asarray([r["terminal"] for r in rows]),
        np.asarray([r["truncated"] for r in rows]),
        horizon_years=config["horizon_years"],
        lambda_ref=config["lambda_ref"],
    )
    for row, a, y in zip(rows, adv, ret, strict=True):
        row.update(advantage=float(a), **{"return": float(y)})


class EpisodeView:
    def __init__(self, model, state):
        self.model, self.state = model, state
        self.population = state.initial_population

    def __call__(self, obs, *args, **kwargs):
        return self.model(context(obs, self.state, self.population), *args, **kwargs)

    def value(self, obs):
        return self.model.value(context(obs, self.state, self.population))


class OwnerReceiptTransport:
    """Use exact completed owner counts, never invert the MLP's log-compressed input."""

    def __init__(self, transport, view, focal):
        self.transport, self.view, self.focal = transport, view, focal

    def __getattr__(self, name):
        return getattr(self.transport, name)

    def capture(self, response, reset=False):
        if reset:
            # Native Context stores integer resident counts as float; reject beyond
            # its exact integer domain. Subsequent counts use the authoritative
            # double-encoded integer CompletedHistory receipt.
            p = response["observations"][str(self.focal)]["Context"][0]
            if p != self.view.state.initial_population or p > 2**24 or p != int(p):
                raise ValueError("Source snapshot/reset population differs")
            self.view.population = p
        elif "result" in response and "CompletedHistory" in response["result"]:
            h = response["result"]["CompletedHistory"][str(self.focal)]
            self.view.population = h[SLICES["population_end"]][0]
        return response

    def reset_world(self, *args, **kwargs):
        return self.capture(self.transport.reset_world(*args, **kwargs), reset=True)

    def step_world(self, *args, **kwargs):
        return self.capture(self.transport.step_world(*args, **kwargs))

    def _request(self, payload):
        return self.capture(self.transport._request(payload))


def objective_episode(
    model,
    transport,
    bank,
    entry,
    generator,
    config,
    *,
    collect=True,
    verify=False,
    peer_policy=None,
):
    from anthropocene_rl.autonomous_v6.reward_objectives import Objective

    state = EpisodeObjectiveState(
        entry["population"], Objective.from_wire(config["training_objective"])
    )
    view = EpisodeView(model, state)
    wrapped = OwnerReceiptTransport(transport, view, entry["focal"])
    rows, summary = native_episode(
        view,
        wrapped,
        bank,
        entry,
        generator,
        config,
        collect=False,
        verify=verify,
        peer_policy=peer_policy,
    )
    assign_learning_rewards(rows, summary, state)
    if collect:
        assign_gae(rows, config)
    return rows, summary


@torch.no_grad()
def revalue_episode(model, rows, summary, state, config):
    """Objective-specific frozen-policy calibration using shared physical trajectories."""
    assign_learning_rewards(rows, summary, state)
    x = {k: torch.cat([r["obs"][k] for r in rows]) for k in rows[0]["obs"]}
    values = model.value(x).tolist()
    tail = 0.0 if summary["terminal"] else float(model.value(summary["final_obs"]))
    for i, row in enumerate(rows):
        row["value"] = values[i]
        row["next_value"] = values[i + 1] if i + 1 < len(values) else tail
    summary["tail_value"] = tail
    assign_gae(rows, config)
    return rows, summary
