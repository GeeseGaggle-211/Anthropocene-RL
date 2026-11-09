"""Head tempering: a deterministic, recorded exploration transform of the actor's final layers.

The PPO pilot's start actor has collapsed non-labor heads (iteration-0 rollout, 580 rows):
fission logits ~[+9.0, -14.5, -14.7] (P(non-neutral) ~ 4e-10), trade-share latent mean
~ -9.2 (export share sigmoid(-9.2) ~ 1e-4). Entropy bonuses cannot reopen heads at those
gaps (the gradients vanish), so this command rescales/shifts the final layers of a COPY of
a trainer checkpoint, chosen from targets measured on a reference rollout:

* ``--fission-explore P`` / ``--fission-temperature T``: ``fission.weight`` and
  ``fission.bias`` scaled by 1/T, so every fission logit is divided by T (exactly up to
  float32 rounding). With a target, T >= 1 is found by bisection on log T so that the mean
  over the reference rows of P(non-neutral) = 1 - softmax(logits/T)[0] equals P.
* ``--trade-share-explore s``: the export labour share on the wire is
  ``tradeShare = sigmoid(z)``, z ~ N(m, exp(clamp(l, -5, 1))^2), where (m, l) =
  ``trade_share(hidden)`` = (W0 h + b0, W1 h + b1) (``labor_model.sample_nonlabor_policy``
  and ``schema12_initial_action``). Shifting ``trade_share.bias[0]`` by d moves every row's
  m by exactly d; d is chosen (bisection; the expected share is increasing in d) so that
  the mean over the reference rows of E[sigmoid(m_i + d + sigma_i Z)] (Gauss-Hermite, 64
  nodes) equals s, with sigma_i after any log-std change below.
* ``--trade-share-log-std L``: shifts ``trade_share.bias[1]`` so the mean over the
  reference rows of the clamped log std equals L (applied before the share shift).
* ``--trade-gate-explore P`` / ``--trade-gate-temperature T`` (default off):
  ``trade_token.weight[2:]``/``bias[2:]`` (the 3 gate logits) scaled by 1/T; the target is
  the mean over rows and the 7 groups of P(gate != make).
* ``--trade-value-log-std L`` (default off): shifts ``trade_token.bias[1]`` so the mean
  clamped trade-value log std over rows and groups equals L.

The reference outputs are RECOMPUTED by running the checkpoint's own actor over the
reference rollout's stored ``initial_tensors`` (forward hooks capture the pre-clamp
``fission``/``trade_token``/``trade_share`` layer outputs); the rollout's stored
``initial_actor_outputs`` belong to whichever actor collected it and are only used as a
cross-check: when the checkpoint's behaviour digest equals the rollout's, recomputed and
stored outputs must agree (else refused). The rollout's normalization binding must equal the
checkpoint's (stored tensors are normalized inputs).

Every other parameter is left bitwise unchanged (checked). The actor optimizer's Adam
state entries of the changed parameters are removed (their moments were accumulated for the
pre-transform parameterisation; a rescaled weight with old second moments would take steps
of the wrong size), so those parameters restart Adam from step 0; all other entries and the
saved layout are kept bitwise. The warm-up factor (``labor_actor_optim``) reads the largest
step over all parameters and is therefore unchanged. A ``headTempering`` list entry records
the inputs, targets, achieved values, the reference rollout sha256, the source actor and
rollout-collector digests, and digests of the changed parameters before/after. The result
passes the trainer-checkpoint loader and is loadable by ``labor_training collect``/``train``
(the rollout used for training must be collected from the tempered checkpoint).

Usage::

    python -m anthropocene_rl.autonomous_v7.labor_head_temper --checkpoint start.pt \
        --reference-rollout rollout.pt --fission-explore 0.02 --trade-share-explore 0.1 \
        --out start-tempered.pt [--report temper.json]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import contracts

TEMPER_SCHEMA = "labor-head-tempering-v1"
TEMPER_KEY = "headTempering"
REFERENCE_CHUNK = 64
HERMITE_NODES = 64
TARGET_TOLERANCE = 1e-3
STORED_OUTPUT_ATOL = 1e-4
STORED_OUTPUT_RTOL = 1e-4
MAX_TEMPERATURE = 1e6
LOG_STD_BOUNDS = (-5.0, 1.0)  # LaborActor.forward's clamp


def tensor_digest(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(f"{tuple(data.shape)}|{data.dtype}".encode())
    h.update(data.numpy().tobytes())
    return h.hexdigest()


# --- reference outputs ---------------------------------------------------------------------


def reference_head_outputs(actor: torch.nn.Module, tensors: Mapping[str, torch.Tensor],
                           rows: int, chunk: int = REFERENCE_CHUNK) -> dict[str, torch.Tensor]:
    """Run ``actor`` over the stored initial tensors (no_grad, fixed chunks) and return the
    raw (pre-clamp) final-layer outputs: ``fission`` [N,3], ``trade_token`` [N,7,5] and
    ``trade_share`` [N,2], plus the forward's own outputs for the cross-check."""
    from .labor_ppo import _rows

    captured: dict[str, list[torch.Tensor]] = {"fission": [], "trade_token": [], "trade_share": []}
    outputs: dict[str, list[torch.Tensor]] = {}
    keys = ("fission", "trade_gate", "trade_share_mean", "trade_share_log_std",
            "trade_value_mean", "trade_value_log_std")

    def hook(name: str) -> Callable[..., None]:
        def record(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            captured[name].append(output.detach().clone())
        return record

    handles = [getattr(actor, name).register_forward_hook(hook(name)) for name in captured]
    try:
        with torch.no_grad():
            for start in range(0, rows, chunk):
                index = torch.arange(start, min(rows, start + chunk), dtype=torch.long)
                output = actor(_rows(tensors, index))
                for key in keys:
                    outputs.setdefault(key, []).append(output[key].detach().clone())
    finally:
        for handle in handles:
            handle.remove()
    result = {name: torch.cat(values) for name, values in captured.items()}
    result.update({f"output:{key}": torch.cat(values) for key, values in outputs.items()})
    if any(len(value) != rows for value in result.values()):
        raise AssertionError("reference outputs are not row-aligned with the rollout")
    return result


def stored_output_check(reference: Mapping[str, torch.Tensor],
                        stored: Mapping[str, torch.Tensor] | None) -> dict[str, Any]:
    """Compare recomputed head outputs with a rollout's stored ``initial_actor_outputs``."""
    if stored is None:
        return {"available": False}
    diffs: dict[str, float] = {}
    for key in ("fission", "trade_gate", "trade_share_mean", "trade_share_log_std",
                "trade_value_mean", "trade_value_log_std"):
        if key not in stored:
            continue
        mine = reference[f"output:{key}"].double()
        theirs = stored[key].double().reshape(mine.shape)
        diffs[key] = float((mine - theirs).abs().max()) if mine.numel() else 0.0
    close = all(
        torch.allclose(reference[f"output:{key}"].double(),
                       stored[key].double().reshape(reference[f"output:{key}"].shape),
                       rtol=STORED_OUTPUT_RTOL, atol=STORED_OUTPUT_ATOL)
        for key in diffs
    )
    return {"available": True, "maxAbsDiff": diffs, "match": bool(close),
            "tolerance": {"rtol": STORED_OUTPUT_RTOL, "atol": STORED_OUTPUT_ATOL}}


# --- targets -------------------------------------------------------------------------------


def _bisect(fn: Callable[[float], float], target: float, low: float, high: float,
            *, iterations: int = 200) -> float:
    """x in [low, high] with fn(x) = target for increasing fn (bracket checked)."""
    f_low, f_high = fn(low), fn(high)
    if not f_low <= target <= f_high:
        raise ValueError(
            f"target {target} is not bracketed: f({low})={f_low}, f({high})={f_high}"
        )
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        if fn(mid) < target:
            low = mid
        else:
            high = mid
        if high - low <= 1e-12 * max(1.0, abs(mid)):
            break
    return 0.5 * (low + high)


def categorical_mean_probabilities(logits: torch.Tensor, temperature: float) -> np.ndarray:
    """Mean over every leading row of softmax(logits / T), float64."""
    flat = logits.double().reshape(-1, logits.shape[-1])
    return torch.softmax(flat / float(temperature), dim=-1).mean(dim=0).numpy()


def solve_temperature(logits: torch.Tensor, keep: int, target: float) -> float:
    """T >= 1 with mean P(option != keep) under logits/T equal to ``target``."""
    if not 0.0 < target < 1.0:
        raise ValueError("an exploration target must be in (0, 1)")

    def explore(log_t: float) -> float:
        return 1.0 - float(categorical_mean_probabilities(logits, math.exp(log_t))[keep])

    if explore(0.0) >= target:
        raise ValueError(
            f"the head already explores {explore(0.0):.3g} >= target {target} at T = 1; "
            "tempering only raises temperature"
        )
    return math.exp(_bisect(explore, target, 0.0, math.log(MAX_TEMPERATURE)))


def expected_share(mean: np.ndarray, log_std: np.ndarray) -> float:
    """Mean over rows of E[sigmoid(mean + exp(log_std) Z)], Z ~ N(0, 1) (Gauss-Hermite)."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(HERMITE_NODES)
    weights = weights / math.sqrt(2.0 * math.pi)
    z = mean[:, None] + np.exp(log_std)[:, None] * nodes[None, :]
    return float(((1.0 / (1.0 + np.exp(-z))) * weights[None, :]).sum(axis=1).mean())


def _clamped(values: np.ndarray) -> np.ndarray:
    return np.clip(values, *LOG_STD_BOUNDS)


def solve_log_std_shift(raw: np.ndarray, target: float) -> float:
    """d with mean(clamp(raw + d)) = target (the forward's clamp applies after the shift)."""
    low, high = LOG_STD_BOUNDS
    if not low < target < high:
        raise ValueError(f"a log-std target must be inside the clamp ({low}, {high})")
    span = float(np.max(np.abs(raw))) + (high - low) + 1.0
    return _bisect(lambda d: float(_clamped(raw + d).mean()), target, -span, span)


def solve_share_shift(mean: np.ndarray, log_std: np.ndarray, target: float) -> float:
    if not 0.0 < target < 1.0:
        raise ValueError("a trade-share target must be in (0, 1)")
    span = float(np.max(np.abs(mean))) + 60.0
    return _bisect(lambda d: expected_share(mean + d, log_std), target, -span, span)


# --- optimizer ------------------------------------------------------------------------------


def _saved_index_names(actor: torch.nn.Module, saved: Mapping[str, Any]) -> dict[int, str]:
    """The saved actor Adam state's parameter index -> name (legacy or two-group layout)."""
    natural = [name for name, _ in actor.named_parameters()]
    groups = list(saved["param_groups"])
    if len(groups) == 1:
        names = groups[0].get("param_names") or natural
        if list(names) != natural or len(groups[0]["params"]) != len(natural):
            raise ValueError("legacy actor optimizer state does not match the actor")
        return dict(zip(groups[0]["params"], natural, strict=True))
    out: dict[int, str] = {}
    for group in groups:
        names = group.get("param_names")
        if names is None or len(names) != len(group["params"]):
            raise ValueError("two-group actor optimizer state lacks parameter names")
        out.update(zip(group["params"], names, strict=True))
    if set(out.values()) != set(natural):
        raise ValueError("actor optimizer state does not cover the actor's parameters")
    return out


def reset_adam_entries(actor: torch.nn.Module, saved: Mapping[str, Any],
                       changed: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``saved`` with the Adam state entries of ``changed`` parameters removed (layout and
    every other entry kept); returns it and a record."""
    index_names = _saved_index_names(actor, saved)
    out = copy.deepcopy(dict(saved))
    out["state"] = dict(out["state"])
    removed = {}
    for index, name in index_names.items():
        if name in changed and index in out["state"]:
            entry = out["state"].pop(index)
            step = entry.get("step")
            removed[name] = None if step is None else int(float(step))
    return out, {
        "changedParameters": list(changed),
        "removedAdamEntries": removed,
        "untouched": [name for name in changed if name not in removed],
        "reason": "moments of the pre-transform parameterisation; those parameters restart "
                  "Adam at step 0; every other entry and the layout are kept bitwise",
    }


# --- transform -----------------------------------------------------------------------------


def temper_actor_state(
    actor_state: Mapping[str, torch.Tensor],
    *,
    fission_temperature: float | None = None,
    gate_temperature: float | None = None,
    share_mean_shift: float | None = None,
    share_log_std_shift: float | None = None,
    value_log_std_shift: float | None = None,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """The tempered actor state dict and the names of the changed tensors."""
    out = {name: value.detach().clone() for name, value in actor_state.items()}
    changed: list[str] = []

    def scale(name: str, factor: float, rows: slice | None = None) -> None:
        value = out[name]
        if rows is None:
            out[name] = (value.double() / factor).to(value.dtype)
        else:
            updated = value.clone()
            updated[rows] = (value[rows].double() / factor).to(value.dtype)
            out[name] = updated
        changed.append(name)

    def shift(name: str, index: int, delta: float) -> None:
        value = out[name].clone()
        value[index] = (value[index].double() + delta).to(value.dtype)
        out[name] = value
        if name not in changed:
            changed.append(name)

    for value, label in ((fission_temperature, "fission"), (gate_temperature, "trade gate")):
        if value is not None and not (math.isfinite(value) and value > 0):
            raise ValueError(f"{label} temperature must be finite and positive")
    if fission_temperature is not None:
        scale("fission.weight", fission_temperature)
        scale("fission.bias", fission_temperature)
    gate = slice(2, 2 + contracts.TRADE_GATES)
    if gate_temperature is not None:
        scale("trade_token.weight", gate_temperature, gate)
        scale("trade_token.bias", gate_temperature, gate)
    if value_log_std_shift is not None:
        shift("trade_token.bias", 1, value_log_std_shift)
    if share_log_std_shift is not None:
        shift("trade_share.bias", 1, share_log_std_shift)
    if share_mean_shift is not None:
        shift("trade_share.bias", 0, share_mean_shift)
    return out, changed


def _head_measures(reference: Mapping[str, torch.Tensor], *, fission_t: float = 1.0,
                   gate_t: float = 1.0) -> dict[str, Any]:
    token = reference["trade_token"].double()
    share = reference["trade_share"].double()
    share_mean = share[:, 0].numpy()
    share_log_std = _clamped(share[:, 1].numpy())
    fission = categorical_mean_probabilities(reference["fission"], fission_t)
    gate = categorical_mean_probabilities(token[..., 2:], gate_t)
    return {
        "fissionMeanProbabilities": {
            name: float(fission[k]) for k, name in enumerate(("neutral", "discourage", "encourage"))
        },
        "fissionNonNeutral": float(1.0 - fission[0]),
        "tradeGateMeanProbabilities": {
            name: float(gate[k]) for k, name in enumerate(("withhold", "stock", "make"))
        },
        "tradeGateNonMake": float(1.0 - gate[contracts.TRADE_MAKE]),
        "tradeShareLatentMean": float(share_mean.mean()),
        "tradeShareLogStdMean": float(share_log_std.mean()),
        "tradeShareAtLatentMean": float((1.0 / (1.0 + np.exp(-share_mean))).mean()),
        "tradeShareExpected": expected_share(share_mean, share_log_std),
        "tradeValueLogStdMean": float(_clamped(token[..., 1].numpy()).mean()),
    }


def temper_checkpoint(
    state: Mapping[str, Any],
    rollout_path: Path,
    *,
    fission_explore: float | None = None,
    fission_temperature: float | None = None,
    share_explore: float | None = None,
    share_log_std: float | None = None,
    gate_explore: float | None = None,
    gate_temperature: float | None = None,
    value_log_std: float | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (tempered trainer checkpoint, record)."""
    from .labor_model import labor_checkpoint_personality
    from .labor_training import (
        _load_rollout,
        _models_from_checkpoint,
        _normalization_digest,
        _require_trainer_checkpoint,
        labor_behavior_configuration,
    )
    from .provenance import checkpoint_hash
    from .value_calibration import behavior_policy_digest, policy_digest

    if fission_explore is not None and fission_temperature is not None:
        raise ValueError("--fission-explore and --fission-temperature are exclusive")
    if gate_explore is not None and gate_temperature is not None:
        raise ValueError("--trade-gate-explore and --trade-gate-temperature are exclusive")
    if all(v is None for v in (fission_explore, fission_temperature, share_explore, share_log_std,
                               gate_explore, gate_temperature, value_log_std)):
        raise ValueError("no tempering requested")
    _require_trainer_checkpoint(state)
    _, actor, task, costs, normalizer = _models_from_checkpoint(state)
    behavior = labor_behavior_configuration(labor_checkpoint_personality(state))
    source_digest = behavior_policy_digest(actor, behavior)
    batch, _, _, rollout_metadata = _load_rollout(rollout_path)
    if batch.normalization_binding_digest != _normalization_digest(normalizer):
        raise ValueError(
            "the reference rollout's normalization binding differs from the checkpoint's: its "
            "stored (normalized) tensors cannot be re-evaluated by this actor"
        )
    reference = reference_head_outputs(actor, batch.initial_tensors, batch.parent_count)
    collector = batch.behavior_policy_digest == source_digest
    cross = stored_output_check(reference, batch.initial_actor_outputs)
    cross["checkpointIsCollector"] = bool(collector)
    if collector and cross["available"] and not cross["match"]:
        raise ValueError(
            f"recomputed head outputs differ from the collector's stored outputs: {cross}"
        )

    token = reference["trade_token"].double()
    share = reference["trade_share"].double()
    targets: dict[str, Any] = {}
    fission_t = fission_temperature
    if fission_explore is not None:
        fission_t = solve_temperature(reference["fission"], 0, fission_explore)
        targets["fissionNonNeutral"] = float(fission_explore)
    gate_t = gate_temperature
    if gate_explore is not None:
        gate_t = solve_temperature(token[..., 2:], contracts.TRADE_MAKE, gate_explore)
        targets["tradeGateNonMake"] = float(gate_explore)
    value_shift = None
    if value_log_std is not None:
        value_shift = solve_log_std_shift(token[..., 1].reshape(-1).numpy(), value_log_std)
        targets["tradeValueLogStdMean"] = float(value_log_std)
    std_shift = None
    raw_log_std = share[:, 1].numpy()
    if share_log_std is not None:
        std_shift = solve_log_std_shift(raw_log_std, share_log_std)
        targets["tradeShareLogStdMean"] = float(share_log_std)
    effective_log_std = _clamped(raw_log_std + (std_shift or 0.0))
    mean_shift = None
    if share_explore is not None:
        mean_shift = solve_share_shift(share[:, 0].numpy(), effective_log_std, share_explore)
        targets["tradeShareExpected"] = float(share_explore)

    before_actor = {name: value.detach().clone() for name, value in state["actor"].items()}
    new_actor, changed = temper_actor_state(
        before_actor, fission_temperature=fission_t, gate_temperature=gate_t,
        share_mean_shift=mean_shift, share_log_std_shift=std_shift,
        value_log_std_shift=value_shift,
    )
    for name, value in before_actor.items():
        if name not in changed and not torch.equal(value, new_actor[name]):
            raise AssertionError(f"tempering changed an untargeted parameter {name}")
    tempered = dict(state)
    tempered["actor"] = new_actor
    optimizer_state, optimizer_record = reset_adam_entries(actor, state["actor_optimizer"], changed)
    tempered["actor_optimizer"] = optimizer_state
    _, tempered_actor, tempered_task, tempered_costs, tempered_normalizer = (
        _models_from_checkpoint(tempered)
    )
    after_reference = reference_head_outputs(
        tempered_actor, batch.initial_tensors, batch.parent_count
    )
    measured_before = _head_measures(reference)
    predicted = _head_measures(reference, fission_t=fission_t or 1.0, gate_t=gate_t or 1.0)
    achieved = _head_measures(after_reference)
    for key, target in targets.items():
        if abs(achieved[key] - target) > TARGET_TOLERANCE:
            raise AssertionError(f"tempered {key} {achieved[key]} misses its target {target}")
    from .labor_value_calibration import labor_critic_digest

    if labor_critic_digest(tempered_task, tempered_costs) != labor_critic_digest(task, costs):
        raise AssertionError("tempering changed the critics")
    if _normalization_digest(tempered_normalizer) != _normalization_digest(normalizer):
        raise AssertionError("tempering changed the normalizer")
    record = {
        "schema": TEMPER_SCHEMA,
        "inputs": {
            "fissionExplore": fission_explore,
            "fissionTemperature": fission_temperature,
            "tradeShareExplore": share_explore,
            "tradeShareLogStd": share_log_std,
            "tradeGateExplore": gate_explore,
            "tradeGateTemperature": gate_temperature,
            "tradeValueLogStd": value_log_std,
        },
        "targets": targets,
        "transform": {
            "fissionTemperature": fission_t,
            "tradeGateTemperature": gate_t,
            "tradeShareMeanBiasShift": mean_shift,
            "tradeShareLogStdBiasShift": std_shift,
            "tradeValueLogStdBiasShift": value_shift,
            "tradeShareMapping": "tradeShare = sigmoid(z), z ~ N(W0 h + b0, exp(clamp(W1 h + b1,"
                                 " -5, 1))^2); target = row mean of E[sigmoid(z)]",
        },
        "referenceRollout": {
            "sha256": checkpoint_hash(rollout_path),
            "name": Path(rollout_path).name,
            "rows": int(batch.parent_count),
            "collectorBehaviorPolicyDigest": batch.behavior_policy_digest,
            "collectorCheckpointSha256": rollout_metadata.get("sourceCheckpointSha256"),
            "worldSeeds": rollout_metadata.get("worldSeeds"),
            "outputs": "recomputed by this checkpoint's actor over the stored initial_tensors",
            "storedOutputCrossCheck": cross,
        },
        "sourceActor": {
            "behaviorPolicyDigest": source_digest,
            "actorTensorDigest": policy_digest(actor),
        },
        "temperedActor": {
            "behaviorPolicyDigest": behavior_policy_digest(tempered_actor, behavior),
            "actorTensorDigest": policy_digest(tempered_actor),
        },
        "measuredBefore": measured_before,
        "predictedAfter": predicted,
        "achievedAfter": achieved,
        "changedParameters": {
            name: {"before": tensor_digest(before_actor[name]),
                   "after": tensor_digest(new_actor[name])}
            for name in changed
        },
        "unchangedParameters": len(before_actor) - len(changed),
        "actorOptimizer": optimizer_record,
        **({} if provenance is None else {"provenance": dict(provenance)}),
    }
    tempered[TEMPER_KEY] = [*list(state.get(TEMPER_KEY, []) or []), record]
    _require_trainer_checkpoint(tempered)
    verify_tempered(tempered, state)
    return tempered, record


def verify_tempered(tempered: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    """The loaders collect/train use, plus bitwise identity of everything not tempered."""
    from .labor_actor_optim import ActorLrConfig, build_actor_optimizer, load_actor_optimizer_state
    from .labor_compose_checkpoint import _equal
    from .labor_training import _models_from_checkpoint, _require_trainer_checkpoint

    _require_trainer_checkpoint(tempered)
    _, actor, *_ = _models_from_checkpoint(tempered)
    record = tempered[TEMPER_KEY][-1]
    changed = set(record["changedParameters"])
    for name, value in base["actor"].items():
        if name not in changed and not _equal(value, tempered["actor"][name]):
            raise AssertionError(f"tempered checkpoint changed untargeted actor tensor {name}")
    unchanged = [k for k in base if k not in ("actor", "actor_optimizer", TEMPER_KEY)]
    differing = [k for k in unchanged if not _equal(base[k], tempered[k])]
    if differing or set(tempered) - set(base) - {TEMPER_KEY}:
        raise AssertionError(f"tempering changed checkpoint fields {differing}")
    lr = ActorLrConfig(3e-4, 3e-4)
    load_actor_optimizer_state(build_actor_optimizer(actor, lr), actor,
                               tempered["actor_optimizer"], lr)  # train's load
    sampling = torch.Generator()
    sampling.set_state(tempered["sampling_generator_state"])  # collect's
    return {"ok": True}


def main(argv: Sequence[str] | None = None) -> int:
    from .provenance import checkpoint_hash

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="trainer checkpoint")
    parser.add_argument("--reference-rollout", type=Path, required=True,
                        help="labor_training collect rollout (its stored initial_tensors)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--fission-explore", type=float, default=None,
                        help="target mean P(non-neutral fission) over the reference rows")
    parser.add_argument("--fission-temperature", type=float, default=None)
    parser.add_argument("--trade-share-explore", type=float, default=None,
                        help="target mean expected export share over the reference rows")
    parser.add_argument("--trade-share-log-std", type=float, default=None,
                        help="target mean (clamped) trade-share latent log std")
    parser.add_argument("--trade-gate-explore", type=float, default=None,
                        help="target mean P(gate != make) over rows and groups (default off)")
    parser.add_argument("--trade-gate-temperature", type=float, default=None)
    parser.add_argument("--trade-value-log-std", type=float, default=None,
                        help="target mean (clamped) trade-value log std (default off)")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError(f"{args.out} exists; tempering never overwrites")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tempered, record = temper_checkpoint(
        state, args.reference_rollout,
        fission_explore=args.fission_explore, fission_temperature=args.fission_temperature,
        share_explore=args.trade_share_explore, share_log_std=args.trade_share_log_std,
        gate_explore=args.trade_gate_explore, gate_temperature=args.trade_gate_temperature,
        value_log_std=args.trade_value_log_std,
        provenance={"inputCheckpointSha256": checkpoint_hash(args.checkpoint),
                    "inputCheckpointName": args.checkpoint.name},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    torch.save(tempered, temporary)
    temporary.replace(args.out)
    reloaded = torch.load(args.out, map_location="cpu", weights_only=True)
    verify_tempered(reloaded, state)
    report = {**record, "outSha256": checkpoint_hash(args.out)}
    text = json.dumps(report, indent=1, sort_keys=True, default=str)
    if args.report is not None:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
