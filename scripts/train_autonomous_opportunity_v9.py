"""Independent MLPs, own-stock trade warmup, native critic gate and paired PPO."""

import argparse
import copy
import hashlib
import json
import time
from functools import partial
from pathlib import Path

import torch
from run_autonomous_causal_gate_v7 import (
    combine_inputs,
    sha,
    summarize_episodes,
    summary_without_tensors,
)
from run_autonomous_continuation_v4 import measured_update

from anthropocene_rl.autonomous_v4.collection import episode_seed
from anthropocene_rl.autonomous_v5.causal import (
    episode,
    n_step_targets,
    protected_state,
)
from anthropocene_rl.autonomous_v5.causal_budget import BudgetLedger, CausalCollector
from anthropocene_rl.autonomous_v5.causal_diagnostics import (
    signature,
    signature_change,
    snapshot_health,
)
from anthropocene_rl.autonomous_v6.contracts import observation
from anthropocene_rl.autonomous_v6.opportunity import market_peer, warmup_trade
from anthropocene_rl.autonomous_v6.policy import inputs_with_mode
from anthropocene_rl.autonomous_v6.readiness import (
    fit_critic,
    fit_intercept,
    protected_calibration,
    ready,
)
from anthropocene_rl.autonomous_v6.readiness import metrics as critic_metrics
from anthropocene_rl.training.autonomous_ppo_v7 import AutonomousTrainerV7


class OpportunityCollector(CausalCollector):
    episode_runner = staticmethod(partial(episode, peer_policy=market_peer))


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, value.shape)).encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def pair_hash(trainer):
    """Everything except the declared actor LR must match within each pair."""
    state = copy.deepcopy(trainer.optimizer.state_dict())
    state["param_groups"][0]["lr"] = 0
    digest = hashlib.sha256(tensor_hash(trainer.model.state_dict()).encode())

    def add(value):
        if isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, value.shape)).encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=repr):
                digest.update(repr(key).encode())
                add(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                add(item)
        else:
            digest.update(repr(value).encode())

    add(state)
    add(trainer.sampler.get_state())
    add(trainer.shuffle.get_state())
    add((trainer.episodes, trainer.transitions, trainer.updates))
    config = dict(trainer.config, actor_lr=0)
    add(config)
    return digest.hexdigest()


def reversal_with_adam(parent, probe, *, steps=16, deadline=None):
    from anthropocene_rl.autonomous.policy import MaskedDirichlet
    from anthropocene_rl.autonomous_v4.diagnostics import probe_face

    clone = copy.deepcopy(parent)
    subset, labels, roles, feature = probe_face(probe)
    phases = []
    for direction in [1, -1]:
        selected = torch.where(labels * direction > 0, roles[0], roles[1])
        mask = subset["WorkforceMask"]
        target = mask.double() * 0.15 / (mask.sum(-1, keepdim=True) - 1)
        target.scatter_(1, selected[:, None], 0.85)

        def measure(direction=direction, target=target, mask=mask):
            raw = clone.model.heads["simplex"](clone.model.actor(subset["x"]))[:, :13]
            alpha = clone.model.concentrations(raw)
            loss = -MaskedDirichlet(alpha, mask).log_prob(target).mean()
            means = alpha * mask
            means /= means.sum(-1, keepdim=True)
            gap = ((means[:, roles[0]] - means[:, roles[1]]) * labels * direction).mean()
            return loss, gap

        before = [float(v.detach()) for v in measure()]
        for _ in range(steps):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Reversal probe exceeded the shared deadline")
            loss, _ = measure()
            clone.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            parameters = [*clone.model.actor.parameters(), *clone.model.heads.parameters()]
            torch.nn.utils.clip_grad_norm_(parameters, 0.5, error_if_nonfinite=True)
            clone.optimizer.step()
        after = [float(v.detach()) for v in measure()]
        phases.append(
            dict(
                direction=direction,
                before=before,
                after=after,
                passed=after[0] < before[0] and after[1] > before[1],
            )
        )
    return dict(
        passed=all(p["passed"] for p in phases),
        phases=phases,
        feature=feature,
        roles=roles.tolist(),
        inherited_populated_adam=True,
        steps=steps * 2,
        lr=parent.optimizer.param_groups[0]["lr"],
        retained_updates=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.output.exists():
        raise FileExistsError(args.output)
    for pin in cfg["inputs"].values():
        if sha(pin["path"]) != pin["sha256"]:
            raise ValueError(f"Changed pinned input: {pin['path']}")
    if (
        cfg["seeds"] != [101, 211, 307]
        or cfg["actor_lrs"] != [0.0003, 0.00003]
        or cfg["transitions_per_cell"] != 1024
    ):
        raise ValueError("Frozen independent-initialization/LR budget")
    admission = json.loads(Path(cfg["inputs"]["admission"]["path"]).read_text())
    if admission["status"] != "PASS":
        raise ValueError("Actual native admission required")
    bank = Path(cfg["inputs"]["bank"]["path"]).parent
    entries = {e["id"]: e for e in json.loads((bank / "manifest.json").read_text())["entries"]}
    selected = admission["selected"]
    worlds = json.loads(Path(cfg["inputs"]["selection"]["path"]).read_text())
    sets = [set(worlds[k]) for k in ["fit_worlds", "validation_worlds", "development_worlds"]]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("World partition leakage")
    for role, ids in selected.items():
        for id_ in ids:
            expected = (
                worlds["development_worlds"]
                if role == "development"
                else worlds["fit_worlds"]
                if role == "ppo"
                else worlds["fit_worlds"] + worlds["validation_worlds"]
            )
            if entries[id_]["world_seed"] not in expected:
                raise ValueError("Actual reset world violates frozen partition")
    corpus = json.loads(Path(cfg["inputs"]["corpus"]["path"]).read_text())
    demo = inputs_with_mode([observation(r["observation"]) for r in corpus])
    targets = torch.tensor([r["target"] for r in corpus])
    probe = inputs_with_mode(
        [observation(w) for w in json.loads(Path(cfg["inputs"]["probe"]["path"]).read_text())]
    )
    adaptation_probe = inputs_with_mode(
        [
            observation(w)
            for w in json.loads(Path(cfg["inputs"]["adaptation_probe"]["path"]).read_text())
        ]
    )
    prior = json.loads(Path(cfg["inputs"]["prior_budget"]["path"]).read_text())
    if prior["total_resolves"] + sum(cfg["caps"].values()) > 27991:
        raise ValueError("Global resource envelope")
    torch.set_num_threads(1)
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
        initializations=[],
        calibrations=[],
        cells=[],
        baselines=[],
    )

    def save():
        ledger.save()
        result["budget"] = json.loads(ledger.path.read_text())
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    parents = []
    try:
        # Mechanism proof is separately pinned/reserved; no learner calls occur here.
        import subprocess

        ledger.phase = "mechanical"
        ledger.reserve("mechanical", 3)
        ledger.reserve("mechanical_replay", 3)
        try:
            proof = subprocess.run(
                [
                    cfg["dotnet"],
                    cfg["inputs"]["native_tool"]["path"],
                    "--autonomous-opportunity-only",
                ],
                capture_output=True,
                text=True,
                timeout=ledger.deadline - time.monotonic(),
            )
        except subprocess.TimeoutExpired as error:
            for suffix, value in [("stdout", error.stdout), ("stderr", error.stderr)]:
                text = value.decode(errors="replace") if isinstance(value, bytes) else value
                (args.output / f"mechanical.{suffix}").write_text(text or "")
            raise
        (args.output / "mechanical.stdout").write_text(proof.stdout)
        (args.output / "mechanical.stderr").write_text(proof.stderr)
        proof.check_returncode()
        result["mechanical"] = json.loads(proof.stdout.strip().splitlines()[-1])
        if (
            result["mechanical"]["result"] != "PASS"
            or result["mechanical"]["validationResolves"] != 3
            or result["mechanical"]["replayResolves"] != 3
        ):
            raise ValueError("Native owner outcome proof failed")
        save()
        with OpportunityCollector(
            [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]],
            ledger=ledger,
            workers=4,
            deadline=ledger.deadline,
        ) as collector:
            from anthropocene_rl.policies.autonomous_ppo_v7 import AutonomousPPOV7

            anchor = torch.load(
                cfg["inputs"]["boundary_anchor"]["path"], map_location="cpu", weights_only=False
            )
            result["boundary_diagnostics"] = []
            ledger.phase = "boundary"
            for floor in [0.5, 0.1]:
                for deterministic in [False, True]:
                    diagnostic = AutonomousPPOV7(
                        anchor["width"],
                        critic_variant=anchor["config"]["critic_variant"],
                        uniform_mix=anchor["config"]["uniform_mix"],
                        concentration_floor=floor,
                    )
                    missing, unexpected = diagnostic.load_state_dict(anchor["model"], strict=False)
                    if set(missing) != {"value_offset", "value_scale"} or unexpected:
                        raise ValueError("Unexpected frozen boundary actor mapping")
                    if deterministic:
                        diagnostic.uniform_mix = 0
                    jobs = [
                        dict(entry=entries[id_], sampler_seed=episode_seed(71107, i), limit=8)
                        for i, id_ in enumerate(selected["floor_diagnostic"])
                    ]
                    _, summaries = collector.collect(
                        diagnostic,
                        bank,
                        jobs,
                        dict(cfg["ppo_config"], deterministic_evaluation=deterministic),
                        collect=False,
                        verify=True,
                    )
                    result["boundary_diagnostics"].append(
                        dict(
                            floor=floor,
                            deterministic=deterministic,
                            summary=summarize_episodes(summaries),
                            episodes=[summary_without_tensors(e) for e in summaries],
                        )
                    )
                    save()
            for seed in cfg["seeds"]:
                trainer = AutonomousTrainerV7(
                    cfg["ppo_config"],
                    seed,
                    dict(
                        gate_config_sha256=sha(args.config),
                        independent_initialization_seed=seed,
                        bank_sha256=cfg["inputs"]["bank"]["sha256"],
                    ),
                    width=256,
                )
                init = tensor_hash(trainer.model.state_dict())
                warmup = warmup_trade(
                    trainer, demo, targets, seed=seed + 77000, deadline=ledger.deadline
                )
                result["initializations"].append(
                    dict(seed=seed, original_model_sha256=init, warmup=warmup)
                )
                save()
                ledger.phase = "calibration"
                protected = protected_state(trainer)
                jobs = [
                    dict(entry=entries[id_], sampler_seed=episode_seed(seed + 60000, i), limit=25)
                    for i, id_ in enumerate(selected["calibration"])
                ]
                rows, episodes = collector.collect(
                    trainer.model, bank, jobs, trainer.config, collect=False, verify=True
                )
                if protected_state(trainer) != protected:
                    raise ValueError("Frozen actor collection changed protected state")
                ys, fitting, tails, zero_tails, double_tails, h100_zero = [], [], [], [], [], []
                for e in episodes:
                    t = e["trace"]
                    y = n_step_targets(
                        [r["reward"] for r in t],
                        [r["elapsed"] for r in t],
                        [r["terminal"] for r in t],
                        e["tail_value"],
                        400,
                        120,
                    )
                    ys.extend(y["target"])
                    tails.extend(y["tail_weight"])
                    zero_tails.extend(y["zero_tail"])
                    double_tails.extend(y["doubled_tail"])
                    h100_zero.extend(
                        n_step_targets(
                            [r["reward"] for r in t],
                            [r["elapsed"] for r in t],
                            [r["terminal"] for r in t],
                            0.0,
                            100,
                            120,
                        )["zero_tail"]
                    )
                    fitting.extend([e["world_seed"] in worlds["fit_worlds"]] * len(t))
                x, y, fit = (
                    combine_inputs(rows),
                    torch.tensor(ys, dtype=torch.float32),
                    torch.tensor(fitting),
                )
                train = ({k: v[fit] for k, v in x.items()}, y[fit])
                validation = ({k: v[~fit] for k, v in x.items()}, y[~fit])
                torch.save(
                    dict(
                        obs=x,
                        targets=y,
                        fit=fit,
                        tail_weights=tails,
                        zero_tail_targets=zero_tails,
                        doubled_tail_targets=double_tails,
                        h100_zero_tail_targets=h100_zero,
                        episodes=[summary_without_tensors(e) for e in episodes],
                    ),
                    args.output / f"native-targets-seed{seed}.pt",
                )
                entry = critic_metrics(trainer, train, validation)
                normalization = trainer.model.fit_value_scale(train[1])
                normalized_entry = critic_metrics(trainer, train, validation)
                calibrated_protected = protected_calibration(trainer)
                first = fit_intercept(trainer, train)
                report = fit_critic(
                    trainer,
                    train,
                    validation,
                    original_mse=entry["mse"],
                    max_epochs=256,
                    seed=seed + 80000,
                    deadline=ledger.deadline,
                )
                second = fit_intercept(trainer, train)
                final = critic_metrics(trainer, train, validation)
                passed = ready(final, entry["mse"])
                checkpoint = args.output / f"calibrated-seed{seed}.pt"
                trainer.checkpoint(checkpoint)
                result["calibrations"].append(
                    dict(
                        seed=seed,
                        passed=passed,
                        entry=entry,
                        final=final,
                        normalization=normalization,
                        after_normalization_before_fit=normalized_entry,
                        normalized_state_protected=protected_calibration(trainer)
                        == calibrated_protected,
                        tail_weight_min=min(tails),
                        tail_weight_max=max(tails),
                        zero_tail_target_mean=float(torch.tensor(zero_tails).mean()),
                        doubled_tail_target_mean=float(torch.tensor(double_tails).mean()),
                        h100_zero_tail_mean=float(torch.tensor(h100_zero).mean()),
                        cached_h100_semantics=(
                            "same physical rewards, H100 zero tail; no H400 tail reuse"
                        ),
                        before_intercept=first,
                        gradient_fit=report,
                        after_intercept=second,
                        actor_protected=protected_state(trainer) == protected,
                        native_summary=summarize_episodes(episodes),
                        checkpoint=str(checkpoint),
                        checkpoint_sha256=sha(checkpoint),
                    )
                )
                parents.append(trainer)
                save()
            if len({r["original_model_sha256"] for r in result["initializations"]}) != 3:
                raise ValueError("Initialization diversity missing")
            if not all(c["passed"] and c["actor_protected"] for c in result["calibrations"]):
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
                    trainer.deadline = ledger.deadline
                    trainer.config = dict(trainer.config, actor_lr=lr)
                    trainer.optimizer.param_groups[0]["lr"] = lr
                    if pair_hash(trainer) != reference:
                        raise ValueError("LR pair differs beyond its actor step size")
                    trainer.provenance = dict(trainer.provenance, paired_lr=lr)
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
                    opening = signature(trainer.model, probe)
                    result["cells"].append(cell)
                    all_episodes = []
                    ledger.phase = "training"
                    value_buffers = tensor_hash(
                        {k: trainer.model.state_dict()[k] for k in ["value_offset", "value_scale"]}
                    )
                    while cell["transitions"] < 1024:
                        trainer.set_rollout_mix()
                        rows, eps = collector.rollout(
                            trainer,
                            bank,
                            [entries[id_] for id_ in selected["ppo"]],
                            seed,
                            min(256, 1024 - cell["transitions"]),
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
                            raise ValueError("PPO changed fixed TRAIN value normalization")
                        save()
                    cell.update(
                        training_summary=summarize_episodes(all_episodes),
                        health_after=snapshot_health(trainer.model, probe, "full"),
                        distribution_change=signature_change(
                            opening, signature(trainer.model, probe)
                        ),
                        reversal=reversal_with_adam(
                            trainer, adaptation_probe, deadline=ledger.deadline
                        ),
                    )
                    (args.output / f"{cell['id']}-training-traces.json").write_text(
                        json.dumps([summary_without_tensors(e) for e in all_episodes]) + "\n"
                    )
                    checkpoint = args.output / f"{cell['id']}-step1024.pt"
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
