"""Per-task gradient norms of the actor BC loss on the shared trunk (diagnostic only).

For a checkpoint's actor and a world-balanced row sample of a labor dataset, every task of
:data:`~.labor_actor_gradnorm.ACTOR_BC_TASKS` (the 13 role activity CEs as weighted in the
current loss, specialty, movement, fission, trade gate) gets the L2 norm of its gradient on

* ``trunk``: every encoder parameter (everything upstream of the head output layers), and
* ``gradnorm_w``: the GradNorm proxy W (last global attention layer + output norm),

as the mean over training-size batches of the per-batch norm (what an optimizer step sees)
and as the norm of the whole sample's mean gradient. The same is reported with the role terms
re-weighted ``uniform`` per staffed role (each staffed role of a row gets an equal part of the
row's activity weight). Shares are each task's norm over the sum of the tasks' norms (the
stability dominance detector's share definition). Nothing is trained; the dataset is read by
memory map and only the sampled rows are copied.

    python -m anthropocene_rl.autonomous_v7.labor_actor_task_gradients \
        --checkpoint actor.pt --dataset cal8-b01 --rows-per-world 400 --out grads.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .labor_actor_gradnorm import (
    ACTOR_BC_TASKS,
    ACTOR_ROLE_TASKS,
    actor_gradnorm_parameters,
    actor_trunk_parameters,
)

REPORT_SCHEMA = "labor-actor-task-gradients-v1"
ROLE_WEIGHTINGS = ("share", "uniform")
PARAMETER_SETS = ("trunk", "gradnorm_w")


def load_labor_rows(
    directory: Path | str,
    *,
    rows_per_world: int,
    seed: int = 0,
    expected_normalizer_digest: str | None = None,
    training: bool = False,
    verify: bool = True,
) -> dict[str, Any]:
    """A world-balanced row sample of one published labor dataset, without loading it whole.

    Per world: ``rows_per_world`` rows (all, if fewer) drawn without replacement from the
    non-quarantined rows with a valid specialty label, by ``default_rng(seed + world)``, in row
    order. Actor inputs and labels only (critic-only privileged arrays are skipped).
    ``row_weight`` is recomputed world-balanced over the sample. ``verify`` checks the
    manifest and every shard hash first (``verify_dataset``); ``training`` applies the
    training reader's reserved-world/partition guard."""
    from .labor_bc import world_balanced_weights
    from .labor_privileged import is_privileged_key
    from .labor_teacher_data import _training_view, verify_dataset
    from .labor_worlds import CALIBRATION_PARTITION, is_calibration_set, require_training_record

    directory = Path(directory)
    if rows_per_world < 1:
        raise ValueError("rows_per_world must be positive")
    manifest = (
        verify_dataset(directory)
        if verify
        else json.loads((directory / "dataset.json").read_text())
    )
    if training:
        manifest = _training_view(manifest, directory, training=True, drop=False)
        require_training_record(manifest["source_record"], f"labor rows {directory}")
    parts: dict[str, list[torch.Tensor]] = {}
    sampled: dict[str, int] = {}
    digests = set()
    for world, entry in sorted(manifest["shards"].items(), key=lambda item: int(item[0])):
        shard = directory / entry["path"]
        meta = json.loads((shard / "meta.json").read_text())
        digests.add(meta["normalizer_digest"])
        quarantined = np.load(shard / "row_quarantined.npy").reshape(-1).astype(bool)
        valid_path = shard / "label_specialty_valid.npy"
        valid = (
            np.load(valid_path).reshape(-1).astype(bool)
            if valid_path.is_file()
            else np.ones_like(quarantined)
        )
        candidates = np.flatnonzero(valid & ~quarantined)
        rng = np.random.default_rng(int(seed) + int(world))
        take = min(int(rows_per_world), len(candidates))
        rows = np.sort(rng.choice(candidates, size=take, replace=False))
        sampled[str(world)] = int(take)
        for name in meta["payload"]:
            key = name[: -len(".npy")]
            if is_privileged_key(key):
                continue
            array = np.load(shard / name, allow_pickle=False, mmap_mode="r")
            parts.setdefault(key, []).append(torch.from_numpy(np.ascontiguousarray(array[rows])))
    if len(digests) != 1:
        raise ValueError("labor shards use different normalizers")
    digest_value = next(iter(digests))
    if expected_normalizer_digest is not None and digest_value != expected_normalizer_digest:
        raise ValueError("labor dataset normalizer differs from the checkpoint normalizer")
    dataset: dict[str, Any] = {k: torch.cat(v, dim=0) for k, v in parts.items()}
    from .labor_bc import ELAPSED_YEARS_FIELD, LEGACY_ELAPSED_DAYS_FIELD

    if LEGACY_ELAPSED_DAYS_FIELD in dataset and ELAPSED_YEARS_FIELD not in dataset:
        # Shards published before the rename store years under the legacy key.
        dataset[ELAPSED_YEARS_FIELD] = dataset.pop(LEGACY_ELAPSED_DAYS_FIELD)
    keep = ~dataset["row_quarantined"].reshape(-1).bool()
    dataset["row_weight"] = world_balanced_weights(dataset["row_world"], keep)
    source = manifest["source_record"]
    dataset["collection"] = {
        "source_kind": source.get("source_kind"),
        "dataset_digest": manifest["dataset_digest"],
        "normalizer_digest": digest_value,
        "partition": CALIBRATION_PARTITION if is_calibration_set(source) else "fitting",
        "rows": len(keep),
        "world_seeds": sorted(int(w) for w in manifest["shards"]),
        "row_sample": {"rows_per_world": int(rows_per_world), "seed": int(seed), **sampled},
        "total_decisions": len(keep),
        "initial_accepted": len(keep),
    }
    if is_calibration_set(source):
        dataset["collection"]["calibration_set"] = True
    return dataset


def inverse_frequency_weights(
    dataset: Mapping[str, Any], power: float = 0.5
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """``pretrain_labor_actor_bc``'s specialty/activity inverse-frequency weights."""
    if power <= 0.0:
        return None, None
    n = int(dataset["target_specialty"].shape[0])
    valid = dataset.get("label_specialty_valid")
    keep = torch.ones(n, dtype=torch.bool) if valid is None else valid.reshape(n).bool()
    if "row_quarantined" in dataset:
        keep = keep & ~dataset["row_quarantined"].reshape(n).bool()
    if not bool(keep.any()):
        keep = torch.ones(n, dtype=torch.bool)
    spec = (dataset["target_specialty"][keep].mean(dim=0) + 0.02).pow(-power)
    act = (dataset["target_activity"][keep].mean(dim=0) + 0.02).pow(-power)
    return spec / spec.mean().clamp(min=1e-8), act / act.mean().clamp(min=1e-8)


def _flat_norm(grads: Sequence[torch.Tensor | None]) -> float:
    return math.sqrt(sum(float(g.double().pow(2).sum()) for g in grads if g is not None))


def task_gradient_report(
    actor: torch.nn.Module,
    dataset: Mapping[str, Any],
    *,
    batch_size: int = 256,
    seed: int = 0,
    target_log_std: float = -2.0,
    inverse_freq_power: float = 0.5,
    label_smoothing: float = 1e-6,
    inactive_role_weight: float = 0.0,
    weightings: Sequence[str] = ROLE_WEIGHTINGS,
) -> dict[str, Any]:
    """Per-task trunk gradient norms of the BC loss over shuffled ``batch_size`` batches."""
    from .labor_bc import _batch_loss_and_l1

    actor.eval()  # dropout-free encoder; eval keeps the forward deterministic
    params = {
        "trunk": actor_trunk_parameters(actor),
        "gradnorm_w": actor_gradnorm_parameters(actor),
    }
    spec_w, act_w = inverse_frequency_weights(dataset, inverse_freq_power)
    keys = [k for k, v in dataset.items() if isinstance(v, torch.Tensor)]
    n = int(dataset["target_specialty"].shape[0])
    order = np.random.default_rng(seed).permutation(n)
    batches = [order[s : s + batch_size] for s in range(0, n, batch_size)]
    # Full batches only for the per-batch statistics (a short tail batch is noisier).
    if len(batches) > 1 and len(batches[-1]) < batch_size:
        batches = batches[:-1]
    per_batch: dict[str, dict[str, dict[str, list[float]]]] = {
        w: {p: {t: [] for t in ACTOR_BC_TASKS} for p in PARAMETER_SETS} for w in weightings
    }
    sums: dict[str, dict[str, dict[str, list[torch.Tensor] | None]]] = {
        w: {p: {t: None for t in ACTOR_BC_TASKS} for p in PARAMETER_SETS} for w in weightings
    }
    staffed = {t: 0 for t in ACTOR_ROLE_TASKS}
    staffed_batches = {t: 0 for t in ACTOR_ROLE_TASKS}
    losses: dict[str, dict[str, float]] = {w: {t: 0.0 for t in ACTOR_BC_TASKS} for w in weightings}
    mean_ce = {t: [] for t in ACTOR_ROLE_TASKS}
    l1_sum = torch.zeros(len(ACTOR_ROLE_TASKS), dtype=torch.float64)
    l1_weight = torch.zeros(len(ACTOR_ROLE_TASKS), dtype=torch.float64)
    started = time.time()
    for index in batches:
        batch = {k: dataset[k][torch.as_tensor(index, dtype=torch.long)] for k in keys}
        l1, weight = _role_activity_l1(actor, batch)
        l1_sum += l1
        l1_weight += weight
        for weighting in weightings:
            terms: dict[str, Any] = {}
            _batch_loss_and_l1(
                actor,
                batch,
                target_log_std=target_log_std,
                inactive_role_weight=inactive_role_weight,
                label_smoothing=label_smoothing,
                spec_weights=spec_w,
                act_weights=act_w,
                task_terms=terms,
                role_weighting=weighting,
            )
            if weighting == weightings[0]:
                for t in ACTOR_ROLE_TASKS:
                    staffed[t] += terms["staffed_rows"][t]
                    staffed_batches[t] += int(terms["staffed_rows"][t] > 0)
                    if terms["staffed_rows"][t] > 0:
                        mean_ce[t].append(terms["mean_ce"][t])
            for task in ACTOR_BC_TASKS:
                loss = terms["loss"][task]
                losses[weighting][task] += float(loss.detach()) / len(batches)
                for pset in PARAMETER_SETS:
                    if not loss.requires_grad:
                        grads: list[torch.Tensor | None] = [None] * len(params[pset])
                    else:
                        grads = list(
                            torch.autograd.grad(
                                loss, params[pset], retain_graph=True, allow_unused=True
                            )
                        )
                    per_batch[weighting][pset][task].append(_flat_norm(grads))
                    dense = [
                        torch.zeros_like(p) if g is None else g.detach()
                        for p, g in zip(params[pset], grads, strict=True)
                    ]
                    held = sums[weighting][pset][task]
                    sums[weighting][pset][task] = (
                        dense if held is None else [a + b for a, b in zip(held, dense, strict=True)]
                    )
    elapsed = time.time() - started
    rows = sum(len(b) for b in batches)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "rows": int(rows),
        "batches": len(batches),
        "batch_size": int(batch_size),
        "tasks": list(ACTOR_BC_TASKS),
        "parameter_sets": {p: int(sum(x.numel() for x in params[p])) for p in PARAMETER_SETS},
        "inverse_freq_power": float(inverse_freq_power),
        "inverse_freq_source": "the measured sample (the fit computes it on its fitting data)",
        "seconds": elapsed,
        "staffed_rows": staffed,
        "staffed_batches": staffed_batches,
        "role_mean_ce_staffed_batches": {
            t: (sum(v) / len(v) if v else None) for t, v in mean_ce.items()
        },
        "role_activity_l1_staffed": {
            t: (float(l1_sum[i] / l1_weight[i]) if l1_weight[i] > 0 else None)
            for i, t in enumerate(ACTOR_ROLE_TASKS)
        },
        "weightings": {},
    }
    for weighting in weightings:
        entry: dict[str, Any] = {"mean_loss": losses[weighting]}
        for pset in PARAMETER_SETS:
            batch_norms = {t: float(np.mean(v)) for t, v in per_batch[weighting][pset].items()}
            sample_norms = {
                t: _flat_norm([g / len(batches) for g in grads])
                for t, grads in sums[weighting][pset].items()
            }
            entry[pset] = {
                "mean_batch_norm": batch_norms,
                "mean_batch_share": _shares(batch_norms),
                "median_batch_norm": {
                    t: float(np.median(v)) for t, v in per_batch[weighting][pset].items()
                },
                "sample_gradient_norm": sample_norms,
                "sample_gradient_share": _shares(sample_norms),
            }
        report["weightings"][weighting] = entry
    report["per_staffed_row"] = _per_staffed_row(report, rows)
    return report


def _role_activity_l1(
    actor: torch.nn.Module, batch: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-role sums of the activity-share L1 (max 2) over rows staffing the role, weighted by
    ``row_weight``, and the weight sums (deterministic masked-softmax decoding)."""
    from .labor_model import _masked_softmax, _masks
    from .labor_privileged import actor_inputs

    with torch.no_grad():
        output = actor(actor_inputs(batch))
        n = len(batch["target_specialty"])
        _, act_mask = _masks(batch, n)
        pred = _masked_softmax(output["activity_logits"], act_mask)
        l1 = (torch.abs(pred - batch["target_activity"]) * act_mask.float()).sum(dim=-1)
        row_weight = batch.get("row_weight")
        row_weight = torch.ones(n) if row_weight is None else row_weight.reshape(n).float()
        staffed = (batch["target_specialty"][:, 1:] > 0) & act_mask.any(dim=-1)
        if "label_activity_role_valid" in batch:
            staffed = staffed & batch["label_activity_role_valid"].reshape(staffed.shape).bool()
        weight = staffed.double() * row_weight.double().unsqueeze(-1)
        return (l1.double() * weight).sum(dim=0), weight.sum(dim=0)


def _shares(norms: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(0.0, v) for v in norms.values())
    return {k: (max(0.0, v) / total if total > 0 else 0.0) for k, v in norms.items()}


def _per_staffed_row(report: Mapping[str, Any], rows: int) -> dict[str, Any]:
    """Each role's trunk gradient share per staffed-row fraction, relative to Gatherer's."""
    out: dict[str, Any] = {}
    for weighting, entry in report["weightings"].items():
        shares = entry["trunk"]["mean_batch_share"]
        intensity = {
            t: (
                shares[t] / (report["staffed_rows"][t] / rows)
                if report["staffed_rows"][t]
                else None
            )
            for t in ACTOR_ROLE_TASKS
        }
        out[weighting] = {"share_per_staffed_fraction": intensity}
        for role in ("Gatherer", "Farmer"):
            reference = intensity.get(f"activity:{role}") or None
            out[weighting][f"relative_to_{role.lower()}"] = {
                t: (v / reference if v is not None and reference else None)
                for t, v in intensity.items()
            }
    return out


def format_table(report: Mapping[str, Any]) -> str:
    share = report["weightings"].get("share", {})
    uniform = report["weightings"].get("uniform", {})
    rel = report["per_staffed_row"]
    lines = [
        f"rows={report['rows']} batches={report['batches']}x{report['batch_size']} "
        f"trunk params={report['parameter_sets']['trunk']}",
        f"{'task':26s} {'staffed':>7s} {'share%':>7s} {'samp%':>7s} {'W%':>7s} "
        f"{'unif%':>7s} {'unifW%':>7s} {'row/F':>7s} {'unif/F':>7s} {'L1':>6s}",
    ]
    for task in report["tasks"]:
        staffed = report["staffed_rows"].get(task, report["rows"])

        def pct(entry: Mapping[str, Any], pset: str, key: str) -> str:
            value = entry.get(pset, {}).get(key, {}).get(task)  # noqa: B023 - late-binding closure kept as written (called within the iteration)
            return "-" if value is None else f"{100 * value:7.2f}"

        def relative(weighting: str) -> str:
            value = rel.get(weighting, {}).get("relative_to_farmer", {}).get(task)  # noqa: B023 - late-binding closure kept as written (called within the iteration)
            return "-" if value is None else f"{value:.3g}"

        l1 = report.get("role_activity_l1_staffed", {}).get(task)

        lines.append(
            f"{task:26s} {staffed:7d} {pct(share, 'trunk', 'mean_batch_share')} "
            f"{pct(share, 'trunk', 'sample_gradient_share')} "
            f"{pct(share, 'gradnorm_w', 'mean_batch_share')} "
            f"{pct(uniform, 'trunk', 'mean_batch_share')} "
            f"{pct(uniform, 'gradnorm_w', 'mean_batch_share')} "
            f"{relative('share'):>7s} {relative('uniform'):>7s} "
            f"{'-' if l1 is None else format(l1, '6.3f'):>6s}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rows-per-world", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inverse-freq-power", type=float, default=0.5)
    parser.add_argument("--target-log-std", type=float, default=-2.0)
    parser.add_argument("--no-verify", action="store_true", default=False)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(int(args.threads))
    from .labor_pretrain_eval import _check_world_policy, _load_models
    from .labor_training import _normalization_digest

    _, actor, _, _, normalizer = _load_models(args.checkpoint)
    dataset = load_labor_rows(
        args.dataset,
        rows_per_world=args.rows_per_world,
        seed=args.seed,
        expected_normalizer_digest=_normalization_digest(normalizer),
        verify=not args.no_verify,
    )
    _check_world_policy(dataset["row_world"].tolist(), f"dataset {args.dataset}")
    report = task_gradient_report(
        actor,
        dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        target_log_std=args.target_log_std,
        inverse_freq_power=args.inverse_freq_power,
    )
    report.update(
        checkpoint=str(args.checkpoint),
        dataset=str(args.dataset),
        dataset_digest=dataset["collection"]["dataset_digest"],
        row_sample=dataset["collection"]["row_sample"],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    table = format_table(report)
    args.out.with_suffix(".txt").write_text(table + "\n")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
