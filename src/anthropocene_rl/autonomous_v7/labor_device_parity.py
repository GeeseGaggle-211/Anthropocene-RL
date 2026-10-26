"""Cross-device parity metrics for the stored-data fits (CPU versus CUDA).

Used by ``tests/test_labor_device.py`` and ``tools/cuda_parity_diag.py``. All functions are
pure and run on CPU copies of the tensors.

* :func:`function_drift` (output space) is the multi-step gate: per output,
  ``max|f_test - f_ref| / max|f_ref - f_init|`` on a fixed probe batch.
* :func:`state_drift` (parameter space) is reported, not gated. Its per-tensor plain
  relative metric is ill-posed for gauge parameters, whose true loss gradient is exactly
  zero (attention key-projection bias, a centred head's last bias): Adam turns their
  roundoff gradients into learning-rate-sized steps of arbitrary sign
  (``diag-proposal.md``).
* :func:`grad_compare` (one step from the same state): the global-L2 relative error is
  gated; the worst-tensor relative error is reported only (it hits gauge tensors).
"""

from __future__ import annotations

import math
from typing import Any

import torch

EPS = 1e-12


def attention_key_bias_gauge(key: str, tensor: torch.Tensor) -> torch.Tensor | None:
    """Gauge mask of ``nn.MultiheadAttention.in_proj_bias``: its key slice [E:2E].

    Softmax is invariant to a per-query constant shift, so the key bias has exactly zero
    true gradient and cannot change the fitted function.
    """
    if key.endswith("self_attn.in_proj_bias"):
        e = tensor.shape[0] // 3
        mask = torch.zeros(tensor.shape, dtype=torch.bool)
        mask[e : 2 * e] = True
        return mask
    return None


def _f(x: float | None) -> float | None:
    if x is None or not math.isfinite(x):
        return None if x is None else (float("inf") if x > 0 else float("-inf"))
    return float(x)


def state_drift(
    test: dict[str, torch.Tensor],
    ref: dict[str, torch.Tensor],
    init: dict[str, torch.Tensor],
    gauge=None,
) -> dict[str, Any]:
    """Drift of ``test`` (CUDA) against ``ref`` (CPU) with ``init`` the shared start.

    ``gauge(key, tensor)`` returns a boolean mask of elements whose loss gradient is
    analytically zero (they do not change the fitted function); the ``*_non_gauge``
    metrics exclude them.
    """
    per: dict[str, dict[str, Any]] = {}
    diff_sq = ref_sq = upd_sq = 0.0
    for key, b in ref.items():
        if not b.is_floating_point():
            continue
        a = test[key].double()
        b = b.double()
        i = init[key].double()
        d = float((a - b).abs().max()) if b.numel() else 0.0
        bmax = float(b.abs().max()) if b.numel() else 0.0
        imax = float(i.abs().max()) if b.numel() else 0.0
        umax = float((b - i).abs().max()) if b.numel() else 0.0
        diff_sq += float((a - b).square().sum())
        ref_sq += float(b.square().sum())
        upd_sq += float((b - i).square().sum())
        per[key] = {
            "numel": int(b.numel()),
            "max_abs_diff": d,
            "max_abs_ref": bmax,
            "max_abs_init": imax,
            "max_abs_update": umax,
            "zero_init": imax == 0.0,
            "plain": d / max(bmax, EPS),
            "scale": d / max(bmax + imax, EPS),
            "update": (d / umax) if umax > 0 else (0.0 if d == 0 else math.inf),
        }
        mask = None if gauge is None else gauge(key, b)
        if mask is not None and bool(mask.any()):
            keep = ~mask
            per[key]["gauge_elements"] = int(mask.sum())
            if bool(keep.any()):
                dn = float((a - b)[keep].abs().max())
                bn = float(b[keep].abs().max())
                un = float((b - i)[keep].abs().max())
                per[key]["plain_non_gauge"] = dn / max(bn, EPS)
                per[key]["update_non_gauge"] = (
                    (dn / un) if un > 0 else (0.0 if dn == 0 else math.inf)
                )
            else:
                per[key]["plain_non_gauge"] = per[key]["update_non_gauge"] = 0.0

    def worst(metric: str, keys=None) -> dict[str, Any]:
        base = metric.replace("_non_gauge", "")
        items = [(k, v.get(metric, v[base])) for k, v in per.items() if keys is None or k in keys]
        if not items:
            return {"value": None, "tensor": None}
        k, v = max(items, key=lambda kv: kv[1])
        return {"value": _f(v), "tensor": k}

    nonzero_ref = {k for k, v in per.items() if v["max_abs_ref"] > 0}
    return {
        "worst_plain": worst("plain"),
        "worst_plain_nonzero_ref": worst("plain", nonzero_ref),  # tests 2/3 exclude zero refs
        "worst_scale": worst("scale"),
        "worst_update": worst("update"),
        "worst_plain_non_gauge": worst("plain_non_gauge"),
        "worst_update_non_gauge": worst("update_non_gauge"),
        "global_l2_relative": math.sqrt(diff_sq) / max(math.sqrt(ref_sq), EPS),
        "global_l2_update_relative": (
            math.sqrt(diff_sq) / math.sqrt(upd_sq) if upd_sq > 0 else None
        ),
        "per_tensor": per,
    }


def function_drift(
    test: dict[str, torch.Tensor], ref: dict[str, torch.Tensor], init: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Output-space drift on a fixed probe batch (every fixture row), evaluated on CPU.

    ``worst_change_relative`` is over the outputs the reference fit moved
    (``max|f_test - f_ref| / max|f_ref - f_init|``); an output the fit left exactly at
    its initial value has no change scale, so it is reported separately as
    ``worst_unmoved_plain`` (``max|f_test - f_ref| / max|f_ref|``), to be held to the
    one-step forward tolerance.
    """
    per = {}
    diff_sq = change_sq = ref_sq = 0.0
    for key, b in ref.items():
        a, i = test[key], init[key]
        d = float((a - b).abs().max())
        change = float((b - i).abs().max())
        diff_sq += float((a - b).square().sum())
        change_sq += float((b - i).square().sum())
        ref_sq += float(b.square().sum())
        per[key] = {
            "plain": d / max(float(b.abs().max()), EPS),
            "change_relative": (d / change) if change > 0 else (0.0 if d == 0 else math.inf),
            "max_abs_change": change,
        }
    moved = [(k, v) for k, v in per.items() if v["max_abs_change"] > 0]
    unmoved = [(k, v) for k, v in per.items() if v["max_abs_change"] == 0]
    worst_change = max(moved, key=lambda kv: kv[1]["change_relative"], default=(None, None))
    worst_unmoved = max(unmoved, key=lambda kv: kv[1]["plain"], default=(None, None))
    worst_plain = max(per.items(), key=lambda kv: kv[1]["plain"])
    return {
        "worst_change_relative": {
            "value": None if worst_change[1] is None else _f(worst_change[1]["change_relative"]),
            "output": worst_change[0],
        },
        "worst_unmoved_plain": {
            "value": 0.0 if worst_unmoved[1] is None else _f(worst_unmoved[1]["plain"]),
            "output": worst_unmoved[0],
        },
        "worst_plain": {"value": _f(worst_plain[1]["plain"]), "output": worst_plain[0]},
        "global_l2_relative": math.sqrt(diff_sq) / max(math.sqrt(ref_sq), EPS),
        "global_l2_change_relative": (
            math.sqrt(diff_sq) / math.sqrt(change_sq) if change_sq > 0 else None
        ),
        "per_output": per,
    }


def compact(drift: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in drift.items() if k not in ("per_tensor", "per_output")}


def repeat_check(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, Any]:
    bitwise = all(torch.equal(a[k], b[k]) for k in a)
    worst = 0.0
    for k in a:
        if a[k].is_floating_point() and not torch.equal(a[k], b[k]):
            d = float((a[k].double() - b[k].double()).abs().max())
            worst = max(worst, d / max(float(a[k].double().abs().max()), EPS))
    return {"bitwise": bitwise, "max_relative": worst}


def grad_compare(
    test: dict[str, torch.Tensor | None], ref: dict[str, torch.Tensor | None]
) -> dict[str, Any]:
    worst, worst_key = 0.0, None
    diff_sq = ref_sq = 0.0
    sign_flips = elements = big_rel = 0
    for key, g in ref.items():
        h = test[key]
        if g is None or h is None:
            if (g is None) != (h is None):
                return {"error": f"gradient presence differs at {key}"}
            continue
        g, h = g.double(), h.double()
        gmax = float(g.abs().max())
        d = float((g - h).abs().max())
        diff_sq += float((g - h).square().sum())
        ref_sq += float(g.square().sum())
        nz = (g != 0) | (h != 0)
        elements += int(nz.sum())
        sign_flips += int((torch.sign(g) != torch.sign(h))[nz].sum())
        big_rel += int(((g - h).abs() > 0.1 * g.abs())[nz].sum())
        if gmax > 0 and d / gmax > worst:
            worst, worst_key = d / gmax, key
    return {
        "worst_tensor_relative": worst,
        "worst_tensor": worst_key,
        "global_l2_relative": math.sqrt(diff_sq) / max(math.sqrt(ref_sq), EPS),
        "nonzero_elements": elements,
        "sign_disagreements": sign_flips,
        "elements_rel_diff_gt_10pct": big_rel,
    }


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), EPS)
