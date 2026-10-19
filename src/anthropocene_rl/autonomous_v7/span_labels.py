"""Span labels from the heuristic's own persistence (contract 9 bootstrap).

The heuristic decides every turn; where its capital decision repeats, a commitment would have reproduced it. Walking
each village's decision sequence, a decision starts a commitment whose span is the longest allowed span (1, 2, 3, 4
or 6) that the following decisions match, and each matched turn inside it is labelled a hold. The walk also yields
the commitment features and hold masks the simulator would have shown, so cloning sees consistent inputs.

Capital controls compared: the three unit strategic levels, the economy-mix shares and the clothing and shelter
routes, each only where both decisions carry a label for it, within ``tolerance``.
"""

from __future__ import annotations

import numpy as np

from . import contracts, data


def capital(batch: dict[str, np.ndarray], i: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    return batch["strategic"][i], batch["strategic_mask"][i].astype(bool), int(batch["clothing_route"][i]), int(batch["shelter_route"][i])


def matches(batch: dict[str, np.ndarray], i: int, j: int, tolerance: float) -> bool:
    si, mi, ci, hi = capital(batch, i)
    sj, mj, cj, hj = capital(batch, j)
    both = mi & mj
    if np.any(np.abs(si[both] - sj[both]) > tolerance):
        return False
    if bool(batch["clothing_route_label_mask"][i]) and bool(batch["clothing_route_label_mask"][j]) and ci != cj:
        return False
    if bool(batch["shelter_route_label_mask"][i]) and bool(batch["shelter_route_label_mask"][j]) and hi != hj:
        return False
    return True


def derive(batch: dict[str, np.ndarray], tolerance: float = 0.05) -> dict[str, np.ndarray]:
    """Span label index (into contracts.SPANS), hold mask before the decision, and the 11 commitment features."""
    n = len(batch["camp"])
    span_label = np.full(n, contracts.SPAN_SINGLE, np.int64)
    hold_mask = np.zeros(n, bool)
    features = np.zeros((n, contracts.COMMITMENT_FEATURES), np.float32)
    present = np.zeros((n, contracts.COMMITMENT_FEATURES), bool)
    present[:, [0, 1, 2, 10]] = True
    allowed = [s for s in contracts.SPANS if s >= 1]
    for rows in data.sequences(batch):
        position = 0
        while position < len(rows):
            start = rows[position]
            run = 1
            while position + run < len(rows) and run < max(allowed) and \
                    batch["turn"][rows[position + run]] == batch["turn"][start] + run and matches(batch, start, rows[position + run], tolerance):
                run += 1
            span = max(s for s in allowed if s <= run)
            span_label[start] = contracts.SPANS.index(span)
            strategic, mask, cloth, shelter = capital(batch, start)
            for k in range(1, span):
                i = rows[position + k]
                span_label[i] = contracts.SPAN_HOLD
                hold_mask[i] = True
                remaining = span - k
                features[i, :3] = (1.0, remaining, k)
                features[i, 3:8] = strategic[:5]
                features[i, 8] = 1.0 if cloth == 1 else 0.0
                features[i, 9] = 1.0 if shelter == 1 else 0.0
                present[i, 3:8] = True
                present[i, 8] = cloth >= 0; present[i, 9] = shelter >= 0
            position += span
    return {"span_label": span_label, "commit_hold_mask": hold_mask, "commitment_features": features, "commitment_present": present}


def extend(batch: dict[str, np.ndarray], tolerance: float = 0.05) -> dict[str, np.ndarray]:
    """A contract-8 shard batch with contract-9 core features, hold masks and span labels."""
    derived = derive(batch, tolerance)
    out = dict(batch)
    out["core"] = np.concatenate([batch["core"], derived["commitment_features"]], axis=1)
    out["core_present"] = np.concatenate([batch["core_present"], derived["commitment_present"]], axis=1)
    out["commit_hold_mask"] = derived["commit_hold_mask"]
    out["span_label"] = derived["span_label"]
    return out


if __name__ == "__main__":
    import json, sys
    from pathlib import Path
    root = Path(sys.argv[1])
    dev = data.load(p for p in sorted(root.iterdir()) if p.is_dir())
    for tolerance in (0.02, 0.05, 0.1):
        d = derive(dev, tolerance)
        labels = d["span_label"]
        print(json.dumps({"tolerance": tolerance, "rows": int(len(labels)),
                          "shares": {str(s): round(float((labels == i).mean()), 3) for i, s in enumerate(contracts.SPANS)}}))
