"""Search teacher (expert iteration) pieces that need no native bridge."""

from __future__ import annotations

import numpy as np
import torch

from anthropocene_rl.autonomous_v7 import contracts, ppo, search_teacher as st
from anthropocene_rl.autonomous_v7.model import Actor, Normalizer, to_tensors
from test_autonomous_v7 import batch


def village(regions=(3, 5), legal=(True, True, False)):
    mask = [True, True] + [False] * contracts.REGION_ROWS
    for slot in regions:
        mask[slot] = True
    return {"camp": 7, "society": 3, "observation": {"regionMovementMask": mask, "economyMixMask": list(legal)}}


def test_alternatives_list_policy_native_legal_regions_and_distinct_vertices() -> None:
    base = {"strategic": [0.5, 0.5, 0.5, 1.0, 0.0], "movement": contracts.NATIVE_MOVEMENT}  # all fields
    alts = st.alternatives(village(), base, st.SearchConfig())
    kinds = [a["kind"] for a in alts]
    assert kinds[0] == st.POLICY and kinds[1] == st.NATIVE
    assert [a["slot"] for a in alts if a["kind"] == st.REGION] == [3, 5]
    # The cultivation vertex is the policy's own mix; only the herding vertex is new; fishing is illegal.
    assert [a["mix"].tolist() for a in alts if a["kind"] == st.MIX] == [[0.0, 1.0, 0.0]]
    assert len(alts) <= st.MAX_ALTERNATIVES


def test_override_applies_only_the_alternative() -> None:
    actions = {7: {"strategic": [0.1, 0.2, 0.3, 1.0, 0.0], "movement": contracts.NATIVE_MOVEMENT}, 8: {"strategic": [0] * 5}}
    v = village()
    assert st.override(actions, 7, {"kind": st.POLICY}, v, True) == actions
    assert 7 not in st.override(actions, 7, {"kind": st.NATIVE}, v, True) and 8 in st.override(actions, 7, {"kind": st.NATIVE}, v, True)
    region = {"kind": st.REGION, "slot": 5}
    assert st.override(actions, 7, region, v, True)[7]["movement"] == contracts.REGION_BASE + 3
    assert st.override(actions, 7, region, v, False)[7]["movement"] == contracts.KEEP_INTENT
    herd = st.override(actions, 7, {"kind": st.MIX, "mix": np.array([0.0, 1.0, 0.0])}, v, False)[7]["strategic"]
    assert herd[:3] == [0.1, 0.2, 0.3] and herd[3] == 0.0 and herd[4] == 0.0
    assert actions[7]["strategic"][3] == 1.0  # the input is not modified


def test_labels_keep_only_winners_over_the_margin_and_the_loss_moves_toward_them() -> None:
    b = batch(); n = len(b["core"])
    b["region_movement_mask"] = np.ones((n, contracts.REGION_MOVEMENT_SLOTS), dtype=bool)
    b["economy_mix_mask"][:] = True
    kinds = np.array([st.REGION, st.MIX, st.POLICY, st.NATIVE] * (n // 4) + [st.REGION] * (n % 4))
    rows = dict(b, label_kind=kinds, label_slot=np.where(kinds == st.REGION, 4, np.where(kinds == st.NATIVE, 0, -1)),
                label_mix=np.where((kinds == st.MIX)[:, None], [[0.0, 1.0, 0.0]], np.nan),
                score_policy=np.full(n, 100.0), score_best=np.where(np.arange(n) % 2 == 0, 110.0, 100.2))
    import tempfile, pathlib
    path = pathlib.Path(tempfile.mkdtemp()) / "labels.npz"
    np.savez(path, **rows)
    kept = ppo.load_search_labels([path], margin=0.005)
    assert kept is not None and (kept["label_kind"] != st.POLICY).all() and (kept["score_best"] == 110.0).all()
    normalizer = Normalizer.fit(b)
    torch.manual_seed(0); actor = Actor(b["core"].shape[1])
    t = to_tensors({k: v for k, v in kept.items()}, normalizer)
    args = (torch.as_tensor(kept["label_kind"]), torch.as_tensor(kept["label_slot"]), torch.as_tensor(kept["label_mix"], dtype=torch.float32))
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-2)
    first = float(ppo.search_loss(actor(t), t, *args))
    for _ in range(30):
        loss = ppo.search_loss(actor(t), t, *args); optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert float(ppo.search_loss(actor(t), t, *args)) < first
    with torch.no_grad():
        out = actor(t)
    region_rows = kept["label_kind"] == st.REGION
    assert (out["region"].argmax(dim=1).numpy()[region_rows] == 4).mean() > 0.9
