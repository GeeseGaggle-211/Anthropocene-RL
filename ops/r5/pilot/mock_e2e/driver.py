"""Mock native stages for the local end-to-end run of ppo_pilot_job.sh (no bridge DLL).

``labor_training collect``: the real collect command over the synthetic WP-A1 fixture rows
(fake bridge execution, the checkpoint's real actor/critics), relabelled to the requested
worlds. ``labor_anchor_producer``: the real producer with the test fake bridge, a scripted
continuation and the in-process serial executor (its --jobs path, without spawned
processes). Everything downstream (rollout/anchor files, train, calibration, the pilot
checks) is real. MOCK_FAIL_WORLDS=<csv> makes a native stage on exactly those worlds fail.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(os.environ["MOCK_REPO"])
sys.path.insert(0, str(REPO / "tests"))

import torch  # noqa: E402
from _pytest.monkeypatch import MonkeyPatch  # noqa: E402

from anthropocene_rl.autonomous_v7 import labor_training as training  # noqa: E402


def _arg(rest: list[str], name: str) -> str:
    return rest[rest.index(name) + 1]


def main() -> None:
    argv = sys.argv[1:]
    assert argv[0] == "-m", argv
    module, rest = argv[1], argv[2:]
    fail = os.environ.get("MOCK_FAIL_WORLDS")
    if fail and "--worlds" in rest and _arg(rest, "--worlds") == fail:
        sys.exit(f"mock native failure on worlds {fail}")
    patch = MonkeyPatch()
    if module.endswith("labor_training") and "--help" in rest:
        training.main(rest)  # the job's capability probe (collect --help)
    elif module.endswith("labor_training"):
        from test_labor_ppo_perhead import _collect

        state = torch.load(_arg(rest, "--checkpoint"), map_location="cpu", weights_only=True)
        _, actor, task, costs, _ = training._models_from_checkpoint(state)
        worlds = [int(w) for w in _arg(rest, "--worlds").split(",") if w]
        plan = ((False, 71), (True, 72), (False, 73), (False, 74), (True, 75))
        identity = [(worlds[0], 3), (worlds[0], 4), (worlds[1 % len(worlds)], 3),
                    (worlds[1 % len(worlds)], 4), (worlds[2 % len(worlds)], 3)]
        if len(set(identity)) < len(identity):  # fewer than 3 worlds (branch_cost_probe)
            identity = [(worlds[i % len(worlds)], 3 + i // len(worlds)) for i in range(5)]
        rows, bootstrap = [], {}
        for (retry, seed), (world, turn) in zip(plan, identity):
            row, tail, *_ = _collect(patch, retry, (actor, task, costs), seed=seed, world=world)
            row.turn = turn
            rows.append(row)
            bootstrap[(world, 7)] = tail
        def collect(**kw):
            brancher = kw.get("brancher")
            if brancher is not None:
                # WP-B: the real branch collector on the fake multi-camp bridge
                # (tests/labor_wpb_fake.py; its stand-in H12 shadow is a keyed actor
                # sample), one world x two decisions; the rollout rows stay the fixture's.
                from labor_wpb_fake import FakeLaborBridge, fake_shadow_factory, install

                from anthropocene_rl.autonomous_v7 import labor_branch_collect as branch

                normalizer = training._models_from_checkpoint(state)[4]
                install(patch)
                patch.setattr(branch, "SHADOW_FACTORY", fake_shadow_factory(actor, normalizer))
                # Its own generator: the fixture rollout's sampling state stays untouched.
                original(**{**kw, "bridge_factory": FakeLaborBridge,
                            "world_seeds": kw["world_seeds"][:1], "turns": 2,
                            "generator": torch.Generator().manual_seed(0)})
            return rows, bootstrap

        original = training.collect_labor_rollout
        patch.setattr(training, "collect_labor_rollout", collect)
        training.main(rest)
    elif module.endswith("labor_anchor_producer"):
        from test_labor_anchor_parallel import Bridge, SerialExecutor, _install

        from anthropocene_rl.autonomous_v7 import labor_anchor_producer as producer

        if "--help" not in rest:
            state = torch.load(_arg(rest, "--checkpoint"), map_location="cpu", weights_only=True)
            _install(producer, patch, training._models_from_checkpoint(state))
        original = producer.collect_labor_anchor_arrays
        patch.setattr(
            producer, "collect_labor_anchor_arrays",
            lambda **kw: original(**{**kw, "bridge_factory": Bridge,
                                     "executor_factory": SerialExecutor}),
        )
        producer.main(rest)
    else:
        sys.exit(f"mock driver: unexpected module {module}")


if __name__ == "__main__":
    main()
