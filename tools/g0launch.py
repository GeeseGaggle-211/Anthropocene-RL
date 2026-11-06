"""Run a labor_training pretrain subcommand with progress lines and preemption-safe partial checkpoints.

Every 25 optimizer steps it prints timing to stderr. Every SAVE_EVERY steps it writes <out>.partial:
the input checkpoint with the models being trained replaced by their current parameters, plus
"g0_partial" metadata. This format is a valid --checkpoint for resuming after a Spot preemption
(the Adam moments restart). It changes no computation of the run itself.
"""  # noqa: E501 - docstring/prose line kept verbatim

import os
import sys
import time

# CUDA fits (--device cuda): keep torch 2.14's runtime-compiled torch._native kernels
# unregistered (they need a C compiler the GPU VM lacks). Read once at `import torch`, so
# it must be set first; labor_device.configure refuses CUDA without it. No effect on CPU.
os.environ["TORCH_DISABLE_NATIVE_JIT"] = "1"

import torch

SAVE_EVERY = int(os.environ.get("G0_SAVE_EVERY", "250"))
_count = [0]
_start = [None]
_orig = torch.optim.Adam.step
argv = sys.argv[1:]
mode = argv[0]
ckpt_path = argv[argv.index("--checkpoint") + 1]
out_path = argv[argv.index("--out") + 1]
_cache = {}


def _save_partial(optimizer):
    from anthropocene_rl.autonomous_v7 import labor_training as lt

    if not _cache:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        layout, actor, task, costs, normalizer = lt._models_from_checkpoint(state)
        _cache.update(
            state=state, layout=layout, actor=actor, task=task, costs=costs, normalizer=normalizer
        )
    c = _cache
    targets = (
        list(c["actor"].parameters())
        if mode == "pretrain-bc"
        else list(c["task"].parameters()) + list(c["costs"].parameters())
    )
    sources = [p for group in optimizer.param_groups for p in group["params"]]
    if len(targets) != len(sources) or any(
        t.shape != s.shape for t, s in zip(targets, sources, strict=False)
    ):
        print(
            "[progress] partial save skipped: parameter layout mismatch",
            file=sys.stderr,
            flush=True,
        )
        return
    with torch.no_grad():
        for t, s in zip(targets, sources, strict=False):
            t.copy_(s)
    out = dict(c["state"])
    out.update(
        lt.labor_checkpoint_payload(
            c["actor"],
            c["task"],
            c["costs"],
            c["normalizer"],
            c["layout"],
            personality=lt.labor_checkpoint_personality(c["state"]),
        )
    )
    prior = (
        int(c["state"].get("g0_partial", {}).get("steps", 0))
        if isinstance(c["state"].get("g0_partial"), dict)
        else 0
    )
    out["g0_partial"] = {"mode": mode, "steps": prior + _count[0], "source": ckpt_path}
    tmp = out_path + ".partial.tmp"
    torch.save(out, tmp)
    os.replace(tmp, out_path + ".partial")
    print(f"[progress] partial saved at steps={prior + _count[0]}", file=sys.stderr, flush=True)


def _step(self, *args, **kwargs):
    if _start[0] is None:
        _start[0] = time.time()
    result = _orig(self, *args, **kwargs)
    _count[0] += 1
    if _count[0] % 25 == 0:
        el = time.time() - _start[0]
        print(
            f"[progress] steps={_count[0]} elapsed_s={el:.0f} s_per_step={el / _count[0]:.3f}",
            file=sys.stderr,
            flush=True,
        )
    if _count[0] % SAVE_EVERY == 0:
        try:
            _save_partial(self)
        except Exception as error:  # never let checkpointing kill the run
            print(f"[progress] partial save failed: {error!r}", file=sys.stderr, flush=True)
    return result


torch.optim.Adam.step = _step
from anthropocene_rl.autonomous_v7 import labor_training  # noqa: E402

labor_training.main(argv)
