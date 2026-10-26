# CPU-vs-CUDA multi-step parity: diagnosis and proposed check

Status: the reviewer approved the proposal on 26 Oct after the GPU run, and it
is implemented on `feature/r5-cuda-diag` (see "Implemented"). The CPU analysis
that motivated it follows below, unchanged.

## GPU results (26 Oct; L4, driver 580, torch 2.14.1+cu130)

`tools/cuda_parity_diag.py --probe-native` ran at 51c139a, exit 0, 194 s.

* **Critic and A head:** the hypothesis is confirmed.
  * Output-space drift is at most 1.5e-5 (critic) and 2e-7 (A head).
  * The teleported one-step loss differs by about 1e-6 and the global-L2
    gradient by about 1e-5 at every step.
  * The worst tensors are gauge tensors.
* **CUDA repeats:** bitwise for all four problems, under both the default and
  the MATH attention backend.
* **Native JIT:** with `TORCH_DISABLE_NATIVE_JIT=1`, no Triton op is registered,
  and the bmm probe passes with the variable set and fails without it.
* **BC is the outlier.** Its worst plain drift is 8.2e-3, on a non-gauge tensor
  (`attention.layers.1.linear1.weight`, update-relative 3.1e-2). Its output
  drift is 1.5e-4 on `specialty_logits`, about 150 times the multiplicative CPU
  proxy, and its step loss drifts by up to 1.7e-5. Its teleported steps are
  clean: loss at most 2e-7, global gradient at most 1.6e-6.

### BC trajectory sensitivity

Every teleported BC step from the same state agrees to about 1e-7 in loss and
about 1e-6 in global gradient. The drift therefore comes from following two
trajectories, not from a kernel defect.

It is also not a kink, a mask or a discrete branch. With SGD, the same GPU run
gives output drift of 6e-6, and per-step parameter drift stays at about 4e-7,
with no jump at any step. A crossed branch would show up under SGD as well, and
as a teleport spike.

The cause is Adam acting on gradient elements whose size is at the roundoff
level:
* On the first steps Adam's update is close to `lr * sign(g)` for every element
  with |g| well above eps (1e-8).
* Each teleported BC step has about 250 sign disagreements and about 500
  elements whose CPU and CUDA gradients differ by more than 10%, out of 8e5
  nonzero elements.
* Each such element takes a step of plus or minus lr in a different direction on
  each device. lr is 5e-3 here, five times the critic's.
* After the first step the worst plain drift is already 5.4e-3.

The multiplicative CPU proxy preserves signs, so it could not show this. An
additive proxy can, and it reproduces the BC numbers on CPU. It adds
`1e-7 * rms(g) * N(0,1)` to every nonzero gradient element, which flips the
signs of roundoff-sized elements and leaves exact zeros at zero.

| run (CPU, additive noise 1e-7) | update-relative drift, non-gauge | output drift | step loss drift |
|---|---|---|---|
| BC, Adam | 2.4e-2 (`local_activity_attention.layers.0.linear1.weight`) | 6.7e-5 | 1.9e-6 |
| BC, SGD | 5e-4 | 3e-6 | 7e-8 |
| critic, Adam | 1.7e-4 | 6.7e-6 | 9e-8 |

The critic's outputs stay at 7e-6 under the same noise. The BC actor is more
sensitive because of its larger learning rate and its many FFN weights with
roundoff-sized gradients. BC's 1.5e-4 output drift is still 6.7 times below the
1e-3 output gate.

## Implemented (feature/r5-cuda-diag)

* `tests/test_labor_device.py`:
  * The one-step parity test is kept.
  * A new teleported mid-trajectory step test covers the critic, the A head and
    BC. It gates the loss and the global-L2 gradient at 1e-4; the worst-tensor
    gradient error is printed only.
  * The three multi-step per-tensor checks are replaced by an output-space check
    on the fixture rows. The worst per-output change-relative drift must be at
    most `CROSS_DEVICE_OUTPUT_TOLERANCE` = 1e-3. An output the fit left exactly
    unchanged is held to 1e-4 plain relative instead.
  * Parameter drift (plain, non-gauge, global L2) is printed (`-s`), not gated.
  * The critic CUDA repeat stays on the approved 1e-5 plain compare.
* `labor_device_parity.py` holds the metrics. `tests/labor_parity_problems.py`
  holds the fixtures and replica loops, shared by the tests and the diagnostic.
* `labor_device.configure(cuda)` raises if `torch._native` has registered JIT
  kernels. `TORCH_DISABLE_NATIVE_JIT=1` is set in `tools/r5_critic_acceptance.sh`
  and in `tools/g0launch.py` before `import torch`, and in the gpu-vm
  `bootstrap.sh` and README.
* The `labor_device` docstring now says that the remedy for a failing CUDA
  determinism gate is pinning `SDPBackend.MATH`, never loosening the tolerance.

## What fails, and why (CPU evidence)

The diagnostic replays the three failing fits with explicit step loops. On CPU
each replica is bitwise equal to the real fit function (`pretrain_critic_on_dataset`,
`fit_advantage_head`, `pretrain_labor_actor_bc`). It then reruns each fit with
every gradient element multiplied by `1 + eps*N(0,1)` before clipping. This
stands in for fp32 reassociation error, which is what a different device
produces.

Here is the final drift after the test's own 9/9/6 steps. "plain" is the tests'
metric. "upd-ng" is the update-relative drift with gauge elements excluded (see
below). "fn" is output-space drift relative to the output's own change from
initialisation, measured on all fixture rows.

| fit | optimizer | eps | plain | upd-ng | global L2 / update | fn (worst output) |
|---|---|---|---|---|---|---|
| critic (TD) | Adam | 1e-7 | 3.0e-2 | 2.1e-5 | 5.4e-4 | 6.7e-6 |
| critic (TD) | Adam | 1e-6 | 2.9e-2 | 2.1e-5 | 5.9e-4 | 4.5e-6 |
| critic (TD) | SGD | 1e-6 | 3.4e-6 | 5.1e-4 | 2.7e-6 | 7.5e-6 |
| critic (MC) | Adam | 1e-6 | 2.1e-2 | 3.6e-5 | 4.9e-4 | 7.4e-6 |
| A head | AdamW | 1e-7 | 1.1 | 3.8e-5 | 3.5e-4 | 1.7e-7 |
| A head | AdamW | 1e-6 | 6.0 | 3.7e-5 | 7.1e-4 | 1.7e-7 |
| A head | SGD | 1e-6 | 1.5e2 | 6.1e-5 | 1.2e-6 | 3.3e-7 |
| BC | Adam | 1e-6 | 8.7e-4 | 1.6e-3 | 1.4e-5 | 6.0e-6 |

A perturbation of 1e-7, about fp32 epsilon, reproduces the observed failures on
CPU. The critic gives 0.03 (GPU observed 0.078) and the A head gives 1 to 6 (GPU
observed 6.36). The per-step trace shows the critic's plain drift jumping from
2e-7 to 0.02 in one step, at step 2.

The worst tensors are always **gauge directions**: parameters whose true loss
gradient is exactly zero, so they cannot change the fitted function.

* **Attention key-projection bias** (`self_attn.in_proj_bias[E:2E]`). Softmax is
  invariant to it. It starts at zero, and its computed gradient is
  1e-9 to 1e-11, against 1e-2 for the q and v slices.
* **A-head last-layer bias** (`heads.<j>.2.bias`). The pairwise loss sees only
  A_i - A_k, and A is centred per root before use. It starts at zero, and its
  computed gradient is about 1e-8.

Adam (eps 1e-8) normalises roundoff-sized gradients into steps of up to the
learning rate, with signs set by roundoff. Any difference in roundoff, whether
from a different device or a 1e-7 perturbation, therefore gives these tensors
O(1) relative differences. Their reference values are close to zero, so the
plain metric |a-b|/|b| is large. The scale-normalised metric does not help,
because these tensors start at zero. SGD removes the amplification in parameter
space, but the plain metric still blows up on the A-head bias, whose value is
pure noise. In output space every case stays at or below 1.2e-5.

So, pending the GPU run, the 1e-3 multi-step per-tensor threshold is ill-posed.
It measures gauge noise, not fit quality.

## What the GPU run must confirm (read the JSON in this order)

1. `real_fit_cpu_vs_cuda.test_metric` reproduces 0.0779 and 6.36. This
   validates the replica on CUDA (`replica_cuda_vs_real_fit_cuda`).
2. Teleported one-step parity (`configs.*.teleport`). The CPU state at every
   step k is loaded on both devices and one step is compared. The loss and the
   global-L2 gradient relative error should stay at or below 1e-4 at every k, not
   only at k=0. If mid-trajectory steps fail where k=0 passes, that is a
   state-dependent kernel defect, not amplification.
3. `final_drift.function` and `worst_update_non_gauge` should be small (about
   1e-5) for both Adam and SGD. Large output-space drift under SGD would rule
   out the amplification explanation.
4. `cuda_repeat` under `default` against `math`. If default SDPA is not bitwise
   but MATH is, the memory-efficient attention backward is the non-deterministic
   kernel the warning names.

## Proposed well-posed check (the co-architect decides)

1. **Kernel parity (gate for the device path):** the existing one-step test,
   plus the teleported form. Load the same state, including the optimizer and
   TD targets, at step 0 and at a mid-trajectory step. Require the loss relative
   error and the global-L2 gradient relative error to be at or below
   `CROSS_DEVICE_STEP_TOLERANCE` (1e-4). This tests kernels without trajectory
   divergence.
2. **Multi-step fit parity (replaces the per-tensor 1e-3 check):** compare
   output space on the fixture rows. Use the per-output
   `max|f_cuda - f_cpu| / max|f_cpu - f_init|` and its global-L2 form. Use the
   critic's decoded values and cost outputs, the A head's centred A, and the
   actor's output logits. I suggest a threshold of 1e-3: the CPU proxy measures
   at most 1.2e-5, roughly 80x margin. Set the final value from the GPU numbers.
3. **Parameter drift:** report it, but do not gate on it. If a parameter check
   is kept, use the update-relative metric with gauge elements excluded. The
   proxy reaches 1.6e-3 on BC, because Adam also amplifies non-gauge elements
   whose gradients are near zero. A parameter gate tighter than about 1e-2 would
   therefore stay fragile.
4. **Determinism probe (r5 acceptance, CUDA 1e-5):** the same flaw applies to
   any probe that uses `compare_state_dicts` (plain max relative). If any CUDA
   kernel is non-deterministic, as the memory-efficient SDPA backward is, gauge
   tensors in a 50-step repeat can differ by O(1) relative and fail the 1e-5
   probe. Depending on step 4 of the GPU readout, the options are:
   1. pin `sdpa_kernel(SDPBackend.MATH)` for CUDA fits in `labor_device`; or
   2. evaluate the probe in output space, or exclude gauge parameters.

   This needs a decision before the 50-step CUDA determinism probe runs.
5. **Not proposed, but noted:** the gauge parameters could be removed at the
   source, for example by dropping the A head's last-layer bias, which centring
   already makes a no-op. That changes checkpoints, so it is out of scope here.

## BC: Triton needs a C compiler

torch 2.14 registers a Triton override for `aten::bmm` outer products on CUDA,
in `torch/_native/ops/bmm_outer_product`. The actor's einsum backward hits it.
torch's own switch is **`TORCH_DISABLE_NATIVE_JIT=1`**:
`torch/_native/common_utils.check_native_jit_disabled`, documented in its
docstring as the single point that disables native DSL ops globally. It is read
and cached once while `import torch` runs, so it must be set in the process
environment, not in Python after torch is imported. When set, no `torch._native`
override is registered, and `aten::bmm` stays on stock cuBLAS. It also disables
the other `torch._native` DSL overrides (cuteDSL, nvmath, helion), which need
optional packages.

I found it by reading the installed torch 2.14.1 source. I could not exercise it
on CUDA locally, because the CPU build registers no override. The diagnostic
sets it by default and, with `--probe-native`, runs a bmm outer-product backward
in subprocesses with and without the variable, to show the effect on the VM.
If it works, set it in the GPU launch environment for `pretrain-bc`. I prepared
no gcc bundle, because the switch exists. I can prepare one if the probe shows
the switch is ineffective.
