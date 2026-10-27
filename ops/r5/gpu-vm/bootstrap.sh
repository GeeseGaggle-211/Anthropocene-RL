#!/usr/bin/env bash
# Bootstrap the r5 GPU fitting VM. Run ON the GPU VM (as the login user) after
# create-gpu-vm.sh. PREPARED, NOT RUN. Idempotent: safe to rerun after a Spot preemption.
#
# Steps
#   1. GPU check (nvidia-smi; the accelerator image ships the NVIDIA 580 driver).
#   2. Code: a `git archive` tarball of feature/r5-integration, fetched from GCS
#      (CODE_TARBALL=gs://...) or already copied with `gcloud compute scp --tunnel-through-iap`
#      (CODE_TARBALL=/path/on/vm). Never a deploy key or a git clone. Its sha256 is checked
#      when CODE_SHA256 is given.
#   3. venv with CUDA torch at the CPU VM's version (torch==2.14.1, pinned in pyproject.toml)
#      plus the learner's pinned base dependencies and pytest, installed ONLY from the
#      wheelhouse that build-wheelhouse.sh staged in GCS (`pip install --no-index
#      --find-links`). The VM has no external IP and no NAT; only Private Google Access
#      reaches GCS. There is no online fallback: a missing wheel is a hard error.
#   4. r5 inputs from GCS: datasets, payloads, normalizers, control report, checkpoints.
#   5. Smoke: nvidia-smi, torch.cuda, then the CPU-vs-CUDA parity tests
#      (tests/test_labor_device.py: they skip without a GPU, so here they must all RUN).
#
# Required:
#   CODE_TARBALL   gs://anthropocene-rl-datasets-508912/model9/r5/code/r5-int-rl-<sha>.tar.gz
#                  or a local path
# Optional:
#   CODE_SHA256    expected sha256 of the tarball
#   R5_PREFIX      gs://anthropocene-rl-datasets-508912/model9/r5/inputs   (staged by the CPU VM)
#   WHEELHOUSE     gs://anthropocene-rl-datasets-508912/model9/r5/wheelhouse/ (default)
#   TORCH_VERSION  2.14.1
#   ROOT           $HOME/r5gpu
set -euo pipefail

# torch 2.14's torch._native Triton kernels compile with a C compiler at first use; the VM
# has none. Read once at `import torch`, inherited by every child. labor_device.configure
# refuses CUDA without it. Export it in any other shell that runs CUDA fits or tests.
export TORCH_DISABLE_NATIVE_JIT=1

ROOT="${ROOT:-$HOME/r5gpu}"
CODE_DIR="$ROOT/code/rl"
VENV="$ROOT/venv"
DATA="$ROOT/inputs"
LOG="$ROOT/bootstrap.log"
TORCH_VERSION="${TORCH_VERSION:-2.14.1}"
WHEELHOUSE="${WHEELHOUSE:-gs://anthropocene-rl-datasets-508912/model9/r5/wheelhouse/}"
R5_PREFIX="${R5_PREFIX:-gs://anthropocene-rl-datasets-508912/model9/r5/inputs}"
: "${CODE_TARBALL:?set CODE_TARBALL (gs:// URL or local path of the git archive tarball)}"

mkdir -p "$ROOT" "$DATA" "$(dirname "$CODE_DIR")"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "== $(date -u +%FT%TZ) $*"; }

# 1. GPU ---------------------------------------------------------------------------------
step "GPU"
if ! nvidia-smi; then
  echo "nvidia-smi failed: the driver is not loaded (Secure Boot + unsigned module?)." >&2
  echo "See README 'Risks: driver'. Not continuing." >&2
  exit 2
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# 2. Code --------------------------------------------------------------------------------
step "code"
tarball="$ROOT/code/rl.tar.gz"
case "$CODE_TARBALL" in
  gs://*) gcloud storage cp "$CODE_TARBALL" "$tarball" ;;
  *) [ "$CODE_TARBALL" = "$tarball" ] || cp "$CODE_TARBALL" "$tarball" ;;
esac
if [ -n "${CODE_SHA256:-}" ]; then
  echo "${CODE_SHA256}  $tarball" | sha256sum -c -
fi
rm -rf "$CODE_DIR.new" && mkdir -p "$CODE_DIR.new"
tar -xzf "$tarball" -C "$CODE_DIR.new" --strip-components=1
rm -rf "$CODE_DIR" && mv "$CODE_DIR.new" "$CODE_DIR"
sha256sum "$tarball" | tee "$ROOT/code/rl.tar.gz.sha256"

# 3. venv --------------------------------------------------------------------------------
step "venv"
PY="${PYTHON:-python3}"
"$PY" --version
if [ ! -x "$VENV/bin/python" ]; then
  # No apt fallback: the VM has no internet. The accelerator image ships python3-venv.
  "$PY" -m venv "$VENV"
fi
pip=("$VENV/bin/python" -m pip)
have_torch() {
  "$VENV/bin/python" - "$TORCH_VERSION" <<'PY'
import sys, torch
ok = torch.__version__.split("+")[0] == sys.argv[1] and torch.version.cuda is not None
print("torch", torch.__version__, "cuda", torch.version.cuda)
sys.exit(0 if ok else 1)
PY
}
if ! have_torch 2>/dev/null; then
  mkdir -p "$ROOT/wheelhouse"
  gcloud storage rsync -r "$WHEELHOUSE" "$ROOT/wheelhouse"
  # Verify every wheel against the manifest before installing anything.
  (cd "$ROOT/wheelhouse" && sha256sum -c MANIFEST.sha256)
  # Wheelhouse only: --no-index means no PyPI, no download.pytorch.org, no fallback.
  "${pip[@]}" install --no-index --find-links "$ROOT/wheelhouse"     "torch==$TORCH_VERSION" -r "$ROOT/wheelhouse/requirements.txt"
  have_torch
fi
"${pip[@]}" freeze > "$ROOT/venv-freeze.txt"

# 4. r5 inputs ---------------------------------------------------------------------------
step "inputs from $R5_PREFIX"
# Expected layout (staged from the CPU VM, see README "Data staging"):
#   inputs/normalizer.pt inputs/privileged-normalizer.pt inputs/control-report.json
#   inputs/fit-data/<dir>... inputs/cal-data/ inputs/fit-payloads/ inputs/reserved-payloads/
#   inputs/bc-data/<dir>...  inputs/checkpoints/   (BC start checkpoint, critic receipts)
# rsync is incremental: a rerun after preemption only fetches what is missing.
gcloud storage rsync -r "$R5_PREFIX" "$DATA"
du -sh "$DATA"/* 2>/dev/null || true

# 5. Smoke -------------------------------------------------------------------------------
step "smoke"
cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR/src"
"$VENV/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
from anthropocene_rl.autonomous_v7 import labor_device as ld
device = ld.setup("cuda")
print(ld.configure(device))
x = torch.randn(1024, 1024, device=device)
torch.cuda.synchronize()
print("matmul ok", float((x @ x).abs().mean()))
print("free/total GiB", [round(v / 2**30, 2) for v in torch.cuda.mem_get_info()])
PY
# The parity tests must RUN here (no 's'): -rs lists any skip; a skip is a failure.
"$VENV/bin/python" -m pytest -q -p no:cacheprovider -rs tests/test_labor_device.py \
  | tee "$ROOT/parity-smoke.txt"
if grep -q "skipped" "$ROOT/parity-smoke.txt"; then
  echo "parity tests were skipped: CUDA not visible to pytest" >&2; exit 4
fi
# CPU regression on the core fit tests (same code path as the CPU VM).
"$VENV/bin/python" -m pytest -q -p no:cacheprovider \
  tests/test_labor_stability.py tests/test_labor_r5_acceptance.py tests/test_labor_qol_head.py \
  | tail -3
step "bootstrap complete: code $(cut -c1-12 "$ROOT/code/rl.tar.gz.sha256"), venv $VENV, inputs $DATA"
