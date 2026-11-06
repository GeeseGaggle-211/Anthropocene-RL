#!/usr/bin/env bash
# Build the offline wheelhouse for the r5 GPU fitting VM. Run ON the existing CPU VM
# (anthropocene-trainer-60), which has internet access. PREPARED, NOT RUN.
#
# The GPU VM has no external IP and no NAT, so bootstrap.sh installs only from this
# wheelhouse (`pip install --no-index --find-links`). This script:
#   1. downloads CUDA torch ${TORCH_VERSION} (cu128 by default: a cu12x build that the
#      L4's NVIDIA 580 driver, CUDA 13 capable, runs) with ALL of its dependencies
#      (nvidia-* CUDA libraries, triton, sympy, networkx, jinja2, fsspec, filelock, ...),
#      plus the repo's runtime dependencies and pytest, for CPython ${PY_VERSION} on
#      Ubuntu 24.04 x86_64 (manylinux wheels only, no source builds);
#   2. proves the set is complete: installs it with --no-index into a scratch venv;
#   3. writes requirements.txt and MANIFEST.sha256 (sha256 of every file);
#   4. uploads to gs://anthropocene-rl-datasets-508912/model9/r5/wheelhouse/.
#
# Usage (CPU VM):   bash build-wheelhouse.sh            # build + verify + upload
#                   UPLOAD=0 bash build-wheelhouse.sh   # build + verify only
# Optional env: TORCH_VERSION (2.14.1), CUDA_TAG (cu128; cu126 or cu130 if cu128 lacks
# the wheel), PY_VERSION (3.12), WORK (~/wheelhouse-build), DEST, PYTHON (python3.12).
set -euo pipefail

TORCH_VERSION="${TORCH_VERSION:-2.14.1}"
CUDA_TAG="${CUDA_TAG:-cu128}"
PY_VERSION="${PY_VERSION:-3.12}"
PYTHON="${PYTHON:-python${PY_VERSION}}"
WORK="${WORK:-$HOME/wheelhouse-build}"
DEST="${DEST:-gs://anthropocene-rl-datasets-508912/model9/r5/wheelhouse/}"
UPLOAD="${UPLOAD:-1}"
WH="$WORK/wheelhouse"

# The learner's pinned runtime dependencies (pyproject.toml) and pytest. Keep identical to
# the pins the CPU VM runs; torch is pinned separately above.
RUNTIME_DEPS=(numpy==2.4.6 orjson==3.13.0 pydantic==2.14.1 PyYAML==6.0.3
              "typing-extensions>=4.12,<5" pytest==9.1.1)

mkdir -p "$WORK"
rm -rf "$WH" && mkdir -p "$WH"
"$PYTHON" -m venv "$WORK/dl-venv"
DLPIP=("$WORK/dl-venv/bin/python" -m pip)
"${DLPIP[@]}" install --quiet --upgrade pip

# Target platform: the L4 image is Ubuntu 24.04 (glibc 2.39), x86_64, CPython 3.12.
# Several manylinux tags are accepted, newest first.
# Every glibc tag the image (glibc 2.39) can load. NCCL/cuDNN wheels use tags such as
# manylinux_2_18, which a short list misses (the first build failed resolving nvidia-nccl).
PLAT=(--platform manylinux2014_x86_64)
for m in $(seq 39 -1 17); do PLAT+=(--platform "manylinux_2_${m}_x86_64"); done
TARGET=("${PLAT[@]}" --python-version "$PY_VERSION" --implementation cp --only-binary=:all:)

# 1a. torch + every dependency in one resolution: torch comes from the PyTorch CUDA index,
#     everything else (nvidia-*, triton, sympy, ...) from PyPI.
"${DLPIP[@]}" download -d "$WH" "${TARGET[@]}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
  --extra-index-url "https://pypi.org/simple" \
  "torch==${TORCH_VERSION}"

# 1b. runtime dependencies and pytest (PyPI).
"${DLPIP[@]}" download -d "$WH" "${TARGET[@]}" "${RUNTIME_DEPS[@]}"

# 3a. requirements.txt (read by bootstrap.sh; torch is requested by name there).
printf '%s\n' "${RUNTIME_DEPS[@]}" > "$WH/requirements.txt"

# 2. Prove completeness: a clean venv must install torch + requirements with no network
#    index (this machine has internet, so --no-index is what makes the test meaningful).
rm -rf "$WORK/check-venv"
"$PYTHON" -m venv "$WORK/check-venv"
"$WORK/check-venv/bin/python" -m pip install --no-index --find-links "$WH" \
  "torch==${TORCH_VERSION}" -r "$WH/requirements.txt"
"$WORK/check-venv/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
assert torch.version.cuda is not None, "wheelhouse torch is a CPU build"
PY

# 3b. sha256 manifest (relative paths, verified on the GPU VM before installing).
(cd "$WH" && find . -maxdepth 1 -type f ! -name MANIFEST.sha256 -printf '%P\n' | sort \
  | xargs sha256sum > MANIFEST.sha256)
du -sh "$WH"; wc -l "$WH/MANIFEST.sha256"
# The CPU VM's own driver-independent version note, for the receipt:
echo "torch==${TORCH_VERSION} ${CUDA_TAG} cp${PY_VERSION/./} manylinux x86_64 built $(date -u +%FT%TZ)" \
  > "$WH/BUILD-INFO.txt"
(cd "$WH" && sha256sum BUILD-INFO.txt >> MANIFEST.sha256)

# 4. Upload. rsync is incremental; --delete-unmatched-destination-objects keeps the bucket
#    prefix identical to this build, so no stale wheel can be picked up by find-links.
if [ "$UPLOAD" = "1" ]; then
  gcloud storage rsync -r --delete-unmatched-destination-objects "$WH" "$DEST"
  gcloud storage ls -l "${DEST}MANIFEST.sha256"
  echo "uploaded $(wc -l < "$WH/MANIFEST.sha256") files to $DEST"
else
  echo "UPLOAD=0: built and verified only; not uploaded"
fi
