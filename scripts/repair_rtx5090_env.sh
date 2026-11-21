#!/usr/bin/env bash
# Repair an existing infinity-modulation environment for RTX 5090 / sm_120.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -n "${INFINITY_PYTHON:-}" ]]; then
  PYTHON_BIN="$INFINITY_PYTHON"
elif [[ -x /venv/infinity-modulation/bin/python ]]; then
  PYTHON_BIN=/venv/infinity-modulation/bin/python
elif command -v conda >/dev/null 2>&1; then
  PYTHON_BIN="$(conda info --base)/envs/infinity-modulation/bin/python"
else
  PYTHON_BIN=""
fi
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || {
  echo "Cannot find infinity-modulation Python; set INFINITY_PYTHON explicitly." >&2
  exit 1
}
command -v nvcc >/dev/null 2>&1 || {
  echo "CUDA toolkit/nvcc is required to build FlashAttention for sm_120." >&2
  exit 1
}

compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')"
[[ "$compute_cap" == "12.0" ]] || {
  echo "This repair script is only for sm_120; detected compute capability $compute_cap." >&2
  exit 1
}
nvcc_release="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | tail -n1)"
"$PYTHON_BIN" - "$nvcc_release" <<'PY'
import sys
major, minor = map(int, sys.argv[1].split("."))
if (major, minor) < (12, 8):
    raise SystemExit(f"CUDA toolkit >= 12.8 is required for sm_120; found {sys.argv[1]}")
PY

TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
FLASH_ATTENTION_VERSION="${FLASH_ATTENTION_VERSION:-2.8.3}"
MAX_JOBS="${FLASH_ATTENTION_MAX_JOBS:-2}"
nvcc_major="${nvcc_release%%.*}"
if [[ -z "${TORCH_INDEX_URL:-}" ]]; then
  if (( nvcc_major >= 13 )); then
    TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
  else
    TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
  fi
fi

echo "[repair] installing PyTorch ${TORCH_VERSION} from ${TORCH_INDEX_URL}"
"$PYTHON_BIN" -m pip uninstall -y flash-attn torch torchvision || true
"$PYTHON_BIN" -m pip install --upgrade \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
  --index-url "$TORCH_INDEX_URL"
"$PYTHON_BIN" -m pip install --upgrade ninja packaging psutil

echo "[repair] building/installing FlashAttention ${FLASH_ATTENTION_VERSION} for sm_120"
env MAX_JOBS="$MAX_JOBS" FLASH_ATTN_CUDA_ARCHS=120 \
  "$PYTHON_BIN" -m pip install --no-build-isolation --no-deps \
  "flash-attn==${FLASH_ATTENTION_VERSION}"

"$PYTHON_BIN" - <<'PY'
import torch
import flash_attn
from flash_attn import flash_attn_varlen_kvpacked_func

assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.get_device_capability() == (12, 0)
assert "sm_120" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
value = (torch.ones(1, device="cuda") + 1).item()
q = torch.randn(2, 2, 128, device="cuda", dtype=torch.bfloat16)
kv = torch.randn(3, 2, 2, 128, device="cuda", dtype=torch.bfloat16)
cu_q = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
cu_k = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
out = flash_attn_varlen_kvpacked_func(
    q=q, kv=kv, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
    max_seqlen_q=2, max_seqlen_k=3, dropout_p=0.0,
)
torch.cuda.synchronize()
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("architectures:", torch.cuda.get_arch_list())
print("flash_attn:", flash_attn.__version__)
print("CUDA smoke:", value)
print("FlashAttention CA smoke:", tuple(out.shape))
PY

echo "[repair] complete; rerun scripts/run_fat_dog_grey_cat_spatial_test.sh"
