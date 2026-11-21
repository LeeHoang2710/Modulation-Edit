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
compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')"
[[ "$compute_cap" == "12.0" ]] || {
  echo "This repair script is only for sm_120; detected compute capability $compute_cap." >&2
  exit 1
}

TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
FLASH_ATTENTION_VERSION="${FLASH_ATTENTION_VERSION:-2.8.3}"
FLASH_ATTENTION_WHEEL="${FLASH_ATTENTION_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"

# Refuse to clean while an earlier source build is still running. Interrupt it
# with Ctrl-C first so its temporary directories are no longer in use.
if pgrep -af 'pip.*flash-attn|ninja.*flash|nvcc.*flash' >/dev/null 2>&1; then
  echo "A FlashAttention source build is still running. Press Ctrl-C in that terminal, then rerun this script." >&2
  pgrep -af 'pip.*flash-attn|ninja.*flash|nvcc.*flash' >&2 || true
  exit 1
fi

echo "[repair] removing abandoned pip build directories and download cache"
"$PYTHON_BIN" -m pip cache purge >/dev/null 2>&1 || true
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import shutil
import tempfile

root = Path(tempfile.gettempdir()).resolve()
prefixes = ("pip-build-env-", "pip-install-", "pip-ephem-wheel-cache-", "pip-unpack-")
for path in root.iterdir():
    if path.is_dir() and path.name.startswith(prefixes):
        print("removing", path)
        shutil.rmtree(path)
PY

echo "[repair] installing PyTorch ${TORCH_VERSION} from ${TORCH_INDEX_URL}"
"$PYTHON_BIN" -m pip uninstall -y flash-attn torch torchvision || true
"$PYTHON_BIN" -m pip install --no-cache-dir --upgrade \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
  --index-url "$TORCH_INDEX_URL"
"$PYTHON_BIN" -m pip install --no-cache-dir --upgrade ninja packaging psutil

echo "[repair] installing prebuilt FlashAttention ${FLASH_ATTENTION_VERSION} wheel (no compilation)"
"$PYTHON_BIN" -m pip install --no-cache-dir --no-deps "$FLASH_ATTENTION_WHEEL"

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
