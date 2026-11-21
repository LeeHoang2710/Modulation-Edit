#!/usr/bin/env bash
set -euo pipefail

# Create a clean Miniconda environment for the original Modulation-Edit
# AdaLN-offset experiments on Vast.ai.
# Usage:
#   bash scripts/setup_vastai_env.sh
#   INFINITY_CONDA_ENV=my-infinity bash scripts/setup_vastai_env.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${INFINITY_CONDA_ENV:-infinity-modulation}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
FLASH_ATTENTION_VERSION="${FLASH_ATTENTION_VERSION:-2.8.3}"
FLASH_ATTENTION_WHEEL="${FLASH_ATTENTION_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found. Install Miniconda in the Vast.ai instance first." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "Using existing conda environment: ${ENV_NAME}"
else
    echo "Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

echo "Installing PyTorch ${TORCH_VERSION} and torchvision ${TORCHVISION_VERSION} from ${TORCH_INDEX_URL}"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel
conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

# requirements.txt documents the legacy stack and pins torch/flash_attn. The
# environment script owns those GPU packages so a later pip install cannot
# silently downgrade the Blackwell-compatible wheel.
echo "Installing project requirements without GPU-stack pins"
FILTERED_REQUIREMENTS="$(mktemp)"
trap 'rm -f "${FILTERED_REQUIREMENTS}"' EXIT
sed -E '/^[[:space:]]*(torch|torchvision|flash[_-]attn)([<=>[:space:]]|$)/d' \
    "${REPO_DIR}/requirements.txt" > "${FILTERED_REQUIREMENTS}"
conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir -r "${FILTERED_REQUIREMENTS}"
conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir \
    "huggingface_hub" \
    "ninja" \
    "packaging" \
    "psutil" \
    "opencv-fixer==0.2.5"

# Infinity uses varlen FlashAttention for prompt CA-KV. Use the upstream
# Python-3.12/Torch-2.9/CUDA-12 wheel; never fall back to a multi-gigabyte local
# CUDA source build on a rented Vast volume.
echo "Installing prebuilt FlashAttention ${FLASH_ATTENTION_VERSION} wheel"
conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir --no-deps \
    "${FLASH_ATTENTION_WHEEL}"

echo "Checking the environment"
conda run -n "${ENV_NAME}" python - <<'PY'
import torch
import transformers
import flash_attn
from flash_attn import flash_attn_varlen_kvpacked_func

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("flash_attn:", flash_attn.__version__)
capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
architectures = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
print("capability:", capability)
print("compiled_architectures:", architectures)
if capability == (12, 0) and "sm_120" not in architectures:
    raise SystemExit("Installed PyTorch wheel does not contain sm_120 support")
if torch.cuda.is_available():
    print("cuda_smoke:", (torch.ones(1, device="cuda") + 1).item())
    q = torch.randn(2, 2, 128, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(3, 2, 2, 128, device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    out = flash_attn_varlen_kvpacked_func(
        q=q, kv=kv, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=2, max_seqlen_k=3, dropout_p=0.0,
    )
    torch.cuda.synchronize()
    print("flash_attn_ca_smoke:", tuple(out.shape))
PY

echo
echo "Environment setup completed. Activate it with:"
echo "  conda activate ${ENV_NAME}"
echo "Then download the model weights with:"
echo "  bash scripts/download_vastai_weights.sh"
