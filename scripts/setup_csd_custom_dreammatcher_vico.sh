#!/usr/bin/env bash
# Install the two isolated environments used by the three five-seed baselines.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

git submodule update --init --recursive models/CSD-VAR models/Custom-Diffusion models/DreamMatcher
bash scripts/setup_csd_var_vico.sh

ENV_NAME="${PERSONALIZATION_ENV_NAME:-benchmark-personalization}"
CUDA_WHL="${PERSONALIZATION_CUDA_WHL:-cu118}"
command -v conda >/dev/null 2>&1 || { echo "Conda is required." >&2; exit 1; }
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    conda create --yes --name "$ENV_NAME" python=3.10 pip
fi
PY=(conda run --no-capture-output -n "$ENV_NAME" python)
"${PY[@]}" -m pip install --upgrade 'pip<24.1' setuptools wheel
"${PY[@]}" -m pip install \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url "https://download.pytorch.org/whl/${CUDA_WHL}"
"${PY[@]}" -m pip install \
    accelerate==0.24.1 transformers==4.31.0 huggingface-hub==0.17.3 \
    safetensors==0.4.5 pytorch-lightning==1.9.5 importlib-metadata einops matplotlib jmespath \
    opencv-python-headless pillow==9.5.0 gdown omegaconf scipy
"${PY[@]}" -m pip install --no-deps -e models/DreamMatcher/diffusers
"${PY[@]}" -m pip install 'git+https://github.com/openai/CLIP.git@main'
"${PY[@]}" -m pip check
"${PY[@]}" - <<'PY'
import torch, torchvision, transformers, diffusers, gdown
assert torch.cuda.is_available(), "CUDA is not available in benchmark-personalization"
print("[ready] torch", torch.__version__, "diffusers", diffusers.__version__, "GPU", torch.cuda.get_device_name())
PY
echo "[ready] environments: benchmark-csdvar and ${ENV_NAME}"
